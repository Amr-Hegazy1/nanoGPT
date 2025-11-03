"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# ---- Recurrent Shared Weights Sampling ----
def sample_recurrent_depth(max_depth, scale_loss_by_n_layer=False, min_depth=1, peak=32):
    """Sample the number of recurrent steps to expand."""
    if max_depth < min_depth:
        raise ValueError(f"max_depth ({max_depth}) must be >= min_depth ({min_depth})")
    if scale_loss_by_n_layer:
        return np.random.randint(min_depth, max_depth + 1)
    # Samples from a log-normal distribution with params to match prior behaviour:
    # mean=32, median=29, mode=24. min=1, max=max_depth (default 32, legacy capped at 100)
    sigma = 0.435
    mu = np.log(peak) + sigma**2
    sample = np.random.lognormal(mu, sigma)
    return int(np.clip(round(sample), min_depth, max_depth))

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# tensorboard logging
tensorboard_log = True # disabled by default
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
share_parameters_across_layers = False
recurrent_shared_weights = False
recurrent_depth = 32
# MoE parameters
moe = False
moe_num_experts = 4
moe_top_k = 2
moe_hard_routing = False
share_moe_experts = False
scale_loss_by_n_layer = False
layer_dropout = 0.0
sticky_dropout = 0.0
learned_stopping = False
learned_stopping_warmup_steps = 0
learned_stopping_controller_weight = 0.0
learned_stopping_entropy_weight = 0.0
learned_stopping_target_depth = None
learned_stopping_temperature = 1.0
learned_stopping_min_prob = 1e-4
learned_stopping_use_threshold = False
learned_stopping_threshold = 0.5
attentive_stopping = False
attentive_stopping_warmup_steps = 0
attentive_stopping_controller_weight = 0.0
attentive_stopping_entropy_weight = 0.0
attentive_stopping_target_depth = None
attentive_stopping_temperature = 1.0
attentive_stopping_min_prob = 1e-4
attentive_stopping_use_threshold = False
attentive_stopping_threshold = 0.5
fixed_edge_blocks = False
n_layers_prelude = 1
n_layers_coda = 1
use_rmsnorm = False
log_correlation = False
recurrent_depth_peak = 32
recurrent_prelude_injection = False
recurrent_noise_mode = 'none'
recurrent_noise_std = 0.0
recurrent_noise_concat_dim = None
recurrent_extra_layernorm = False
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str)) and k != 'log_dist_peak']
for optional_key in ['learned_stopping_target_depth', 'attentive_stopping_target_depth', 'recurrent_noise_concat_dim', 'n_layers_prelude', 'n_layers_coda', 'log_correlation', 'recurrent_depth_peak', 'recurrent_prelude_injection']:
    if optional_key not in config_keys and optional_key in globals():
        config_keys.append(optional_key)
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    if tensorboard_log:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=out_dir)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  share_parameters_across_layers=share_parameters_across_layers,
                  recurrent_shared_weights=recurrent_shared_weights,
                  recurrent_depth=recurrent_depth,
                  moe=moe, moe_num_experts=moe_num_experts, moe_top_k=moe_top_k, moe_hard_routing=moe_hard_routing, share_moe_experts=share_moe_experts,
                  scale_loss_by_n_layer=scale_loss_by_n_layer,
                  layer_dropout=layer_dropout, sticky_dropout=sticky_dropout, learned_stopping=learned_stopping,
                  learned_stopping_warmup_steps=learned_stopping_warmup_steps,
                  learned_stopping_controller_weight=learned_stopping_controller_weight,
                  learned_stopping_entropy_weight=learned_stopping_entropy_weight,
                  learned_stopping_target_depth=learned_stopping_target_depth,
                  learned_stopping_temperature=learned_stopping_temperature,
                  learned_stopping_min_prob=learned_stopping_min_prob,
                  learned_stopping_use_threshold=learned_stopping_use_threshold,
                  learned_stopping_threshold=learned_stopping_threshold,
                  fixed_edge_blocks=fixed_edge_blocks,
                  use_rmsnorm=use_rmsnorm,
                  recurrent_noise_mode=recurrent_noise_mode,
                  recurrent_noise_std=recurrent_noise_std,
                  recurrent_noise_concat_dim=recurrent_noise_concat_dim,
                  recurrent_extra_layernorm=recurrent_extra_layernorm,
                  recurrent_prelude_injection=recurrent_prelude_injection,
                  attentive_stopping=attentive_stopping,
                  attentive_stopping_warmup_steps=attentive_stopping_warmup_steps,
                  attentive_stopping_controller_weight=attentive_stopping_controller_weight,
                  attentive_stopping_entropy_weight=attentive_stopping_entropy_weight,
                  attentive_stopping_target_depth=attentive_stopping_target_depth,
                  attentive_stopping_temperature=attentive_stopping_temperature,
                  attentive_stopping_min_prob=attentive_stopping_min_prob,
                  attentive_stopping_use_threshold=attentive_stopping_use_threshold,
                  attentive_stopping_threshold=attentive_stopping_threshold)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # also carry over shared-weights flag if present (default False for older checkpoints)
    if 'share_parameters_across_layers' in checkpoint_model_args:
        model_args['share_parameters_across_layers'] = checkpoint_model_args['share_parameters_across_layers']
    if 'recurrent_shared_weights' in checkpoint_model_args:
        model_args['recurrent_shared_weights'] = checkpoint_model_args['recurrent_shared_weights']
    if 'recurrent_depth' in checkpoint_model_args:
        model_args['recurrent_depth'] = checkpoint_model_args['recurrent_depth']
    if 'moe' in checkpoint_model_args:
        model_args['moe'] = checkpoint_model_args['moe']
    if 'moe_num_experts' in checkpoint_model_args:
        model_args['moe_num_experts'] = checkpoint_model_args['moe_num_experts']
    if 'moe_top_k' in checkpoint_model_args:
        model_args['moe_top_k'] = checkpoint_model_args['moe_top_k']
    if 'moe_hard_routing' in checkpoint_model_args:
        model_args['moe_hard_routing'] = checkpoint_model_args['moe_hard_routing']
    if 'share_moe_experts' in checkpoint_model_args:
        model_args['share_moe_experts'] = checkpoint_model_args['share_moe_experts']
    if 'scale_loss_by_n_layer' in checkpoint_model_args:
        model_args['scale_loss_by_n_layer'] = checkpoint_model_args['scale_loss_by_n_layer']
    if 'layer_dropout' in checkpoint_model_args:
        model_args['layer_dropout'] = checkpoint_model_args['layer_dropout']
    if 'sticky_dropout' in checkpoint_model_args:
        model_args['sticky_dropout'] = checkpoint_model_args['sticky_dropout']
    if 'learned_stopping' in checkpoint_model_args:
        model_args['learned_stopping'] = checkpoint_model_args['learned_stopping']
    if 'learned_stopping_warmup_steps' in checkpoint_model_args:
        model_args['learned_stopping_warmup_steps'] = checkpoint_model_args['learned_stopping_warmup_steps']
    if 'learned_stopping_controller_weight' in checkpoint_model_args:
        model_args['learned_stopping_controller_weight'] = checkpoint_model_args['learned_stopping_controller_weight']
    if 'learned_stopping_entropy_weight' in checkpoint_model_args:
        model_args['learned_stopping_entropy_weight'] = checkpoint_model_args['learned_stopping_entropy_weight']
    if 'learned_stopping_target_depth' in checkpoint_model_args:
        model_args['learned_stopping_target_depth'] = checkpoint_model_args['learned_stopping_target_depth']
    if 'learned_stopping_temperature' in checkpoint_model_args:
        model_args['learned_stopping_temperature'] = checkpoint_model_args['learned_stopping_temperature']
    if 'learned_stopping_min_prob' in checkpoint_model_args:
        model_args['learned_stopping_min_prob'] = checkpoint_model_args['learned_stopping_min_prob']
    if 'learned_stopping_use_threshold' in checkpoint_model_args:
        model_args['learned_stopping_use_threshold'] = checkpoint_model_args['learned_stopping_use_threshold']
    if 'learned_stopping_threshold' in checkpoint_model_args:
        model_args['learned_stopping_threshold'] = checkpoint_model_args['learned_stopping_threshold']
    if 'fixed_edge_blocks' in checkpoint_model_args:
        model_args['fixed_edge_blocks'] = checkpoint_model_args['fixed_edge_blocks']
    if 'use_rmsnorm' in checkpoint_model_args:
        model_args['use_rmsnorm'] = checkpoint_model_args['use_rmsnorm']
    if 'recurrent_noise_mode' in checkpoint_model_args:
        model_args['recurrent_noise_mode'] = checkpoint_model_args['recurrent_noise_mode']
    if 'recurrent_noise_std' in checkpoint_model_args:
        model_args['recurrent_noise_std'] = checkpoint_model_args['recurrent_noise_std']
    if 'recurrent_noise_concat_dim' in checkpoint_model_args:
        model_args['recurrent_noise_concat_dim'] = checkpoint_model_args['recurrent_noise_concat_dim']
    if 'recurrent_extra_layernorm' in checkpoint_model_args:
        model_args['recurrent_extra_layernorm'] = checkpoint_model_args['recurrent_extra_layernorm']
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
    model_args['share_parameters_across_layers'] = getattr(model.config, 'share_parameters_across_layers', False)
    model_args['recurrent_shared_weights'] = getattr(model.config, 'recurrent_shared_weights', False)
    model_args['recurrent_depth'] = getattr(model.config, 'recurrent_depth', 32)
    model_args['moe'] = getattr(model.config, 'moe', False)
    model_args['moe_num_experts'] = getattr(model.config, 'moe_num_experts', 4)
    model_args['moe_top_k'] = getattr(model.config, 'moe_top_k', 2)
    model_args['moe_hard_routing'] = getattr(model.config, 'moe_hard_routing', False)
    model_args['share_moe_experts'] = getattr(model.config, 'share_moe_experts', False)
    model_args['scale_loss_by_n_layer'] = getattr(model.config, 'scale_loss_by_n_layer', False)
    model_args['layer_dropout'] = getattr(model.config, 'layer_dropout', 0.0)
    model_args['sticky_dropout'] = getattr(model.config, 'sticky_dropout', 0.0)
    model_args['learned_stopping'] = getattr(model.config, 'learned_stopping', False)
    model_args['learned_stopping_warmup_steps'] = getattr(model.config, 'learned_stopping_warmup_steps', 0)
    model_args['learned_stopping_controller_weight'] = getattr(model.config, 'learned_stopping_controller_weight', 0.0)
    model_args['learned_stopping_entropy_weight'] = getattr(model.config, 'learned_stopping_entropy_weight', 0.0)
    model_args['learned_stopping_target_depth'] = getattr(model.config, 'learned_stopping_target_depth', None)
    model_args['learned_stopping_temperature'] = getattr(model.config, 'learned_stopping_temperature', 1.0)
    model_args['learned_stopping_min_prob'] = getattr(model.config, 'learned_stopping_min_prob', 1e-4)
    model_args['learned_stopping_use_threshold'] = getattr(model.config, 'learned_stopping_use_threshold', False)
    model_args['learned_stopping_threshold'] = getattr(model.config, 'learned_stopping_threshold', 0.5)
    model_args['fixed_edge_blocks'] = getattr(model.config, 'fixed_edge_blocks', False)
    model_args['use_rmsnorm'] = getattr(model.config, 'use_rmsnorm', False)
    model_args['recurrent_noise_mode'] = getattr(model.config, 'recurrent_noise_mode', 'none')
    model_args['recurrent_noise_std'] = getattr(model.config, 'recurrent_noise_std', 0.0)
    model_args['recurrent_noise_concat_dim'] = getattr(model.config, 'recurrent_noise_concat_dim', None)
    model_args['recurrent_extra_layernorm'] = getattr(model.config, 'recurrent_extra_layernorm', False)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    find_unused_parameters = config['moe'] or config['recurrent_shared_weights']
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=find_unused_parameters)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                n = recurrent_depth if recurrent_shared_weights else None
                logits, loss, _ = model(X, Y, n=n)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out, logits

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it is past the decay phase, return min learning rate
    if it >= lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    if warmup_iters == lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
sampled_depths = []
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses, val_logits = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "val/perplexity": torch.exp(losses['val']),
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            }
            if len(sampled_depths) > 0:
                log_dict["train/sampled_depth_distribution"] = wandb.Histogram(sampled_depths)
                sampled_depths = []
            # Add stopping metrics if they exist
            stopping_metrics = getattr(raw_model, 'stopping_metrics', None)
            if stopping_metrics:
                log_dict.update({f"train/stopping/{k}": v for k, v in stopping_metrics.items()})
            
            attentive_stopping_metrics = getattr(raw_model, 'attentive_stopping_metrics', None)
            if attentive_stopping_metrics:
                log_dict.update({f"train/attentive_stopping/{k}": v for k, v in attentive_stopping_metrics.items()})

            if log_correlation:
                # calculate token embedding correlation
                x_c = val_logits - val_logits.mean(dim=-1, keepdim=True)
                normed_x = x_c / x_c.norm(dim=-1, keepdim=True)
                token_corr = (normed_x @ normed_x.transpose(-2, -1)).mean()
                log_dict['val/token_correlation'] = token_corr.item()

                # calculate batch correlation
                # Reshape the logits tensor to 2D
                logits_2d = val_logits.view(-1, val_logits.size(-1))
                # Calculate the correlation matrix
                corr_matrix = torch.corrcoef(logits_2d)
                # Extract the upper triangular part of the matrix, excluding the diagonal
                upper_tri = torch.triu(corr_matrix, diagonal=1)
                # Calculate the mean of the correlations
                mean_corr = upper_tri.sum() / (upper_tri.numel() - val_logits.size(0))
                log_dict['val/batch_correlation'] = mean_corr.item()

            wandb.log(log_dict)
        if tensorboard_log and master_process:
            writer.add_scalar('train/loss', losses['train'], iter_num)
            writer.add_scalar('val/loss', losses['val'], iter_num)
            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('mfu', running_mfu*100, iter_num)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            n = sample_recurrent_depth(recurrent_depth, scale_loss_by_n_layer, peak=recurrent_depth_peak) if recurrent_shared_weights else None
            if n is not None:
                sampled_depths.append(n)
            logits, loss, num_expanded_layers = model(X, Y, n=n)
            if scale_loss_by_n_layer and num_expanded_layers is not None:
                if n is not None and n > 0:
                    ref_layers = n
                else:
                    ref_layers = getattr(model.config, 'recurrent_depth', None)
                    if not ref_layers or ref_layers <= 0:
                        ref_layers = getattr(model.config, 'n_layer', None)
                if not ref_layers or ref_layers <= 0:
                    ref_layers = 1
                scale = num_expanded_layers / ref_layers
                loss = loss * scale
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")





        if tensorboard_log:
            writer.add_scalar('train/loss', lossf, iter_num)
            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('mfu', running_mfu*100, iter_num)
            # Add stopping metrics if they exist
            stopping_metrics = getattr(raw_model, 'stopping_metrics', None)
            if stopping_metrics:
                for k, v in stopping_metrics.items():
                    writer.add_scalar(f"train/stopping/{k}", v, iter_num)
            attentive_stopping_metrics = getattr(raw_model, 'attentive_stopping_metrics', None)
            if attentive_stopping_metrics:
                for k, v in attentive_stopping_metrics.items():
                    writer.add_scalar(f"train/attentive_stopping/{k}", v, iter_num)

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if tensorboard_log and master_process:
    writer.close()

if ddp:
    destroy_process_group()