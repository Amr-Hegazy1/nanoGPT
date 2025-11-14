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
import sys
import time
import math
import json
import pickle
import random
import shutil
import subprocess
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from curriculum import determine_recurrent_depth, parse_schedule_options

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
wandb_run_id = ''
wandb_resume = ''
wandb_init_timeout = 120.0
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
hard_attentive_stopping = False
hard_attentive_stopping_threshold = 0.5
oracle_stopping = False
oracle_bootstrap_checkpoint = ''
oracle_update_interval = 1000
oracle_max_depth = None
oracle_stop_weight = 1.0
oracle_difficulty_weight = 1.0
oracle_temperature = 1.0
oracle_min_prob = 1e-4
oracle_use_threshold = False
oracle_threshold = 0.5
stopping_tokenwise = False
fixed_edge_blocks = False
n_layers_prelude = 1
n_layers_coda = 1
use_rmsnorm = False
sandwich_norm = False
log_correlation = False
recurrent_depth_peak = 32
recurrent_prelude_injection = False
recurrent_prelude_injection_mode = 'add'
recurrent_noise_mode = 'none'
recurrent_noise_std = 0.0
recurrent_noise_concat_dim = None
recurrent_extra_layernorm = False
# curriculum scheduling for recurrent depth (ascending/descending/random)
recurrent_depth_schedule = 'random'  # options: 'random', 'ascending', 'descending'
recurrent_depth_schedule_interval = 0  # iterations per curriculum step; <=0 treated as 1
recurrent_depth_schedule_min_depth = 1
recurrent_depth_schedule_resample_prob = 0.0  # probability of falling back to random sampling
recurrent_depth_schedule_opts = ''  # JSON or comma-separated key=value pairs for advanced strategies
recurrent_depth_schedule_feedback_alpha = 0.05  # EMA smoothing for performance-aware schedules
recurrent_depth_schedule_difficulty_min_ratio = 0.5
recurrent_depth_schedule_difficulty_max_ratio = 1.5
# hyperparameter tuning (W&B sweeps)
hyperparameter_tuning = False
hyperparameter_tuning_trials = 10
hyperparameter_tuning_max_iters = 200
hyperparameter_tuning_resume = False
hyperparameter_tuning_result_path = ''
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
for optional_key in ['learned_stopping_target_depth', 'attentive_stopping_target_depth', 'oracle_bootstrap_checkpoint', 'oracle_max_depth',
                     'recurrent_noise_concat_dim', 'n_layers_prelude', 'n_layers_coda', 'log_correlation',
                     'recurrent_depth_peak', 'recurrent_prelude_injection', 'recurrent_prelude_injection_mode']:
    if optional_key not in config_keys and optional_key in globals():
        config_keys.append(optional_key)
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Hyperparameter tuning via local random search
if hyperparameter_tuning:
    rank_env = int(os.environ.get('RANK', -1))
    if rank_env not in (-1, 0):
        sys.exit(0)
    base_seed = 1337 + int(time.time())
    rng = random.Random(base_seed)
    os.makedirs(out_dir, exist_ok=True)

    skip_prefixes = (
        "--hyperparameter_tuning",
        "--hyperparameter_tuning_trials",
        "--hyperparameter_tuning_max_iters",
        "--hyperparameter_tuning_result_path",
        "--learning_rate",
        "--batch_size",
        "--weight_decay",
    )

    base_args: list[str] = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if any(arg.startswith(pref) for pref in skip_prefixes):
            # skip following token as well if the flag is provided without '='
            if "=" not in arg and (i + 1) < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        base_args.append(arg)
        i += 1

    def sample_log_uniform(min_val: float, max_val: float) -> float:
        return math.exp(rng.uniform(math.log(min_val), math.log(max_val)))

    trial_summaries: list[dict[str, float | int | str]] = []
    tuning_iters = int(hyperparameter_tuning_max_iters)
    if tuning_iters <= 0:
        tuning_iters = 1

    for trial_idx in range(hyperparameter_tuning_trials):
        trial_num = trial_idx + 1
        lr = sample_log_uniform(1e-5, 5e-4)
        wd = sample_log_uniform(1e-6, 1e-2)
        bs = int(rng.choice([1, 2, 3, 4]))
        trial_suffix = f"tuning_trial_{trial_num}_{int(time.time())}_{rng.randrange(10_000)}"
        trial_out_dir = os.path.join(out_dir, trial_suffix)
        result_path = os.path.join(trial_out_dir, "result.json")
        os.makedirs(trial_out_dir, exist_ok=True)

        run_args = list(base_args)
        run_args.extend([
            f"--learning_rate={lr}",
            f"--batch_size={bs}",
            f"--weight_decay={wd}",
            "--hyperparameter_tuning=False",
            "--hyperparameter_tuning_resume=True",
            f"--hyperparameter_tuning_result_path={result_path}",
            f"--out_dir={trial_out_dir}",
        ])
        if not any(a.startswith("--max_iters") for a in run_args):
            run_args.append(f"--max_iters={tuning_iters}")
        if not any(a.startswith("--eval_interval") for a in run_args):
            run_args.append(f"--eval_interval={tuning_iters}")
        if not any(a.startswith("--log_interval") for a in run_args):
            log_interval = max(1, tuning_iters // 5)
            run_args.append(f"--log_interval={log_interval}")
        run_args.append("--wandb_log=False")

        print(f"[Tuning] Trial {trial_num}/{hyperparameter_tuning_trials}: "
              f"lr={lr:.3e}, batch_size={bs}, weight_decay={wd:.3e}")

        result_data = None
        try:
            subprocess.run([sys.executable, __file__] + run_args, check=True)
            with open(result_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
        except subprocess.CalledProcessError as exc:
            print(f"[Tuning] Trial {trial_num} failed with exit code {exc.returncode}")
        except FileNotFoundError:
            print(f"[Tuning] Trial {trial_num} did not produce result file at {result_path}")
        except json.JSONDecodeError as exc:
            print(f"[Tuning] Trial {trial_num} produced invalid JSON: {exc}")
        else:
            val_loss = result_data.get("val_loss")
            if val_loss is None:
                print(f"[Tuning] Trial {trial_num} result missing val_loss")
            else:
                try:
                    val_loss = float(val_loss)
                except (TypeError, ValueError):
                    print(f"[Tuning] Trial {trial_num} val_loss is not numeric: {val_loss}")
                    val_loss = None
            if val_loss is not None:
                trial_summaries.append({
                    "trial": trial_num,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "weight_decay": wd,
                    "val_loss": val_loss,
                })
                print(f"[Tuning] Trial {trial_num} completed with val_loss={val_loss:.4f}")
        finally:
            try:
                if os.path.exists(result_path):
                    os.remove(result_path)
            except OSError:
                pass
            shutil.rmtree(trial_out_dir, ignore_errors=True)

    successful_trials = [t for t in trial_summaries if math.isfinite(t["val_loss"])]
    if not successful_trials:
        print("[Tuning] All trials failed; aborting.")
        sys.exit(1)

    best_trial = min(successful_trials, key=lambda t: t["val_loss"])
    print(f"[Tuning] Best trial #{best_trial['trial']} -> "
          f"val_loss={best_trial['val_loss']:.4f}, "
          f"lr={best_trial['learning_rate']:.3e}, "
          f"batch_size={int(best_trial['batch_size'])}, "
          f"weight_decay={best_trial['weight_decay']:.3e}")

    final_args = list(base_args)
    final_args.extend([
        f"--learning_rate={best_trial['learning_rate']}",
        f"--batch_size={int(best_trial['batch_size'])}",
        f"--weight_decay={best_trial['weight_decay']}",
        "--hyperparameter_tuning=False",
        "--hyperparameter_tuning_resume=False",
    ])
    if not any(a.startswith("--wandb_log") for a in final_args):
        final_args.append("--wandb_log=True")

    print("[Tuning] Launching full training run with best hyperparameters.")
    subprocess.run([sys.executable, __file__] + final_args, check=True)
    sys.exit(0)

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
schedule_options = parse_schedule_options(recurrent_depth_schedule_opts)
try:
    curriculum_feedback_alpha = float(recurrent_depth_schedule_feedback_alpha)
except (TypeError, ValueError):
    curriculum_feedback_alpha = 0.05
if not math.isfinite(curriculum_feedback_alpha) or curriculum_feedback_alpha <= 0.0:
    curriculum_feedback_alpha = 0.05
elif curriculum_feedback_alpha > 1.0:
    curriculum_feedback_alpha = 1.0
try:
    difficulty_min_ratio = float(recurrent_depth_schedule_difficulty_min_ratio)
except (TypeError, ValueError):
    difficulty_min_ratio = 0.5
try:
    difficulty_max_ratio = float(recurrent_depth_schedule_difficulty_max_ratio)
except (TypeError, ValueError):
    difficulty_max_ratio = difficulty_min_ratio + 1.0
if not math.isfinite(difficulty_min_ratio):
    difficulty_min_ratio = 0.5
if not math.isfinite(difficulty_max_ratio):
    difficulty_max_ratio = difficulty_min_ratio + 1.0
if difficulty_max_ratio <= difficulty_min_ratio:
    difficulty_max_ratio = difficulty_min_ratio + 1.0
difficulty_ratio_span = max(1e-6, difficulty_max_ratio - difficulty_min_ratio)
curriculum_feedback_metric: float | None = None
curriculum_difficulty_score: float | None = None
curriculum_loss_ema: float | None = None
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
                  sticky_dropout=sticky_dropout, learned_stopping=learned_stopping,
                  learned_stopping_warmup_steps=learned_stopping_warmup_steps,
                  learned_stopping_controller_weight=learned_stopping_controller_weight,
                  learned_stopping_entropy_weight=learned_stopping_entropy_weight,
                  learned_stopping_target_depth=learned_stopping_target_depth,
                  learned_stopping_temperature=learned_stopping_temperature,
                  learned_stopping_min_prob=learned_stopping_min_prob,
                  learned_stopping_use_threshold=learned_stopping_use_threshold,
                  learned_stopping_threshold=learned_stopping_threshold,
                  oracle_stopping=oracle_stopping,
                  oracle_bootstrap_checkpoint=oracle_bootstrap_checkpoint,
                  oracle_update_interval=oracle_update_interval,
                  oracle_max_depth=oracle_max_depth,
                  oracle_stop_weight=oracle_stop_weight,
                  oracle_difficulty_weight=oracle_difficulty_weight,
                  oracle_temperature=oracle_temperature,
                  oracle_min_prob=oracle_min_prob,
                  oracle_use_threshold=oracle_use_threshold,
                  oracle_threshold=oracle_threshold,
                  fixed_edge_blocks=fixed_edge_blocks,
                  use_rmsnorm=use_rmsnorm,
                  sandwich_norm=sandwich_norm,
                  recurrent_noise_mode=recurrent_noise_mode,
                  recurrent_noise_std=recurrent_noise_std,
                  recurrent_noise_concat_dim=recurrent_noise_concat_dim,
                  recurrent_extra_layernorm=recurrent_extra_layernorm,
                  recurrent_prelude_injection=recurrent_prelude_injection,
                  recurrent_prelude_injection_mode=recurrent_prelude_injection_mode,
                  attentive_stopping=attentive_stopping,
                  attentive_stopping_warmup_steps=attentive_stopping_warmup_steps,
                  attentive_stopping_controller_weight=attentive_stopping_controller_weight,
                  attentive_stopping_entropy_weight=attentive_stopping_entropy_weight,
                  attentive_stopping_target_depth=attentive_stopping_target_depth,
                  attentive_stopping_temperature=attentive_stopping_temperature,
                  attentive_stopping_min_prob=attentive_stopping_min_prob,
                  attentive_stopping_use_threshold=attentive_stopping_use_threshold,
                  attentive_stopping_threshold=attentive_stopping_threshold,
                  hard_attentive_stopping=hard_attentive_stopping,
                  hard_attentive_stopping_threshold=hard_attentive_stopping_threshold,
                  stopping_tokenwise=stopping_tokenwise)
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
    if 'sticky_dropout' in checkpoint_model_args:
        model_args['sticky_dropout'] = checkpoint_model_args['sticky_dropout']
    if 'stopping_tokenwise' in checkpoint_model_args:
        model_args['stopping_tokenwise'] = checkpoint_model_args['stopping_tokenwise']
    if 'stopping_tokenwise' in checkpoint_model_args:
        model_args['stopping_tokenwise'] = checkpoint_model_args['stopping_tokenwise']
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
    if 'oracle_stopping' in checkpoint_model_args:
        model_args['oracle_stopping'] = checkpoint_model_args['oracle_stopping']
    if 'oracle_bootstrap_checkpoint' in checkpoint_model_args:
        model_args['oracle_bootstrap_checkpoint'] = checkpoint_model_args['oracle_bootstrap_checkpoint']
    if 'oracle_update_interval' in checkpoint_model_args:
        model_args['oracle_update_interval'] = checkpoint_model_args['oracle_update_interval']
    if 'oracle_max_depth' in checkpoint_model_args:
        model_args['oracle_max_depth'] = checkpoint_model_args['oracle_max_depth']
    if 'oracle_stop_weight' in checkpoint_model_args:
        model_args['oracle_stop_weight'] = checkpoint_model_args['oracle_stop_weight']
    if 'oracle_difficulty_weight' in checkpoint_model_args:
        model_args['oracle_difficulty_weight'] = checkpoint_model_args['oracle_difficulty_weight']
    if 'oracle_temperature' in checkpoint_model_args:
        model_args['oracle_temperature'] = checkpoint_model_args['oracle_temperature']
    if 'oracle_min_prob' in checkpoint_model_args:
        model_args['oracle_min_prob'] = checkpoint_model_args['oracle_min_prob']
    if 'oracle_use_threshold' in checkpoint_model_args:
        model_args['oracle_use_threshold'] = checkpoint_model_args['oracle_use_threshold']
    if 'oracle_threshold' in checkpoint_model_args:
        model_args['oracle_threshold'] = checkpoint_model_args['oracle_threshold']
    if 'fixed_edge_blocks' in checkpoint_model_args:
        model_args['fixed_edge_blocks'] = checkpoint_model_args['fixed_edge_blocks']
    if 'use_rmsnorm' in checkpoint_model_args:
        model_args['use_rmsnorm'] = checkpoint_model_args['use_rmsnorm']
    if 'sandwich_norm' in checkpoint_model_args:
        model_args['sandwich_norm'] = checkpoint_model_args['sandwich_norm']
    if 'recurrent_noise_mode' in checkpoint_model_args:
        model_args['recurrent_noise_mode'] = checkpoint_model_args['recurrent_noise_mode']
    if 'recurrent_noise_std' in checkpoint_model_args:
        model_args['recurrent_noise_std'] = checkpoint_model_args['recurrent_noise_std']
    if 'recurrent_noise_concat_dim' in checkpoint_model_args:
        model_args['recurrent_noise_concat_dim'] = checkpoint_model_args['recurrent_noise_concat_dim']
    if 'recurrent_extra_layernorm' in checkpoint_model_args:
        model_args['recurrent_extra_layernorm'] = checkpoint_model_args['recurrent_extra_layernorm']
    if 'recurrent_prelude_injection' in checkpoint_model_args:
        model_args['recurrent_prelude_injection'] = checkpoint_model_args['recurrent_prelude_injection']
    if 'recurrent_prelude_injection_mode' in checkpoint_model_args:
        model_args['recurrent_prelude_injection_mode'] = checkpoint_model_args['recurrent_prelude_injection_mode']
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
    model.load_oracle_state(checkpoint.get('oracle_state'))
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
    model_args['sticky_dropout'] = getattr(model.config, 'sticky_dropout', 0.0)
    model_args['stopping_tokenwise'] = getattr(model.config, 'stopping_tokenwise', False)
    model_args['stopping_tokenwise'] = getattr(model.config, 'stopping_tokenwise', False)
    model_args['learned_stopping'] = getattr(model.config, 'learned_stopping', False)
    model_args['learned_stopping_warmup_steps'] = getattr(model.config, 'learned_stopping_warmup_steps', 0)
    model_args['learned_stopping_controller_weight'] = getattr(model.config, 'learned_stopping_controller_weight', 0.0)
    model_args['learned_stopping_entropy_weight'] = getattr(model.config, 'learned_stopping_entropy_weight', 0.0)
    model_args['learned_stopping_target_depth'] = getattr(model.config, 'learned_stopping_target_depth', None)
    model_args['learned_stopping_temperature'] = getattr(model.config, 'learned_stopping_temperature', 1.0)
    model_args['learned_stopping_min_prob'] = getattr(model.config, 'learned_stopping_min_prob', 1e-4)
    model_args['learned_stopping_use_threshold'] = getattr(model.config, 'learned_stopping_use_threshold', False)
    model_args['learned_stopping_threshold'] = getattr(model.config, 'learned_stopping_threshold', 0.5)
    model_args['oracle_stopping'] = getattr(model.config, 'oracle_stopping', False)
    model_args['oracle_bootstrap_checkpoint'] = getattr(model.config, 'oracle_bootstrap_checkpoint', '')
    model_args['oracle_update_interval'] = getattr(model.config, 'oracle_update_interval', 1000)
    model_args['oracle_max_depth'] = getattr(model.config, 'oracle_max_depth', None)
    model_args['oracle_stop_weight'] = getattr(model.config, 'oracle_stop_weight', 1.0)
    model_args['oracle_difficulty_weight'] = getattr(model.config, 'oracle_difficulty_weight', 1.0)
    model_args['oracle_temperature'] = getattr(model.config, 'oracle_temperature', 1.0)
    model_args['oracle_min_prob'] = getattr(model.config, 'oracle_min_prob', 1e-4)
    model_args['oracle_use_threshold'] = getattr(model.config, 'oracle_use_threshold', False)
    model_args['oracle_threshold'] = getattr(model.config, 'oracle_threshold', 0.5)
    model_args['fixed_edge_blocks'] = getattr(model.config, 'fixed_edge_blocks', False)
    model_args['use_rmsnorm'] = getattr(model.config, 'use_rmsnorm', False)
    model_args['sandwich_norm'] = getattr(model.config, 'sandwich_norm', False)
    model_args['recurrent_noise_mode'] = getattr(model.config, 'recurrent_noise_mode', 'none')
    model_args['recurrent_noise_std'] = getattr(model.config, 'recurrent_noise_std', 0.0)
    model_args['recurrent_noise_concat_dim'] = getattr(model.config, 'recurrent_noise_concat_dim', None)
    model_args['recurrent_extra_layernorm'] = getattr(model.config, 'recurrent_extra_layernorm', False)
    model_args['recurrent_prelude_injection'] = getattr(model.config, 'recurrent_prelude_injection', False)
    model_args['recurrent_prelude_injection_mode'] = getattr(model.config, 'recurrent_prelude_injection_mode', 'add')
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

    init_kwargs = {
        "project": wandb_project,
        "name": wandb_run_name,
    }
    run_id = wandb_run_id.strip() if isinstance(wandb_run_id, str) else wandb_run_id
    resume_mode = wandb_resume.strip() if isinstance(wandb_resume, str) else wandb_resume
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = resume_mode if resume_mode else "allow"
    else:
        init_kwargs["config"] = config
    timeout_val = None
    try:
        timeout_val = float(wandb_init_timeout)
    except (TypeError, ValueError):
        timeout_val = None
    if timeout_val and timeout_val > 0:
        init_kwargs["settings"] = wandb.Settings(init_timeout=timeout_val)

    comm_error = getattr(wandb, "errors", None)
    comm_exception = getattr(comm_error, "CommError", Exception) if comm_error else Exception
    try:
        wandb.init(**init_kwargs)
        if run_id and isinstance(config, dict):
            try:
                wandb.config.update(config, allow_val_change=True)
            except Exception:
                pass
    except comm_exception as err:
        print(f"wandb.init failed ({err}); disabling wandb logging.")
        wandb_log = False
        if isinstance(config, dict):
            config['wandb_log'] = False
    except Exception as err:
        print(f"wandb.init failed ({err.__class__.__name__}: {err}); disabling wandb logging.")
        wandb_log = False
        if isinstance(config, dict):
            config['wandb_log'] = False

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
            if curriculum_feedback_metric is not None:
                log_dict["train/curriculum_feedback"] = curriculum_feedback_metric
            if curriculum_difficulty_score is not None:
                log_dict["train/curriculum_difficulty"] = curriculum_difficulty_score
            # Add stopping metrics if they exist
            stopping_metrics = getattr(raw_model, 'stopping_metrics', None)
            if stopping_metrics:
                log_dict.update({f"train/stopping/{k}": v for k, v in stopping_metrics.items()})
            
            attentive_stopping_metrics = getattr(raw_model, 'attentive_stopping_metrics', None)
            if attentive_stopping_metrics:
                log_dict.update({f"train/attentive_stopping/{k}": v for k, v in attentive_stopping_metrics.items()})
            hard_attentive_metrics = getattr(raw_model, 'hard_attentive_stopping_metrics', None)
            if hard_attentive_metrics:
                log_dict.update({f"train/hard_attentive_stopping/{k}": v for k, v in hard_attentive_metrics.items()})

            oracle_metrics = getattr(raw_model, 'oracle_metrics', None)
            if oracle_metrics:
                log_dict.update({f"train/oracle/{k}": v for k, v in oracle_metrics.items()})

            oracle_metrics = getattr(raw_model, 'oracle_metrics', None)
            if oracle_metrics:
                log_dict.update({f"train/oracle/{k}": v for k, v in oracle_metrics.items()})

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
                    'oracle_state': raw_model.get_oracle_state(),
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    last_micro_loss_value = None
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            if recurrent_shared_weights:
                n = determine_recurrent_depth(
                    iter_num,
                    recurrent_depth,
                    schedule=recurrent_depth_schedule,
                    interval=recurrent_depth_schedule_interval,
                    min_depth=recurrent_depth_schedule_min_depth,
                    resample_prob=recurrent_depth_schedule_resample_prob,
                    scale_loss_by_n_layer=scale_loss_by_n_layer,
                    peak=recurrent_depth_peak,
                    schedule_options=schedule_options,
                    feedback_metric=curriculum_feedback_metric,
                    difficulty_score=curriculum_difficulty_score,
                )
            else:
                n = None
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
            try:
                last_micro_loss_value = float(loss.detach().float().item())
            except (AttributeError, RuntimeError):
                last_micro_loss_value = None
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
    if hasattr(raw_model, "maybe_update_oracle"):
        raw_model.maybe_update_oracle(iter_num)
    if hasattr(raw_model, "maybe_update_oracle"):
        raw_model.maybe_update_oracle(iter_num)

    if last_micro_loss_value is not None:
        if curriculum_loss_ema is None:
            difficulty_ratio = 1.0
            curriculum_loss_ema = last_micro_loss_value
        else:
            denominator = curriculum_loss_ema if abs(curriculum_loss_ema) > 1e-8 else 1e-8
            difficulty_ratio = last_micro_loss_value / denominator
            if curriculum_feedback_alpha >= 1.0:
                curriculum_loss_ema = last_micro_loss_value
            else:
                ema_alpha = curriculum_feedback_alpha
                curriculum_loss_ema = (1 - ema_alpha) * curriculum_loss_ema + ema_alpha * last_micro_loss_value
        curriculum_feedback_metric = curriculum_loss_ema
        ratio_clamped = max(difficulty_min_ratio, min(difficulty_max_ratio, difficulty_ratio))
        curriculum_difficulty_score = (ratio_clamped - difficulty_min_ratio) / difficulty_ratio_span

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
            if curriculum_feedback_metric is not None:
                writer.add_scalar('train/curriculum_feedback', curriculum_feedback_metric, iter_num)
            if curriculum_difficulty_score is not None:
                writer.add_scalar('train/curriculum_difficulty', curriculum_difficulty_score, iter_num)
            # Add stopping metrics if they exist
            stopping_metrics = getattr(raw_model, 'stopping_metrics', None)
            if stopping_metrics:
                for k, v in stopping_metrics.items():
                    writer.add_scalar(f"train/stopping/{k}", v, iter_num)
            attentive_stopping_metrics = getattr(raw_model, 'attentive_stopping_metrics', None)
            if attentive_stopping_metrics:
                for k, v in attentive_stopping_metrics.items():
                    writer.add_scalar(f"train/attentive_stopping/{k}", v, iter_num)
            hard_attentive_metrics = getattr(raw_model, 'hard_attentive_stopping_metrics', None)
            if hard_attentive_metrics:
                for k, v in hard_attentive_metrics.items():
                    writer.add_scalar(f"train/hard_attentive_stopping/{k}", v, iter_num)
            oracle_metrics = getattr(raw_model, 'oracle_metrics', None)
            if oracle_metrics:
                for k, v in oracle_metrics.items():
                    writer.add_scalar(f"train/oracle/{k}", v, iter_num)
            oracle_metrics = getattr(raw_model, 'oracle_metrics', None)
            if oracle_metrics:
                for k, v in oracle_metrics.items():
                    writer.add_scalar(f"train/oracle/{k}", v, iter_num)

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if hyperparameter_tuning_result_path:
    tuning_payload = {
        "val_loss": float(best_val_loss),
        "completed_iters": int(iter_num),
        "timestamp": time.time(),
    }
    result_dir = os.path.dirname(hyperparameter_tuning_result_path)
    try:
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        with open(hyperparameter_tuning_result_path, 'w', encoding='utf-8') as f:
            json.dump(tuning_payload, f)
    except OSError as exc:
        print(f"Warning: could not write tuning result to {hyperparameter_tuning_result_path}: {exc}")

if tensorboard_log and master_process:
    writer.close()

if ddp:
    destroy_process_group()
