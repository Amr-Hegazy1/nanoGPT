import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out'
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# data loader
data_dir = os.path.join('data', checkpoint['config']['dataset'])
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = val_data
    ix = torch.randint(len(data) - gptconf.block_size, (checkpoint['config']['batch_size'],))
    x = torch.stack([torch.from_numpy((data[i:i+gptconf.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+gptconf.block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# evaluation
@torch.no_grad()
def estimate_loss(n_values):
    out = {}
    model.eval()
    for n in n_values:
        losses = torch.zeros(100) # 100 iterations for each n
        for k in range(100):
            X, Y = get_batch('val')
            with ctx:
                logits, loss, _ = model(X, Y, n=n)
            losses[k] = loss.item()
        out[n] = losses.mean()
    model.train()
    return out

n_values = range(1, 40) if gptconf.recurrent_shared_weights else range(1, 40)
losses = estimate_loss(n_values)

# plot
plt.figure()
plt.plot(list(losses.keys()), list(losses.values()))
plt.xlabel("Number of expanded layers")
plt.ylabel("Validation loss")
plt.title("Validation loss vs. number of expanded layers")
plt.savefig(os.path.join(out_dir, 'recurrent_loss.png'))
