# nanoGPT

![nanoGPT](assets/nanogpt.jpg)

The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm tensorboard
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of Shakespeare. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
python train.py config/train_shakespeare_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-shakespeare-char
```

This generates a few samples, for example:

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

## Logging

This repository supports logging to both [Weights & Biases](https://wandb.ai/) and [TensorBoard](https://www.tensorflow.org/tensorboard) for monitoring and comparing experiments. In addition to the loss, the perplexity is also logged to Weights & Biases.

-   **Weights & Biases:** Enable by adding the `--wandb_log=True` flag to your `train.py` command. You can also specify a project and run name with `--wandb_project` and `--wandb_run_name`.
-   **TensorBoard:** Enable by adding the `--tensorboard_log=True` flag. The TensorBoard logs will be saved in the `out_dir`. You can view them by running `tensorboard --logdir out_dir` (replace `out_dir` with your output directory).

You can use either or both at the same time.

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/shakespeare` and run `prepare.py` to download the tiny shakespeare dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_shakespeare.py
```

This will load the config parameter overrides in `config/finetune_shakespeare.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-shakespeare` by default, per the config file. You can then run the code in `sample.py --out_dir=out-shakespeare`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## Shared parameters across Transformer layers (optional)

You can share a single Transformer block’s parameters across all layers (ALBERT-style). This reduces parameter count and memory while keeping depth the same.

- Enable via CLI:
  - Training: `python train.py --share_parameters_across_layers=True`
  - Benchmarking: `python bench.py --share_parameters_across_layers=True`
- Checkpoints: the flag is saved in `model_args` and is respected on resume. If resuming, the checkpoint value takes precedence.
- Pretrained GPT-2 weights: cross-layer sharing is not supported when initializing from OpenAI GPT-2 checkpoints (`--init_from=gpt2*`). It is intended for from-scratch training.

### Recurrent Shared Weights (Experimental)

This experiment allows you to reuse a single shared transformer block `n` times. `n` is sampled from a log-normal distribution during training, and can be set manually during inference. This allows for a dynamic depth during training.

*   **Loss Scaling:** The loss can be scaled by the number of expanded layers using the `--scale_loss_by_n_layer` flag. When this switch is active, recurrent depths are sampled uniformly between 1 and `--recurrent_depth`, yielding a consistent range of layer counts.

*   **Validation Loss Plot:** After training a model with recurrent shared weights, a plot of the validation loss versus the number of expanded layers is generated and saved to the output directory. This can help visualize the effect of the number of layers on the model's performance.

*   **Fixed Edge Blocks:** Use `--fixed_edge_blocks=True` to keep a configurable number of Transformer blocks unshared at the beginning and end of the network, while looping a single shared block in the middle. The number of prelude and coda layers can be set with `--n_layers_prelude` and `--n_layers_coda`.

*   **Prelude Injection:** Use `--recurrent_prelude_injection=True` to inject the output of the prelude block into each recurrent block. This is similar to the `add` injection type in the `recurrent-pretraining` repository. This option requires `--fixed_edge_blocks=True` to be set.

*   **Recurrent Noise:** Inject noise before each shared layer with `--recurrent_noise_mode=add|concat`, combining it with `--recurrent_noise_std=<float>` (and `--recurrent_noise_concat_dim=<int>` when concatenating).

*   **Extra LayerNorm:** Activate `--recurrent_extra_layernorm=True` to append an additional LayerNorm after every shared recurrence step.

*   **How to use:**
    *   To enable this during training, use the following flags:
        ```bash
        python train.py --share_parameters_across_layers=True --recurrent_shared_weights=True
        ```
    *   You can optionally change the default mean of the sampling distribution with `--recurrent_depth=<value>`.
    *   During inference, you can specify the number of recurrent steps `n` with:
        ```bash
        python sample.py --recurrent_depth=<value>
        ```

### Mixture of Experts (MoE) (Experimental)

This experiment implements a Mixture of Experts (MoE) model. At each layer, a router selects a subset of "expert" MLPs to process the input. This allows for a much larger model capacity without a proportional increase in computational cost.

*   **Top-k Routing:** The router can select the top-k experts based on the router logits.
*   **Hard Routing:** Instead of a weighted average, you can use hard routing to select a single expert for each token.
*   **Dummy Expert:** A dummy expert (an identity function) is included to allow tokens to be skipped, which can improve efficiency.

*   **How to use:**
    *   To enable MoE during training, use the following flag:
        ```bash
        python train.py --moe=True
        ```
    *   You can control the number of experts with `--moe_num_experts=<value>` (default is 4).
    *   You can set the number of top experts to use with `--moe_top_k=<value>` (default is 2).
    *   To use hard routing, add the `--moe_hard_routing=True` flag.

### Sandwich Norm (Experimental)

This experiment adds an extra normalization layer after the main operation in each sub-block of the transformer. This can help stabilize training and improve performance in some cases.

*   **How to use:**
    *   To enable Sandwich Norm during training, use the following flag:
        ```bash
        python train.py --sandwich_norm=True
        ```

### Shared MoE (Experimental)

This experiment allows you to share the same set of experts across all layers of the model. This can significantly reduce the number of parameters and memory usage, especially for models with a large number of experts.

*   **How to use:**
    *   To enable shared MoE during training, use the following flags:
        ```bash
        python train.py --moe=True --share_moe_experts=True
        ```

### Layer Dropout (Experimental)

This experiment applies dropout to entire layers within a recurrent block. It can be used as a form of regularization.

*   **How to use:**
    *   To enable Layer Dropout during training, use the following flag with a recurrent model:
        ```bash
        python train.py --recurrent_shared_weights=True --layer_dropout=<prob>
        ```
    *   `<prob>` is the probability of dropping a layer (e.g., 0.1).

### Sticky Dropout (Experimental)

In a recurrent setting, this experiment implements a "sticky" dropout where once a sequence in a batch is "dropped," it remains dropped for all subsequent recurrent steps. This can be used to simulate sequences "finishing" at different times.

*   **How to use:**
    *   To enable Sticky Dropout during training, use the following flag with a recurrent model:
        ```bash
        python train.py --recurrent_shared_weights=True --sticky_dropout=<prob>
        ```
    *   `<prob>` is the probability of dropping a sequence at each recurrent step (e.g., 0.1).

### Learned Stopping (Experimental)

This experiment introduces a differentiable gating head that blends the shared block output with the current hidden state, allowing the model to *learn* an expected recursion depth. The gate is supervised with optional controller and entropy regularizers inspired by skip-middle and Mixture-of-Recursions.

*   **How to use:**
    *   Enable the head with:
        ```bash
        python train.py --recurrent_shared_weights=True --learned_stopping=True
        ```
    *   Optionally stabilise training with
        *   `--learned_stopping_warmup_steps=1000` to keep the gate closed for early iterations.
        *   `--learned_stopping_controller_weight=0.1` and `--learned_stopping_target_depth=<float>` to keep the expected depth near a desired value (defaults to the uniform baseline when unspecified).
        *   `--learned_stopping_entropy_weight=0.01` to encourage high-entropy exit distributions.
        *   `--learned_stopping_temperature` and `--learned_stopping_min_prob` to tune the smoothness of the gate.
        *   `--learned_stopping_use_threshold=True --learned_stopping_threshold=0.6` to snap the gate to a hard decision at evaluation time.
*   **Logging:** When enabled, the training loop logs `mean_depth`, controller loss, entropy, and the instantaneous target depth so you can track convergence of the learned policy.


## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

- Cross-layer sharing reduces parameters roughly by ~n_layer× for the block stack, improving memory footprint. Combine with `torch.compile` for best speed.

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
