# Repository Guidelines

## Project Structure & Module Organization
- Core training logic lives in `train.py`, while `model.py` houses the GPT blocks (`GPTConfig`, `GPT`) and should stay import-safe.
- Reusable experiment settings sit under `config/`; treat them as single sources of truth for hyperparameters instead of hardcoding flags in scripts.
- Dataset tooling is kept in `data/<dataset_name>/` with each directory exposing a `prepare.py` that writes `train.bin` / `val.bin`.
- Outputs default to `out/` or dataset-specific folders such as `out-shakespeare-char`; keep large checkpoints out of version control.
- Utility scripts (`bench.py`, `sample.py`, `run_experiments_*.sh`) assume the repo root as the working directory.

## Build, Test, and Development Commands
- ```sh
  python data/shakespeare_char/prepare.py
  ```  
  Download and tokenize Tiny Shakespeare into binary shards; mirrors how other datasets should be staged.
- ```sh
  python train.py config/train_shakespeare_char.py
  ```  
  Reference training run for single-GPU debugging; override flags inline for quick experiments.
- ```sh
  torchrun --standalone --nproc_per_node=4 train.py
  ```  
  Distributed launch template; adjust `nnodes`, `master_addr`, and `out_dir` before committing.
- ```sh
  python sample.py --out_dir=out-shakespeare-char
  ```  
  Smoke-test checkpoints and surface regressions in generated text.
- ```sh
  python bench.py --device=cuda --compile=True
  ```  
  Sanity-check PyTorch compile speedups before pushing performance claims.

## Coding Style & Naming Conventions
- Follow existing Python style: 4-space indent, explicit imports, and lowercase_snake_case for variables, configs, and CLI flags.
- Keep configuration dictionaries or `argparse` overrides together near the top of files; document non-obvious magic numbers inline.
- Prefer torch/numpy vectorization over Python loops; wrap experimental code with feature flags so defaults remain stable.
- Type hints are optional but appreciated for public functions and dataclasses that appear in configs.

## Testing Guidelines
- Treat short training runs (`--max_iters` small) as integration tests; they must finish without CUDA OOMs and log decreasing loss.
- For regression checks, run `python train.py <config> --eval_only=True --init_from=<checkpoint>` to ensure loading and metrics stay stable.
- Add reproducible seeds in new scripts and capture expected metrics in the PR description; if adding datasets, include a `prepare.py` that can run offline.

## Commit & Pull Request Guidelines
- Follow the prevalent Git style: subject lines in imperative mood (`"add fused mha kernel"`), <=72 chars, body explaining motivation and validation.
- Keep PRs focused (model change vs. data tooling) and describe: what changed, configs used, hardware, and validation commands/log snippets.
- Link tracking issues or research notes, flag breaking API changes up top, and include screenshots of dashboards (W&B/TensorBoard) when visual evidence supports the claim.
