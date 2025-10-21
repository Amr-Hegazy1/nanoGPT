# Repository Guidelines

## Project Structure & Module Organization
- `model.py` hosts the GPT backbone plus experimental modules (Mixture of Experts, recurrent shared weights, stop predictors).
- `train.py` contains the training loop; reusable configurations live in `config/`, and shell helpers like `run_experiments_gpt2.sh` stage common sweeps.
- `data/` stores tokenized corpora produced by `configurator.py`; `assets/` holds plots and shared artifacts; notebooks (`scaling_laws.ipynb`, `transformer_sizing.ipynb`) capture exploratory research.
- Utility entry points include `bench.py` for throughput checks and `sample.py` for text generation demos.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates an isolated environment.
- `pip install -r requirements.txt` installs the PyTorch and logging stack used across scripts.
- `python train.py config/train_shakespeare_char.py` runs the smallest end-to-end training smoke test.
- `python train.py --share_parameters_across_layers=True --recurrent_shared_weights=True --max_iters=100 --eval_iters=20` quickly exercises the recurrent-depth pathway.
- `python bench.py` benchmarks forward/backward passes on the current hardware.

## Coding Style & Naming Conventions
- Use Python 3.10+ with 4-space indentation and follow the existing PEP8-style layout; keep imports grouped (stdlib, third-party, local).
- Modules, configs, and flags use snake_case (e.g., `recurrent_shared_weights`, `train_gpt2.py`); experiment toggles should mirror the corresponding config attribute names.
- Prefer dataclasses for new configuration surfaces and high-signal comments for non-obvious experimental logic; avoid verbose inline commentary.
- Maintain reproducibility by documenting new command-line flags in the relevant config or script docstring.

## Testing Guidelines
- There is no formal unit-test suite; validate changes with targeted training runs and log traces.
- Add or update a config under `config/` to reproduce new behaviour, then run `python train.py <config_path> --max_iters=200 --eval_only=True` to sanity-check loss paths.
- Use `plot_recurrent_loss.py --out_dir=out/diagnostics` when tuning expansion heuristics and attach the generated plot to discussion threads if it shifts behaviour.
- Capture tensorboard or wandb screenshots for significant training changes and reference them in the PR body.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit style (`feat:`, `refactor:`, `fix:`) in imperative mood and keep each commit focused on one logical change.
- Include paired updates (configs, assets, docs) in the same PR to keep experiments reproducible; note any migration steps in the description.
- PR summaries should cover motivation, new flags, reproduction commands, and observed metrics; link issues or experiment notes when available.
- Highlight compatibility considerations (checkpoint format, required datasets) and flag follow-up work or risks before requesting review.
