# nanoGPT Project Overview

This project is a lightweight and fast implementation for training and fine-tuning medium-sized GPT (Generative Pre-trained Transformer) models. It is designed for simplicity and hackability, with the core logic contained in `train.py` (training loop) and `model.py` (GPT model definition).

## Building and Running

### 1. Installation

The project dependencies are listed in `requirements.txt`. Install them using pip:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

The project includes scripts to prepare datasets for training. These are located in the `data/` directory. For example, to prepare the Shakespeare dataset, run:

```bash
python data/shakespeare_char/prepare.py
```

### 3. Training

The main script for training is `train.py`. You can configure a training run using command-line arguments or by specifying a configuration file from the `config/` directory.

**Example: Training a character-level Shakespeare model on a GPU**

```bash
python train.py config/train_shakespeare_char.py
```

**Example: Training on a CPU**

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

### 4. Sampling

To generate samples from a trained model, use the `sample.py` script.

**Example: Sampling from a trained model**

```bash
python sample.py --out_dir=out-shakespeare-char
```

## Experiments

The `run_experiments_gpt2.sh` script runs a series of experiments with different model configurations. The experiments are as follows:

*   **Baseline:** A standard GPT-2 model.
*   **Shared Parameters:** A model with shared parameters across layers (ALBERT-style).
*   **Recurrent Shared Weights:** A model with recurrent shared weights.
*   **MoE with Soft Routing:** A Mixture of Experts (MoE) model with soft routing.
*   **MoE with Hard Routing:** A Mixture of Experts (MoE) model with hard routing.
*   **Random 2D Recurrence (Flat):** A model with flat random 2D recurrence.
*   **Random 2D Recurrence (Hierarchical):** A model with hierarchical random 2D recurrence.

To run the experiments, execute the following command:

```bash
./run_experiments_gpt2.sh
```

You can also run the experiments with Distributed Data Parallel (DDP) by using the `--ddp` flag:

```bash
./run_experiments_gpt2.sh --ddp --nproc <number_of_processes>
```

## Development Conventions

*   **Code Structure:** The project follows a clear structure with a separation of concerns:
    *   `model.py`: Defines the GPT model.
    *   `train.py`: Implements the training loop.
    *   `config/`: Contains configuration files for different training runs.
    *   `data/`: Includes scripts for data preparation.
*   **Performance:** The project uses `torch.compile` for performance optimization.
*   **Logging:** Training progress and results can be logged using either Weights & Bienses or TensorBoard.
*   **Configuration:** Training runs are configured through a combination of default settings in `train.py`, configuration files in `config/`, and command-line arguments.