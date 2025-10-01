#!/bin/bash

# Base configuration
CONFIG="config/train_gpt2.py"
WANDB_PROJECT="nanoGPT-recurrence-experiments-gpt2"

# Experiment 1: Baseline
echo "Running experiment 1: Baseline"
EXP1_DIR="out-gpt2-baseline"
python train.py $CONFIG --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="baseline" --out_dir=$EXP1_DIR --compile=False
echo "Sampling from experiment 1"
python sample.py --out_dir=$EXP1_DIR > $EXP1_DIR/samples.txt

# Experiment 2: Share parameters across layers
echo "Running experiment 2: Share parameters across layers"
EXP2_DIR="out-gpt2-shared-params"
python train.py $CONFIG --share_parameters_across_layers=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="shared-params" --out_dir=$EXP2_DIR --compile=False
echo "Sampling from experiment 2"
python sample.py --out_dir=$EXP2_DIR > $EXP2_DIR/samples.txt

# Experiment 3: Recurrent shared weights
echo "Running experiment 3: Recurrent shared weights"
EXP3_DIR="out-gpt2-recurrent-shared"
python train.py $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared" --out_dir=$EXP3_DIR --compile=False
echo "Sampling from experiment 3"
python sample.py --out_dir=$EXP3_DIR > $EXP3_DIR/samples.txt

# Experiment 4: 2D recurrence
echo "Running experiment 4: 2D recurrence"
EXP4_DIR="out-gpt2-2d-recurrence"
python train.py $CONFIG --enable_2d_recurrence=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="2d-recurrence" --out_dir=$EXP4_DIR --compile=False
echo "Sampling from experiment 4"
python sample.py --out_dir=$EXP4_DIR > $EXP4_DIR/samples.txt

echo "All experiments complete."
