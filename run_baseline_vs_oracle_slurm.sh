#!/bin/bash
set -e

# Base configuration
CONFIG="config/train_gpt2.py"
WANDB_PROJECT="nanoGPT-experiments-gpt2"

# DDP settings (set to "true" or "false")
DDP=true
NPROC_PER_NODE=8

# Slurm configuration
slurm_cluster=gpu-a10
slurm_partition=a10x8-192c768m #a10

MAX_ITERS=20000

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ddp)
            if [[ -n "${2:-}" && "${2:0:2}" != "--" ]]; then
                DDP="$2"
                shift 2
            else
                DDP=true
                shift
            fi
            ;;
        --ddp=true|--ddp=True|--ddp=1)
            DDP=true
            shift
            ;;
        --ddp=false|--ddp=False|--ddp=0|--no-ddp)
            DDP=false
            shift
            ;;
        --nproc)
            NPROC_PER_NODE="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
    esac
done

# Normalise DDP flag
case "$DDP" in
    true|True|1)   DDP_ENABLED=true ;;
    false|False|0) DDP_ENABLED=false ;;
    *)             echo "Invalid value for --ddp: $DDP (expected true/false)"; exit 1 ;;
esac

# Set training command based on DDP flag
if [ "$DDP_ENABLED" = true ]; then
    TRAIN_CMD="torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train.py"
else
    TRAIN_CMD="python train.py"
fi



# --- Base Experiments ---

# Experiment 1: Baseline
echo "Running experiment 1: Baseline"
EXP1_NAME="baseline-steps${MAX_ITERS}"
EXP1_DIR=logs/$EXP1_NAME
mkdir -p $EXP1_DIR
bash_cmd="$TRAIN_CMD $CONFIG --max_iters=$MAX_ITERS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name=$EXP1_NAME --out_dir=$EXP1_DIR --compile=False --batch_size=2 --gradient_accumulation_steps=80 --log_correlation=True"
cbrun srund -t ${slurm_cluster} -x "-p ${slurm_partition} -c 4 -J ${EXP1_NAME} -o ${EXP1_DIR}/slurm-%j.out --time 8:00:00 --wckey sparse_scaling_law" -e "${bash_cmd}"
# TODO: Add inside a script?
# echo "Sampling from experiment 1"
# python sample.py --out_dir=$EXP1_DIR > $EXP1_DIR/samples.txt



# Base for this section: Recurrent Shared Weights
BASE_RECURRENT_ARGS="$CONFIG --max_iters=$MAX_ITERS --share_parameters_across_layers=True --recurrent_shared_weights=True --compile=False --batch_size=2 --gradient_accumulation_steps=80 --log_correlation=True --recurrent_depth_peak=32"






# --- Oracle Stopping Experiments ---

BASE_ORACLE_ARGS="$BASE_RECURRENT_ARGS --oracle_stopping=True --oracle_update_interval=50 --oracle_stop_weight=0.3 --oracle_difficulty_weight=0.1 --recurrent_depth_schedule_min_depth=24"



# Experiment 2: Oracle Stopping (Tokenwise)
echo "Running experiment 2: Oracle Stopping (Tokenwise)"
EXP34_NAME="oracle-stopping-tokenwise-fixed-edge-noise-predlude-injection-concat-steps${MAX_ITERS}"
EXP34_DIR=logs/$EXP34_NAME
mkdir -p $EXP34_DIR
bash_cmd="$TRAIN_CMD $BASE_ORACLE_ARGS --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name=$EXP34_NAME --recurrent_prelude_injection=True --recurrent_prelude_injection_mode=concat --fixed_edge_blocks=True --recurrent_noise_mode=add --recurrent_noise_std=0.1 --out_dir=$EXP34_DIR"
cbrun srund -t ${slurm_cluster} -x "-p ${slurm_partition} -c 4 -J ${EXP34_NAME} -o ${EXP34_DIR}/slurm-%j.out --time 16:00:00 --wckey sparse_scaling_law" -e "${bash_cmd}"
# TODO: add inside a script?
# python sample.py --out_dir=$EXP34_DIR > $EXP34_DIR/samples.txt
# python plot_recurrent_loss.py --out_dir=$EXP34_DIR



echo "All experiments submitted."
