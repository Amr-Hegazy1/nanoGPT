#!/bin/bash

SCRIPT_SOURCE="${BASH_SOURCE[0]:-$0}"
SCRIPT_NAME_WITH_EXTENSION=$(basename "$SCRIPT_SOURCE")
SCRIPT_NAME="${SCRIPT_NAME_WITH_EXTENSION%.*}"

# Slurm configuration
slurm_cluster=gpu-a10
slurm_partition=a10x8-192c768m #a10
JOB_NAME=${SCRIPT_NAME}
LOG_DIR=./logs/${JOB_NAME}

# Args
CONFIG="config/train_gpt2.py"
WANDB_PROJECT="nanoGPT-experiments-gpt2"

MODEL_ARGS="--n_layer=24 --n_head=16 --n_embd=1024" # GPT-2 Medium
MAX_ITERS=20000
LEARNING_RATE_ARGS="--lr_decay_iters=$MAX_ITERS"
BATCH_ARGS="--batch_size=5 --gradient_accumulation_steps=32"

# Encapsulate all runtime commands in a function so remote runner can invoke it with -e "run_job"
run_job() {
    # Display GPU information
    nvidia-smi

    mkdir -p "${LOG_DIR}"

    # Launch the PyTorch distributed training command using srun and torchrun
    torchrun \
        train.py \
        $CONFIG \
        $MODEL_ARGS \
        --max_iters=$MAX_ITERS $LEARNING_RATE_ARGS $BATCH_ARGS \
        --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name=$JOB_NAME \
        --out_dir=${LOG_DIR} \
        --compile=False \
        --log_correlation=True \
        --share_parameters_across_layers=True \
        --stopping_tokenwise=True \
        --recurrent_shared_weights=True \
        --recurrent_depth_schedule_min_depth=2 \
        --recurrent_depth_peak=22 \
        --recurrent_prelude_injection=True \
        --recurrent_prelude_injection_mode=concat \
        --fixed_edge_blocks=True \
        --recurrent_noise_mode=add \
        --recurrent_noise_std=0.1 \
        --oracle_stopping=True --oracle_dummy=True --oracle_update_interval=50 --oracle_stop_weight=0.3 --oracle_difficulty_weight=0.1


    # Sample command to generate text after training
    python sample.py \
        --init_from=resume \
        --out_dir=${LOG_DIR} \
        --start="The capital of Egypt is" \
        --num_samples=1 \
        --max_new_tokens=200 \
        --temperature=0.2 | tee ${LOG_DIR}/sample_temp0p2.txt

    python sample.py \
        --init_from=resume \
        --out_dir=${LOG_DIR} \
        --start="The capital of Egypt is" \
        --num_samples=1 \
        --max_new_tokens=200 \
        --temperature=0.5 | tee ${LOG_DIR}/sample_temp0p5.txt

    python sample.py \
        --init_from=resume \
        --out_dir=logs/$JOB_NAME \
        --start="The capital of Egypt is" \
        --num_samples=1 \
        --max_new_tokens=200 \
        --temperature=0.8 | tee ${LOG_DIR}/sample_temp0p8.txt
}

# export function so remote runner that sources the script can call it
export -f run_job

# Only run the cbrun launcher when the script is executed directly (not when sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]:-$0}")
    cbrun srund \
        -t ${slurm_cluster} \
        -x "-p ${slurm_partition} -c 4 -J ${JOB_NAME} -o ${LOG_DIR}/slurm-%j.out --time 72:00:00 --wckey sparse_scaling_law --gres=gpu:8 --nodes=1 --tasks-per-node=8 --exclusive" \
        -e "bash -lc 'source ${SCRIPT_PATH} && run_job'"
fi