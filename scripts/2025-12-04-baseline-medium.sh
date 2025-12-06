#!/bin/bash
#SBATCH --job-name=nanogpt_multinode      # Job name
#SBATCH --output=logs/2025-12-05-baseline-medium/slurm-%j.out  # Standard output and error log
#SBATCH --nodes=2                         # Request N nodes
#SBATCH --tasks-per-node=8                # Number of processes (GPUs) per node
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --gres=gpu:8                      # Request N GPUs per node
#SBATCH --time=04:00:00                   # Time limit hrs:min:sec
#SBATCH --partition=a10x8-192c768m        # Specify the partition (queue)
#SBATCH --wckey sparse_scaling_law
#SBATCH --exclusive                       # Exclusive access to nodes. No other jobs will share these nodes.

# Command to run script on Cerebras cluster:
# cbrun -t gpu-a10 -- sbatch ./scripts/2025-12-05-baseline-medium.sh

# Activate your custom python environment if necessary
# conda activate nanogpt

# Environment Args
## Set environment variables for PyTorch Distributed Data Parallel (DDP)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
## --- ADD THESE ENVIRONMENT VARIABLES TO FORCE IPV4 AND SPECIFY THE INTERFACE ---
export NCCL_SOCKET_FAMILY=4
export NCCL_IB_DISABLE=1          # Optional: Disable InfiniBand if experiencing issues, force standard Ethernet
export NCCL_SOCKET_IFNAME=eth0  # Optional: Uncomment and adjust 'eth0' to your cluster's network interface name if needed

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"

# Training Args
CONFIG="config/train_gpt2.py"
JOB_NAME="2025-12-05-baseline-medium"
WANDB_PROJECT=$JOB_NAME

MODEL_ARGS="--n_layer=24 --n_head=16 --n_embd=1024" # GPT-2 Medium
MAX_ITERS=20000
LEARNING_RATE_ARGS="--lr_decay_iters=$MAX_ITERS"
BATCH_ARGS="--batch_size=2 --gradient_accumulation_steps=256"

# Launch the PyTorch distributed training command using srun and torchrun
torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $SLURM_NTASKS_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    train.py \
    $CONFIG \
    $MODEL_ARGS \
    --max_iters=$MAX_ITERS $LEARNING_RATE_ARGS $BATCH_ARGS \
    --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name=$JOB_NAME \
    --out_dir=logs/$JOB_NAME \
    --compile=False \
    --log_correlation=True

# Sample command to generate text after training
python sample.py \
    --init_from=resume \
    --out_dir=logs/$JOB_NAME \
    --start="The capital of Egypt is" \
    --num_samples=1 \
    --max_new_tokens=200 \
    --temperature=0.2 | tee logs/$JOB_NAME/sample_temp0p2.txt

python sample.py \
    --init_from=resume \
    --out_dir=logs/$JOB_NAME \
    --start="The capital of Egypt is" \
    --num_samples=1 \
    --max_new_tokens=200 \
    --temperature=0.5 | tee logs/$JOB_NAME/sample_temp0p5.txt

python sample.py \
    --init_from=resume \
    --out_dir=logs/$JOB_NAME \
    --start="The capital of Egypt is" \
    --num_samples=1 \
    --max_new_tokens=200 \
    --temperature=0.8 | tee logs/$JOB_NAME/sample_temp0p8.txt
