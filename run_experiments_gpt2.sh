#!/bin/bash

# Base configuration
CONFIG="config/train_gpt2.py"
WANDB_PROJECT="nanoGPT-experiments-gpt2"

# DDP settings
DDP=true
NPROC_PER_NODE=2

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --ddp)
            DDP=true
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

# Set training command based on DDP flag
if [ "$DDP" = true ]; then
    TRAIN_CMD="torchrun --standalone --nproc_per_node=$NPROC_PER_NODE train.py"
else
    TRAIN_CMD="python train.py"
fi

# Experiment 1: Baseline
echo "Running experiment 1: Baseline"
EXP1_DIR="out-gpt2-baseline"
$TRAIN_CMD $CONFIG --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="baseline" --out_dir=$EXP1_DIR --compile=False --batch_size=4
echo "Sampling from experiment 1"
python sample.py --out_dir=$EXP1_DIR > $EXP1_DIR/samples.txt

# Experiment 2: Shared Parameters
echo "Running experiment 2: Shared Parameters"
EXP2_DIR="out-gpt2-shared-params"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="shared-params" --out_dir=$EXP2_DIR --compile=False --batch_size=4
echo "Sampling from experiment 2"
python sample.py --out_dir=$EXP2_DIR > $EXP2_DIR/samples.txt

# Experiment 3: Recurrent Shared Weights
echo "Running experiment 3: Recurrent Shared Weights"
EXP3_DIR="out-gpt2-recurrent-shared"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared" --out_dir=$EXP3_DIR --compile=False --batch_size=4
echo "Sampling from experiment 3"
python sample.py --out_dir=$EXP3_DIR > $EXP3_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP3_DIR

# Experiment 4: MoE with Soft Routing
echo "Running experiment 4: MoE with Soft Routing"
EXP4_DIR="out-gpt2-moe-soft-routing"
$TRAIN_CMD $CONFIG --moe=True --moe_top_k=1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="moe-soft-routing" --out_dir=$EXP4_DIR --compile=False --batch_size=4
echo "Sampling from experiment 4"
python sample.py --out_dir=$EXP4_DIR > $EXP4_DIR/samples.txt

# Experiment 5: MoE with Hard Routing
echo "Running experiment 5: MoE with Hard Routing"
EXP5_DIR="out-gpt2-moe-hard-routing"
$TRAIN_CMD $CONFIG --moe=True --moe_hard_routing=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="moe-hard-routing" --out_dir=$EXP5_DIR --compile=False --batch_size=4 --moe_top_k=1
echo "Sampling from experiment 5"
python sample.py --out_dir=$EXP5_DIR > $EXP5_DIR/samples.txt

# Experiment 6: Shared MoE
echo "Running experiment 6: Shared MoE"
EXP6_DIR="out-gpt2-shared-moe"
$TRAIN_CMD $CONFIG --moe=True --share_moe_experts=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="shared-moe" --out_dir=$EXP6_DIR --compile=False --batch_size=4
echo "Sampling from experiment 6"
python sample.py --out_dir=$EXP6_DIR > $EXP6_DIR/samples.txt

# Experiment 7: Recurrent Shared Weights with loss scaling
echo "Running experiment 7: Recurrent Shared Weights with loss scaling"
EXP7_DIR="out-gpt2-recurrent-shared-loss-scaling"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-loss-scaling" --out_dir=$EXP7_DIR --compile=False --batch_size=4
echo "Sampling from experiment 7"
python sample.py --out_dir=$EXP7_DIR > $EXP7_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP7_DIR



# Experiment 8: Recurrent Shared MoE
echo "Running experiment 8: Recurrent Shared MoE"
EXP8_DIR="out-gpt2-recurrent-shared-moe"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --moe=True --share_moe_experts=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-moe" --out_dir=$EXP8_DIR --compile=False --batch_size=4
echo "Sampling from experiment 8"
python sample.py --out_dir=$EXP8_DIR > $EXP8_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP8_DIR



# Experiment 9: Recurrent Shared MoE with loss scaling
echo "Running experiment 9: Recurrent Shared MoE with loss scaling"
EXP9_DIR="out-gpt2-recurrent-shared-moe-loss-scaling"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --moe=True --share_moe_experts=True --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-moe-loss-scaling" --out_dir=$EXP9_DIR --compile=False --batch_size=4
echo "Sampling from experiment 9"
python sample.py --out_dir=$EXP9_DIR > $EXP9_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP9_DIR



# Experiment 10: Recurrent Shared Weights with Layer Dropout
echo "Running experiment 10: Recurrent Shared Weights with Layer Dropout"
EXP10_DIR="out-gpt2-recurrent-shared-layer-dropout"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --layer_dropout=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-layer-dropout" --out_dir=$EXP10_DIR --compile=False --batch_size=4
echo "Sampling from experiment 10"
python sample.py --out_dir=$EXP10_DIR > $EXP10_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP10_DIR

# Experiment 11: Recurrent Shared Weights with Sticky Dropout
echo "Running experiment 11: Recurrent Shared Weights with Sticky Dropout"
EXP11_DIR="out-gpt2-recurrent-shared-sticky-dropout"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --sticky_dropout=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-sticky-dropout" --out_dir=$EXP11_DIR --compile=False --batch_size=4
echo "Sampling from experiment 11"
python sample.py --out_dir=$EXP11_DIR > $EXP11_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP11_DIR

# Experiment 12: Recurrent Shared Weights with Learned Stopping
echo "Running experiment 12: Recurrent Shared Weights with Learned Stopping"
EXP12_DIR="out-gpt2-recurrent-shared-learned-stopping"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --learned_stopping=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-learned-stopping" --out_dir=$EXP12_DIR --compile=False --batch_size=4
echo "Sampling from experiment 12"
python sample.py --out_dir=$EXP12_DIR > $EXP12_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP12_DIR

# Experiment 13: Recurrent Shared MoE with Layer Dropout
echo "Running experiment 13: Recurrent Shared MoE with Layer Dropout"
EXP13_DIR="out-gpt2-recurrent-shared-moe-layer-dropout"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --moe=True --share_moe_experts=True --layer_dropout=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-moe-layer-dropout" --out_dir=$EXP13_DIR --compile=False --batch_size=4
echo "Sampling from experiment 13"
python sample.py --out_dir=$EXP13_DIR > $EXP13_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP13_DIR

# Experiment 14: Recurrent Shared MoE with Sticky Dropout
echo "Running experiment 14: Recurrent Shared MoE with Sticky Dropout"
EXP14_DIR="out-gpt2-recurrent-shared-moe-sticky-dropout"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --moe=True --share_moe_experts=True --sticky_dropout=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-moe-sticky-dropout" --out_dir=$EXP14_DIR --compile=False --batch_size=4
echo "Sampling from experiment 14"
python sample.py --out_dir=$EXP14_DIR > $EXP14_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP14_DIR

# Experiment 15: Recurrent Shared MoE with Learned Stopping
echo "Running experiment 15: Recurrent Shared MoE with Learned Stopping"
EXP15_DIR="out-gpt2-recurrent-shared-moe-learned-stopping"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --moe=True --share_moe_experts=True --learned_stopping=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-moe-learned-stopping" --out_dir=$EXP15_DIR --compile=False --batch_size=4
echo "Sampling from experiment 15"
python sample.py --out_dir=$EXP15_DIR > $EXP15_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP15_DIR

echo "All experiments complete."
