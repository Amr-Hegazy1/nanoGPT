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

# --- Base Experiments ---

# Experiment 1: Baseline
echo "Running experiment 1: Baseline"
EXP1_DIR="out-gpt2-baseline"
$TRAIN_CMD $CONFIG --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="baseline" --out_dir=$EXP1_DIR --compile=False --batch_size=4 --log_correlation=True
echo "Sampling from experiment 1"
python sample.py --out_dir=$EXP1_DIR > $EXP1_DIR/samples.txt

# Experiment 2: Shared Parameters
echo "Running experiment 2: Shared Parameters"
EXP2_DIR="out-gpt2-shared-params"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="shared-params" --out_dir=$EXP2_DIR --compile=False --batch_size=4 --log_correlation=True
echo "Sampling from experiment 2"
python sample.py --out_dir=$EXP2_DIR > $EXP2_DIR/samples.txt

# Experiment 3: Recurrent Shared Weights
echo "Running experiment 3: Recurrent Shared Weights"
EXP3_DIR="out-gpt2-recurrent-shared"
$TRAIN_CMD $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared" --out_dir=$EXP3_DIR --compile=False --batch_size=4 --log_correlation=True
echo "Sampling from experiment 3"
python sample.py --out_dir=$EXP3_DIR > $EXP3_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP3_DIR

# --- MoE Experiments ---

# Experiment 4: MoE with Soft Routing
echo "Running experiment 4: MoE with Soft Routing"
EXP4_DIR="out-gpt2-moe-soft-routing"
$TRAIN_CMD $CONFIG --moe=True --moe_top_k=1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="moe-soft-routing" --out_dir=$EXP4_DIR --compile=False --batch_size=4 --log_correlation=True
echo "Sampling from experiment 4"
python sample.py --out_dir=$EXP4_DIR > $EXP4_DIR/samples.txt

# Experiment 5: MoE with Hard Routing
echo "Running experiment 5: MoE with Hard Routing"
EXP5_DIR="out-gpt2-moe-hard-routing"
$TRAIN_CMD $CONFIG --moe=True --moe_hard_routing=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="moe-hard-routing" --out_dir=$EXP5_DIR --compile=False --batch_size=4 --moe_top_k=1 --log_correlation=True
echo "Sampling from experiment 5"
python sample.py --out_dir=$EXP5_DIR > $EXP5_DIR/samples.txt

# Experiment 6: Shared MoE
echo "Running experiment 6: Shared MoE"
EXP6_DIR="out-gpt2-shared-moe"
$TRAIN_CMD $CONFIG --moe=True --share_moe_experts=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="shared-moe" --out_dir=$EXP6_DIR --compile=False --batch_size=4 --log_correlation=True
echo "Sampling from experiment 6"
python sample.py --out_dir=$EXP6_DIR > $EXP6_DIR/samples.txt

# --- Recurrent Shared Weights Experiments (Isolated Features) ---

# Base for this section: Recurrent Shared Weights (like Exp 3)
BASE_RECURRENT_ARGS="$CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --compile=False --batch_size=4 --log_correlation=True --recurrent_depth_peak=32"

# Experiment 7: Recurrent Shared Weights with Layer Dropout
echo "Running experiment 7: Recurrent Shared Weights with Layer Dropout"
EXP7_DIR="out-gpt2-recurrent-layer-dropout"
$TRAIN_CMD $BASE_RECURRENT_ARGS --layer_dropout=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-layer-dropout" --out_dir=$EXP7_DIR
python sample.py --out_dir=$EXP7_DIR > $EXP7_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP7_DIR

# Experiment 8: Recurrent Shared Weights with Layer Dropout + Loss Scaling
echo "Running experiment 8: Recurrent Shared Weights with Layer Dropout + Loss Scaling"
EXP8_DIR="out-gpt2-recurrent-layer-dropout-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_ARGS --layer_dropout=0.1 --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-layer-dropout-loss-scaling" --out_dir=$EXP8_DIR
python sample.py --out_dir=$EXP8_DIR > $EXP8_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP8_DIR

# Experiment 9: Recurrent Shared Weights with Sticky Dropout
echo "Running experiment 9: Recurrent Shared Weights with Sticky Dropout"
EXP9_DIR="out-gpt2-recurrent-sticky-dropout"
$TRAIN_CMD $BASE_RECURRENT_ARGS --sticky_dropout=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-sticky-dropout" --out_dir=$EXP9_DIR
python sample.py --out_dir=$EXP9_DIR > $EXP9_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP9_DIR

# Experiment 10: Recurrent Shared Weights with Sticky Dropout + Loss Scaling
echo "Running experiment 10: Recurrent Shared Weights with Sticky Dropout + Loss Scaling"
EXP10_DIR="out-gpt2-recurrent-sticky-dropout-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_ARGS --sticky_dropout=0.1 --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-sticky-dropout-loss-scaling" --out_dir=$EXP10_DIR
python sample.py --out_dir=$EXP10_DIR > $EXP10_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP10_DIR

# Experiment 11: Recurrent Shared Weights with Learned Stopping
echo "Running experiment 11: Recurrent Shared Weights with Learned Stopping"
EXP11_DIR="out-gpt2-recurrent-learned-stopping"
$TRAIN_CMD $BASE_RECURRENT_ARGS --learned_stopping=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping" --out_dir=$EXP11_DIR
python sample.py --out_dir=$EXP11_DIR > $EXP11_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP11_DIR

# Experiment 12: Recurrent Shared Weights with Learned Stopping + Loss Scaling
echo "Running experiment 12: Recurrent Shared Weights with Learned Stopping + Loss Scaling"
EXP12_DIR="out-gpt2-recurrent-learned-stopping-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_ARGS --learned_stopping=True --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping-loss-scaling" --out_dir=$EXP12_DIR
python sample.py --out_dir=$EXP12_DIR > $EXP12_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP12_DIR

# Experiment 13: Recurrent Shared Weights with Attentive Stopping
echo "Running experiment 13: Recurrent Shared Weights with Attentive Stopping"
EXP13_DIR="out-gpt2-recurrent-attentive-stopping"
$TRAIN_CMD $BASE_RECURRENT_ARGS --attentive_stopping=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-attentive-stopping" --out_dir=$EXP13_DIR
python sample.py --out_dir=$EXP13_DIR > $EXP13_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP13_DIR

# Experiment 14: Recurrent Shared Weights with Attentive Stopping + Loss Scaling
echo "Running experiment 14: Recurrent Shared Weights with Attentive Stopping + Loss Scaling"
EXP14_DIR="out-gpt2-recurrent-attentive-stopping-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_ARGS --attentive_stopping=True --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-attentive-stopping-loss-scaling" --out_dir=$EXP14_DIR
python sample.py --out_dir=$EXP14_DIR > $EXP14_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP14_DIR


# --- Combined Recurrent + MoE Experiments ---

# Base for this section: Recurrent Shared Weights + Shared MoE
BASE_RECURRENT_MOE_ARGS="$BASE_RECURRENT_ARGS --moe=True --share_moe_experts=True"

# Experiment 15: Recurrent Shared MoE
echo "Running experiment 15: Recurrent Shared MoE"
EXP15_DIR="out-gpt2-recurrent-shared-moe"
$TRAIN_CMD $BASE_RECURRENT_MOE_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-moe" --out_dir=$EXP15_DIR
python sample.py --out_dir=$EXP15_DIR > $EXP15_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP15_DIR

# Experiment 16: Recurrent Shared MoE with Loss Scaling
echo "Running experiment 16: Recurrent Shared MoE with Loss Scaling"
EXP16_DIR="out-gpt2-recurrent-shared-moe-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_MOE_ARGS --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-moe-loss-scaling" --out_dir=$EXP16_DIR
python sample.py --out_dir=$EXP16_DIR > $EXP16_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP16_DIR

# --- New Experiments ---

# Experiment 17: Recurrent Shared Weights with (1, 2, 1) configuration
echo "Running experiment 17: Recurrent Shared Weights with (1, 2, 1) configuration"
EXP17_DIR="out-gpt2-recurrent-1-2-1"
$TRAIN_CMD $BASE_RECURRENT_ARGS --n_layers_prelude=1 --n_layer=2 --n_layers_coda=1 --log_correlation=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-1-2-1" --out_dir=$EXP17_DIR --recurrent_depth_peak=16
python sample.py --out_dir=$EXP17_DIR > $EXP17_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP17_DIR

# Experiment 18: Recurrent Shared Weights with (2, 4, 2) configuration
echo "Running experiment 18: Recurrent Shared Weights with (2, 4, 2) configuration"
EXP18_DIR="out-gpt2-recurrent-2-4-2"
$TRAIN_CMD $BASE_RECURRENT_ARGS --n_layers_prelude=2 --n_layer=4 --n_layers_coda=2 --log_correlation=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-2-4-2" --out_dir=$EXP18_DIR
python sample.py --out_dir=$EXP18_DIR > $EXP18_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP18_DIR

# Experiment 19: Recurrent Shared Weights with Learned Stopping, Fixed Edge, and Noise
echo "Running experiment 19: Recurrent Shared Weights with Learned Stopping, Fixed Edge, and Noise"
EXP19_DIR="out-gpt2-recurrent-learned-stopping-fixed-edge-noise"
$TRAIN_CMD $BASE_RECURRENT_ARGS --learned_stopping=True --fixed_edge_blocks=True --recurrent_noise_mode=add --recurrent_noise_std=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping-fixed-edge-noise" --out_dir=$EXP19_DIR
python sample.py --out_dir=$EXP19_DIR > $EXP19_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP19_DIR

# Experiment 20: Recurrent Shared Weights with Prelude Injection
echo "Running experiment 20: Recurrent Shared Weights with Prelude Injection"
EXP20_DIR="out-gpt2-recurrent-prelude-injection"
$TRAIN_CMD $BASE_RECURRENT_ARGS --recurrent_prelude_injection=True --fixed_edge_blocks=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-prelude-injection" --out_dir=$EXP20_DIR
python sample.py --out_dir=$EXP20_DIR > $EXP20_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP20_DIR

# --- RMS Norm Experiments ---

# Base for this section: Recurrent Shared Weights (like Exp 3) + RMS Norm
BASE_RECURRENT_RMS_ARGS="$CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --use_rmsnorm=True --compile=False --batch_size=4 --log_correlation=True --recurrent_depth_peak=32"

# Experiment 21: Recurrent Shared Weights with RMS Norm
echo "Running experiment 21: Recurrent Shared Weights with RMS Norm"
EXP21_DIR="out-gpt2-recurrent-rms-norm"
$TRAIN_CMD $BASE_RECURRENT_RMS_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-rms-norm" --out_dir=$EXP21_DIR
python sample.py --out_dir=$EXP21_DIR > $EXP21_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP21_DIR

# Experiment 22: Recurrent Shared Weights with RMS Norm + Loss Scaling
echo "Running experiment 22: Recurrent Shared Weights with RMS Norm + Loss Scaling"
EXP22_DIR="out-gpt2-recurrent-rms-norm-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_RMS_ARGS --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-rms-norm-loss-scaling" --out_dir=$EXP22_DIR
python sample.py --out_dir=$EXP22_DIR > $EXP22_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP22_DIR

# Experiment 23: Recurrent Shared Weights with RMS Norm + Learned Stopping
echo "Running experiment 23: Recurrent Shared Weights with RMS Norm + Learned Stopping"
EXP23_DIR="out-gpt2-recurrent-rms-norm-learned-stopping"
$TRAIN_CMD $BASE_RECURRENT_RMS_ARGS --learned_stopping=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-rms-norm-learned-stopping" --out_dir=$EXP23_DIR
python sample.py --out_dir=$EXP23_DIR > $EXP23_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP23_DIR

# Experiment 24: Recurrent Shared Weights with RMS Norm + Learned Stopping + Loss Scaling
echo "Running experiment 24: Recurrent Shared Weights with RMS Norm + Learned Stopping + Loss Scaling"
EXP24_DIR="out-gpt2-recurrent-rms-norm-learned-stopping-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_RMS_ARGS --learned_stopping=True --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-rms-norm-learned-stopping-loss-scaling" --out_dir=$EXP24_DIR
python sample.py --out_dir=$EXP24_DIR > $EXP24_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP24_DIR

# Experiment 25: Attentive Stopping with Fixed Edge and Noise
echo "Running experiment 25: Attentive Stopping with Fixed Edge and Noise"
EXP25_DIR="out-gpt2-attentive-stopping-fixed-edge-noise"
$TRAIN_CMD $BASE_RECURRENT_ARGS --attentive_stopping=True --fixed_edge_blocks=True --recurrent_noise_mode=add --recurrent_noise_std=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="attentive-stopping-fixed-edge-noise" --out_dir=$EXP25_DIR
python sample.py --out_dir=$EXP25_DIR > $EXP25_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP25_DIR

echo "All experiments complete."