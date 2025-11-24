#!/bin/bash

# Base configuration
CONFIG="config/train_gpt2.py"
WANDB_PROJECT="nanoGPT-experiments-gpt2"

# DDP settings (set to "true" or "false")
DDP=false
NPROC_PER_NODE=1

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

# Experiment 9: Recurrent Shared Weights with Sticky Dropout
echo "Running experiment 9: Recurrent Shared Weights with Sticky Dropout"
EXP9_DIR="out-gpt2-recurrent-sticky-dropout"
$TRAIN_CMD $BASE_RECURRENT_ARGS --sticky_dropout=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-sticky-dropout" --out_dir=$EXP9_DIR
python sample.py --out_dir=$EXP9_DIR > $EXP9_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP9_DIR

# Experiment 9 (Tokenwise): Recurrent Shared Weights with Sticky Dropout
echo "Running experiment 9 (Tokenwise): Recurrent Shared Weights with Sticky Dropout"
EXP9_TOKENWISE_DIR="out-gpt2-recurrent-sticky-dropout-tokenwise"
$TRAIN_CMD $BASE_RECURRENT_ARGS --sticky_dropout=0.1 --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-sticky-dropout-tokenwise" --out_dir=$EXP9_TOKENWISE_DIR
python sample.py --out_dir=$EXP9_TOKENWISE_DIR > $EXP9_TOKENWISE_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP9_TOKENWISE_DIR

# Experiment 10: Recurrent Shared Weights with Sticky Dropout + Loss Scaling
echo "Running experiment 10: Recurrent Shared Weights with Sticky Dropout + Loss Scaling"
EXP10_DIR="out-gpt2-recurrent-sticky-dropout-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_ARGS --sticky_dropout=0.1 --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-sticky-dropout-loss-scaling" --out_dir=$EXP10_DIR
python sample.py --out_dir=$EXP10_DIR > $EXP10_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP10_DIR

# Experiment 10 (Tokenwise): Recurrent Shared Weights with Sticky Dropout + Loss Scaling
echo "Running experiment 10 (Tokenwise): Recurrent Shared Weights with Sticky Dropout + Loss Scaling"
EXP10_TOKENWISE_DIR="out-gpt2-recurrent-sticky-dropout-loss-scaling-tokenwise"
$TRAIN_CMD $BASE_RECURRENT_ARGS --sticky_dropout=0.1 --scale_loss_by_n_layer=True --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-sticky-dropout-loss-scaling-tokenwise" --out_dir=$EXP10_TOKENWISE_DIR
python sample.py --out_dir=$EXP10_TOKENWISE_DIR > $EXP10_TOKENWISE_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP10_TOKENWISE_DIR

# Experiment 11: Recurrent Shared Weights with Learned Stopping
echo "Running experiment 11: Recurrent Shared Weights with Learned Stopping"
EXP11_DIR="out-gpt2-recurrent-learned-stopping"
$TRAIN_CMD $BASE_RECURRENT_ARGS --learned_stopping=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping" --out_dir=$EXP11_DIR
python sample.py --out_dir=$EXP11_DIR > $EXP11_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP11_DIR

# Experiment 11 (Tokenwise): Recurrent Shared Weights with Learned Stopping
echo "Running experiment 11 (Tokenwise): Recurrent Shared Weights with Learned Stopping"
EXP11_TOKENWISE_DIR="out-gpt2-recurrent-learned-stopping-tokenwise"
$TRAIN_CMD $BASE_RECURRENT_ARGS --learned_stopping=True --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping-tokenwise" --out_dir=$EXP11_TOKENWISE_DIR
python sample.py --out_dir=$EXP11_TOKENWISE_DIR > $EXP11_TOKENWISE_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP11_TOKENWISE_DIR

# Experiment 12: Recurrent Shared Weights with Learned Stopping + Loss Scaling
echo "Running experiment 12: Recurrent Shared Weights with Learned Stopping + Loss Scaling"
EXP12_DIR="out-gpt2-recurrent-learned-stopping-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_ARGS --learned_stopping=True --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping-loss-scaling" --out_dir=$EXP12_DIR
python sample.py --out_dir=$EXP12_DIR > $EXP12_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP12_DIR

# Experiment 12 (Tokenwise): Recurrent Shared Weights with Learned Stopping + Loss Scaling
echo "Running experiment 12 (Tokenwise): Recurrent Shared Weights with Learned Stopping + Loss Scaling"
EXP12_TOKENWISE_DIR="out-gpt2-recurrent-learned-stopping-loss-scaling-tokenwise"
$TRAIN_CMD $BASE_RECURRENT_ARGS --learned_stopping=True --scale_loss_by_n_layer=True --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping-loss-scaling-tokenwise" --out_dir=$EXP12_TOKENWISE_DIR
python sample.py --out_dir=$EXP12_TOKENWISE_DIR > $EXP12_TOKENWISE_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP12_TOKENWISE_DIR

# Experiment 13: Recurrent Shared Weights with Attentive Stopping
echo "Running experiment 13: Recurrent Shared Weights with Attentive Stopping"
EXP13_DIR="out-gpt2-recurrent-attentive-stopping"
$TRAIN_CMD $BASE_RECURRENT_ARGS --attentive_stopping=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-attentive-stopping" --out_dir=$EXP13_DIR
python sample.py --out_dir=$EXP13_DIR > $EXP13_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP13_DIR

# Experiment 13 (Tokenwise): Recurrent Shared Weights with Attentive Stopping
echo "Running experiment 13 (Tokenwise): Recurrent Shared Weights with Attentive Stopping"
EXP13_TOKENWISE_DIR="out-gpt2-recurrent-attentive-stopping-tokenwise"
$TRAIN_CMD $BASE_RECURRENT_ARGS --attentive_stopping=True --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-attentive-stopping-tokenwise" --out_dir=$EXP13_TOKENWISE_DIR
python sample.py --out_dir=$EXP13_TOKENWISE_DIR > $EXP13_TOKENWISE_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP13_TOKENWISE_DIR

# Experiment 14: Recurrent Shared Weights with Attentive Stopping + Loss Scaling
echo "Running experiment 14: Recurrent Shared Weights with Attentive Stopping + Loss Scaling"
EXP14_DIR="out-gpt2-recurrent-attentive-stopping-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_ARGS --attentive_stopping=True --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-attentive-stopping-loss-scaling" --out_dir=$EXP14_DIR
python sample.py --out_dir=$EXP14_DIR > $EXP14_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP14_DIR

# Experiment 14 (Tokenwise): Recurrent Shared Weights with Attentive Stopping + Loss Scaling
echo "Running experiment 14 (Tokenwise): Recurrent Shared Weights with Attentive Stopping + Loss Scaling"
EXP14_TOKENWISE_DIR="out-gpt2-recurrent-attentive-stopping-loss-scaling-tokenwise"
$TRAIN_CMD $BASE_RECURRENT_ARGS --attentive_stopping=True --scale_loss_by_n_layer=True --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-attentive-stopping-loss-scaling-tokenwise" --out_dir=$EXP14_TOKENWISE_DIR
python sample.py --out_dir=$EXP14_TOKENWISE_DIR > $EXP14_TOKENWISE_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP14_TOKENWISE_DIR


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

# Experiment 18: Recurrent Shared Weights with cumulative stop features
echo "Running experiment 18: Recurrent Shared Weights with cumulative stop features"
EXP18_DIR="out-gpt2-recurrent-cumsum-stop"
$TRAIN_CMD $BASE_RECURRENT_ARGS --stop_use_cumsum_pooling=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-cumsum-stop" --out_dir=$EXP18_DIR
python sample.py --out_dir=$EXP18_DIR > $EXP18_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP18_DIR

# Experiment 19: Recurrent Shared Weights with pooled features disabled
echo "Running experiment 19: Recurrent Shared Weights with pooled stop features disabled"
EXP19_DIR="out-gpt2-recurrent-stop-token-only"
$TRAIN_CMD $BASE_RECURRENT_ARGS --stop_disable_pooled_features=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-stop-token-only" --out_dir=$EXP19_DIR
python sample.py --out_dir=$EXP19_DIR > $EXP19_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP19_DIR

# Experiment 20: Recurrent Shared Weights with cumulative stop features + no pooled context
echo "Running experiment 20: Recurrent Shared Weights with cumulative stop features + no pooled context"
EXP20_DIR="out-gpt2-recurrent-stop-cumsum-token-only"
$TRAIN_CMD $BASE_RECURRENT_ARGS --stop_use_cumsum_pooling=True --stop_disable_pooled_features=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-stop-cumsum-token-only" --out_dir=$EXP20_DIR
python sample.py --out_dir=$EXP20_DIR > $EXP20_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP20_DIR

# Experiment 21: Recurrent Shared Weights with truncated BPTT (depth 2)
echo "Running experiment 21: Recurrent Shared Weights with truncated BPTT (depth 2)"
EXP21_DIR="out-gpt2-recurrent-truncated-bptt-2"
$TRAIN_CMD $BASE_RECURRENT_ARGS --bp_truncate_depth=2 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-truncated-bptt-2" --out_dir=$EXP21_DIR
python sample.py --out_dir=$EXP21_DIR > $EXP21_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP21_DIR

# Experiment 22: Recurrent Shared Weights with truncated BPTT (depth 4)
echo "Running experiment 22: Recurrent Shared Weights with truncated BPTT (depth 4)"
EXP22_DIR="out-gpt2-recurrent-truncated-bptt-4"
$TRAIN_CMD $BASE_RECURRENT_ARGS --bp_truncate_depth=4 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-truncated-bptt-4" --out_dir=$EXP22_DIR
python sample.py --out_dir=$EXP22_DIR > $EXP22_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP22_DIR

# Experiment 21: Recurrent Shared Weights with (2, 4, 2) configuration
echo "Running experiment 21: Recurrent Shared Weights with (2, 4, 2) configuration"
EXP21_DIR="out-gpt2-recurrent-2-4-2"
$TRAIN_CMD $BASE_RECURRENT_ARGS --n_layers_prelude=2 --n_layer=4 --n_layers_coda=2 --log_correlation=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-2-4-2" --out_dir=$EXP21_DIR
python sample.py --out_dir=$EXP21_DIR > $EXP21_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP21_DIR

# Experiment 22: Recurrent Shared Weights with Learned Stopping, Fixed Edge, and Noise
echo "Running experiment 22: Recurrent Shared Weights with Learned Stopping, Fixed Edge, and Noise"
EXP22_DIR="out-gpt2-recurrent-learned-stopping-fixed-edge-noise"
$TRAIN_CMD $BASE_RECURRENT_ARGS --learned_stopping=True --fixed_edge_blocks=True --recurrent_noise_mode=add --recurrent_noise_std=0.1 --hyperparameter_tuning=True --hyperparameter_tuning_trials=12 --hyperparameter_tuning_max_iters=300 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping-fixed-edge-noise" --out_dir=$EXP22_DIR
python sample.py --out_dir=$EXP22_DIR > $EXP22_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP22_DIR

# Experiment 23: Recurrent Shared Weights with Prelude Injection
echo "Running experiment 23: Recurrent Shared Weights with Prelude Injection"
EXP23_DIR="out-gpt2-recurrent-prelude-injection"
$TRAIN_CMD $BASE_RECURRENT_ARGS --recurrent_prelude_injection=True --fixed_edge_blocks=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-prelude-injection" --out_dir=$EXP23_DIR
python sample.py --out_dir=$EXP23_DIR > $EXP23_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP23_DIR

# Experiment 24: Recurrent Shared Weights with Prelude Injection (Concat)
echo "Running experiment 24: Recurrent Shared Weights with Prelude Injection (Concat)"
EXP24_DIR="out-gpt2-recurrent-prelude-injection-concat"
$TRAIN_CMD $BASE_RECURRENT_ARGS --recurrent_prelude_injection=True --recurrent_prelude_injection_mode=concat --fixed_edge_blocks=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-prelude-injection-concat" --out_dir=$EXP24_DIR
python sample.py --out_dir=$EXP24_DIR > $EXP24_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP24_DIR

# --- RMS Norm Experiments ---

# Base for this section: Recurrent Shared Weights (like Exp 3) + RMS Norm
BASE_RECURRENT_RMS_ARGS="$CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --use_rmsnorm=True --compile=False --batch_size=4 --log_correlation=True --recurrent_depth_peak=32"

# Experiment 22: Recurrent Shared Weights with RMS Norm
echo "Running experiment 22: Recurrent Shared Weights with RMS Norm"
EXP22_DIR="out-gpt2-recurrent-rms-norm"
$TRAIN_CMD $BASE_RECURRENT_RMS_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-rms-norm" --out_dir=$EXP22_DIR
python sample.py --out_dir=$EXP22_DIR > $EXP22_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP22_DIR

# Experiment 23: Recurrent Shared Weights with RMS Norm + Loss Scaling
echo "Running experiment 23: Recurrent Shared Weights with RMS Norm + Loss Scaling"
EXP23_DIR="out-gpt2-recurrent-rms-norm-loss-scaling"
$TRAIN_CMD $BASE_RECURRENT_RMS_ARGS --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-rms-norm-loss-scaling" --out_dir=$EXP23_DIR
python sample.py --out_dir=$EXP23_DIR > $EXP23_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP23_DIR

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

# Experiment 26: Recurrent Sandwich Norm with LayerNorm
echo "Running experiment 26: Recurrent Sandwich Norm with LayerNorm"
EXP26_DIR="out-gpt2-recurrent-sandwich-norm-layernorm"
$TRAIN_CMD $BASE_RECURRENT_ARGS --sandwich_norm=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-sandwich-norm-layernorm" --out_dir=$EXP26_DIR
echo "Sampling from experiment 26"
python sample.py --out_dir=$EXP26_DIR > $EXP26_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP26_DIR

# Experiment 27: Recurrent Sandwich Norm with RMSNorm
echo "Running experiment 27: Recurrent Sandwich Norm with RMSNorm"
EXP27_DIR="out-gpt2-recurrent-sandwich-norm-rmsnorm"
$TRAIN_CMD $BASE_RECURRENT_RMS_ARGS --sandwich_norm=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-sandwich-norm-rmsnorm" --out_dir=$EXP27_DIR
echo "Sampling from experiment 27"
python sample.py --out_dir=$EXP27_DIR > $EXP27_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP27_DIR

# Experiment 28: Recurrent Depth Curriculum (Ascending)
echo "Running experiment 28: Recurrent Depth Curriculum (Ascending)"
EXP28_DIR="out-gpt2-recurrent-depth-curriculum-ascending"
$TRAIN_CMD $BASE_RECURRENT_ARGS --recurrent_depth_schedule=ascending --recurrent_depth_schedule_interval=200 --recurrent_depth_schedule_resample_prob=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-depth-curriculum-ascending" --out_dir=$EXP28_DIR
python sample.py --out_dir=$EXP28_DIR > $EXP28_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP28_DIR

# Experiment 29: Recurrent Depth Curriculum (Descending)
echo "Running experiment 29: Recurrent Depth Curriculum (Descending)"
EXP29_DIR="out-gpt2-recurrent-depth-curriculum-descending"
$TRAIN_CMD $BASE_RECURRENT_ARGS --recurrent_depth_schedule=descending --recurrent_depth_schedule_interval=200 --recurrent_depth_schedule_resample_prob=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-depth-curriculum-descending" --out_dir=$EXP29_DIR
python sample.py --out_dir=$EXP29_DIR > $EXP29_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP29_DIR

# Experiment 30: Recurrent Depth Curriculum (Cyclical)
echo "Running experiment 30: Recurrent Depth Curriculum (Cyclical)"
EXP30_DIR="out-gpt2-recurrent-depth-curriculum-cyclical"
$TRAIN_CMD $BASE_RECURRENT_ARGS --recurrent_depth_schedule=cyclical --recurrent_depth_schedule_interval=150 --recurrent_depth_schedule_opts='cycle_length=6' --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-depth-curriculum-cyclical" --out_dir=$EXP30_DIR
python sample.py --out_dir=$EXP30_DIR > $EXP30_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP30_DIR

# Experiment 31: Recurrent Depth Curriculum (Random Walk)
echo "Running experiment 31: Recurrent Depth Curriculum (Random Walk)"
EXP31_DIR="out-gpt2-recurrent-depth-curriculum-random-walk"
$TRAIN_CMD $BASE_RECURRENT_ARGS --recurrent_depth_schedule=random_walk --recurrent_depth_schedule_interval=150 --recurrent_depth_schedule_opts='step_size=3,reset_prob=0.1' --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-depth-curriculum-random-walk" --out_dir=$EXP31_DIR
python sample.py --out_dir=$EXP31_DIR > $EXP31_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP31_DIR

# Experiment 32: Performance-Aware Curriculum
echo "Running experiment 32: Performance-Aware Curriculum"
EXP32_DIR="out-gpt2-recurrent-depth-curriculum-performance"
$TRAIN_CMD $BASE_RECURRENT_ARGS --recurrent_depth_schedule=performance --recurrent_depth_schedule_interval=150 --recurrent_depth_schedule_opts='patience=4,tolerance=5e-4,warmup_intervals=2' --recurrent_depth_schedule_feedback_alpha=0.2 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-depth-curriculum-performance" --out_dir=$EXP32_DIR
python sample.py --out_dir=$EXP32_DIR > $EXP32_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP32_DIR

# --- Oracle Stopping Experiments ---

BASE_ORACLE_ARGS="$BASE_RECURRENT_ARGS --oracle_stopping=True --oracle_update_interval=50 --oracle_stop_weight=0.3 --oracle_difficulty_weight=0.1 --recurrent_depth_schedule_min_depth=24"
BASE_ORACLE_EMA_ARGS="$BASE_ORACLE_ARGS --oracle_teacher_use_ema=True --oracle_teacher_ema_decay=0.999 --oracle_teacher_ema_decay_min=0.99 --oracle_teacher_ema_decay_schedule=2000"
BASE_ORACLE_CONF_ARGS="$BASE_ORACLE_ARGS --oracle_confidence_weighting=True --oracle_confidence_floor=0.05 --oracle_confidence_exponent=1.0 --oracle_confidence_ceiling=1.0 --oracle_stop_adv_clip=3.0"
BASE_ORACLE_ADAPTIVE_ARGS="$BASE_ORACLE_ARGS --oracle_adaptive_update_interval=True --oracle_update_interval_min=25 --oracle_update_interval_max=200 --oracle_update_interval_shrink=0.8 --oracle_update_interval_growth=1.2 --oracle_update_interval_tolerance=0.05"
BASE_ORACLE_CURRICULUM_ARGS="$BASE_ORACLE_ARGS --oracle_curriculum_depth_start=12 --oracle_curriculum_depth_warmup_steps=2000 --oracle_temperature_final=0.8 --oracle_temperature_schedule_steps=2000"
BASE_ORACLE_ENHANCED_ARGS="$BASE_ORACLE_EMA_ARGS --oracle_confidence_weighting=True --oracle_confidence_floor=0.05 --oracle_confidence_exponent=1.0 --oracle_confidence_ceiling=1.0 --oracle_stop_adv_clip=3.0 --oracle_adaptive_update_interval=True --oracle_update_interval_min=25 --oracle_update_interval_max=200 --oracle_update_interval_shrink=0.8 --oracle_update_interval_growth=1.2 --oracle_update_interval_tolerance=0.05 --oracle_curriculum_depth_start=12 --oracle_curriculum_depth_warmup_steps=2000 --oracle_temperature_final=0.8 --oracle_temperature_schedule_steps=2000"

# Experiment 33: Oracle Stopping (Sequence-Level)
echo "Running experiment 33: Oracle Stopping (Sequence-Level)"
EXP33_DIR="out-gpt2-oracle-stopping-seq"
$TRAIN_CMD $BASE_ORACLE_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-seq" --out_dir=$EXP33_DIR
python sample.py --out_dir=$EXP33_DIR > $EXP33_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP33_DIR

# Experiment 34: Oracle Stopping (Tokenwise)
echo "Running experiment 34: Oracle Stopping (Tokenwise)"
EXP34_DIR="out-gpt2-oracle-stopping-tokenwise"
$TRAIN_CMD $BASE_ORACLE_ARGS --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-tokenwise" --out_dir=$EXP34_DIR
python sample.py --out_dir=$EXP34_DIR > $EXP34_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP34_DIR

# Experiment 34b: Oracle Stopping Tokenwise + Fixed Edge Prelude + Noise (best configs)
echo "Running experiment 34b: Oracle Stopping Tokenwise + Fixed Edge Prelude + Noise"
EXP34B_DIR="out-gpt2-oracle-stopping-tokenwise-fixed-edge"
$TRAIN_CMD $BASE_ORACLE_ARGS --stopping_tokenwise=True --recurrent_prelude_injection=True --recurrent_prelude_injection_mode=concat --fixed_edge_blocks=True --recurrent_noise_mode=add --recurrent_noise_std=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-tokenwise-fixed-edge-noise-prelude-concat" --out_dir=$EXP34B_DIR
python sample.py --out_dir=$EXP34B_DIR > $EXP34B_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP34B_DIR

# Experiment 34c: Oracle Stopping Tokenwise + Fixed Edge Prelude + Noise (enhanced teacher/loss curriculum)
echo "Running experiment 34c: Oracle Stopping Tokenwise + Fixed Edge Prelude + Noise (enhanced)"
EXP34C_DIR="out-gpt2-oracle-stopping-tokenwise-fixed-edge-enhanced"
$TRAIN_CMD $BASE_ORACLE_ENHANCED_ARGS --stopping_tokenwise=True --recurrent_prelude_injection=True --recurrent_prelude_injection_mode=concat --fixed_edge_blocks=True --recurrent_noise_mode=add --recurrent_noise_std=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-tokenwise-fixed-edge-enhanced" --out_dir=$EXP34C_DIR
python sample.py --out_dir=$EXP34C_DIR > $EXP34C_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP34C_DIR

# --- Attentive Stopping Experiments ---

BASE_ATTENTIVE_ARGS="$BASE_RECURRENT_ARGS --attentive_stopping=True --attentive_stopping_controller_weight=0.05 --attentive_stopping_entropy_weight=0.01 --attentive_stopping_warmup_steps=200"

# Experiment 35: Soft Attentive Stopping (Sequence-Level)
echo "Running experiment 35: Soft Attentive Stopping (Sequence-Level)"
EXP35_DIR="out-gpt2-attentive-stopping-soft"
$TRAIN_CMD $BASE_ATTENTIVE_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="attentive-stopping-soft" --out_dir=$EXP35_DIR
python sample.py --out_dir=$EXP35_DIR > $EXP35_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP35_DIR

# Experiment 36: Hard Attentive Stopping (Tokenwise)
echo "Running experiment 36: Hard Attentive Stopping (Tokenwise)"
EXP36_DIR="out-gpt2-attentive-stopping-hard-tokenwise"
$TRAIN_CMD $BASE_ATTENTIVE_ARGS --hard_attentive_stopping=True --hard_attentive_stopping_threshold=0.6 --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="attentive-stopping-hard-tokenwise" --out_dir=$EXP36_DIR
python sample.py --out_dir=$EXP36_DIR > $EXP36_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP36_DIR

# --- Oracle + Attentive Stopping Experiments ---

BASE_ORACLE_ATTENTIVE_ARGS="$BASE_RECURRENT_ARGS --oracle_stopping=True --oracle_update_interval=25 --oracle_stop_weight=0.3 --oracle_difficulty_weight=0.1 --attentive_stopping=True --attentive_stopping_controller_weight=0.05 --attentive_stopping_entropy_weight=0.01 --attentive_stopping_warmup_steps=200"

# Experiment 37: Oracle + Attentive Stopping (Sequence-Level)
echo "Running experiment 37: Oracle + Attentive Stopping (Sequence-Level)"
EXP37_DIR="out-gpt2-oracle-attentive-seq"
$TRAIN_CMD $BASE_ORACLE_ATTENTIVE_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-attentive-seq" --out_dir=$EXP37_DIR
python sample.py --out_dir=$EXP37_DIR > $EXP37_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP37_DIR

# Experiment 38: Oracle + Attentive Stopping (Tokenwise)
echo "Running experiment 38: Oracle + Attentive Stopping (Tokenwise)"
EXP38_DIR="out-gpt2-oracle-attentive-tokenwise"
$TRAIN_CMD $BASE_ORACLE_ATTENTIVE_ARGS --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-attentive-tokenwise" --out_dir=$EXP38_DIR
python sample.py --out_dir=$EXP38_DIR > $EXP38_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP38_DIR

# Experiment 40: Oracle Stopping with EMA teacher (Sequence-Level)
echo "Running experiment 40: Oracle Stopping with EMA teacher (Sequence-Level)"
EXP40_DIR="out-gpt2-oracle-stopping-seq-ema"
$TRAIN_CMD $BASE_ORACLE_EMA_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-seq-ema" --out_dir=$EXP40_DIR
python sample.py --out_dir=$EXP40_DIR > $EXP40_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP40_DIR

# Experiment 41: Oracle Stopping with EMA teacher (Tokenwise)
echo "Running experiment 41: Oracle Stopping with EMA teacher (Tokenwise)"
EXP41_DIR="out-gpt2-oracle-stopping-tokenwise-ema"
$TRAIN_CMD $BASE_ORACLE_EMA_ARGS --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-tokenwise-ema" --out_dir=$EXP41_DIR
python sample.py --out_dir=$EXP41_DIR > $EXP41_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP41_DIR

# Experiment 42: Oracle Stopping with confidence weighting + advantage clipping (Sequence-Level)
echo "Running experiment 42: Oracle Stopping with confidence weighting (Sequence-Level)"
EXP42_DIR="out-gpt2-oracle-stopping-seq-confidence"
$TRAIN_CMD $BASE_ORACLE_CONF_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-seq-confidence" --out_dir=$EXP42_DIR
python sample.py --out_dir=$EXP42_DIR > $EXP42_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP42_DIR

# Experiment 43: Oracle Stopping with confidence weighting + advantage clipping (Tokenwise)
echo "Running experiment 43: Oracle Stopping with confidence weighting (Tokenwise)"
EXP43_DIR="out-gpt2-oracle-stopping-tokenwise-confidence"
$TRAIN_CMD $BASE_ORACLE_CONF_ARGS --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-tokenwise-confidence" --out_dir=$EXP43_DIR
python sample.py --out_dir=$EXP43_DIR > $EXP43_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP43_DIR

# Experiment 44: Oracle Stopping with adaptive update interval (Sequence-Level)
echo "Running experiment 44: Oracle Stopping with adaptive update interval (Sequence-Level)"
EXP44_DIR="out-gpt2-oracle-stopping-seq-adaptive"
$TRAIN_CMD $BASE_ORACLE_ADAPTIVE_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-seq-adaptive" --out_dir=$EXP44_DIR
python sample.py --out_dir=$EXP44_DIR > $EXP44_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP44_DIR

# Experiment 45: Oracle Stopping with adaptive update interval (Tokenwise)
echo "Running experiment 45: Oracle Stopping with adaptive update interval (Tokenwise)"
EXP45_DIR="out-gpt2-oracle-stopping-tokenwise-adaptive"
$TRAIN_CMD $BASE_ORACLE_ADAPTIVE_ARGS --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-tokenwise-adaptive" --out_dir=$EXP45_DIR
python sample.py --out_dir=$EXP45_DIR > $EXP45_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP45_DIR

# Experiment 46: Oracle Stopping with curriculum (Sequence-Level)
echo "Running experiment 46: Oracle Stopping with curriculum (Sequence-Level)"
EXP46_DIR="out-gpt2-oracle-stopping-seq-curriculum"
$TRAIN_CMD $BASE_ORACLE_CURRICULUM_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-seq-curriculum" --out_dir=$EXP46_DIR
python sample.py --out_dir=$EXP46_DIR > $EXP46_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP46_DIR

# Experiment 47: Oracle Stopping with curriculum (Tokenwise)
echo "Running experiment 47: Oracle Stopping with curriculum (Tokenwise)"
EXP47_DIR="out-gpt2-oracle-stopping-tokenwise-curriculum"
$TRAIN_CMD $BASE_ORACLE_CURRICULUM_ARGS --stopping_tokenwise=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="oracle-stopping-tokenwise-curriculum" --out_dir=$EXP47_DIR
python sample.py --out_dir=$EXP47_DIR > $EXP47_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP47_DIR

# --- Learned Stopping (best configs) ---

BASE_LEARNED_FIXED_ARGS="$BASE_RECURRENT_ARGS --learned_stopping=True --fixed_edge_blocks=True --stopping_tokenwise=True"

echo "Running experiment 39: Learned Stopping Tokenwise + Fixed Edge"
EXP39_DIR="out-gpt2-recurrent-learned-stopping-tokenwise-fixed-edge"
$TRAIN_CMD $BASE_LEARNED_FIXED_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping-tokenwise-fixed-edge" --out_dir=$EXP39_DIR
python sample.py --out_dir=$EXP39_DIR > $EXP39_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP39_DIR


echo "All experiments complete."
