#!/bin/bash

# Base configuration
CONFIG="config/train_shakespeare_char.py"
WANDB_PROJECT="nanoGPT-experiments-shakespeare"

# --- Base Experiments ---

# Experiment 1: Baseline
echo "Running experiment 1: Baseline"
EXP1_DIR="out-shakespeare-char-baseline"
python train.py $CONFIG --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="baseline" --out_dir=$EXP1_DIR --compile=False --batch_size=12
echo "Sampling from experiment 1"
python sample.py --out_dir=$EXP1_DIR > $EXP1_DIR/samples.txt

# Experiment 2: Shared Parameters
echo "Running experiment 2: Shared Parameters"
EXP2_DIR="out-shakespeare-char-shared-params"
python train.py $CONFIG --share_parameters_across_layers=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="shared-params" --out_dir=$EXP2_DIR --compile=False --batch_size=24
echo "Sampling from experiment 2"
python sample.py --out_dir=$EXP2_DIR > $EXP2_DIR/samples.txt

# Experiment 3: Recurrent Shared Weights
echo "Running experiment 3: Recurrent Shared Weights"
EXP3_DIR="out-shakespeare-char-recurrent-shared"
python train.py $CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared" --out_dir=$EXP3_DIR --compile=False --batch_size=24
echo "Sampling from experiment 3"
python sample.py --out_dir=$EXP3_DIR > $EXP3_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP3_DIR

# --- MoE Experiments ---

# Experiment 4: MoE with Soft Routing
echo "Running experiment 4: MoE with Soft Routing"
EXP4_DIR="out-shakespeare-char-moe-soft-routing"
python train.py $CONFIG --moe=True --moe_top_k=1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="moe-soft-routing" --out_dir=$EXP4_DIR --compile=False --batch_size=12
echo "Sampling from experiment 4"
python sample.py --out_dir=$EXP4_DIR > $EXP4_DIR/samples.txt

# Experiment 5: MoE with Hard Routing
echo "Running experiment 5: MoE with Hard Routing"
EXP5_DIR="out-shakespeare-char-moe-hard-routing"
python train.py $CONFIG --moe=True --moe_hard_routing=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="moe-hard-routing" --out_dir=$EXP5_DIR --compile=False --batch_size=16 --moe_top_k=1
echo "Sampling from experiment 5"
python sample.py --out_dir=$EXP5_DIR > $EXP5_DIR/samples.txt

# Experiment 6: Shared MoE
echo "Running experiment 6: Shared MoE"
EXP6_DIR="out-shakespeare-char-shared-moe"
python train.py $CONFIG --moe=True --share_moe_experts=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="shared-moe" --out_dir=$EXP6_DIR --compile=False --batch_size=12
echo "Sampling from experiment 6"
python sample.py --out_dir=$EXP6_DIR > $EXP6_DIR/samples.txt

# --- Recurrent Shared Weights Experiments (Isolated Features) ---

# Base for this section: Recurrent Shared Weights (like Exp 3)
BASE_RECURRENT_ARGS="$CONFIG --share_parameters_across_layers=True --recurrent_shared_weights=True --compile=False --batch_size=24"

# Experiment 7: Recurrent Shared Weights with Layer Dropout
echo "Running experiment 7: Recurrent Shared Weights with Layer Dropout"
EXP7_DIR="out-shakespeare-char-recurrent-layer-dropout"
python train.py $BASE_RECURRENT_ARGS --layer_dropout=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-layer-dropout" --out_dir=$EXP7_DIR
python sample.py --out_dir=$EXP7_DIR > $EXP7_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP7_DIR

# Experiment 8: Recurrent Shared Weights with Layer Dropout + Loss Scaling
echo "Running experiment 8: Recurrent Shared Weights with Layer Dropout + Loss Scaling"
EXP8_DIR="out-shakespeare-char-recurrent-layer-dropout-loss-scaling"
python train.py $BASE_RECURRENT_ARGS --layer_dropout=0.1 --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-layer-dropout-loss-scaling" --out_dir=$EXP8_DIR
python sample.py --out_dir=$EXP8_DIR > $EXP8_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP8_DIR

# Experiment 9: Recurrent Shared Weights with Sticky Dropout
echo "Running experiment 9: Recurrent Shared Weights with Sticky Dropout"
EXP9_DIR="out-shakespeare-char-recurrent-sticky-dropout"
python train.py $BASE_RECURRENT_ARGS --sticky_dropout=0.1 --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-sticky-dropout" --out_dir=$EXP9_DIR
python sample.py --out_dir=$EXP9_DIR > $EXP9_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP9_DIR

# Experiment 10: Recurrent Shared Weights with Sticky Dropout + Loss Scaling
echo "Running experiment 10: Recurrent Shared Weights with Sticky Dropout + Loss Scaling"
EXP10_DIR="out-shakespeare-char-recurrent-sticky-dropout-loss-scaling"
python train.py $BASE_RECURRENT_ARGS --sticky_dropout=0.1 --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-sticky-dropout-loss-scaling" --out_dir=$EXP10_DIR
python sample.py --out_dir=$EXP10_DIR > $EXP10_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP10_DIR

# Experiment 11: Recurrent Shared Weights with Learned Stopping
echo "Running experiment 11: Recurrent Shared Weights with Learned Stopping"
EXP11_DIR="out-shakespeare-char-recurrent-learned-stopping"
python train.py $BASE_RECURRENT_ARGS --learned_stopping=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping" --out_dir=$EXP11_DIR
python sample.py --out_dir=$EXP11_DIR > $EXP11_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP11_DIR

# Experiment 12: Recurrent Shared Weights with Learned Stopping + Loss Scaling
echo "Running experiment 12: Recurrent Shared Weights with Learned Stopping + Loss Scaling"
EXP12_DIR="out-shakespeare-char-recurrent-learned-stopping-loss-scaling"
python train.py $BASE_RECURRENT_ARGS --learned_stopping=True --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-learned-stopping-loss-scaling" --out_dir=$EXP12_DIR
python sample.py --out_dir=$EXP12_DIR > $EXP12_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP12_DIR

# Experiment 13: Recurrent Shared Weights with Attentive Stopping
echo "Running experiment 13: Recurrent Shared Weights with Attentive Stopping"
EXP13_DIR="out-shakespeare-char-recurrent-attentive-stopping"
python train.py $BASE_RECURRENT_ARGS --attentive_stopping=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-attentive-stopping" --out_dir=$EXP13_DIR
python sample.py --out_dir=$EXP13_DIR > $EXP13_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP13_DIR

# Experiment 14: Recurrent Shared Weights with Attentive Stopping + Loss Scaling
echo "Running experiment 14: Recurrent Shared Weights with Attentive Stopping + Loss Scaling"
EXP14_DIR="out-shakespeare-char-recurrent-attentive-stopping-loss-scaling"
python train.py $BASE_RECURRENT_ARGS --attentive_stopping=True --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-attentive-stopping-loss-scaling" --out_dir=$EXP14_DIR
python sample.py --out_dir=$EXP14_DIR > $EXP14_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP14_DIR


# --- Combined Recurrent + MoE Experiments ---

# Base for this section: Recurrent Shared Weights + Shared MoE
BASE_RECURRENT_MOE_ARGS="$BASE_RECURRENT_ARGS --moe=True --share_moe_experts=True --batch_size=12"

# Experiment 15: Recurrent Shared MoE
echo "Running experiment 15: Recurrent Shared MoE"
EXP15_DIR="out-shakespeare-char-recurrent-shared-moe"
python train.py $BASE_RECURRENT_MOE_ARGS --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-moe" --out_dir=$EXP15_DIR
python sample.py --out_dir=$EXP15_DIR > $EXP15_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP15_DIR

# Experiment 16: Recurrent Shared MoE with Loss Scaling
echo "Running experiment 16: Recurrent Shared MoE with Loss Scaling"
EXP16_DIR="out-shakespeare-char-recurrent-shared-moe-loss-scaling"
python train.py $BASE_RECURRENT_MOE_ARGS --scale_loss_by_n_layer=True --wandb_log=True --wandb_project=$WANDB_PROJECT --wandb_run_name="recurrent-shared-moe-loss-scaling" --out_dir=$EXP16_DIR
python sample.py --out_dir=$EXP16_DIR > $EXP16_DIR/samples.txt
python plot_recurrent_loss.py --out_dir=$EXP16_DIR

echo "All experiments complete."