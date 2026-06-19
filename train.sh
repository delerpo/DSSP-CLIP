#!/bin/bash
# ============================================================================
# DSSP-CLIP Training Script
# Trains on one dataset and evaluates on the remaining six zero-shot.
# Paper settings: ViT-L/14@336px, 5 epochs, Adam lr=1e-4, batch_size=8, 518x518
#
# Usage:
#   bash train.sh          # train on VisA, test on the other 6 datasets
#   bash train.sh mvtec    # train on MVTec AD, test on VisA + 5 others
# ============================================================================

DATA_DIR="${DATA_DIR:-./data}"
CLIP_DIR="${CLIP_DIR:-$HOME/.cache/clip}"
TRAIN_DATASET="${1:-visa}"

if [ "$TRAIN_DATASET" = "mvtec" ]; then
    # Train on MVTec AD, test on VisA + the other 5 datasets
    TEST_DATASETS="visa btad dtd dagm mpdd sdd"
else
    # Train on VisA, test on MVTec AD + the other 5 datasets
    TEST_DATASETS="mvtec btad dtd dagm mpdd sdd"
fi

echo "============================================"
echo "DSSP-CLIP Training"
echo "  Train dataset : $TRAIN_DATASET"
echo "  Test datasets : $TEST_DATASETS"
echo "  Data dir      : $DATA_DIR"
echo "  CLIP cache    : $CLIP_DIR"
echo "============================================"

python main.py \
    --clip_download_dir "$CLIP_DIR" \
    --data_dir "$DATA_DIR" \
    --dataset "$TRAIN_DATASET" \
    --model "ViT-L/14@336px" \
    --img_size 518 \
    --epochs 5 \
    --lr 0.0001 \
    --batch_size 8 \
    --prompt_len 12 \
    --feature_layers 6 12 18 24 \
    --cspf_start 3 \
    --cspf_end 13 \
    --lambda1 1.0 \
    --lambda2 1.0 \
    --val_freq 1 \
    --log_dir ./train_log \
    --test_dataset $TEST_DATASETS \
    --seed 122

echo "Training completed."
