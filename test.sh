#!/bin/bash
# ============================================================================
# DSSP-CLIP Zero-Shot Evaluation Script
# Loads pre-trained weights and evaluates on unseen test datasets.
# Paper: Train on one dataset → zero-shot test on the remaining six.
#
# Usage:
#   bash test.sh                              # test VisA (weights from MVTec AD)
#   bash test.sh mvtec visa                   # test single: train=mvtec test=visa
#   bash test.sh mvtec "visa btad dtd"        # test multiple datasets
#   bash test.sh visa mvtec                   # test reverse
# ============================================================================

DATA_DIR="${DATA_DIR:-./data}"
CLIP_DIR="${CLIP_DIR:-$HOME/.cache/clip}"
WEIGHT_DIR="${WEIGHT_DIR:-./weights}"

# Defaults: weights trained on MVTec AD, test on VisA
TRAIN_DATASET="${1:-mvtec}"
TEST_DATASETS="${2:-visa}"

echo "============================================"
echo "DSSP-CLIP Zero-Shot Evaluation"
echo "  Weight source : $TRAIN_DATASET"
echo "  Test datasets : $TEST_DATASETS"
echo "  Weight dir    : $WEIGHT_DIR"
echo "  Data dir      : $DATA_DIR"
echo "  CLIP cache    : $CLIP_DIR"
echo "============================================"

python main.py \
    --clip_download_dir "$CLIP_DIR" \
    --data_dir "$DATA_DIR" \
    --dataset "$TRAIN_DATASET" \
    --model "ViT-L/14@336px" \
    --img_size 518 \
    --feature_layers 6 12 18 24 \
    --cspf_start 3 \
    --cspf_end 13 \
    --prompt_len 12 \
    --batch_size 8 \
    --weight "$WEIGHT_DIR" \
    --log_dir ./test_log \
    --test_dataset $TEST_DATASETS \
    --seed 122

echo "Evaluation completed."
