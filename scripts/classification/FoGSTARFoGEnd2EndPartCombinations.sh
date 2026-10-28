#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
ROOT_PATH=${ROOT_PATH:-./data/FoG-STAR}
OUTPUT_DIR=${OUTPUT_DIR:-./results_end2end_part_combinations}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-./ckpt/gaitfm_pretrain_JointFM_ClinicalGait_FoG_mPower_WearGaitPD_COMPWALK_GSTRIDE_MobiliseD_MAETiny_GaitFMPretrain_sl250_mc6_mr0.75_0/checkpoint.pth}
MAE_CKPT=${MAE_CKPT:-./checkpoints/backbones/mae_tiny_400e.pth.tar}
MAE_MODEL=${MAE_MODEL:-mae_vit_tiny_patch16}
MAE_TAG=${MAE_TAG:-MAETiny_GaitFM}
PRETRAIN_EPOCH=${PRETRAIN_EPOCH:-bestval}
POOLING=${POOLING:-cls}

TASK_IDS=${TASK_IDS:-"3 4 5 6"}
TARGET_RATE=${TARGET_RATE:-50}
WINDOW_SECONDS=${WINDOW_SECONDS:-5}
STRIDE=${STRIDE:-50}
FOG_THRESHOLD=${FOG_THRESHOLD:-0.5}
MAX_INTERP_GAP_SECONDS=${MAX_INTERP_GAP_SECONDS:-0.1}
MAX_COMBO_SIZE=${MAX_COMBO_SIZE:-0}
SPLIT_SEED=${SPLIT_SEED:-2021}
EPOCHS=20
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-0.00001}
CLASSIFIER_LR=${CLASSIFIER_LR:-0.0001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
HIDDEN_DIM=${HIDDEN_DIM:-256}
DROPOUT=${DROPOUT:-0.2}
NUM_WORKERS=${NUM_WORKERS:-0}
SEED=${SEED:-2021}
GPU=${GPU:-0}
INSPECT_ONLY=${INSPECT_ONLY:-0}

if [[ "$INSPECT_ONLY" == "1" ]]; then
  set -- --inspect_only
else
  set --
fi

"$PYTHON" scripts/train_fogstar_end2end_part_combinations.py \
  --root_path "$ROOT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --model_id "FoGSTAR_FoG_${MAE_TAG}_End2EndPartCombinations_epoch${PRETRAIN_EPOCH}_${POOLING}_task3456_seed${SEED}" \
  --pretrain_ckpt "$PRETRAIN_CKPT" \
  --mae_checkpoint_path "$MAE_CKPT" \
  --mae_model "$MAE_MODEL" \
  --mae_tag "$MAE_TAG" \
  --mae_finetune_pooling "$POOLING" \
  --task_ids ${TASK_IDS} \
  --target_rate "$TARGET_RATE" \
  --window_seconds "$WINDOW_SECONDS" \
  --stride "$STRIDE" \
  --fog_threshold "$FOG_THRESHOLD" \
  --max_interp_gap_seconds "$MAX_INTERP_GAP_SECONDS" \
  --max_combo_size "$MAX_COMBO_SIZE" \
  --split_seed "$SPLIT_SEED" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" \
  --classifier_learning_rate "$CLASSIFIER_LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --hidden_dim "$HIDDEN_DIM" \
  --dropout "$DROPOUT" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --gpu "$GPU" \
  "$@"
