#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
ROOT_PATH=${ROOT_PATH:-./data/WearGait-PD_full_50hz_5s}
OUTPUT_DIR=${OUTPUT_DIR:-./results_end2end_part_combinations}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-./ckpt/gaitfm_pretrain_JointFM_ClinicalGait_FoG_mPower_WearGaitPD_COMPWALK_GSTRIDE_MobiliseD_MAETiny_GaitFMPretrain_sl250_mc6_mr0.75_0/checkpoint.pth}
MAE_CKPT=${MAE_CKPT:-./checkpoints/backbones/mae_tiny_400e.pth.tar}
MAE_MODEL=${MAE_MODEL:-mae_vit_tiny_patch16}
MAE_TAG=${MAE_TAG:-MAETiny}
PRETRAIN_EPOCH=${PRETRAIN_EPOCH:-bestval}
SEQ_LEN=${SEQ_LEN:-250}
EPOCHS=20
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.00001}
CLASSIFIER_LR=${CLASSIFIER_LR:-0.0001}
HIDDEN_DIM=${HIDDEN_DIM:-256}
DROPOUT=${DROPOUT:-0.2}
NUM_WORKERS=${NUM_WORKERS:-0}
POOLING=${POOLING:-cls}
LOAD_SHARDS_TO_MEMORY=${LOAD_SHARDS_TO_MEMORY:-1}
MAX_COMBO_SIZE=${MAX_COMBO_SIZE:-5}
WEARGAIT_BODY_PARTS=${WEARGAIT_BODY_PARTS:-"R_Wrist L_Wrist R_MidLatThigh L_MidLatThigh Forehead"}
WEARGAIT_TASKS=${WEARGAIT_TASKS:-"SelfPace SP HurriedPace HP SelfPace_mat SPm HurriedPace_mat HPm SelfPace_doorpat SPdoorpat"}
SEED=${SEED:-2021}
SPLIT_SEED=${SPLIT_SEED:-2021}

read -r -a BODY_PART_LIST <<< "$WEARGAIT_BODY_PARTS"
read -r -a TASK_LIST <<< "$WEARGAIT_TASKS"

"$PYTHON" scripts/train_weargaitpd_end2end_part_combinations.py \
  --root_path "$ROOT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --model_id "WearGaitPD_${MAE_TAG}_End2EndPartCombinations_epoch${PRETRAIN_EPOCH}_${POOLING}_bp5_wrists_thighs_forehead_max${MAX_COMBO_SIZE}_seed${SEED}" \
  --pretrain_ckpt "$PRETRAIN_CKPT" \
  --mae_checkpoint_path "$MAE_CKPT" \
  --mae_model "$MAE_MODEL" \
  --mae_tag "$MAE_TAG" \
  --mae_finetune_pooling "$POOLING" \
  --seq_len "$SEQ_LEN" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" \
  --classifier_learning_rate "$CLASSIFIER_LR" \
  --hidden_dim "$HIDDEN_DIM" \
  --dropout "$DROPOUT" \
  --num_workers "$NUM_WORKERS" \
  --load_shards_to_memory "$LOAD_SHARDS_TO_MEMORY" \
  --weargait_tasks "${TASK_LIST[@]}" \
  --max_combo_size "$MAX_COMBO_SIZE" \
  --weargait_body_parts "${BODY_PART_LIST[@]}" \
  --seed "$SEED" \
  --weargait_split_seed "$SPLIT_SEED"
