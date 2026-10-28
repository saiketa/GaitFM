#!/usr/bin/env bash
set -euo pipefail

TASKS=${TASKS:-${GSTRIDE_TASK:-"frailty subsequent_year_fall"}}
PYTHON=${PYTHON:-python3}
ROOT_PATH=${ROOT_PATH:-./data/GSTRIDE}
FOLLOWUP_LABELS=${FOLLOWUP_LABELS:-${ROOT_PATH}/GSTRIDE_subsequent_year_falls.csv}
MAE_CKPT=${MAE_CKPT:-./checkpoints/backbones/mae_tiny_400e.pth.tar}
MAE_MODEL=${MAE_MODEL:-mae_vit_tiny_patch16}
MAE_TAG=${MAE_TAG:-MAETiny}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-./ckpt/gaitfm_pretrain_JointFM_ClinicalGait_FoG_mPower_WearGaitPD_COMPWALK_GSTRIDE_MobiliseD_${MAE_TAG}_GaitFMPretrain_sl250_mc6_mr0.75_0/checkpoint.pth}
SEQ_LEN=${SEQ_LEN:-250}
WINDOW_SECONDS=${WINDOW_SECONDS:-5}
STRIDE=${STRIDE:-250}
EPOCHS=20
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-0.00001}
NUM_WORKERS=${NUM_WORKERS:-10}
POOLING=${POOLING:-cls}
SEED=${SEED:-2021}
SPLIT_SEED=${SPLIT_SEED:-2021}

read -r -a TASK_LIST <<< "$TASKS"

for TASK in "${TASK_LIST[@]}"; do
  "$PYTHON" -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path "$ROOT_PATH" \
    --model_id "GSTRIDE_${TASK}_${MAE_TAG}SensorPretrain_seed${SEED}" \
    --model GaitFMSensorPretrainFinetune \
    --data GSTRIDE \
    --gstride_task "$TASK" \
    --gstride_followup_labels "$FOLLOWUP_LABELS" \
    --gstride_target_rate 50 \
    --gstride_window_seconds "$WINDOW_SECONDS" \
    --gstride_stride "$STRIDE" \
    --gstride_val_ratio 0.2 \
    --gstride_test_ratio 0.2 \
    --gstride_split_seed "$SPLIT_SEED" \
    --seq_len "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --dropout 0.5 \
    --experiment_name "GaitFMSensorPretrain_${MAE_TAG}_GSTRIDE_${TASK}" \
    --learning_rate "$LR" \
    --train_epochs "$EPOCHS" \
    --gait_image_size 224 \
    --gait_segment_stride 10 \
    --mae_checkpoint_path "$MAE_CKPT" \
    --mae_model "$MAE_MODEL" \
    --mae_finetune_pooling "$POOLING" \
    --gaitfm_pretrain_checkpoint "$PRETRAIN_CKPT" \
    --seed "$SEED"
done
