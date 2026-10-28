#!/usr/bin/env bash
set -euo pipefail

DISEASES=${DISEASES:-"PD MS COPD CHF"}
PYTHON=${PYTHON:-python3}
ROOT_PATH=${ROOT_PATH:-./data/Mobilise-D-TVS_50hz_5s}
MAE_CKPT=${MAE_CKPT:-./checkpoints/backbones/mae_tiny_400e.pth.tar}
MAE_MODEL=${MAE_MODEL:-mae_vit_tiny_patch16}
MAE_TAG=${MAE_TAG:-MAETiny}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-./ckpt/gaitfm_pretrain_JointFM_ClinicalGait_FoG_mPower_WearGaitPD_COMPWALK_GSTRIDE_MobiliseD_${MAE_TAG}_GaitFMPretrain_sl250_mc6_mr0.75_0/checkpoint.pth}
SEQ_LEN=${SEQ_LEN:-250}
EPOCHS=20
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-0.00001}
NUM_WORKERS=${NUM_WORKERS:-10}
LOAD_SHARDS_TO_MEMORY=${LOAD_SHARDS_TO_MEMORY:-1}
POOLING=${POOLING:-cls}
SEED=${SEED:-2021}
SPLIT_SEED=${SPLIT_SEED:-2021}

read -r -a DISEASE_LIST <<< "$DISEASES"

for DISEASE in "${DISEASE_LIST[@]}"; do
  "$PYTHON" -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path "$ROOT_PATH" \
    --model_id "MobiliseDTVS_HA_vs_${DISEASE}_${MAE_TAG}SensorPretrain_seed${SEED}" \
    --model GaitFMSensorPretrainFinetune \
    --data MobiliseDTVSProcessed \
    --mobilised_subset finetune \
    --mobilised_disease "$DISEASE" \
    --mobilised_val_ratio 0.2 \
    --mobilised_test_ratio 0.2 \
    --mobilised_split_seed "$SPLIT_SEED" \
    --seq_len "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --dropout 0.5 \
    --experiment_name "GaitFMSensorPretrain_${MAE_TAG}_MobiliseDTVS_${DISEASE}" \
    --learning_rate "$LR" \
    --train_epochs "$EPOCHS" \
    --gait_image_size 224 \
    --gait_segment_stride 10 \
    --mae_checkpoint_path "$MAE_CKPT" \
    --mae_model "$MAE_MODEL" \
    --mae_finetune_pooling "$POOLING" \
    --gaitfm_pretrain_checkpoint "$PRETRAIN_CKPT" \
    --load_shards_to_memory "$LOAD_SHARDS_TO_MEMORY" \
    --seed "$SEED"
done
