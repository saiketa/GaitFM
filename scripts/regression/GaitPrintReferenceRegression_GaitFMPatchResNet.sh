#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

ROOT_PATH=${ROOT_PATH:-./data/GaitPrint}
OUTPUT_DIR=${OUTPUT_DIR:-./results_gaitprint_reference_patch_resnet_sensorpretrain}
TARGETS=${TARGETS:-"gait_speed step_length step_time"}
BODY_PART=${BODY_PART:-Pelvis}

MAE_TAG=${MAE_TAG:-MAETiny_GaitFM_GaitPrintPatchResNet}
MAE_MODEL=${MAE_MODEL:-mae_vit_tiny_patch16}
MAE_CKPT=${MAE_CKPT:-./checkpoints/backbones/mae_tiny_400e.pth.tar}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-./ckpt/gaitfm_pretrain_JointFM_ClinicalGait_FoG_mPower_WearGaitPD_COMPWALK_GSTRIDE_MobiliseD_MAETiny_GaitFMPretrain_sl250_mc6_mr0.75_0/checkpoint.pth}

TARGET_RATE=${TARGET_RATE:-50}
WINDOW_SECONDS=${WINDOW_SECONDS:-5}
STRIDE=${STRIDE:-250}
EPOCHS=20
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-1e-5}
HEAD_LR=${HEAD_LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
DROPOUT=${DROPOUT:-0.2}
NUM_WORKERS=${NUM_WORKERS:-0}
SEED=${SEED:-2021}
GPU=${GPU:-0}
STANDARDIZE_TARGET=${STANDARDIZE_TARGET:-1}
CACHE_DIR=${CACHE_DIR:-./cache_gaitprint_reference_regression}

export GAITFM_CHANNEL_MASK_MODE=${GAITFM_CHANNEL_MASK_MODE:-random_acc_only}
export GAITFM_CHANNEL_MASK_PROB=${GAITFM_CHANNEL_MASK_PROB:-0}
export GAITFM_CHANNEL_MASK_APPLY_EVAL=${GAITFM_CHANNEL_MASK_APPLY_EVAL:-0}

for TARGET in ${TARGETS}; do
  CACHE_NPZ="${CACHE_DIR}/gaitprint_${TARGET}_body-${BODY_PART}_sr${TARGET_RATE}_ws${WINDOW_SECONDS}_stride${STRIDE}_seed${SEED}.npz"
  "${PYTHON}" scripts/train_gaitprint_reference_patch_resnet.py \
    --target "${TARGET}" \
    --root_path "${ROOT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_id "GaitPrint_${TARGET}_${BODY_PART}_${MAE_TAG}_SensorPretrainPatchResNet" \
    --mae_checkpoint_path "${MAE_CKPT}" \
    --mae_model "${MAE_MODEL}" \
    --mae_tag "${MAE_TAG}" \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    --body_part "${BODY_PART}" \
    --target_rate "${TARGET_RATE}" \
    --window_seconds "${WINDOW_SECONDS}" \
    --stride "${STRIDE}" \
    --cache_npz "${CACHE_NPZ}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --head_learning_rate "${HEAD_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --dropout "${DROPOUT}" \
    --num_workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --gpu "${GPU}" \
    --standardize_target "${STANDARDIZE_TARGET}"
done
