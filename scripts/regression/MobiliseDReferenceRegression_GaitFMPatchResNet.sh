#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python3}"

ROOT_PATH=${ROOT_PATH:-./data/Mobilise-D-TVS}
OUTPUT_DIR=${OUTPUT_DIR:-./results_mobilised_reference_patch_resnet_sensorpretrain}
TARGETS=${TARGETS:-"gait_speed step_length step_time"}

MAE_TAG=${MAE_TAG:-MAETiny_GaitFM_RefPatchResNet}
MAE_MODEL=${MAE_MODEL:-mae_vit_tiny_patch16}
MAE_CKPT=${MAE_CKPT:-./checkpoints/backbones/mae_tiny_400e.pth.tar}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-./ckpt/gaitfm_pretrain_JointFM_ClinicalGait_FoG_mPower_WearGaitPD_COMPWALK_GSTRIDE_MobiliseD_MAETiny_GaitFMPretrain_sl250_mc6_mr0.75_0/checkpoint.pth}

TESTS=${TESTS:-"Test5 Test6 Test7"}
SENSOR_SYSTEMS=${SENSOR_SYSTEMS:-"INDIP SU"}
BODY_PARTS=${BODY_PARTS:-"LowerBack"}
TARGET_RATE=${TARGET_RATE:-50}
WINDOW_SECONDS=${WINDOW_SECONDS:-5}
STRIDE=${STRIDE:-250}
MIN_REF_COVERAGE=${MIN_REF_COVERAGE:-0.6}
LABEL_AGGREGATION=${LABEL_AGGREGATION:-overlap_weighted}

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
CACHE_DIR=${CACHE_DIR:-./cache_mobilised_reference_regression}

export GAITFM_CHANNEL_MASK_MODE=${GAITFM_CHANNEL_MASK_MODE:-random_acc_only}
export GAITFM_CHANNEL_MASK_PROB=${GAITFM_CHANNEL_MASK_PROB:-0}
export GAITFM_CHANNEL_MASK_APPLY_EVAL=${GAITFM_CHANNEL_MASK_APPLY_EVAL:-0}

for TARGET in ${TARGETS}; do
  CACHE_NPZ="${CACHE_DIR}/mobilised_${TARGET}_tests-${TESTS// /_}_parts-${BODY_PARTS// /_}_sr${TARGET_RATE}_ws${WINDOW_SECONDS}_stride${STRIDE}_seed${SEED}.npz"
  "${PYTHON}" scripts/train_mobilised_tvs_reference_patch_resnet.py \
    --target "${TARGET}" \
    --root_path "${ROOT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --mae_checkpoint_path "${MAE_CKPT}" \
    --mae_model "${MAE_MODEL}" \
    --mae_tag "${MAE_TAG}" \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    --tests ${TESTS} \
    --sensor_systems ${SENSOR_SYSTEMS} \
    --body_parts ${BODY_PARTS} \
    --target_rate "${TARGET_RATE}" \
    --window_seconds "${WINDOW_SECONDS}" \
    --stride "${STRIDE}" \
    --min_ref_coverage "${MIN_REF_COVERAGE}" \
    --label_aggregation "${LABEL_AGGREGATION}" \
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
