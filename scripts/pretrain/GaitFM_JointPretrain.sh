#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
CLINICAL_ROOT=${CLINICAL_ROOT:-./data/ClinicalGait/data}
FOG_ROOT=${FOG_ROOT:-./data/FoGContest}
MPOWER_ROOT=${MPOWER_ROOT:-./data/mPower_full_50hz_5s}
WEARGAIT_ROOT=${WEARGAIT_ROOT:-./data/WearGait-PD_full_50hz_5s}
COMPWALK_ROOT=${COMPWALK_ROOT:-./data/COMPWALK-ACL_50hz_5s}
GSTRIDE_ROOT=${GSTRIDE_ROOT:-./data/GSTRIDE}
MOBILISED_ROOT=${MOBILISED_ROOT:-./data/Mobilise-D-TVS_50hz_5s}
MAE_CKPT=${MAE_CKPT:-./checkpoints/backbones/mae_tiny_400e.pth.tar}
MAE_MODEL=${MAE_MODEL:-mae_vit_tiny_patch16}
MAE_TAG=${MAE_TAG:-MAETiny}

SEQ_LEN=${SEQ_LEN:-250}
IMAGE_SIZE=${IMAGE_SIZE:-224}
EPOCHS=5
BATCH_SIZE=${BATCH_SIZE:-2048}
LR=${LR:-0.00001}
LR_SCHEDULE=${LR_SCHEDULE:-type1}
JOINT_EPOCH_STEPS=${JOINT_EPOCH_STEPS:-0}
JOINT_NUM_WORKERS=${JOINT_NUM_WORKERS:-0}
JOINT_PREFETCH_FACTOR=${JOINT_PREFETCH_FACTOR:-2}
JOINT_LOG_EVERY=${JOINT_LOG_EVERY:-200}
SHARD_CACHE_SIZE=${SHARD_CACHE_SIZE:-2}
LOAD_SHARDS_TO_MEMORY=${LOAD_SHARDS_TO_MEMORY:-1}
MASK_RATIO=${MASK_RATIO:-0.75}
DROP_GYRO_PROB=${DROP_GYRO_PROB:-0.5}

USE_CLINICAL=${USE_CLINICAL:-1}
USE_FOG=${USE_FOG:-1}
USE_MPOWER=${USE_MPOWER:-1}
USE_WEARGAIT=${USE_WEARGAIT:-1}
USE_COMPWALK=${USE_COMPWALK:-1}
USE_GSTRIDE=${USE_GSTRIDE:-1}
USE_MOBILISED=${USE_MOBILISED:-1}
GSTRIDE_TASK=${GSTRIDE_TASK:-pretrain}
DISEASES=${DISEASES:-"ACL CIPN CVA HOA KOA PD RIL"}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export GAITFM_PRETRAIN_DROP_GYRO_PROB="$DROP_GYRO_PROB"

"$PYTHON" -u run_gaitfm_pretrain.py \
  --model_id JointFM_ClinicalGait_FoG_mPower_WearGaitPD_COMPWALK_GSTRIDE_MobiliseD_${MAE_TAG} \
  --clinical_root "$CLINICAL_ROOT" \
  --fog_root "$FOG_ROOT" \
  --mpower_root "$MPOWER_ROOT" \
  --weargait_root "$WEARGAIT_ROOT" \
  --compwalk_root "$COMPWALK_ROOT" \
  --gstride_root "$GSTRIDE_ROOT" \
  --mobilised_root "$MOBILISED_ROOT" \
  --joint_use_clinical "$USE_CLINICAL" \
  --joint_use_fog "$USE_FOG" \
  --joint_use_mpower "$USE_MPOWER" \
  --joint_use_weargait "$USE_WEARGAIT" \
  --joint_use_compwalk "$USE_COMPWALK" \
  --joint_use_gstride "$USE_GSTRIDE" \
  --joint_use_mobilised "$USE_MOBILISED" \
  --gstride_task "$GSTRIDE_TASK" \
  --joint_clinical_diseases $DISEASES \
  --joint_epoch_steps "$JOINT_EPOCH_STEPS" \
  --joint_num_workers "$JOINT_NUM_WORKERS" \
  --joint_prefetch_factor "$JOINT_PREFETCH_FACTOR" \
  --joint_log_every "$JOINT_LOG_EVERY" \
  --shard_cache_size "$SHARD_CACHE_SIZE" \
  --load_shards_to_memory "$LOAD_SHARDS_TO_MEMORY" \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --dropout 0.1 \
  --experiment_name GaitFM_JointPretrain_${MAE_TAG} \
  --learning_rate "$LR" \
  --lr_schedule "$LR_SCHEDULE" \
  --train_epochs "$EPOCHS" \
  --gait_image_size "$IMAGE_SIZE" \
  --gait_segment_stride 10 \
  --gait_max_channels 6 \
  --mae_checkpoint_path "$MAE_CKPT" \
  --mae_model "$MAE_MODEL" \
  --mae_mask_ratio "$MASK_RATIO" \
  --mae_norm_pix_loss 1
