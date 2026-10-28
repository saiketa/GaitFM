#!/usr/bin/env bash
set -euo pipefail

EPOCHS=5
MAE_CKPT=${MAE_CKPT:-none}
MAE_MODEL=${MAE_MODEL:-mae_vit_tiny_patch16}
MAE_TAG=${MAE_TAG:-MAETiny_ScratchSensorPretrain}
MPOWER_ROOT=${MPOWER_ROOT:-./data/mPower_full_50hz_5s}
WEARGAIT_ROOT=${WEARGAIT_ROOT:-./data/WearGait-PD_full_50hz_5s}
COMPWALK_ROOT=${COMPWALK_ROOT:-./data/COMPWALK-ACL_50hz_5s}
GSTRIDE_ROOT=${GSTRIDE_ROOT:-./data/GSTRIDE}
MOBILISED_ROOT=${MOBILISED_ROOT:-./data/Mobilise-D-TVS_50hz_5s}

export EPOCHS
export MAE_CKPT
export MAE_MODEL
export MAE_TAG
export MPOWER_ROOT
export WEARGAIT_ROOT
export COMPWALK_ROOT
export GSTRIDE_ROOT
export MOBILISED_ROOT

bash scripts/pretrain/GaitFM_JointPretrain.sh
