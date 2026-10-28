#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
ROOT_PATH=${ROOT_PATH:-./data/FoGContest}
OUTPUT_DIR=${OUTPUT_DIR:-./results_end2end_single_part}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-./ckpt/gaitfm_pretrain_JointFM_ClinicalGait_FoG_mPower_WearGaitPD_COMPWALK_GSTRIDE_MobiliseD_MAETiny_GaitFMPretrain_sl250_mc6_mr0.75_0/checkpoint.pth}
MAE_CKPT=${MAE_CKPT:-./checkpoints/backbones/mae_tiny_400e.pth.tar}
MAE_MODEL=${MAE_MODEL:-mae_vit_tiny_patch16}
MAE_TAG=${MAE_TAG:-MAETiny}
PRETRAIN_EPOCH=${PRETRAIN_EPOCH:-bestval}
SEQ_LEN=${SEQ_LEN:-250}
STRIDE=${STRIDE:-250}
EPOCHS=20
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.00001}
CLASSIFIER_LR=${CLASSIFIER_LR:-0.0001}
HIDDEN_DIM=${HIDDEN_DIM:-256}
DROPOUT=${DROPOUT:-0.2}
NUM_WORKERS=${NUM_WORKERS:-0}
POOLING=${POOLING:-cls}
SEED=${SEED:-2021}
SPLIT_SEED=${SPLIT_SEED:-2021}

"$PYTHON" scripts/train_singlepart_end2end.py \
  --data FoGContest \
  --root_path "$ROOT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --model_id "FoGContest_${MAE_TAG}_End2EndSinglePart_epoch${PRETRAIN_EPOCH}_${POOLING}_seed${SEED}" \
  --pretrain_ckpt "$PRETRAIN_CKPT" \
  --mae_checkpoint_path "$MAE_CKPT" \
  --mae_model "$MAE_MODEL" \
  --mae_tag "$MAE_TAG" \
  --mae_finetune_pooling "$POOLING" \
  --seq_len "$SEQ_LEN" \
  --stride "$STRIDE" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" \
  --classifier_learning_rate "$CLASSIFIER_LR" \
  --hidden_dim "$HIDDEN_DIM" \
  --dropout "$DROPOUT" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --split_seed "$SPLIT_SEED"
