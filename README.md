# GaitFM: A Cross-Modal Foundation Model for Wearable Sensor-Based Gait Assessment in Healthcare

GaitFM is a cross-modal foundation model for wearable sensor-based gait assessment in clinical and free-living settings. It is built on a Transformer-based masked autoencoder (MAE). To mitigate the limited scale of gait-specific inertial measurement unit (IMU) data, GaitFM transfers knowledge across modalities by initializing from a model pretrained on large-scale natural images and then continuing pretraining on a curated gait IMU corpus collected from approximately 4,000 participants.

This repository provides sensor pretraining, downstream fine-tuning, preprocessing utilities, and experiment launchers for three clinical application domains:

- Gait Parameter Estimation
- Disease and Abnormality Diagnosis
- Clinical Outcome Prediction

Gait-parameter estimation is formulated as regression. Disease and abnormality diagnosis and clinical-outcome prediction are formulated as classification.

## Installation

Python 3.10 or newer is required. From the repository root, run:

```bash
python -m pip install -r requirements.txt
```

## Datasets

Download each dataset from its original provider and comply with its access terms.

### Sensor pretraining and internal validation

- [GSTRIDE](https://zenodo.org/records/17052815) — García-Villamil Neira et al. (2022)
- [FoGContest](https://zenodo.org/records/10959560) — Salomon et al. (2024)
- [COMPWALK-ACL](https://zenodo.org/records/15624356) — Yona et al. (2025)
- [mPower](https://www.synapse.org/Synapse:syn4993293/wiki/) — Bot et al. (2016)
- [WearGait-PD](https://www.synapse.org/Synapse:syn52540892/wiki/623751) — Anderson et al. (2024)
- [Mobilise-D](https://zenodo.org/records/13899386) — Mazzà et al. (2021)
- [ClinicalGait](https://doi.org/10.6084/m9.figshare.28806086) — Voisard et al. (2025)

### External validation

- [LTMM](https://physionet.org/content/ltmm/1.0.0/) — Weiss et al. (2013)
- [FoG-STAR](https://zenodo.org/records/17838806) — Borzì et al. (2026)
- [StrokeRehabilitation](https://zenodo.org/records/10534055) — Zhou et al. (2024)
- [GaitPrint](https://springernature.figshare.com/articles/dataset/NONAN_GaitPrint_An_IMU_gait_database_of_healthy_middle-aged_adults/29371796) — Wiles et al. (2025)

### Preprocessing

Prepare mPower, WearGait-PD, COMPWALK-ACL, and Mobilise-D with the corresponding `scripts/preprocess_*.py` utilities. Inspect each preprocessing interface before running it:

```bash
python scripts/preprocess_mpower_full_50hz_5s.py --help
python scripts/preprocess_weargaitpd_full_50hz_5s.py --help
python scripts/preprocess_compwalk_acl_50hz_5s.py --help
python scripts/preprocess_mobilised_tvs_50hz_5s.py --help
```

## Pretraining

### 1. Download the vision-pretrained MAE initialization

The joint-pretraining launcher uses a ViT-Tiny backbone from [MAE-Lite](https://github.com/wangsr126/MAE-Lite).
The commands below therefore set all three related launcher variables explicitly:

```bash
export MAE_CKPT=./checkpoints/backbones/mae_tiny_400e.pth.tar
export MAE_MODEL=mae_vit_tiny_patch16
export MAE_TAG=MAETiny
```

### 2. Continue pretraining on gait IMU data

```bash
python3 gaitfm.py run pretrain/GaitFM_JointPretrain \
  --set CLINICAL_ROOT=/path/to/ClinicalGait/data \
  --set FOG_ROOT=/path/to/FoGContest \
  --set MPOWER_ROOT=/path/to/mPower_full_50hz_5s \
  --set WEARGAIT_ROOT=/path/to/WearGait-PD_full_50hz_5s \
  --set COMPWALK_ROOT=/path/to/COMPWALK-ACL_50hz_5s \
  --set GSTRIDE_ROOT=/path/to/GSTRIDE \
  --set MOBILISED_ROOT=/path/to/Mobilise-D-TVS_50hz_5s \
  --set MAE_CKPT="$MAE_CKPT" \
  --set MAE_MODEL="$MAE_MODEL" \
  --set MAE_TAG="$MAE_TAG"
```

The joint-pretraining launcher uses five epochs, a batch size of 2,048, a learning rate of `1e-5`, and a masking ratio of 0.75 by default. You may override launcher variables such as `BATCH_SIZE` to fit available hardware.

With the MAE-Tiny settings above, the best validation-loss checkpoint is written to `./ckpt/gaitfm_pretrain_JointFM_ClinicalGait_FoG_mPower_WearGaitPD_COMPWALK_GSTRIDE_MobiliseD_MAETiny_GaitFMPretrain_sl250_mc6_mr0.75_0/checkpoint.pth`。

For convenience:

```bash
export GAITFM_CKPT=./ckpt/gaitfm_pretrain_JointFM_ClinicalGait_FoG_mPower_WearGaitPD_COMPWALK_GSTRIDE_MobiliseD_MAETiny_GaitFMPretrain_sl250_mc6_mr0.75_0/checkpoint.pth
```

## Fine-tuning GaitFM on Downstream Tasks

List all available launchers:

```bash
python gaitfm.py list
python gaitfm.py list --category classification
python gaitfm.py list --category regression
```

Append `--dry-run` to any `gaitfm.py run` command to inspect the selected launcher without starting training.

### Disease and abnormality classification

The following example fine-tunes GaitFM on the seven ClinicalGait binary disease tasks:

```bash
python3 gaitfm.py run classification/ClinicalGaitEnd2EndPartCombinations \
  --set ROOT_PATH=/path/to/ClinicalGait/data \
  --set DISEASES="ACL CIPN CVA HOA KOA PD RIL" \
  --set MAE_CKPT="$MAE_CKPT" \
  --set MAE_MODEL="$MAE_MODEL" \
  --set MAE_TAG="$MAE_TAG" \
  --set PRETRAIN_CKPT="$GAITFM_CKPT"
```

### Clinical outcome prediction

```bash
python3 gaitfm.py run classification/GaitFMSensorPretrain_GSTRIDE \
  --set ROOT_PATH=/path/to/GSTRIDE \
  --set TASKS=subsequent_year_fall \
  --set FOLLOWUP_LABELS=/path/to/GSTRIDE_subsequent_year_falls.csv \
  --set MAE_CKPT="$MAE_CKPT" \
  --set MAE_MODEL="$MAE_MODEL" \
  --set MAE_TAG="$MAE_TAG" \
  --set PRETRAIN_CKPT="$GAITFM_CKPT"
```

### Gait parameter regression

The following example fine-tunes the sensor-pretrained model for gait-speed, step-length, and step-time estimation on GaitPrint:

```bash
python3 gaitfm.py run regression/GaitPrintReferenceRegression_GaitFMPatchResNet \
  --set ROOT_PATH=/path/to/GaitPrint \
  --set TARGETS="gait_speed step_length step_time" \
  --set MAE_CKPT="$MAE_CKPT" \
  --set MAE_MODEL="$MAE_MODEL" \
  --set PRETRAIN_CKPT="$GAITFM_CKPT"
```

See [TASKS.md](TASKS.md) for the complete dataset-endpoint inventory.
