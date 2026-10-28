# Supported task inventory

## Sensor pretraining

The training partitions of these seven internal datasets are used for gradient updates. Their disjoint validation partitions are used only to select the minimum-validation-loss pretraining checkpoint. Test partitions are not accessed during pretraining:

- ClinicalGait
- COMPWALK-ACL
- FoGContest
- GSTRIDE
- Mobilise-D
- WearGait-PD
- mPower

FoG-STAR, GaitPrint, LTMM, and StrokeRehabilitation are external datasets and are used only for downstream fine-tuning and evaluation.

## Gait parameter estimation

| Dataset | Endpoint | Task type |
|---|---|---|
| Mobilise-D | Gait speed | Regression |
| Mobilise-D | Step length | Regression |
| Mobilise-D | Step time | Regression |
| GaitPrint | Gait speed | Regression |
| GaitPrint | Step length | Regression |
| GaitPrint | Step time | Regression |

## Disease and gait abnormality

| Dataset | Endpoint | Task type |
|---|---|---|
| ClinicalGait | Healthy versus ACL, CIPN, CVA, HOA, KOA, PD, or RIL | Seven binary classifications |
| Mobilise-D | Healthy versus PD, MS, COPD, or CHF | Four binary classifications |
| mPower | Parkinson's disease | Binary classification |
| COMPWALK-ACL | ACL abnormality | Binary classification |
| WearGait-PD | Parkinson's disease | Binary classification |
| FoGContest | Freezing of gait | Binary classification |
| FoG-STAR | Freezing of gait | Binary classification |
| StrokeRehabilitation | Post-stroke functional recovery | Binary classification |
| GSTRIDE | Non-frail, pre-frail, or frail | Three-class classification |

## Clinical outcome prediction

| Dataset | Endpoint | Task type |
|---|---|---|
| GSTRIDE | Fall during the subsequent year | Binary classification |
| LTMM LabWalks | One-year fall risk | Binary classification |

The manuscript's headline count of 18 refers to task concepts: three gait parameters, fourteen disease or abnormality concepts, and one clinical outcome concept. The explicit dataset-by-endpoint matrix above contains 26 evaluations because repeated concepts are evaluated on additional internal or external datasets.

`python3 gaitfm.py list` reports the pretraining, classification, and regression launchers used for this scientific scope.
