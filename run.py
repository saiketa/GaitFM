import argparse
import random

import numpy as np
import torch

from utils.print_args import print_args


def build_parser():
    parser = argparse.ArgumentParser(description="GaitFM IMU classification")

    parser.add_argument("--task_name", type=str, required=True, default="classification")
    parser.add_argument("--is_training", type=int, required=True, default=1)
    parser.add_argument("--model_id", type=str, required=True, default="test")
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./data")
    parser.add_argument("--checkpoints", type=str, default="./ckpt/")

    parser.add_argument("--clinical_disease", type=str, default=None)
    parser.add_argument("--clinical_source_rate", type=int, default=100)
    parser.add_argument("--clinical_target_rate", type=int, default=50)
    parser.add_argument("--clinical_window_seconds", type=float, default=5.0)
    parser.add_argument("--clinical_val_ratio", type=float, default=0.2)
    parser.add_argument("--clinical_test_ratio", type=float, default=0.2)
    parser.add_argument("--clinical_split_seed", type=int, default=2021)

    parser.add_argument("--fog_types", type=str, nargs="+", default=["tdcsfog", "defog", "notype"])
    parser.add_argument("--fog_stride", type=int, default=None)
    parser.add_argument("--fog_filter_valid_task", type=int, default=1)
    parser.add_argument("--fog_val_ratio", type=float, default=0.2)
    parser.add_argument("--fog_test_ratio", type=float, default=0.2)
    parser.add_argument("--fog_split_seed", type=int, default=2021)

    parser.add_argument("--weargait_source_rate", type=float, default=100)
    parser.add_argument("--weargait_target_rate", type=float, default=50)
    parser.add_argument("--weargait_window_seconds", type=float, default=5.0)
    parser.add_argument("--weargait_stride", type=int, default=None)
    parser.add_argument("--weargait_val_ratio", type=float, default=0.2)
    parser.add_argument("--weargait_test_ratio", type=float, default=0.2)
    parser.add_argument("--weargait_split_seed", type=int, default=2021)
    parser.add_argument("--weargait_body_parts", type=str, nargs="+", default=None)
    parser.add_argument("--weargait_tasks", type=str, nargs="+", default=None)

    parser.add_argument("--ltmm_stride", type=int, default=None)
    parser.add_argument("--ltmm_target_rate", type=float, default=50)
    parser.add_argument("--ltmm_val_ratio", type=float, default=0.2)
    parser.add_argument("--ltmm_test_ratio", type=float, default=0.2)
    parser.add_argument("--ltmm_split_seed", type=int, default=2021)

    parser.add_argument("--mpower_target_rate", type=float, default=50)
    parser.add_argument("--mpower_window_seconds", type=float, default=5.0)
    parser.add_argument("--mpower_stride", type=int, default=250)
    parser.add_argument("--mpower_val_ratio", type=float, default=0.2)
    parser.add_argument("--mpower_test_ratio", type=float, default=0.2)
    parser.add_argument("--mpower_split_seed", type=int, default=2021)
    parser.add_argument("--mpower_max_records", type=int, default=0)
    parser.add_argument("--mpower_tasks", type=str, nargs="+", default=None)
    parser.add_argument("--mpower_require_label", type=int, default=0)

    parser.add_argument("--gstride_task", type=str, default="frailty", choices=["frailty", "subsequent_year_fall"])
    parser.add_argument("--gstride_followup_labels", type=str, default="")
    parser.add_argument("--gstride_followup_subject_column", type=str, default="subject_id")
    parser.add_argument("--gstride_followup_outcome_column", type=str, default="subsequent_year_fall")
    parser.add_argument("--gstride_target_rate", type=float, default=50)
    parser.add_argument("--gstride_window_seconds", type=float, default=5.0)
    parser.add_argument("--gstride_stride", type=int, default=None)
    parser.add_argument("--gstride_val_ratio", type=float, default=0.2)
    parser.add_argument("--gstride_test_ratio", type=float, default=0.2)
    parser.add_argument("--gstride_split_seed", type=int, default=2021)

    parser.add_argument("--compwalk_val_ratio", type=float, default=0.2)
    parser.add_argument("--compwalk_test_ratio", type=float, default=0.2)
    parser.add_argument("--compwalk_split_seed", type=int, default=2021)

    parser.add_argument("--mobilised_subset", type=str, default="finetune", choices=["pretrain", "finetune", "all"])
    parser.add_argument("--mobilised_disease", type=str, default="PD", choices=["PD", "MS", "COPD", "CHF"])
    parser.add_argument("--mobilised_val_ratio", type=float, default=0.2)
    parser.add_argument("--mobilised_test_ratio", type=float, default=0.2)
    parser.add_argument("--mobilised_split_seed", type=int, default=2021)


    parser.add_argument("--seq_len", type=int, default=250)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--classifier_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--experiment_name", type=str, default="GaitFM")
    parser.add_argument("--load_shards_to_memory", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)

    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu_type", type=str, default="cuda")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    parser.add_argument("--gait_image_size", type=int, default=240)
    parser.add_argument("--gait_segment_stride", type=int, default=10)
    parser.add_argument("--gait_sum_to_one", type=int, default=1)
    parser.add_argument("--gaitfm_pretrain_checkpoint", type=str, default="")

    parser.add_argument("--mae_checkpoint_path", type=str, default="./checkpoints/backbones/mae_tiny_400e.pth.tar")
    parser.add_argument("--mae_model", type=str, default="mae_vit_tiny_patch16")
    parser.add_argument("--mae_norm_pix_loss", type=int, default=1)
    parser.add_argument("--mae_finetune_pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--gait_max_channels", type=int, default=6)

    return parser


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
        print("Using GPU")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(item) for item in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print_args(args)

    if args.task_name != "classification":
        raise ValueError("This simplified FM project only supports --task_name classification.")
    from exp.exp_classification import Exp_Classification as Exp

    if args.is_training:
        exp = Exp(args)
        setting = (
            f"classification_{args.model_id}_{args.model}_{args.data}"
            f"_sl{args.seq_len}_bs{args.batch_size}_lr{args.learning_rate}_{args.experiment_name}_0"
        )
        print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
        exp.train(setting)
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting)
        if args.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        setting = (
            f"classification_{args.model_id}_{args.model}_{args.data}"
            f"_sl{args.seq_len}_bs{args.batch_size}_lr{args.learning_rate}_{args.experiment_name}_0"
        )
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)


if __name__ == "__main__":
    main()
