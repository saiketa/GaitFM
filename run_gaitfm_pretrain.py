import os
import random

import numpy as np
import torch

from exp.exp_gaitfm_pretrain import Exp_GaitFMPretrain
from gaitfm_pretrain_args import build_parser as build_joint_parser


def build_parser():
    parser = build_joint_parser()
    parser.description = "Joint fixed-channel sensor-image MAE pretraining with stop-gradient targets"
    parser.set_defaults(
        task_name="gaitfm_pretrain",
        model="GaitFMPretrain",
        checkpoints="./ckpt/",
        experiment_name="GaitFM_JointPretrain",
        gait_image_size=224,
        batch_size=64,
        learning_rate=1e-5,
    )
    parser.add_argument("--mae_code_path", type=str, default="", help="Deprecated; MAE code is vendored in models/MAEViT.py.")
    parser.add_argument("--mae_checkpoint_path", type=str, default="./checkpoints/backbones/mae_tiny_400e.pth.tar")
    parser.add_argument("--mae_model", type=str, default="mae_vit_tiny_patch16")
    parser.add_argument("--mae_mask_ratio", type=float, default=0.75)
    parser.add_argument("--mae_norm_pix_loss", type=int, default=1)
    return parser


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.use_gpu:
        torch.cuda.manual_seed_all(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
    os.makedirs(args.checkpoints, exist_ok=True)

    setting = "gaitfm_pretrain_{}_{}_sl{}_mc{}_mr{}_0".format(
        args.model_id,
        args.model,
        args.seq_len,
        args.gait_max_channels,
        args.mae_mask_ratio,
    )
    print(">>>>>>>start joint MAE pretraining : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
    exp = Exp_GaitFMPretrain(args)
    exp.train(setting)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
