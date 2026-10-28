
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from models.GaitFMFullFinetune import Model as MAEFinetuneModel
from utils.regression_tools import json_default, save_checkpoint
from mobilised_regression_data import (
    DEFAULT_TESTS,
    WindowDataset,
    build_samples,
    evaluate,
    format_eval,
    load_sensor_checkpoint,
    make_loader,
    target_display_name,
    train_one_epoch,
    validation_loss,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mobilise-D TVS reference gait regression with GaitFM and PatchResNet."
    )
    parser.add_argument("--target", required=True, choices=["gait_speed", "step_length", "step_time"])
    parser.add_argument("--root_path", type=str, default="./data/Mobilise-D-TVS")
    parser.add_argument("--output_dir", type=str, default="./results_mobilised_reference_patch_resnet_sensorpretrain")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--mae_checkpoint_path", type=str, default="./checkpoints/backbones/mae_tiny_400e.pth.tar")
    parser.add_argument("--mae_model", type=str, default="mae_vit_tiny_patch16")
    parser.add_argument("--mae_tag", type=str, default="MAETiny_GaitFM_PatchResNet")
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--tests", nargs="+", default=list(DEFAULT_TESTS))
    parser.add_argument("--sensor_systems", nargs="+", default=["INDIP", "SU"])
    parser.add_argument("--body_parts", nargs="+", default=["LowerBack"])
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--default_source_rate", type=float, default=100.0)
    parser.add_argument("--min_ref_coverage", type=float, default=0.6)
    parser.add_argument("--label_aggregation", choices=["overlap_weighted", "mean"], default="overlap_weighted")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=2021)
    parser.add_argument("--cache_npz", type=str, default="")
    parser.add_argument("--rebuild_cache", type=int, default=0)
    parser.add_argument("--max_subjects", type=int, default=0)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0, help="Encoder LR. Default 0 because encoder is frozen.")
    parser.add_argument("--head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--gait_image_size", type=int, default=224)
    parser.add_argument("--gait_segment_stride", type=int, default=10)
    parser.add_argument("--gait_sum_to_one", type=int, default=1)
    parser.add_argument("--max_channels", type=int, default=6)
    parser.add_argument("--standardize_target", type=int, default=1)
    return parser.parse_args()


class MobiliseReferenceRegressorBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        configs = SimpleNamespace(
            gait_image_size=args.gait_image_size,
            gait_segment_stride=args.gait_segment_stride,
            gait_sum_to_one=args.gait_sum_to_one,
            gait_max_channels=args.max_channels,
            mae_model=args.mae_model,
            mae_checkpoint_path=args.mae_checkpoint_path,
            mae_norm_pix_loss=1,
            mae_finetune_pooling="mean",
            dropout=0.0,
            num_class=2,
        )
        self.encoder = MAEFinetuneModel(configs)
        load_sensor_checkpoint(self.encoder, args.pretrain_ckpt)


def main(model_class):
    args = parse_args()
    os.environ.setdefault("GAITFM_CHANNEL_MASK_MODE", "random_acc_only")
    os.environ.setdefault("GAITFM_CHANNEL_MASK_PROB", "0")
    os.environ.setdefault("GAITFM_CHANNEL_MASK_APPLY_EVAL", "0")
    if args.model_id is None:
        args.model_id = f"MobiliseDTVS_{args.target}_{args.mae_tag}_ReferenceSensorPretrainPatchResNet"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    samples = build_samples(args)
    train_ds = WindowDataset(samples, "TRAIN")
    val_ds = WindowDataset(samples, "VAL", train_ds.mean, train_ds.std)
    test_ds = WindowDataset(samples, "TEST", train_ds.mean, train_ds.std)
    print(
        f"Mobilise-D {target_display_name(args.target)} splits | "
        f"TRAIN windows={len(train_ds)} subjects={len(set(train_ds.subjects.tolist()))} "
        f"VAL windows={len(val_ds)} subjects={len(set(val_ds.subjects.tolist()))} "
        f"TEST windows={len(test_ds)} subjects={len(set(test_ds.subjects.tolist()))}"
    )
    print(
        f"Target stats | TRAIN mean={float(np.mean(train_ds.y)):.6f} std={float(np.std(train_ds.y)):.6f} "
        f"VAL mean={float(np.mean(val_ds.y)):.6f} TEST mean={float(np.mean(test_ds.y)):.6f}"
    )
    target_mean = float(np.mean(train_ds.y)) if int(args.standardize_target) else 0.0
    target_std = float(np.std(train_ds.y)) if int(args.standardize_target) else 1.0
    if not np.isfinite(target_std) or target_std <= 0:
        target_std = 1.0
    print(f"Target standardization: mean={target_mean:.6f}, std={target_std:.6f} (TRAIN only)")

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    model = model_class(args).to(device)
    head_param_ids = {id(param) for param in model.head.parameters()}
    encoder_params = [param for param in model.parameters() if id(param) not in head_param_ids and param.requires_grad]
    head_params = [param for param in model.head.parameters() if param.requires_grad]
    param_groups = []
    if encoder_params and args.learning_rate > 0:
        param_groups.append({"params": encoder_params, "lr": args.learning_rate})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_learning_rate})
    if not param_groups:
        raise ValueError("No trainable parameters remain.")
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    print(f"Trainable parameters: {trainable}/{total} ({100.0 * trainable / max(total, 1):.2f}%)")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    criterion = nn.SmoothL1Loss()
    out_dir = Path(args.output_dir) / args.model_id
    ckpt_dir = out_dir / "ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_records = {"val_loss": None}
    best_paths = {"val_loss": ckpt_dir / "val_loss.pth"}
    best_val_loss = float("inf")
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, target_mean, target_std)
        val_loss = validation_loss(model, val_loader, criterion, device, target_mean, target_std)
        line = (
            f"Epoch {epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"time={time.time() - start:.1f}s"
        )
        print(line)
        log_lines.append(line)
        for name, score in {"val_loss": -val_loss}.items():
            if score is None or not np.isfinite(score):
                continue
            current = best_records[name]
            if current is None or float(score) > current["score"]:
                best_records[name] = {"epoch": epoch, "score": float(score), "val_loss": float(val_loss)}
                save_checkpoint(model, best_paths[name])

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = float(val_loss)

    if best_records["val_loss"] is None:
        raise RuntimeError("No finite validation loss; no checkpoint selected")

    results = {}
    summary_path = out_dir / "summary.log"
    results_json_path = out_dir / "results.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        for line in log_lines:
            handle.write(line + "\n")
        for name, record in best_records.items():
            if record is None:
                continue
            model.load_state_dict(torch.load(best_paths[name], map_location=device))
            result = evaluate(model, val_loader, test_loader, device, target_mean, target_std)
            line = format_eval(f"Best by {name}", record["epoch"], record["val_loss"], result)
            print(line)
            handle.write(line + "\n")
            results[name] = {"epoch": record["epoch"], "val_loss": record["val_loss"], "result": result}
    with open(results_json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=json_default)
    print(f"Saved summary: {summary_path}")
    print(f"Saved structured results: {results_json_path}")
