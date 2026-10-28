
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import gaitprint_regression as base
from models.GaitFMFullFinetune import Model as MAEFinetuneModel
from mobilised_regression_data import load_sensor_checkpoint
from utils.regression_tools import format_reg_metrics, json_default, regression_metrics, save_checkpoint


DAILY_PARTS = ("Head", "L_Hand", "R_Hand", "L_Thigh", "R_Thigh")


def parse_args():
    parser = argparse.ArgumentParser(description="GaitPrint GaitFM dynamic part-combination regression.")
    parser.add_argument("--target", required=True, choices=base.TARGETS)
    parser.add_argument("--root_path", type=str, default="./data/GaitPrint")
    parser.add_argument("--output_dir", type=str, default="./results_gaitprint_reference_part_combinations_sensorpretrain")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--mae_checkpoint_path", type=str, default="./checkpoints/backbones/mae_tiny_400e.pth.tar")
    parser.add_argument("--mae_model", type=str, default="mae_vit_tiny_patch16")
    parser.add_argument("--mae_tag", type=str, default="MAETiny_GaitFM_GaitPrintPartCombinations")
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--parts", nargs="+", default=list(DAILY_PARTS), choices=sorted(base.PART_COLUMNS))
    parser.add_argument("--max_combo_size", type=int, default=0, help="0 means enumerate all non-empty combinations.")
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--min_steps_per_window", type=int, default=3)
    parser.add_argument("--min_strides_per_window", type=int, default=1)
    parser.add_argument("--require_full_trial", type=int, default=1)
    parser.add_argument("--require_contacts", type=int, default=1)
    parser.add_argument("--require_lower_body", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=2021)
    parser.add_argument("--cache_npz", type=str, default="")
    parser.add_argument("--rebuild_cache", type=int, default=0)
    parser.add_argument("--max_subjects", type=int, default=0)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
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


def _part_features_from_frame(df, path, part, args):
    acc_cols, gyro_cols = base.PART_COLUMNS[part]
    required = ["time"] + list(acc_cols) + list(gyro_cols)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required GaitPrint columns for {part}: {missing}")
    timestamp = base.pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=np.float64)
    features = base.numeric_frame(df, list(acc_cols) + list(gyro_cols))
    features[:, :3] = features[:, :3] / 1000.0
    return base.resample(features, timestamp, args.target_rate)


def build_moments(args):
    def attach_subject_splits(samples):
        split_by_subject = base.split_subjects(set(samples["subjects"].astype(str)), args)
        samples["splits"] = np.asarray(
            [split_by_subject[subject] for subject in samples["subjects"].astype(str)],
            dtype="U8",
        )
        return samples

    cache_path = Path(args.cache_npz).expanduser() if args.cache_npz else None
    if cache_path and cache_path.exists() and not int(args.rebuild_cache):
        with np.load(cache_path, allow_pickle=False) as data:
            samples = {key: data[key] for key in data.files}
        target_key = f"y_{args.target}"
        if target_key not in samples:
            raise ValueError(f"Cache does not contain {target_key}; rebuild it with --rebuild_cache 1.")
        samples["y"] = samples[target_key]
        samples.pop("splits", None)
        return attach_subject_splits(samples) if getattr(args, "attach_split", True) else samples

    root = Path(args.root_path)
    table = base.read_trial_table(root, args)
    csv_files = sorted(path for path in root.glob("S*/*.csv") if base._trial_key_from_path(path) is not None)
    records = []
    for path in csv_files:
        key = base._trial_key_from_path(path)
        table_key = (key["id"], key["group"], key["day"], key["block"], key["trial"])
        if table_key in table:
            records.append({"path": path, "subject": key["id"], **key})
    if int(args.max_subjects) > 0:
        keep = set(sorted({record["subject"] for record in records})[: int(args.max_subjects)])
        records = [record for record in records if record["subject"] in keep]
    if not records:
        raise ValueError(f"No valid GaitPrint records found under {root}")
    print(
        f"Building GaitPrint cache from {len(records)} trial CSV files "
        f"for parts={list(args.parts)} ...",
        flush=True,
    )

    window_size = int(round(args.target_rate * args.window_seconds))
    stride = int(args.stride)
    parts = list(args.parts)

    X, subjects, files, starts = [], [], [], []
    targets_by_name = {target: [] for target in base.TARGETS}
    for record_index, record in enumerate(records, start=1):
        df = base.pd.read_csv(record["path"])
        part_arrays = []
        for part in parts:
            features = _part_features_from_frame(df, record["path"], part, args)
            part_arrays.append(features)
        min_len = min(len(features) for features in part_arrays)
        if min_len < window_size:
            continue
        events = base.prepare_trial_events(df)
        stacked = np.stack([features[:min_len] for features in part_arrays], axis=0)
        for start in range(0, min_len - window_size + 1, stride):
            start_sec = float(start / args.target_rate)
            end_sec = float((start + window_size) / args.target_rate)
            targets = base.derive_window_targets(events, start_sec, end_sec, args)
            if targets is None:
                continue
            if not all(np.isfinite(targets[target]) for target in base.TARGETS):
                continue
            X.append(stacked[:, start : start + window_size].astype(np.float32))
            for target in base.TARGETS:
                targets_by_name[target].append(float(targets[target]))
            subjects.append(record["subject"])
            files.append(str(record["path"]))
            starts.append(int(start))
            if int(args.max_windows) > 0 and len(X) >= int(args.max_windows):
                break
        if int(args.max_windows) > 0 and len(X) >= int(args.max_windows):
            break
        if record_index == 1 or record_index % 10 == 0 or record_index == len(records):
            print(
                f"GaitPrint cache progress: {record_index}/{len(records)} trials, "
                f"valid_windows={len(X)}",
                flush=True,
            )
    if not X:
        raise ValueError(f"No GaitPrint windows with valid {args.target} labels.")
    samples = {
        "X": np.stack(X).astype(np.float32),
        "y": np.asarray(targets_by_name[args.target], dtype=np.float32),
        "subjects": np.asarray(subjects, dtype="U32"),
        "files": np.asarray(files, dtype="U512"),
        "starts": np.asarray(starts, dtype=np.int64),
        "parts": np.asarray(parts, dtype="U32"),
    }
    for target, values in targets_by_name.items():
        samples[f"y_{target}"] = np.asarray(values, dtype=np.float32)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing compressed GaitPrint cache: {cache_path}", flush=True)
        np.savez_compressed(cache_path, **samples)
        print(f"Saved GaitPrint part-combination cache: {cache_path}")
    return attach_subject_splits(samples) if getattr(args, "attach_split", True) else samples


def enumerate_combos(num_parts, max_combo_size):
    max_size = num_parts if int(max_combo_size) <= 0 else min(int(max_combo_size), num_parts)
    combos = []
    for size in range(1, max_size + 1):
        combos.extend(itertools.combinations(range(num_parts), size))
    return combos


class CombinationDataset(Dataset):
    def __init__(self, samples, split, combos, mean=None, std=None):
        mask = samples["splits"].astype(str) == split
        self.X = samples["X"][mask]
        self.y = samples["y"][mask].astype(np.float32)
        self.subjects = samples["subjects"][mask].astype(str)
        self.parts = samples["parts"].astype(str).tolist()
        self.combos = [tuple(combo) for combo in combos]
        if len(self.X) == 0:
            raise ValueError(f"No GaitPrint windows for split={split}")
        if mean is None or std is None:
            matrix = self.X.reshape(-1, self.X.shape[-1])
            mean = matrix.mean(axis=0).astype(np.float32)
            std = (matrix.std(axis=0) + np.finfo(np.float32).eps).astype(np.float32)
        self.mean = mean
        self.std = std
        self.X = ((self.X - mean) / std).astype(np.float32)

    def __len__(self):
        return len(self.y) * len(self.combos)

    def __getitem__(self, idx):
        moment_idx = idx // len(self.combos)
        combo = self.combos[idx % len(self.combos)]
        max_parts = len(self.parts)
        x = np.zeros((max_parts, self.X.shape[2], self.X.shape[3]), dtype=np.float32)
        part_ids = np.zeros((max_parts,), dtype=np.int64)
        part_mask = np.zeros((max_parts,), dtype=np.float32)
        for out_idx, part_idx in enumerate(combo):
            x[out_idx] = self.X[moment_idx, part_idx]
            part_ids[out_idx] = int(part_idx)
            part_mask[out_idx] = 1.0
        combo_name = "+".join(self.parts[part_idx] for part_idx in combo)
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(part_ids).long(),
            torch.from_numpy(part_mask).float(),
            torch.tensor(float(self.y[moment_idx]), dtype=torch.float32),
            combo_name,
            self.subjects[moment_idx],
        )


def collate(batch):
    return (
        torch.stack([item[0] for item in batch], dim=0),
        torch.stack([item[1] for item in batch], dim=0),
        torch.stack([item[2] for item in batch], dim=0),
        torch.stack([item[3] for item in batch], dim=0),
        [item[4] for item in batch],
        [item[5] for item in batch],
    )


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate)


class GaitPrintPartCombinationRegressorBase(nn.Module):
    def __init__(self, args, num_parts):
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


def aggregate_subject_regression(y_true, y_pred, subjects):
    subjects = np.asarray(subjects).astype(str)
    out_true, out_pred = [], []
    for subject in sorted(set(subjects.tolist())):
        idx = subjects == subject
        out_true.append(float(np.mean(y_true[idx])))
        out_pred.append(float(np.mean(y_pred[idx])))
    return np.asarray(out_true), np.asarray(out_pred)


def train_one_epoch(model, loader, criterion, optimizer, device, target_mean, target_std):
    model.train()
    losses = []
    for x, part_ids, part_mask, y, _, _ in loader:
        optimizer.zero_grad()
        pred = model(x.to(device), part_ids.to(device), part_mask.to(device))
        yy = y.to(device)
        if target_std > 0:
            yy = (yy - float(target_mean)) / float(target_std)
        loss = criterion(pred, yy)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def predict(model, loader, device, target_mean, target_std):
    model.eval()
    preds, targets, combos, subjects = [], [], [], []
    for x, part_ids, part_mask, y, batch_combos, batch_subjects in loader:
        pred = model(x.to(device), part_ids.to(device), part_mask.to(device)).cpu()
        if target_std > 0:
            pred = pred * float(target_std) + float(target_mean)
        preds.append(pred)
        targets.append(y.float())
        combos.extend(batch_combos)
        subjects.extend(batch_subjects)
    return (
        torch.cat(targets).numpy(),
        torch.cat(preds).numpy(),
        np.asarray(combos).astype(str),
        np.asarray(subjects).astype(str),
    )


@torch.no_grad()
def validation_loss(model, loader, criterion, device, target_mean, target_std):
    model.eval()
    losses = []
    for x, part_ids, part_mask, y, _, _ in loader:
        pred = model(x.to(device), part_ids.to(device), part_mask.to(device))
        yy = y.to(device)
        if target_std > 0:
            yy = (yy - float(target_mean)) / float(target_std)
        losses.append(float(criterion(pred, yy).item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate_arrays(val_y, val_pred, val_subjects, test_y, test_pred, test_subjects):
    val_subject_y, val_subject_pred = aggregate_subject_regression(val_y, val_pred, val_subjects)
    test_subject_y, test_subject_pred = aggregate_subject_regression(test_y, test_pred, test_subjects)
    return {
        "val": regression_metrics(val_y, val_pred),
        "test": regression_metrics(test_y, test_pred),
        "val_subject": regression_metrics(val_subject_y, val_subject_pred),
        "test_subject": regression_metrics(test_subject_y, test_subject_pred),
        "n_val_subjects": int(len(val_subject_y)),
        "n_test_subjects": int(len(test_subject_y)),
    }


def evaluate(model, val_loader, test_loader, device, target_mean, target_std):
    val_y, val_pred, _, val_subjects = predict(model, val_loader, device, target_mean, target_std)
    test_y, test_pred, _, test_subjects = predict(model, test_loader, device, target_mean, target_std)
    return evaluate_arrays(val_y, val_pred, val_subjects, test_y, test_pred, test_subjects)


def evaluate_by_combo(model, val_loader, test_loader, device, target_mean, target_std):
    val_y, val_pred, val_combos, val_subjects = predict(model, val_loader, device, target_mean, target_std)
    test_y, test_pred, test_combos, test_subjects = predict(model, test_loader, device, target_mean, target_std)
    rows = []
    for combo in sorted(set(test_combos.tolist()), key=lambda name: (name.count("+"), name)):
        val_idx = val_combos == combo
        test_idx = test_combos == combo
        if not np.any(val_idx) or not np.any(test_idx):
            continue
        result = evaluate_arrays(
            val_y[val_idx],
            val_pred[val_idx],
            val_subjects[val_idx],
            test_y[test_idx],
            test_pred[test_idx],
            test_subjects[test_idx],
        )
        result.update(
            {
                "combo": combo,
                "combo_size": int(combo.count("+") + 1),
                "n_val": int(np.sum(val_idx)),
                "n_test": int(np.sum(test_idx)),
            }
        )
        rows.append(result)
    return rows


def format_eval(prefix, epoch, val_loss, result):
    return (
        f"{prefix}: Epoch {epoch} | val_loss={val_loss:.6f} "
        f"Sample-Val: {format_reg_metrics(result['val'])} "
        f"Sample-Test: {format_reg_metrics(result['test'])} "
        f"| Subject-Val(n={result['n_val_subjects']}): {format_reg_metrics(result['val_subject'])} "
        f"Subject-Test(n={result['n_test_subjects']}): {format_reg_metrics(result['test_subject'])}"
    )


def main(model_class):
    args = parse_args()
    os.environ.setdefault("GAITFM_CHANNEL_MASK_MODE", "random_acc_only")
    os.environ.setdefault("GAITFM_CHANNEL_MASK_PROB", "0")
    os.environ.setdefault("GAITFM_CHANNEL_MASK_APPLY_EVAL", "0")
    if args.model_id is None:
        args.model_id = f"GaitPrint_{args.target}_{args.mae_tag}_PartCombinations"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    samples = build_moments(args)
    part_names = samples["parts"].astype(str).tolist()
    combos = enumerate_combos(len(part_names), args.max_combo_size)
    train_ds = CombinationDataset(samples, "TRAIN", combos)
    val_ds = CombinationDataset(samples, "VAL", combos, train_ds.mean, train_ds.std)
    test_ds = CombinationDataset(samples, "TEST", combos, train_ds.mean, train_ds.std)
    print(f"GaitPrint target: {base.target_display_name(args.target)}")
    print(f"GaitPrint selected part order ({len(part_names)}): {part_names}")
    print(f"Enumerated combinations: {len(combos)}")
    for split, ds in (("TRAIN", train_ds), ("VAL", val_ds), ("TEST", test_ds)):
        print(
            f"{split}: moments={len(ds.y)} combinations={len(ds)} "
            f"subjects={len(set(ds.subjects.tolist()))}"
        )
    target_mean = float(np.mean(train_ds.y)) if int(args.standardize_target) else 0.0
    target_std = float(np.std(train_ds.y)) if int(args.standardize_target) else 1.0
    if not np.isfinite(target_std) or target_std <= 0:
        target_std = 1.0
    print(f"Target standardization: mean={target_mean:.6f}, std={target_std:.6f} (TRAIN moments only)")

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)
    model = model_class(args, len(part_names)).to(device)
    head_ids = {id(param) for param in list(model.head.parameters()) + list(model.part_embed.parameters())}
    encoder_params = [param for param in model.parameters() if id(param) not in head_ids and param.requires_grad]
    head_params = [param for param in list(model.head.parameters()) + list(model.part_embed.parameters()) if param.requires_grad]
    param_groups = []
    if encoder_params and args.learning_rate > 0:
        param_groups.append({"params": encoder_params, "lr": args.learning_rate})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_learning_rate})
    if not param_groups:
        raise ValueError("No trainable parameters remain.")
    print(
        "Trainable parameters: {0}/{1} ({2:.2f}%)".format(
            sum(param.numel() for param in model.parameters() if param.requires_grad),
            sum(param.numel() for param in model.parameters()),
            100.0
            * sum(param.numel() for param in model.parameters() if param.requires_grad)
            / max(sum(param.numel() for param in model.parameters()), 1),
        )
    )

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

    summary_path = out_dir / "summary.log"
    results_json_path = out_dir / "combo_results.json"
    combo_results = {}
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
            combo_rows = evaluate_by_combo(model, val_loader, test_loader, device, target_mean, target_std)
            combo_results[name] = {
                "epoch": record["epoch"],
                "val_loss": record["val_loss"],
                "overall": result,
                "combos": combo_rows,
            }
            for row in combo_rows:
                sample_line = (
                    f"  {row['combo']} sample n={row['n_test']} "
                    f"Val: {format_reg_metrics(row['val'])} "
                    f"Test: {format_reg_metrics(row['test'])}"
                )
                subject_line = (
                    f"  {row['combo']} subject n={row['n_test_subjects']} "
                    f"Val: {format_reg_metrics(row['val_subject'])} "
                    f"Test: {format_reg_metrics(row['test_subject'])}"
                )
                print(sample_line)
                print(subject_line)
                handle.write(sample_line + "\n")
                handle.write(subject_line + "\n")
    with open(results_json_path, "w", encoding="utf-8") as handle:
        json.dump(combo_results, handle, indent=2, default=json_default)
    print(f"Saved summary: {summary_path}")
    print(f"Saved structured combo results: {results_json_path}")
