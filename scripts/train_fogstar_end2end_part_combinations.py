
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PART_ORDER = ("ankleL", "ankleR", "back", "wrist")
PART_COLUMNS = {
    part: tuple(f"{part}_{sensor}_{axis}" for sensor in ("acc", "gyro") for axis in ("x", "y", "z"))
    for part in PART_ORDER
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train FoG-STAR dynamic body-part FoG classifier.")
    parser.add_argument("--root_path", default="./data/FoG-STAR")
    parser.add_argument("--data_file", default="sensor_data.csv")
    parser.add_argument("--output_dir", default="./results_end2end_part_combinations")
    parser.add_argument("--model_id", default="FoGSTAR_FoG_End2EndPartCombinations")
    parser.add_argument("--pretrain_ckpt", required=True)
    parser.add_argument("--mae_checkpoint_path", default="./checkpoints/backbones/mae_tiny_400e.pth.tar")
    parser.add_argument("--mae_model", default="mae_vit_tiny_patch16")
    parser.add_argument("--mae_tag", default="MAETiny")
    parser.add_argument("--mae_finetune_pooling", default="cls", choices=["cls", "mean"])
    parser.add_argument("--task_ids", nargs="+", type=int, default=[3, 4, 5, 6])
    parser.add_argument("--source_rate", type=float, default=60.0)
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=50, help="Stride in target-rate samples.")
    parser.add_argument("--fog_threshold", type=float, default=0.5)
    parser.add_argument("--max_interp_gap_seconds", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=2021)
    parser.add_argument("--max_combo_size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--classifier_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--gait_image_size", type=int, default=224)
    parser.add_argument("--gait_segment_stride", type=int, default=10)
    parser.add_argument("--gait_sum_to_one", type=int, default=1)
    parser.add_argument("--max_channels", type=int, default=6)
    return parser.parse_args()


def resolve_data_path(root_path, data_file):
    root = Path(root_path).expanduser()
    path = root / data_file if root.is_dir() else root
    if not path.exists():
        raise FileNotFoundError(f"FoG-STAR sensor data not found: {path}")
    return path


def validate_columns(frame):
    required = {"timestamp", "fog", "subjectID", "sessionID", "taskID"}
    required.update(column for columns in PART_COLUMNS.values() for column in columns)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"FoG-STAR data is missing columns: {missing}")


def nearest_indices(source_time, target_time):
    right = np.searchsorted(source_time, target_time, side="left")
    right = np.clip(right, 0, len(source_time) - 1)
    left = np.clip(right - 1, 0, len(source_time) - 1)
    choose_left = np.abs(target_time - source_time[left]) <= np.abs(source_time[right] - target_time)
    return np.where(choose_left, left, right)


def resample_group(group, args):
    group = group.sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    source_time = pd.to_numeric(group["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(source_time)
    group = group.loc[keep].reset_index(drop=True)
    source_time = source_time[keep]
    if len(source_time) < 2:
        return None
    source_time = source_time - source_time[0]
    duration = float(source_time[-1])
    target_time = np.arange(0.0, duration + 1e-9, 1.0 / args.target_rate, dtype=np.float64)
    if len(target_time) < int(round(args.window_seconds * args.target_rate)):
        return None
    nearest = nearest_indices(source_time, target_time)
    time_valid = np.abs(target_time - source_time[nearest]) <= (1.5 / args.source_rate)
    fog = pd.to_numeric(group["fog"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)[nearest]

    max_gap_samples = max(1, int(round(args.max_interp_gap_seconds * args.source_rate)))
    parts = {}
    for part in PART_ORDER:
        values = group[list(PART_COLUMNS[part])].apply(pd.to_numeric, errors="coerce")
        values = values.interpolate(
            method="linear",
            limit=max_gap_samples,
            limit_direction="both",
            limit_area="inside",
        )
        raw = values.to_numpy(dtype=np.float64)
        valid_rows = np.isfinite(raw).all(axis=1)
        if valid_rows.sum() < 2:
            continue
        output = np.empty((len(target_time), raw.shape[1]), dtype=np.float32)
        for channel in range(raw.shape[1]):
            valid = np.isfinite(raw[:, channel])
            output[:, channel] = np.interp(target_time, source_time[valid], raw[valid, channel])
        target_valid = valid_rows[nearest] & time_valid
        output[~target_valid] = np.nan
        parts[part] = output
    return fog, parts


def load_raw_moments(args):
    path = resolve_data_path(args.root_path, args.data_file)
    frame = pd.read_csv(path)
    validate_columns(frame)
    frame = frame[frame["taskID"].isin(args.task_ids)].copy()
    if frame.empty:
        raise ValueError(f"No FoG-STAR rows remain for task IDs {args.task_ids}.")

    window_size = int(round(args.window_seconds * args.target_rate))
    moments = []
    grouped = frame.groupby(["subjectID", "sessionID", "taskID"], sort=True)
    for (subject, session, task), group in grouped:
        resampled = resample_group(group, args)
        if resampled is None:
            continue
        fog, part_arrays = resampled
        for start in range(0, len(fog) - window_size + 1, args.stride):
            end = start + window_size
            fog_ratio = float(np.mean(fog[start:end] > 0.5))
            label = int(fog_ratio > args.fog_threshold)
            parts = {}
            for part, array in part_arrays.items():
                window = array[start:end]
                if np.isfinite(window).all():
                    parts[part] = window.astype(np.float32, copy=True)
            if not parts:
                continue
            moments.append(
                {
                    "subject": str(subject),
                    "session": str(session),
                    "task": int(task),
                    "start": int(start),
                    "moment_id": f"{subject}::{session}::{task}::{start:08d}",
                    "fog_ratio": fog_ratio,
                    "label": label,
                    "parts": parts,
                }
            )
    if not moments:
        raise ValueError("No valid FoG-STAR windows were generated.")
    return moments


def split_one_group(subjects, seed):
    subjects = np.asarray(sorted(subjects), dtype=str)
    rng = np.random.default_rng(int(seed))
    subjects = subjects[rng.permutation(len(subjects))]
    if len(subjects) < 3:
        raise ValueError("Each subject stratum needs at least three subjects.")
    n_test = max(1, int(round(0.2 * len(subjects))))
    n_val = max(1, int(round(0.2 * len(subjects))))
    while n_test + n_val >= len(subjects):
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            break
    return {
        "TEST": subjects[:n_test].tolist(),
        "VAL": subjects[n_test : n_test + n_val].tolist(),
        "TRAIN": subjects[n_test + n_val :].tolist(),
    }


def stratified_subject_split(moments, seed):
    by_subject = {}
    for moment in moments:
        by_subject.setdefault(moment["subject"], []).append(moment["label"])
    positive = [subject for subject, labels in by_subject.items() if any(labels)]
    negative_only = [subject for subject, labels in by_subject.items() if not any(labels)]
    positive_split = split_one_group(positive, seed)
    negative_split = split_one_group(negative_only, seed + 1)
    split = {
        name: sorted(positive_split[name] + negative_split[name])
        for name in ("TRAIN", "VAL", "TEST")
    }
    return split


def normalize_moments(moments, train_subjects):
    train_subjects = set(train_subjects)
    arrays = [
        window
        for moment in moments
        if moment["subject"] in train_subjects
        for window in moment["parts"].values()
    ]
    if not arrays:
        raise ValueError("No TRAIN windows are available for normalization.")
    matrix = np.concatenate(arrays, axis=0).astype(np.float64)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = (matrix.std(axis=0) + np.finfo(np.float32).eps).astype(np.float32)
    for moment in moments:
        moment["parts"] = {
            part: ((window - mean[None, :]) / std[None, :]).astype(np.float32)
            for part, window in moment["parts"].items()
        }
    return mean, std


def build_examples(moments, max_combo_size):
    examples = []
    max_size = len(PART_ORDER) if max_combo_size <= 0 else min(max_combo_size, len(PART_ORDER))
    for moment_index, moment in enumerate(moments):
        available = [part for part in PART_ORDER if part in moment["parts"]]
        for size in range(1, min(max_size, len(available)) + 1):
            for combo in itertools.combinations(available, size):
                examples.append(
                    {
                        "moment_index": moment_index,
                        "parts": combo,
                        "part_ids": [PART_ORDER.index(part) for part in combo],
                        "combo": "+".join(combo),
                        "subject": moment["subject"],
                        "label": moment["label"],
                    }
                )
    return examples


def subjectwise_auroc(y_true, probs, subjects):
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(probs, dtype=np.float64)[:, 1]
    subjects = np.asarray(subjects).astype(str)
    values = []
    for subject in sorted(np.unique(subjects)):
        idx = subjects == subject
        if np.unique(y_true[idx]).size == 2:
            values.append(float(roc_auc_score(y_true[idx], score[idx])))
    return {
        "auroc": float(np.mean(values)) if values else float("nan"),
        "std": float(np.std(values)) if values else float("nan"),
        "n_subjects": len(values),
        "values": values,
    }


def evaluate_overall(model, val_loader, test_loader, device):
    val_y, val_probs, _, val_subjects = cg.predict(model, val_loader, device)
    test_y, test_probs, _, test_subjects = cg.predict(model, test_loader, device)
    threshold, val_metrics = cg.best_f1_threshold(val_y, val_probs)
    return {
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": cg.threshold_metrics(test_y, test_probs, threshold),
        "val_subjectwise": subjectwise_auroc(val_y, val_probs, val_subjects),
        "test_subjectwise": subjectwise_auroc(test_y, test_probs, test_subjects),
    }


def evaluate_by_combo(model, val_loader, test_loader, device):
    val_y, val_probs, val_combos, val_subjects = cg.predict(model, val_loader, device)
    test_y, test_probs, test_combos, test_subjects = cg.predict(model, test_loader, device)
    results = []
    for combo in sorted(set(test_combos.tolist())):
        val_idx = np.where(val_combos == combo)[0]
        test_idx = np.where(test_combos == combo)[0]
        if not len(val_idx) or not len(test_idx):
            continue
        threshold, val_metrics = cg.best_f1_threshold(val_y[val_idx], val_probs[val_idx])
        results.append(
            {
                "combo": combo,
                "part_count": combo.count("+") + 1,
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "threshold": threshold,
                "val": val_metrics,
                "test": cg.threshold_metrics(test_y[test_idx], test_probs[test_idx], threshold),
                "val_subjectwise": subjectwise_auroc(val_y[val_idx], val_probs[val_idx], val_subjects[val_idx]),
                "test_subjectwise": subjectwise_auroc(test_y[test_idx], test_probs[test_idx], test_subjects[test_idx]),
            }
        )
    return results


def format_subjectwise(result):
    return f"mean_AUROC={result['auroc']:.6f} std={result['std']:.6f} n={result['n_subjects']}"


def format_record(prefix, epoch, val_loss, overall):
    return (
        f"{prefix}: Epoch {epoch} | val_loss={val_loss:.6f} "
        f"threshold={overall['threshold']:.2f} "
        f"Sample-Val: {cg.format_metrics(overall['val_metrics'])} "
        f"Sample-Test: {cg.format_metrics(overall['test_metrics'])} | "
        f"Subject-wise-Val: {format_subjectwise(overall['val_subjectwise'])} "
        f"Subject-wise-Test: {format_subjectwise(overall['test_subjectwise'])}"
    )


def print_split_summary(name, moments, examples):
    labels = np.asarray([moment["label"] for moment in moments], dtype=np.int64)
    subjects = sorted({moment["subject"] for moment in moments})
    counts = dict(zip(*np.unique(labels, return_counts=True))) if len(labels) else {}
    positive_subjects = len({moment["subject"] for moment in moments if moment["label"] == 1})
    print(
        f"{name}: subjects={len(subjects)} positive_window_subjects={positive_subjects} "
        f"moments={len(moments)} combinations={len(examples)} label_counts={counts}"
    )


def main():
    global cg, torch, DataLoader
    args = parse_args()
    if not 0.0 <= args.fog_threshold <= 1.0:
        raise ValueError("--fog_threshold must be between 0 and 1.")
    np.random.seed(args.seed)

    all_moments = load_raw_moments(args)
    split_subjects = stratified_subject_split(all_moments, args.split_seed)
    mean, std = normalize_moments(all_moments, split_subjects["TRAIN"])
    moments = {
        split: [moment for moment in all_moments if moment["subject"] in set(subjects)]
        for split, subjects in split_subjects.items()
    }
    examples = {
        split: build_examples(moments[split], args.max_combo_size)
        for split in ("TRAIN", "VAL", "TEST")
    }
    print(f"FoG-STAR tasks={args.task_ids} target_rate={args.target_rate:g}Hz window={args.window_seconds:g}s stride={args.stride}")
    print(f"Window label: FoG iff fog_ratio > {args.fog_threshold:.2f}")
    print(f"Available parts: {list(PART_ORDER)}")
    print(f"TRAIN normalization mean={mean.tolist()} std={std.tolist()}")
    for split in ("TRAIN", "VAL", "TEST"):
        print(f"{split} subjects: {split_subjects[split]}")
        print_split_summary(split, moments[split], examples[split])
        if len(np.unique([moment["label"] for moment in moments[split]])) < 2:
            raise ValueError(f"{split} does not contain both FoG classes.")
    if args.inspect_only:
        print("Inspection complete; model training was skipped.")
        return

    import torch
    from torch.utils.data import DataLoader

    import train_clinicalgait_end2end_part_combinations as cg

    cg.PART_ORDER = PART_ORDER
    torch.manual_seed(args.seed)
    train_ds = cg.RawCombinationDataset(moments["TRAIN"], examples["TRAIN"])
    val_ds = cg.RawCombinationDataset(moments["VAL"], examples["VAL"])
    test_ds = cg.RawCombinationDataset(moments["TEST"], examples["TEST"])
    sampler = cg.weighted_sampler(train_ds.labels, args.seed)
    loader_kwargs = {"num_workers": args.num_workers, "collate_fn": cg.collate_raw_combinations}
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    model = cg.EndToEndPartCombinationModel(args).to(device)
    encoder_params = list(model.encoder.parameters())
    head_ids = {id(parameter) for parameter in model.classifier.parameters()}
    head_ids.update(id(parameter) for parameter in model.part_embedding.parameters())
    head_params = [parameter for parameter in model.parameters() if id(parameter) in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.learning_rate},
            {"params": head_params, "lr": args.classifier_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir) / args.model_id
    ckpt_dir = out_dir / "ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_records = {"val_loss": None}
    best_paths = {"val_loss": ckpt_dir / "val_loss.pth"}
    best_val_loss = float("inf")
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = cg.train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = cg.validation_loss(model, val_loader, criterion, device)
        line = (
            f"Epoch {epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"time={time.time() - started:.1f}s"
        )
        print(line)
        log_lines.append(line)
        for name, score in {"val_loss": -val_loss}.items():
            if score is None or not np.isfinite(score):
                continue
            current = best_records[name]
            if current is None or float(score) > current["score"]:
                best_records[name] = {"epoch": epoch, "score": float(score), "val_loss": float(val_loss)}
                cg.save_checkpoint(model, best_paths[name])
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = float(val_loss)

    combo_results = {}
    if best_records["val_loss"] is None:
        raise RuntimeError("No finite validation loss; no checkpoint selected")

    summary_path = out_dir / "summary.log"
    combo_json_path = out_dir / "combo_results.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines) + "\n")
        for name, record in best_records.items():
            if record is None:
                continue
            model.load_state_dict(torch.load(best_paths[name], map_location=device))
            overall = evaluate_overall(model, val_loader, test_loader, device)
            line = format_record(f"Best by {name}", record["epoch"], record["val_loss"], overall)
            print(line)
            handle.write(line + "\n")
            combo_records = evaluate_by_combo(model, val_loader, test_loader, device)
            combo_results[name] = {"epoch": record["epoch"], "overall": overall, "combos": combo_records}
            handle.write(f"Per-combination results for {name}\n")
            for item in combo_records:
                combo_line = (
                    f"  {item['combo']} parts={item['part_count']} sample n={item['n_test']} "
                    f"threshold={item['threshold']:.2f} Val: {cg.format_metrics(item['val'])} "
                    f"Test: {cg.format_metrics(item['test'])} | "
                    f"Subject-wise-Val: {format_subjectwise(item['val_subjectwise'])} "
                    f"Subject-wise-Test: {format_subjectwise(item['test_subjectwise'])}"
                )
                print(combo_line)
                handle.write(combo_line + "\n")
    with open(combo_json_path, "w", encoding="utf-8") as handle:
        json.dump(combo_results, handle, indent=2, default=cg.json_default)
    print(f"Saved summary: {summary_path}")
    print(f"Saved structured combo results: {combo_json_path}")


if __name__ == "__main__":
    main()
