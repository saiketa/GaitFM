
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train_clinicalgait_end2end_part_combinations as cg


PART_ORDER = ("LF", "RF", "LW", "RW", "SA")
FEATURE_COLUMNS = ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z")


def parse_args():
    parser = argparse.ArgumentParser(description="Train StrokeRehabilitation part-combination classifier.")
    parser.add_argument("--root_path", type=str, default="./data/StrokeRehabilitation")
    parser.add_argument("--target", type=str, default="FAC_improvement", choices=["FAC_improvement"])
    parser.add_argument("--visits", nargs="+", default=["visit1"], choices=["visit1", "visit2"])
    parser.add_argument("--output_dir", type=str, default="./results_end2end_part_combinations")
    parser.add_argument("--model_id", type=str, default="StrokeRehab_FACImprovement_End2EndPartCombinations")
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--mae_checkpoint_path", type=str, default="./checkpoints/backbones/mae_tiny_400e.pth.tar")
    parser.add_argument("--mae_model", type=str, default="mae_vit_tiny_patch16")
    parser.add_argument("--mae_tag", type=str, default="MAETiny")
    parser.add_argument("--mae_finetune_pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--source_rate", type=float, default=120.0)
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--split_seed", type=int, default=2021)
    parser.add_argument("--max_combo_size", type=int, default=0, help="0 means enumerate all available non-empty combinations.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--classifier_learning_rate", type=float, default=1e-4)
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
    return parser.parse_args()


def read_part_csv(path):
    df = pd.read_csv(path, skiprows=6)
    missing = [column for column in ("SampleTimeFine",) + FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    time_us = pd.to_numeric(df["SampleTimeFine"], errors="coerce").to_numpy(dtype=np.float64)
    data = df[list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    data = data.interpolate(method="linear", limit_direction="both").fillna(0.0).to_numpy(dtype=np.float32)
    valid = np.isfinite(time_us)
    time_us, data = time_us[valid], data[valid]
    order = np.argsort(time_us)
    time_us, data = time_us[order], data[order]
    keep = np.concatenate([[True], np.diff(time_us) > 0])
    return (time_us[keep] - time_us[keep][0]) / 1e6, data[keep]


def resample_to_rate(time_sec, data, target_rate):
    if len(time_sec) < 2:
        return data.astype(np.float32)
    duration = float(time_sec[-1] - time_sec[0])
    if duration <= 0:
        return data.astype(np.float32)
    new_t = np.arange(0.0, duration + 1e-9, 1.0 / float(target_rate), dtype=np.float64)
    out = np.empty((len(new_t), data.shape[1]), dtype=np.float32)
    for col in range(data.shape[1]):
        out[:, col] = np.interp(new_t, time_sec, data[:, col])
    return out


def read_window_array(path, target_rate):
    time_sec, data = read_part_csv(path)
    return resample_to_rate(time_sec, data, target_rate)


def load_labels(root_path):
    info_path = Path(root_path) / "raw" / "participant_info.csv"
    if not info_path.exists():
        raise FileNotFoundError(f"participant_info.csv not found: {info_path}")
    table = pd.read_csv(info_path)
    labels = {}
    for _, row in table.iterrows():
        value = str(row["FAC_improvement"]).strip().upper()
        if value not in {"Y", "N"}:
            continue
        labels[str(row["sub"])] = 1 if value == "Y" else 0
    if not labels:
        raise ValueError("No valid FAC_improvement labels found.")
    return labels


def stratified_6_2_2(labels, seed):
    positives = sorted([subject for subject, label in labels.items() if label == 1])
    negatives = sorted([subject for subject, label in labels.items() if label == 0])
    if len(positives) < 2 or len(negatives) < 2:
        raise ValueError("Need at least two positive and two negative subjects for 1-pos/1-neg VAL and TEST.")
    rng = np.random.default_rng(int(seed))
    positives = [positives[i] for i in rng.permutation(len(positives))]
    negatives = [negatives[i] for i in rng.permutation(len(negatives))]
    split = {
        "VAL": [positives[0], negatives[0]],
        "TEST": [positives[1], negatives[1]],
        "TRAIN": positives[2:] + negatives[2:],
    }
    for name in split:
        split[name] = sorted(split[name])
    return split


def collect_recordings(root_path, split_subjects, labels, visits):
    recordings = {name: [] for name in ("TRAIN", "VAL", "TEST")}
    root = Path(root_path)
    for split, subjects in split_subjects.items():
        for subject in subjects:
            for visit in visits:
                imu_dir = root / "raw" / subject / visit / "imu"
                if not imu_dir.is_dir():
                    continue
                part_files = {part: str(imu_dir / f"{part}.csv") for part in PART_ORDER if (imu_dir / f"{part}.csv").exists()}
                if part_files:
                    recordings[split].append(
                        {
                            "subject": subject,
                            "visit": visit,
                            "label": int(labels[subject]),
                            "part_files": part_files,
                        }
                    )
    return recordings


def fit_train_stats(train_recordings, target_rate):
    arrays = []
    for rec in train_recordings:
        for part in PART_ORDER:
            path = rec["part_files"].get(part)
            if path:
                arrays.append(read_window_array(path, target_rate))
    if not arrays:
        raise ValueError("No StrokeRehabilitation training arrays found for normalization.")
    matrix = np.concatenate(arrays, axis=0)
    return matrix.mean(axis=0).astype(np.float32), (matrix.std(axis=0) + np.finfo(np.float32).eps).astype(np.float32)


def build_moments(recordings, mean, std, target_rate, window_size, stride):
    moments = []
    for rec in recordings:
        part_arrays = {}
        min_len = None
        for part in PART_ORDER:
            path = rec["part_files"].get(part)
            if not path:
                continue
            data = read_window_array(path, target_rate)
            data = ((data - mean[None, :]) / std[None, :]).astype(np.float32)
            part_arrays[part] = data
            min_len = len(data) if min_len is None else min(min_len, len(data))
        if min_len is None or min_len < window_size:
            continue
        for start in range(0, min_len - window_size + 1, stride):
            parts = {part: arr[start : start + window_size] for part, arr in part_arrays.items()}
            moments.append(
                {
                    "subject": rec["subject"],
                    "visit": rec["visit"],
                    "moment_id": f"{rec['subject']}::{rec['visit']}::{start:08d}",
                    "label": int(rec["label"]),
                    "parts": parts,
                }
            )
    return moments


def build_examples(moments, max_combo_size):
    examples = []
    max_size = len(PART_ORDER) if int(max_combo_size) <= 0 else min(int(max_combo_size), len(PART_ORDER))
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
                        "label": int(moment["label"]),
                    }
                )
    return examples


def main():
    args = parse_args()
    cg.PART_ORDER = PART_ORDER
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_gpu else "cpu")
    window_size = int(round(float(args.target_rate) * float(args.window_seconds)))
    stride = int(args.stride)

    labels = load_labels(args.root_path)
    split_subjects = stratified_6_2_2(labels, args.split_seed)
    for split, subjects in split_subjects.items():
        print(f"{split} subjects: {subjects} label_counts={dict(zip(*np.unique([labels[s] for s in subjects], return_counts=True)))}")
    recordings = collect_recordings(args.root_path, split_subjects, labels, args.visits)
    train_mean, train_std = fit_train_stats(recordings["TRAIN"], args.target_rate)
    moments = {
        split: build_moments(recordings[split], train_mean, train_std, args.target_rate, window_size, stride)
        for split in ("TRAIN", "VAL", "TEST")
    }
    examples = {split: build_examples(moments[split], args.max_combo_size) for split in ("TRAIN", "VAL", "TEST")}
    for split in ("TRAIN", "VAL", "TEST"):
        y = np.asarray([item["label"] for item in examples[split]], dtype=np.int64)
        counts = dict(zip(*np.unique(y, return_counts=True))) if len(y) else {}
        print(f"{split}: moments={len(moments[split])} combinations={len(examples[split])} label_counts={counts}")
    if not examples["TRAIN"] or not examples["VAL"] or not examples["TEST"]:
        raise ValueError("TRAIN/VAL/TEST combinations must all be non-empty.")

    train_ds = cg.RawCombinationDataset(moments["TRAIN"], examples["TRAIN"])
    val_ds = cg.RawCombinationDataset(moments["VAL"], examples["VAL"])
    test_ds = cg.RawCombinationDataset(moments["TEST"], examples["TEST"])
    sampler = cg.weighted_sampler(train_ds.labels, args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=cg.collate_raw_combinations,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=cg.collate_raw_combinations)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=cg.collate_raw_combinations)

    model = cg.EndToEndPartCombinationModel(args).to(device)
    encoder_params = list(model.encoder.parameters())
    classifier_param_ids = {id(param) for param in model.classifier.parameters()}
    classifier_param_ids.update(id(param) for param in model.part_embedding.parameters())
    classifier_params = [param for param in model.parameters() if id(param) in classifier_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.learning_rate},
            {"params": classifier_params, "lr": args.classifier_learning_rate},
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
        start = time.time()
        train_loss = cg.train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = cg.validation_loss(model, val_loader, criterion, device)
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
                cg.save_checkpoint(model, best_paths[name])
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = float(val_loss)

    if best_records["val_loss"] is None:
        raise RuntimeError("No finite validation loss; no checkpoint selected")

    summary_path = out_dir / "summary.log"
    combo_json_path = out_dir / "combo_results.json"
    combo_results = {}
    with open(summary_path, "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")
        for name, record in best_records.items():
            if record is None:
                continue
            model.load_state_dict(torch.load(best_paths[name], map_location=device))
            overall = cg.evaluate_overall(model, val_loader, test_loader, device)
            line = cg.format_record(f"Best by {name}", record["epoch"], record["val_loss"], overall)
            print(line)
            f.write(line + "\n")
            combo_results[name] = {
                "epoch": record["epoch"],
                "overall": overall,
                "combos": cg.evaluate_by_combo(model, val_loader, test_loader, device),
            }
            f.write(f"Per-combination results for {name}\n")
            for combo_record in combo_results[name]["combos"]:
                sample_line = (
                    f"  {combo_record['combo']} sample n={combo_record['n_test']} "
                    f"threshold={combo_record['threshold']:.2f} "
                    f"Val: {cg.format_metrics(combo_record['val'])} "
                    f"Test: {cg.format_metrics(combo_record['test'])}"
                )
                subject_line = (
                    f"  {combo_record['combo']} subject n={combo_record['n_test_subjects']} "
                    f"threshold={combo_record['subject_threshold']:.2f} "
                    f"Val: {cg.format_metrics(combo_record['val_subject'])} "
                    f"Test: {cg.format_metrics(combo_record['test_subject'])}"
                )
                print(sample_line)
                print(subject_line)
                f.write(sample_line + "\n")
                f.write(subject_line + "\n")
    with open(combo_json_path, "w", encoding="utf-8") as f:
        json.dump(combo_results, f, indent=2, default=cg.json_default)
    print(f"Saved summary: {summary_path}")
    print(f"Saved structured combo results: {combo_json_path}")


if __name__ == "__main__":
    main()
