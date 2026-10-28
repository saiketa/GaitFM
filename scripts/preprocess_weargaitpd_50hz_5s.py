
import argparse
import csv
import json
import os
import warnings
from collections import Counter
from glob import glob

import numpy as np
import pandas as pd


DEFAULT_BODY_PARTS = [
    "LowerBack",
    "R_Wrist",
    "L_Wrist",
    "R_MidLatThigh",
    "L_MidLatThigh",
    "R_LatShank",
    "L_LatShank",
    "R_DorsalFoot",
    "L_DorsalFoot",
    "R_Ankle",
    "L_Ankle",
    "Xiphoid",
    "Forehead",
]
AXES = ["X", "Y", "Z"]
FEATURE_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def subject_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_", 1)[0]


def task_from_path(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.split("_", 1)[1] if "_" in stem else ""


def label_from_subject(subject_id):
    subject = str(subject_id)
    if "(control)" in subject.lower() or subject.startswith(("HC", "WHC")):
        return 0
    if subject.startswith(("WPD", "NLS")):
        return 1
    raise ValueError(f"Cannot infer WearGait-PD label from subject id: {subject_id}")


def raw_files(root, tasks=None):
    selected = []
    task_set = set(tasks) if tasks else None
    for path in sorted(glob(os.path.join(root, "**", "*.csv"), recursive=True)):
        name = os.path.basename(path)
        if name.lower() in {"subjects.csv", "files.csv"} or name.startswith("manifest"):
            continue
        task = task_from_path(path)
        if task_set is None or task in task_set:
            selected.append(path)
    if not selected:
        raise FileNotFoundError(f"No WearGait-PD CSV files found in {root}.")
    return selected


def time_seconds(df):
    if "Time" not in df.columns:
        return None
    values = df["Time"].astype(str).str.extract(r"([-+]?\d*\.?\d+)")[0]
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    if np.isfinite(values).sum() < 2:
        return None
    return values


def body_part_array(df, body_part):
    columns = [f"{body_part}_Acc_{axis}" for axis in AXES] + [f"{body_part}_Gyr_{axis}" for axis in AXES]
    if not all(column in df.columns for column in columns):
        return None
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    values = values.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float32)
    if len(values) < 2:
        return None
    return values


def resample(array, time_sec, target_rate, source_rate):
    if time_sec is None:
        duration = (len(array) - 1) / float(source_rate)
        old_t = np.linspace(0.0, duration, len(array), dtype=np.float64)
    else:
        old_t = np.asarray(time_sec, dtype=np.float64)
        finite = np.isfinite(old_t)
        array = array[finite]
        old_t = old_t[finite]
        if len(old_t) < 2:
            return array.astype(np.float32)
        order = np.argsort(old_t)
        old_t = old_t[order]
        array = array[order]
        _, unique_idx = np.unique(old_t, return_index=True)
        unique_idx = np.sort(unique_idx)
        old_t = old_t[unique_idx]
        array = array[unique_idx]
        old_t = old_t - old_t[0]
    duration = float(old_t[-1])
    if duration <= 0:
        return array.astype(np.float32)
    new_len = int(np.floor(duration * float(target_rate))) + 1
    new_t = np.arange(new_len, dtype=np.float64) / float(target_rate)
    out = np.stack([np.interp(new_t, old_t, array[:, ch]) for ch in range(array.shape[1])], axis=1)
    return out.astype(np.float32)


class ShardWriter:
    def __init__(self, output_dir, shard_size):
        self.output_dir = output_dir
        self.shard_size = int(shard_size)
        self.shard_idx = 0
        self.buffer_x = []
        self.buffer_y = []
        self.buffer_meta = []
        self.manifest_rows = []

    def add(self, window, label, meta):
        self.buffer_x.append(window.astype(np.float32))
        self.buffer_y.append(int(label))
        self.buffer_meta.append(meta)
        if len(self.buffer_x) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buffer_x:
            return
        shard_name = f"windows_50hz_5s_shard_{self.shard_idx:05d}.npz"
        shard_path = os.path.join(self.output_dir, shard_name)
        x = np.stack(self.buffer_x, axis=0).astype(np.float32)
        y = np.asarray(self.buffer_y, dtype=np.int64)
        np.savez(shard_path, X=x, y=y, feature_columns=np.asarray(FEATURE_COLUMNS))
        for local_idx, meta in enumerate(self.buffer_meta):
            row = dict(meta)
            row.update({"shard": shard_name, "shard_index": local_idx, "label": int(y[local_idx])})
            self.manifest_rows.append(row)
        print(f"Wrote {shard_name}: X={x.shape}")
        self.shard_idx += 1
        self.buffer_x.clear()
        self.buffer_y.clear()
        self.buffer_meta.clear()


def write_manifest(path, rows):
    fieldnames = [
        "global_index",
        "shard",
        "shard_index",
        "subject",
        "task",
        "body_part",
        "label",
        "source_path",
        "window_start_sample",
        "window_end_sample",
        "duration_sec",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = {key: row.get(key, "") for key in fieldnames}
            out["global_index"] = idx
            writer.writerow(out)


def preprocess(args):
    os.makedirs(args.output_dir, exist_ok=True)
    files = raw_files(args.root_path, tasks=args.tasks)
    body_parts = args.body_parts or DEFAULT_BODY_PARTS
    window_size = int(round(args.target_rate * args.window_seconds))
    stride = args.stride if args.stride > 0 else window_size

    writer = ShardWriter(args.output_dir, args.shard_size)
    stats = Counter()
    file_label_counts = Counter()
    file_task_counts = Counter()
    window_label_counts = Counter()
    window_task_counts = Counter()
    window_body_counts = Counter()
    durations = []

    for file_idx, path in enumerate(files, start=1):
        subject = subject_from_path(path)
        try:
            label = label_from_subject(subject)
        except ValueError as exc:
            warnings.warn(str(exc))
            stats["files_skipped_bad_subject"] += 1
            continue
        task = task_from_path(path)
        stats["files_seen"] += 1
        file_label_counts[label] += 1
        file_task_counts[task] += 1
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            warnings.warn(f"Failed to read WearGait-PD file {path}: {exc}")
            stats["files_read_failed"] += 1
            continue
        t = time_seconds(df)
        available_parts = 0
        file_windows = 0
        for body_part in body_parts:
            array = body_part_array(df, body_part)
            if array is None:
                continue
            available_parts += 1
            resampled = resample(array, t, args.target_rate, args.source_rate)
            duration = (len(resampled) - 1) / float(args.target_rate) if len(resampled) > 1 else 0.0
            durations.append(duration)
            if duration < 10.0:
                stats["body_recordings_shorter_than_10s"] += 1
            if duration < 5.0:
                stats["body_recordings_shorter_than_5s"] += 1
            if len(resampled) < window_size:
                stats["body_recordings_without_5s_window"] += 1
                continue
            n_windows = 0
            for start in range(0, len(resampled) - window_size + 1, stride):
                end = start + window_size
                writer.add(
                    resampled[start:end],
                    label,
                    {
                        "subject": subject,
                        "task": task,
                        "body_part": body_part,
                        "source_path": path,
                        "window_start_sample": start,
                        "window_end_sample": end,
                        "duration_sec": f"{duration:.6f}",
                    },
                )
                n_windows += 1
                window_label_counts[label] += 1
                window_task_counts[task] += 1
                window_body_counts[body_part] += 1
            file_windows += n_windows
            stats["body_recordings_with_windows"] += int(n_windows > 0)
            stats["windows"] += n_windows
        if available_parts == 0:
            stats["files_without_requested_body_parts"] += 1
        if file_windows > 0:
            stats["files_with_windows"] += 1
        if args.progress_every > 0 and file_idx % args.progress_every == 0:
            print(
                f"Processed {file_idx}/{len(files)} files | windows={stats['windows']} | "
                f"body<10s={stats['body_recordings_shorter_than_10s']} "
                f"body<5s={stats['body_recordings_shorter_than_5s']}"
            )

    writer.flush()
    manifest_path = os.path.join(args.output_dir, "manifest_50hz_5s.csv")
    write_manifest(manifest_path, writer.manifest_rows)

    summary = {
        "root_path": args.root_path,
        "output_dir": args.output_dir,
        "target_rate": args.target_rate,
        "source_rate": args.source_rate,
        "window_seconds": args.window_seconds,
        "window_size": window_size,
        "stride": stride,
        "feature_columns": FEATURE_COLUMNS,
        "body_parts": body_parts,
        "stats": dict(stats),
        "file_label_counts": {str(k): int(v) for k, v in sorted(file_label_counts.items())},
        "file_task_counts": dict(sorted(file_task_counts.items())),
        "window_label_counts": {str(k): int(v) for k, v in sorted(window_label_counts.items())},
        "window_task_counts": dict(sorted(window_task_counts.items())),
        "window_body_counts": dict(sorted(window_body_counts.items())),
        "duration_sec_per_body_recording": {
            "count": len(durations),
            "mean": float(np.mean(durations)) if durations else 0.0,
            "median": float(np.median(durations)) if durations else 0.0,
            "min": float(np.min(durations)) if durations else 0.0,
            "max": float(np.max(durations)) if durations else 0.0,
        },
        "manifest": manifest_path,
        "num_shards": writer.shard_idx,
    }
    summary_path = os.path.join(args.output_dir, "summary_50hz_5s.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("\nSummary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess WearGait-PD raw CSV files to 50 Hz, 5 s windows.")
    parser.add_argument("--root_path", default="./data/WearGait-PD")
    parser.add_argument("--output_dir", default="./data/WearGait-PD_50hz_5s")
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--source_rate", type=float, default=100.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=0, help="0 means non-overlapping windows of window_size.")
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional task names parsed from file names.")
    parser.add_argument("--body_parts", nargs="+", default=None, help="Optional body parts to extract.")
    return parser.parse_args()


def main():
    preprocess(parse_args())


if __name__ == "__main__":
    main()
