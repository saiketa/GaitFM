
import argparse
import csv
import glob
import json
import math
import os
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


DEVICE_MOTION_COLUMNS = [
    "deviceMotion_walking_outbound.json.items",
    "deviceMotion_walking_return.json.items",
    "deviceMotion_walking_rest.json.items",
]
FEATURE_COLUMNS = [
    "user_acc_x",
    "user_acc_y",
    "user_acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


def robust_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError as exc:
        warnings.warn(
            f"Failed to parse {path} with pandas C engine ({exc}). "
            "Retrying with python engine and on_bad_lines='skip'. "
            "Malformed metadata rows will be ignored."
        )
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.setdefault("engine", "python")
        fallback_kwargs.setdefault("on_bad_lines", "skip")
        return pd.read_csv(path, **fallback_kwargs)


def parse_bool_label(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return 1
    if text in {"false", "0", "no", "n"}:
        return 0
    raise ValueError(f"Cannot parse professional-diagnosis value: {value}")


def handle_to_str(value):
    if pd.isna(value):
        return ""
    try:
        fval = float(value)
        if fval.is_integer():
            return str(int(fval))
    except Exception:
        pass
    return str(value).strip()


def read_demographic_labels(root_path):
    path = os.path.join(root_path, "Demographics", "Demographics_metadata.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"mPower Demographics metadata not found: {path}")
    df = robust_read_csv(path)
    required = {"healthCode", "professional-diagnosis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"mPower Demographics missing columns: {missing}")
    df = df[df["professional-diagnosis"].notna()].copy()
    df["healthCode"] = df["healthCode"].astype(str)
    df["label"] = df["professional-diagnosis"].apply(parse_bool_label)
    label_df = df[["healthCode", "label"]].drop_duplicates()
    if label_df["healthCode"].duplicated().any():
        label_df = label_df.groupby("healthCode", as_index=False)["label"].max()
    return dict(zip(label_df["healthCode"], label_df["label"]))


def read_walking_metadata(root_path):
    candidates = [
        os.path.join(root_path, "Walking_Activity.csv"),
        os.path.join(root_path, "WalkingActivity", "WalkingActivity_metadata.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Loaded walking metadata: {path}")
            return robust_read_csv(path)
    raise FileNotFoundError(f"mPower WalkingActivity metadata not found under: {root_path}")


def resolve_mpower_path(root_path, walking_dir, value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    candidates = []
    if os.path.isabs(text):
        candidates.append(text)
    else:
        candidates.append(os.path.join(root_path, text))
        candidates.append(os.path.join(walking_dir, text))
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else text


def load_file_map(root_path, walking_dir):
    candidates = []
    for folder in (root_path, walking_dir):
        for name in ("WalkingActivity_file_map.csv", "file_map.csv", "synapse_file_map.csv", "download_file_map.csv"):
            candidates.append(os.path.join(folder, name))

    for path in candidates:
        if not os.path.exists(path):
            continue
        df = robust_read_csv(path)
        lower = {c.lower(): c for c in df.columns}
        handle_col = None
        for key in ("filehandleid", "file_handle_id", "handle_id", "id"):
            if key in lower:
                handle_col = lower[key]
                break
        path_col = None
        for key in ("path", "file_path", "filepath", "filename", "name"):
            if key in lower:
                path_col = lower[key]
                break
        if handle_col is None or path_col is None:
            continue
        out = {}
        for _, row in df.iterrows():
            handle = handle_to_str(row[handle_col])
            full_path = resolve_mpower_path(root_path, walking_dir, row[path_col])
            if os.path.exists(full_path):
                out[handle] = full_path
        if out:
            print(f"Loaded explicit file map: {path} entries={len(out)}")
            return out
    return {}


def fallback_column_file_maps(walking_dir, walking_meta, columns):
    maps = {}
    for column in columns:
        handles = [handle_to_str(v) for v in walking_meta[column].dropna()] if column in walking_meta.columns else []
        files = sorted(glob.glob(os.path.join(walking_dir, f"{column}-*.tmp")))
        count = min(len(handles), len(files))
        maps[column] = {handles[idx]: files[idx] for idx in range(count)}
        print(f"Fallback map {column}: handles={len(handles)} files={len(files)} paired={count}")
    warnings.warn(
        "Explicit file map not found. Pairing downloaded deviceMotion files to metadata rows by sorted order. "
        "For strict reproducibility, save a Synapse file map with fileHandleId/path columns."
    )
    return maps


def matched_records(root_path, max_records=0, tasks=None, allow_sorted_fallback=False, include_unlabeled=False):
    walking_dir = os.path.join(root_path, "WalkingActivity")
    walking_meta = read_walking_metadata(root_path)
    label_map = read_demographic_labels(root_path)
    columns = tasks or DEVICE_MOTION_COLUMNS
    file_map = load_file_map(root_path, walking_dir)
    if file_map:
        fallback_maps = None
    elif allow_sorted_fallback:
        fallback_maps = fallback_column_file_maps(walking_dir, walking_meta, columns)
    else:
        raise FileNotFoundError(
            "No explicit mPower file map found. Build one with "
            "scripts/build_mpower_synapse_file_map.py and place it at "
            "<root_path>/WalkingActivity_file_map.csv, or pass --allow_sorted_fallback "
            "only for diagnostic runs."
        )

    records = []
    skipped_no_label = 0
    for _, row in walking_meta.iterrows():
        health_code = str(row.get("healthCode", ""))
        if health_code not in label_map:
            skipped_no_label += 1
            if not include_unlabeled:
                continue
        for column in columns:
            if column not in walking_meta.columns or pd.isna(row.get(column)):
                continue
            handle = handle_to_str(row[column])
            path = file_map.get(handle)
            if path is None and fallback_maps is not None:
                path = fallback_maps.get(column, {}).get(handle)
            if path is None or not os.path.exists(path):
                continue
            records.append(
                {
                    "path": path,
                    "subject": health_code,
                    "record_id": str(row.get("recordId", "")),
                    "task": column.replace("deviceMotion_walking_", "").replace(".json.items", ""),
                    "label": int(label_map.get(health_code, -1)),
                    "file_handle": handle,
                }
            )
            if max_records > 0 and len(records) >= max_records:
                break
        if max_records > 0 and len(records) >= max_records:
            break
    print(f"Matched records: {len(records)} subjects={len({r['subject'] for r in records})}")
    print(f"Walking metadata rows skipped because diagnosis label is missing: {skipped_no_label}")
    return records


def load_json_items(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read().strip()
    except Exception as exc:
        warnings.warn(f"Failed to read {path}: {exc}")
        return None
    if not text:
        return None
    try:
        items = json.loads(text)
    except Exception:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                return None
        items = rows

    if items is None:
        return None
    if isinstance(items, dict):
        for key in ("items", "data", "records"):
            if isinstance(items.get(key), list):
                return items[key]
        return None
    if not isinstance(items, list):
        return None
    return items


def read_device_motion(path):
    items = load_json_items(path)
    if not items:
        return None, 0.0, 0
    rows = []
    times = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ts = item.get("timestamp")
        ua = item.get("userAcceleration") or {}
        rr = item.get("rotationRate") or {}
        values = [ua.get(axis) for axis in ("x", "y", "z")] + [rr.get(axis) for axis in ("x", "y", "z")]
        if ts is None or any(v is None for v in values):
            continue
        try:
            times.append(float(ts))
            rows.append([float(v) for v in values])
        except Exception:
            continue
    if len(rows) < 2:
        return None, 0.0, len(rows)

    time_sec = np.asarray(times, dtype=np.float64)
    data = np.asarray(rows, dtype=np.float32)
    order = np.argsort(time_sec)
    time_sec = time_sec[order]
    data = data[order]
    _, unique_idx = np.unique(time_sec, return_index=True)
    unique_idx = np.sort(unique_idx)
    time_sec = time_sec[unique_idx]
    data = data[unique_idx]
    if len(time_sec) < 2:
        return None, 0.0, len(time_sec)
    time_sec = time_sec - time_sec[0]
    duration = float(time_sec[-1])
    return (time_sec, data), duration, len(time_sec)


def resample_to_rate(time_sec, data, target_rate):
    duration = float(time_sec[-1])
    if duration <= 0:
        return None
    new_len = int(math.floor(duration * target_rate)) + 1
    if new_len < 2:
        return None
    new_t = np.arange(new_len, dtype=np.float64) / float(target_rate)
    resampled = np.stack([np.interp(new_t, time_sec, data[:, ch]) for ch in range(data.shape[1])], axis=1)
    return resampled.astype(np.float32)


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
        "record_id",
        "task",
        "label",
        "source_path",
        "file_handle",
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
    records = matched_records(
        args.root_path,
        max_records=args.max_records,
        tasks=args.tasks,
        allow_sorted_fallback=args.allow_sorted_fallback,
        include_unlabeled=args.include_unlabeled,
    )
    writer = ShardWriter(args.output_dir, args.shard_size)

    stats = Counter()
    label_counts = Counter()
    task_counts = Counter()
    window_label_counts = Counter()
    window_task_counts = Counter()
    durations = []

    window_size = int(round(args.target_rate * args.window_seconds))
    stride = args.stride if args.stride > 0 else window_size

    for rec_idx, record in enumerate(records, start=1):
        stats["matched_records"] += 1
        label_counts[int(record["label"])] += 1
        task_counts[record["task"]] += 1
        parsed, duration, valid_points = read_device_motion(record["path"])
        durations.append(duration)
        if duration < 10.0:
            stats["records_shorter_than_10s"] += 1
        if duration < 5.0:
            stats["records_shorter_than_5s"] += 1
        if parsed is None:
            stats["invalid_or_empty_records"] += 1
            continue
        time_sec, data = parsed
        resampled = resample_to_rate(time_sec, data, args.target_rate)
        if resampled is None:
            stats["resample_failed_records"] += 1
            continue
        if len(resampled) < window_size:
            stats["records_without_5s_window_after_resample"] += 1
            continue

        n_windows = 0
        for start in range(0, len(resampled) - window_size + 1, stride):
            end = start + window_size
            writer.add(
                resampled[start:end],
                record["label"],
                {
                    "subject": record["subject"],
                    "record_id": record["record_id"],
                    "task": record["task"],
                    "source_path": record["path"],
                    "file_handle": record["file_handle"],
                    "window_start_sample": start,
                    "window_end_sample": end,
                    "duration_sec": f"{duration:.6f}",
                },
            )
            n_windows += 1
            window_label_counts[int(record["label"])] += 1
            window_task_counts[record["task"]] += 1
        stats["windows"] += n_windows
        stats["valid_records_with_windows"] += int(n_windows > 0)

        if args.progress_every > 0 and rec_idx % args.progress_every == 0:
            print(
                f"Processed {rec_idx}/{len(records)} records | windows={stats['windows']} | "
                f"<10s={stats['records_shorter_than_10s']} <5s={stats['records_shorter_than_5s']}"
            )

    writer.flush()
    manifest_path = os.path.join(args.output_dir, "manifest_50hz_5s.csv")
    write_manifest(manifest_path, writer.manifest_rows)

    summary = {
        "root_path": args.root_path,
        "output_dir": args.output_dir,
        "target_rate": args.target_rate,
        "window_seconds": args.window_seconds,
        "window_size": window_size,
        "stride": stride,
        "include_unlabeled": bool(args.include_unlabeled),
        "feature_columns": FEATURE_COLUMNS,
        "stats": dict(stats),
        "record_label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "record_task_counts": dict(sorted(task_counts.items())),
        "window_label_counts": {str(k): int(v) for k, v in sorted(window_label_counts.items())},
        "window_task_counts": dict(sorted(window_task_counts.items())),
        "duration_sec": {
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
    parser = argparse.ArgumentParser(description="Preprocess mPower WalkingActivity deviceMotion to 50 Hz, 5 s windows.")
    parser.add_argument("--root_path", default="./data/mPower")
    parser.add_argument("--output_dir", default="./data/mPower_walking_50hz_5s")
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=0, help="0 means non-overlapping windows of window_size.")
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument("--tasks", nargs="+", default=None, help="Optional deviceMotion metadata columns to use.")
    parser.add_argument(
        "--allow_sorted_fallback",
        action="store_true",
        help="Allow unsafe sorted metadata/file pairing when no explicit file map exists.",
    )
    parser.add_argument(
        "--include_unlabeled",
        action="store_true",
        help="Keep walking records without professional-diagnosis labels as label=-1 for unlabeled pretraining.",
    )
    return parser.parse_args()


def main():
    preprocess(parse_args())


if __name__ == "__main__":
    main()
