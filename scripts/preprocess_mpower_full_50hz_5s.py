
import argparse
import csv
import glob
import json
import math
import os
from collections import Counter

import numpy as np

from preprocess_mpower_walking_50hz_5s import (
    FEATURE_COLUMNS,
    ShardWriter,
    handle_to_str,
    load_json_items,
    read_demographic_labels,
    robust_read_csv,
)


def discover_metadata_files(root_path):
    files = []
    for path in sorted(glob.glob(os.path.join(root_path, "**", "*.csv"), recursive=True)):
        lower = os.path.basename(path).lower()
        if lower.startswith("manifest") or lower in {
            "demographics_metadata.csv",
            "mds-updrs_metadata.csv",
            "mpower_full_file_map.csv",
            "walkingactivity_file_map.csv",
            "file_map.csv",
            "synapse_file_map.csv",
            "download_file_map.csv",
        }:
            continue
        try:
            header = robust_read_csv(path, nrows=0)
        except Exception:
            continue
        columns = set(header.columns)
        if "healthCode" in columns and any(str(col).startswith("deviceMotion") for col in columns):
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No mPower metadata CSVs with healthCode + deviceMotion columns found under: {root_path}")
    return files


def resolve_path(root_path, value):
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    candidates = [text] if os.path.isabs(text) else [
        os.path.join(root_path, text),
        os.path.join(root_path, "WalkingActivity", text),
        os.path.join(root_path, "TappingActivity", text),
        os.path.join(root_path, "MemoryActivity", text),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else text


def load_global_file_map(root_path):
    candidates = []
    for folder, _, files in os.walk(root_path):
        for name in files:
            if name.lower() in {
                "mpower_full_file_map.csv",
                "walkingactivity_file_map.csv",
                "file_map.csv",
                "synapse_file_map.csv",
                "download_file_map.csv",
            }:
                candidates.append(os.path.join(folder, name))
    out = {}
    for path in sorted(candidates):
        df = robust_read_csv(path)
        lower = {str(c).lower(): c for c in df.columns}
        handle_col = next((lower[k] for k in ("filehandleid", "file_handle_id", "handle_id", "id") if k in lower), None)
        path_col = next((lower[k] for k in ("path", "file_path", "filepath", "filename", "name") if k in lower), None)
        if handle_col is None or path_col is None:
            continue
        before = len(out)
        for _, row in df.iterrows():
            handle = handle_to_str(row[handle_col])
            full_path = resolve_path(root_path, row[path_col])
            if handle and os.path.exists(full_path):
                out[handle] = full_path
        print(f"Loaded file map: {path} added={len(out) - before} total={len(out)}")
    if not out:
        raise FileNotFoundError(
            "No explicit mPower file map found. Build a fileHandleId/path map from Synapse cache "
            "and place it under the mPower root before preprocessing the full dataset."
        )
    return out


def read_device_motion(path):
    items = load_json_items(path)
    if not items:
        return None, 0.0
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
        return None, 0.0
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
        return None, 0.0
    time_sec = time_sec - time_sec[0]
    return (time_sec, data), float(time_sec[-1])


def resample_to_rate(time_sec, data, target_rate):
    duration = float(time_sec[-1])
    if duration <= 0:
        return None
    new_len = int(math.floor(duration * target_rate)) + 1
    if new_len < 2:
        return None
    new_t = np.arange(new_len, dtype=np.float64) / float(target_rate)
    return np.stack([np.interp(new_t, time_sec, data[:, ch]) for ch in range(data.shape[1])], axis=1).astype(np.float32)


def metadata_task_name(path, column):
    column_text = str(column)
    if column_text.startswith("deviceMotion_walking_") and column_text.endswith(".json.items"):
        return column_text.replace("deviceMotion_walking_", "").replace(".json.items", "")
    stem = os.path.splitext(os.path.basename(path))[0]
    col = column_text.replace(".json.items", "")
    if col.startswith("deviceMotion_"):
        col = col[len("deviceMotion_"):]
    return f"{stem}:{col}"


def matched_records(root_path, max_records=0):
    label_map = read_demographic_labels(root_path)
    file_map = load_global_file_map(root_path)
    metadata_files = discover_metadata_files(root_path)
    records = []
    task_counts = Counter()
    missing_file = 0
    for meta_path in metadata_files:
        df = robust_read_csv(meta_path)
        columns = [col for col in df.columns if str(col).startswith("deviceMotion")]
        for _, row in df.iterrows():
            health_code = str(row.get("healthCode", ""))
            label = int(label_map.get(health_code, -1))
            for column in columns:
                handle = handle_to_str(row.get(column))
                if not handle:
                    continue
                path = file_map.get(handle)
                if path is None or not os.path.exists(path):
                    missing_file += 1
                    continue
                task = metadata_task_name(meta_path, column)
                records.append({
                    "path": path,
                    "subject": health_code,
                    "record_id": str(row.get("recordId", "")),
                    "task": task,
                    "label": label,
                    "file_handle": handle,
                })
                task_counts[task] += 1
                if max_records > 0 and len(records) >= max_records:
                    break
            if max_records > 0 and len(records) >= max_records:
                break
        if max_records > 0 and len(records) >= max_records:
            break
    print(f"Metadata files with deviceMotion: {len(metadata_files)}")
    print(f"Matched deviceMotion records: {len(records)} subjects={len({r['subject'] for r in records})}")
    print(f"DeviceMotion handles missing local file mapping: {missing_file}")
    print(f"Task counts: {dict(sorted(task_counts.items()))}")
    return records


def write_manifest(path, rows):
    fieldnames = [
        "global_index", "shard", "shard_index", "subject", "record_id", "task",
        "label", "source_path", "file_handle", "window_start_sample",
        "window_end_sample", "duration_sec",
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
    records = matched_records(args.root_path, max_records=args.max_records)
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
        parsed, duration = read_device_motion(record["path"])
        durations.append(duration)
        if parsed is None:
            stats["invalid_or_empty_records"] += 1
            continue
        time_sec, data = parsed
        resampled = resample_to_rate(time_sec, data, args.target_rate)
        if resampled is None or len(resampled) < window_size:
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
            print(f"Processed {rec_idx}/{len(records)} records | windows={stats['windows']}")
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
    parser = argparse.ArgumentParser(description="Preprocess all mPower deviceMotion streams to 50 Hz, 5 s windows.")
    parser.add_argument("--root_path", default="./data/mPower")
    parser.add_argument("--output_dir", default="./data/mPower_full_50hz_5s")
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=0, help="0 means non-overlapping windows of window_size.")
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=1000)
    return parser.parse_args()


def main():
    preprocess(parse_args())


if __name__ == "__main__":
    main()
