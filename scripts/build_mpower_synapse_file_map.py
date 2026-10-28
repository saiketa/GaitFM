
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DEVICE_MOTION_COLUMNS = [
    "deviceMotion_walking_outbound.json.items",
    "deviceMotion_walking_return.json.items",
    "deviceMotion_walking_rest.json.items",
]


def robust_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError as exc:
        print(
            f"Warning: failed to parse {path} with pandas C engine ({exc}). "
            "Retrying with python engine and on_bad_lines='skip'."
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


def read_walking_metadata(root_path):
    candidates = [
        Path(root_path) / "Walking_Activity.csv",
        Path(root_path) / "WalkingActivity" / "WalkingActivity_metadata.csv",
    ]
    for path in candidates:
        if path.exists():
            print(f"Loaded walking metadata: {path}")
            return robust_read_csv(path)
    raise FileNotFoundError(f"No mPower walking metadata found under {root_path}")


def read_demographic_labels(root_path):
    path = Path(root_path) / "Demographics" / "Demographics_metadata.csv"
    if not path.exists():
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


def collect_needed_handles(walking_meta, columns):
    needed = set()
    for column in columns:
        if column not in walking_meta.columns:
            raise ValueError(f"Walking metadata missing column: {column}")
        needed.update(handle_to_str(value) for value in walking_meta[column].dropna())
    needed.discard("")
    return needed


def portable_path(root_path, cached_path):
    root = Path(root_path).resolve()
    path = Path(cached_path)
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        pass
    parts = path.parts
    if "WalkingActivity" in parts:
        idx = parts.index("WalkingActivity")
        return str(Path(*parts[idx:]))
    return str(path)


def scan_synapse_cache(root_path, cache_path, needed_handles):
    cache_root = Path(cache_path)
    if not cache_root.exists():
        raise FileNotFoundError(f"Synapse cache folder not found: {cache_root}")

    out = {}
    inspected = 0
    for dirpath, _, filenames in os.walk(cache_root):
        if ".cacheMap" not in filenames:
            continue
        file_handle_id = Path(dirpath).name
        if file_handle_id not in needed_handles:
            continue
        inspected += 1
        cache_map_path = Path(dirpath) / ".cacheMap"
        try:
            with open(cache_map_path, "r", encoding="utf-8") as handle:
                cache_map = json.load(handle)
        except Exception:
            continue
        candidates = []
        for cached_path, meta in cache_map.items():
            name = Path(cached_path).name
            if not name.startswith("deviceMotion_walking_") or not name.endswith(".tmp"):
                continue
            rel_path = portable_path(root_path, cached_path)
            candidates.append((rel_path, meta or {}))
        if candidates:
            candidates = sorted(candidates, key=lambda item: item[0])
            out[file_handle_id] = candidates[0]
    print(f"Synapse cache inspected handles: {inspected}")
    print(f"Synapse cache matched handles: {len(out)} / {len(needed_handles)}")
    return out


def build_rows(root_path, columns, cache_map):
    walking_meta = read_walking_metadata(root_path)
    label_map = read_demographic_labels(root_path)
    rows = []
    missing_label = 0
    missing_cache = 0
    task_counts = Counter()
    label_counts = Counter()

    for _, row in walking_meta.iterrows():
        health_code = str(row.get("healthCode", ""))
        if health_code not in label_map:
            missing_label += 1
            continue
        label = int(label_map[health_code])
        for column in columns:
            if column not in walking_meta.columns or pd.isna(row.get(column)):
                continue
            handle = handle_to_str(row[column])
            if not handle:
                continue
            cached = cache_map.get(handle)
            if cached is None:
                missing_cache += 1
                continue
            rel_path, cache_meta = cached
            task = column.replace("deviceMotion_walking_", "").replace(".json.items", "")
            rows.append(
                {
                    "fileHandleId": handle,
                    "path": rel_path,
                    "recordId": str(row.get("recordId", "")),
                    "healthCode": health_code,
                    "label": label,
                    "task_column": column,
                    "task": task,
                    "medTimepoint": str(row.get("medTimepoint", "")),
                    "modified_time": cache_meta.get("modified_time", ""),
                    "content_md5": cache_meta.get("content_md5", ""),
                }
            )
            task_counts[task] += 1
            label_counts[label] += 1

    print(f"Rows written: {len(rows)}")
    print(f"Walking metadata rows skipped because diagnosis label is missing: {missing_label}")
    print(f"File handles skipped because cache mapping is missing: {missing_cache}")
    print(f"Task counts: {dict(sorted(task_counts.items()))}")
    print(f"Label counts: {dict(sorted(label_counts.items()))}")
    return rows


def write_rows(path, rows):
    fieldnames = [
        "fileHandleId",
        "path",
        "recordId",
        "healthCode",
        "label",
        "task_column",
        "task",
        "medTimepoint",
        "modified_time",
        "content_md5",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved mapping: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build mPower Synapse cache mapping CSV.")
    parser.add_argument("--root_path", default="./data/mPower")
    parser.add_argument("--synapse_cache", default=None, help="Default: <root_path>/synapseCache")
    parser.add_argument("--output_csv", default=None, help="Default: <root_path>/WalkingActivity_file_map.csv")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional deviceMotion metadata columns. Default uses outbound, return, and rest.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_path = Path(args.root_path)
    walking_meta = read_walking_metadata(root_path)
    columns = args.tasks or DEVICE_MOTION_COLUMNS
    needed_handles = collect_needed_handles(walking_meta, columns)
    synapse_cache = args.synapse_cache or str(root_path / "synapseCache")
    output_csv = args.output_csv or str(root_path / "WalkingActivity_file_map.csv")
    cache_map = scan_synapse_cache(root_path, synapse_cache, needed_handles)
    rows = build_rows(root_path, columns, cache_map)
    write_rows(output_csv, rows)


if __name__ == "__main__":
    main()
