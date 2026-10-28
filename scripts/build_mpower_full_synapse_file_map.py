
import argparse
import csv
import json
import os
from pathlib import Path

import pandas as pd


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


def discover_metadata_files(root_path):
    skip_names = {
        "demographics_metadata.csv",
        "mds-updrs_metadata.csv",
        "mpower_full_file_map.csv",
        "walkingactivity_file_map.csv",
        "file_map.csv",
        "synapse_file_map.csv",
        "download_file_map.csv",
    }
    out = []
    for path in sorted(Path(root_path).rglob("*.csv")):
        lower = path.name.lower()
        if lower.startswith("manifest") or lower in skip_names:
            continue
        try:
            header = robust_read_csv(path, nrows=0)
        except Exception:
            continue
        columns = set(header.columns)
        if "healthCode" in columns and any(str(col).startswith("deviceMotion") for col in columns):
            out.append(path)
    if not out:
        raise FileNotFoundError(f"No metadata CSVs with healthCode + deviceMotion columns found under: {root_path}")
    return out


def collect_needed_handles(metadata_files):
    needed = {}
    for meta_path in metadata_files:
        df = robust_read_csv(meta_path)
        columns = [col for col in df.columns if str(col).startswith("deviceMotion")]
        for _, row in df.iterrows():
            for column in columns:
                handle = handle_to_str(row.get(column))
                if not handle:
                    continue
                needed.setdefault(handle, set()).add(f"{meta_path.name}:{column}")
    return needed


def portable_path(root_path, cached_path):
    root = Path(root_path).resolve()
    path = Path(cached_path)
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        pass
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
            if not name.startswith("deviceMotion_") or not name.endswith(".tmp"):
                continue
            rel_path = portable_path(root_path, cached_path)
            candidates.append((rel_path, meta or {}))
        if candidates:
            candidates = sorted(candidates, key=lambda item: item[0])
            out[file_handle_id] = candidates[0]
    print(f"Synapse cache inspected needed handles: {inspected}")
    print(f"Synapse cache matched handles: {len(out)} / {len(needed_handles)}")
    return out


def write_rows(path, needed_handles, cache_map):
    fieldnames = ["fileHandleId", "path", "metadata_refs", "modified_time", "content_md5"]
    missing = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for file_handle_id in sorted(needed_handles):
            cached = cache_map.get(file_handle_id)
            if cached is None:
                missing += 1
                continue
            rel_path, cache_meta = cached
            writer.writerow(
                {
                    "fileHandleId": file_handle_id,
                    "path": rel_path,
                    "metadata_refs": "|".join(sorted(needed_handles[file_handle_id])),
                    "modified_time": cache_meta.get("modified_time", ""),
                    "content_md5": cache_meta.get("content_md5", ""),
                }
            )
    print(f"File handles without cache mapping: {missing}")
    print(f"Saved mapping: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build full mPower Synapse cache mapping CSV.")
    parser.add_argument("--root_path", default="./data/mPower")
    parser.add_argument("--synapse_cache", default=None, help="Default: <root_path>/synapseCache")
    parser.add_argument("--output_csv", default=None, help="Default: <root_path>/mPower_full_file_map.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    root_path = Path(args.root_path)
    metadata_files = discover_metadata_files(root_path)
    needed_handles = collect_needed_handles(metadata_files)
    print(f"Metadata files with deviceMotion: {len(metadata_files)}")
    print(f"Unique deviceMotion file handles needed: {len(needed_handles)}")
    synapse_cache = args.synapse_cache or str(root_path / "synapseCache")
    output_csv = args.output_csv or str(root_path / "mPower_full_file_map.csv")
    cache_map = scan_synapse_cache(root_path, synapse_cache, needed_handles)
    write_rows(output_csv, needed_handles, cache_map)


if __name__ == "__main__":
    main()
