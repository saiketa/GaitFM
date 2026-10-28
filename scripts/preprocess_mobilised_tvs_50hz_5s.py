
import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np


COHORTS = ("HA", "PD", "MS", "PFF", "COPD", "CHF")
PRETRAIN_LAB_TESTS = {f"Test{i}" for i in range(4, 12)}
FINETUNE_LAB_TESTS = {"Test5", "Test6", "Test7"}
PRETRAIN_FREE_RECORDINGS = {"Recording4"}
FEATURE_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def import_loadmat():
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise RuntimeError(
            "scipy is required to read Mobilise-D .mat files. Install scipy in the server environment."
        ) from exc
    return loadmat


def normalize_key(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def is_mapping(value):
    return isinstance(value, dict)


def as_array(value):
    if isinstance(value, np.ndarray):
        return value
    return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_mat(path):
    loadmat = import_loadmat()
    data = loadmat(path, simplify_cells=True)
    return {key: value for key, value in data.items() if not key.startswith("__")}


def get_child(obj, token):
    if not is_mapping(obj):
        return None
    if token in obj:
        return obj[token]
    wanted = normalize_key(token)
    for key, value in obj.items():
        if normalize_key(key) == wanted:
            return value
    return None


def find_path(obj, tokens):
    if not tokens:
        return obj
    child = get_child(obj, tokens[0])
    if child is not None:
        found = find_path(child, tokens[1:])
        if found is not None:
            return found
    if is_mapping(obj):
        for value in obj.values():
            found = find_path(value, tokens)
            if found is not None:
                return found
    return None


def iter_arrays(obj, prefix=""):
    if is_mapping(obj):
        for key, value in obj.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            yield from iter_arrays(value, name)
    else:
        arr = as_array(obj)
        if arr is not None and np.issubdtype(arr.dtype, np.number):
            yield prefix, arr


def squeeze_numeric(arr):
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return arr.astype(np.float64)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    return None


def looks_like_axis_matrix(arr):
    arr = squeeze_numeric(arr)
    if arr is None or arr.ndim != 2:
        return False
    return 3 in arr.shape and max(arr.shape) >= 2


def to_n_by_3(arr):
    arr = squeeze_numeric(arr)
    if arr is None or arr.ndim != 2:
        return None
    if arr.shape[1] == 3:
        return arr.astype(np.float32)
    if arr.shape[0] == 3:
        return arr.T.astype(np.float32)
    return None


def find_named_matrix(block, include_terms, exclude_terms=()):
    scored = []
    include_terms = [normalize_key(term) for term in include_terms]
    exclude_terms = [normalize_key(term) for term in exclude_terms]
    for name, arr in iter_arrays(block):
        lname = normalize_key(name)
        if not all(term in lname for term in include_terms):
            continue
        if any(term in lname for term in exclude_terms):
            continue
        if looks_like_axis_matrix(arr):
            scored.append((len(lname), name, arr))
    if not scored:
        return None, ""
    _, name, arr = sorted(scored, key=lambda item: item[0])[0]
    return to_n_by_3(arr), name


def find_time_vector(block, expected_len):
    candidates = []
    for name, arr in iter_arrays(block):
        lname = normalize_key(name)
        if "time" not in lname and "timestamp" not in lname:
            continue
        arr = squeeze_numeric(arr)
        if arr is None or arr.ndim != 1 or len(arr) < 2:
            continue
        if abs(len(arr) - expected_len) <= 2:
            candidates.append((abs(len(arr) - expected_len), name, arr))
    if not candidates:
        return None, ""
    _, name, arr = sorted(candidates, key=lambda item: item[0])[0]
    return np.asarray(arr, dtype=np.float64), name


def extract_sensor_stream(mat_data, key_tokens, sensor_system, sensor_name, default_rate):
    roots = []
    if "data" in mat_data:
        roots.append(mat_data["data"])
    roots.append(mat_data)
    block = None
    for root in roots:
        block = find_path(root, list(key_tokens) + [sensor_system, sensor_name])
        if block is not None:
            break
        block = find_path(root, list(key_tokens) + [sensor_name])
        if block is not None:
            break
    if block is None:
        return None, "sensor block not found"

    acc, acc_name = find_named_matrix(block, ["acc"], exclude_terms=["gyr", "gyro", "angular"])
    if acc is None:
        acc, acc_name = find_named_matrix(block, ["acceler"])
    gyro, gyro_name = find_named_matrix(block, ["gyr"])
    if gyro is None:
        gyro, gyro_name = find_named_matrix(block, ["gyro"])
    if gyro is None:
        gyro, gyro_name = find_named_matrix(block, ["angular"])
    if acc is None or gyro is None:
        return None, f"acc/gyro not found in sensor block; acc={acc_name} gyro={gyro_name}"

    n = min(len(acc), len(gyro))
    acc = acc[:n]
    gyro = gyro[:n]
    time_vec, time_name = find_time_vector(block, n)
    if time_vec is not None:
        time_vec = np.asarray(time_vec[:n], dtype=np.float64)
        finite = np.isfinite(time_vec)
        if finite.sum() >= 2:
            time_sec = time_vec[finite]
            data = np.concatenate([acc[finite], gyro[finite]], axis=1).astype(np.float32)
            order = np.argsort(time_sec)
            time_sec = time_sec[order]
            data = data[order]
            _, unique_idx = np.unique(time_sec, return_index=True)
            unique_idx = np.sort(unique_idx)
            time_sec = time_sec[unique_idx]
            data = data[unique_idx]
            if time_sec[-1] > time_sec[0]:
                time_sec = time_sec - time_sec[0]
                return (time_sec, data), f"acc={acc_name};gyro={gyro_name};time={time_name}"

    time_sec = np.arange(n, dtype=np.float64) / float(default_rate)
    data = np.concatenate([acc, gyro], axis=1).astype(np.float32)
    return (time_sec, data), f"acc={acc_name};gyro={gyro_name};time=synthetic_{default_rate}Hz"


def resample_to_rate(time_sec, data, target_rate):
    if len(time_sec) < 2:
        return None
    duration = float(time_sec[-1] - time_sec[0])
    if duration <= 0:
        return None
    new_len = int(math.floor(duration * target_rate)) + 1
    if new_len < 2:
        return None
    new_t = np.arange(new_len, dtype=np.float64) / float(target_rate)
    return np.stack([np.interp(new_t, time_sec, data[:, ch]) for ch in range(data.shape[1])], axis=1).astype(np.float32)


class ShardWriter:
    def __init__(self, output_dir, shard_size):
        self.output_dir = Path(output_dir)
        self.shard_size = int(shard_size)
        self.x_parts = []
        self.y_parts = []
        self.pending_rows = []
        self.manifest_rows = []
        self.shard_idx = 0
        self.shard_index = 0

    def add(self, x, label, row):
        self.x_parts.append(np.asarray(x, dtype=np.float32))
        self.y_parts.append(int(label))
        item = dict(row)
        item["shard"] = ""
        item["shard_index"] = self.shard_index
        self.pending_rows.append(item)
        self.shard_index += 1
        if len(self.x_parts) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.x_parts:
            return
        shard_name = f"windows_50hz_5s_shard_{self.shard_idx:05d}.npz"
        path = self.output_dir / shard_name
        np.savez(
            path,
            X=np.stack(self.x_parts, axis=0).astype(np.float32),
            y=np.asarray(self.y_parts, dtype=np.int64),
        )
        for row in self.pending_rows:
            row["shard"] = shard_name
            self.manifest_rows.append(row)
        print(f"Wrote {shard_name}: X=({len(self.x_parts)}, {self.x_parts[0].shape[0]}, {self.x_parts[0].shape[1]})")
        self.x_parts = []
        self.y_parts = []
        self.pending_rows = []
        self.shard_idx += 1
        self.shard_index = 0


def iter_records(root_path, sensor_systems, body_parts):
    root = Path(root_path)
    for cohort_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name in COHORTS):
        cohort = cohort_dir.name
        for subject_dir in sorted(path for path in cohort_dir.iterdir() if path.is_dir()):
            subject = f"{cohort}_{subject_dir.name}"
            for measurement_dir in [subject_dir / "Laboratory", subject_dir / "Free-living"]:
                test_list_path = measurement_dir / "test_list.json"
                data_path = measurement_dir / "data.mat"
                if not test_list_path.exists() or not data_path.exists():
                    continue
                measurement = measurement_dir.name
                try:
                    tests = load_json(test_list_path).get("data", [])
                except Exception as exc:
                    print(f"Warning: failed to read {test_list_path}: {exc}")
                    continue
                for entry in tests:
                    key = [str(item) for item in entry.get("key", [])]
                    if len(key) < 2:
                        continue
                    test = key[1]
                    trial = key[2] if len(key) > 2 else ""
                    use_for_pretrain = int(
                        (measurement == "Laboratory" and test in PRETRAIN_LAB_TESTS)
                        or (measurement == "Free-living" and test in PRETRAIN_FREE_RECORDINGS)
                    )
                    use_for_finetune = int(measurement == "Laboratory" and test in FINETUNE_LAB_TESTS)
                    if not use_for_pretrain and not use_for_finetune:
                        continue
                    available = entry.get("value", {}).get("available_sensors", [])
                    for sensor_system, body_part in available:
                        if sensor_systems and sensor_system not in sensor_systems:
                            continue
                        if body_parts and body_part not in body_parts:
                            continue
                        yield {
                            "cohort": cohort,
                            "subject": subject,
                            "subject_raw": subject_dir.name,
                            "measurement": measurement,
                            "key": key,
                            "test": test,
                            "trial": trial,
                            "sensor_system": sensor_system,
                            "body_part": body_part,
                            "data_path": data_path,
                            "use_for_pretrain": use_for_pretrain,
                            "use_for_finetune": use_for_finetune,
                        }


def write_manifest(path, rows):
    fieldnames = [
        "global_index", "shard", "shard_index", "subject", "subject_raw", "cohort", "label",
        "measurement", "test", "trial", "sensor_system", "body_part",
        "use_for_pretrain", "use_for_finetune", "source_file", "window_start_sample",
        "window_end_sample", "duration_sec", "extract_info",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = {key: row.get(key, "") for key in fieldnames}
            out["global_index"] = idx
            writer.writerow(out)


def preprocess(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sensor_systems = [item.strip() for item in args.sensor_systems.split(",") if item.strip()]
    body_parts = [item.strip() for item in args.body_parts.split(",") if item.strip()]
    writer = ShardWriter(output_dir, args.shard_size)
    window_size = int(round(args.target_rate * args.window_seconds))
    stride = args.stride if args.stride > 0 else window_size
    stats = Counter()
    cohort_counts = Counter()
    window_cohort_counts = Counter()
    record_cache = {}

    records = list(iter_records(args.root_path, sensor_systems, body_parts))
    print(f"Mobilise-D selected sensor records: {len(records)}")
    for idx, record in enumerate(records, start=1):
        stats["selected_records"] += 1
        cohort_counts[record["cohort"]] += 1
        data_path = str(record["data_path"])
        if data_path not in record_cache:
            try:
                record_cache[data_path] = load_mat(data_path)
            except Exception as exc:
                stats["failed_mat_files"] += 1
                print(f"Warning: failed to load {data_path}: {exc}")
                continue
        parsed, info = extract_sensor_stream(
            record_cache[data_path],
            record["key"],
            record["sensor_system"],
            record["body_part"],
            args.default_source_rate,
        )
        if parsed is None:
            stats["missing_sensor_streams"] += 1
            if args.verbose_missing:
                print(f"Warning: {record['source_file'] if 'source_file' in record else data_path}: {info}")
            continue
        time_sec, data = parsed
        resampled = resample_to_rate(time_sec, data, args.target_rate)
        if resampled is None or len(resampled) < window_size:
            stats["too_short_after_resample"] += 1
            continue
        n_windows = 0
        label = 0 if record["cohort"] == "HA" else 1
        duration = float(time_sec[-1]) if len(time_sec) else 0.0
        for start in range(0, len(resampled) - window_size + 1, stride):
            end = start + window_size
            writer.add(
                resampled[start:end],
                label,
                {
                    "subject": record["subject"],
                    "subject_raw": record["subject_raw"],
                    "cohort": record["cohort"],
                    "label": label,
                    "measurement": record["measurement"],
                    "test": record["test"],
                    "trial": record["trial"],
                    "sensor_system": record["sensor_system"],
                    "body_part": record["body_part"],
                    "use_for_pretrain": record["use_for_pretrain"],
                    "use_for_finetune": record["use_for_finetune"],
                    "source_file": str(Path(data_path).relative_to(Path(args.root_path))),
                    "window_start_sample": start,
                    "window_end_sample": end,
                    "duration_sec": f"{duration:.6f}",
                    "extract_info": info,
                },
            )
            n_windows += 1
            window_cohort_counts[record["cohort"]] += 1
        stats["windows"] += n_windows
        stats["valid_records_with_windows"] += int(n_windows > 0)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"Processed {idx}/{len(records)} sensor records | windows={stats['windows']}")
    writer.flush()
    if not writer.manifest_rows:
        raise RuntimeError("No Mobilise-D windows were produced. Check .mat structure and sensor selection.")
    manifest_path = output_dir / "manifest_50hz_5s.csv"
    write_manifest(manifest_path, writer.manifest_rows)
    summary = {
        "root_path": args.root_path,
        "output_dir": args.output_dir,
        "target_rate": args.target_rate,
        "window_seconds": args.window_seconds,
        "window_size": window_size,
        "stride": stride,
        "feature_columns": FEATURE_COLUMNS,
        "sensor_systems": sensor_systems,
        "body_parts": body_parts,
        "pretrain_selection": "Laboratory Test4-Test11 + Free-living Recording4",
        "finetune_selection": "Laboratory Test5-Test7",
        "stats": dict(stats),
        "record_cohort_counts": dict(cohort_counts),
        "window_cohort_counts": dict(window_cohort_counts),
        "manifest": str(manifest_path),
        "num_shards": writer.shard_idx,
    }
    summary_path = output_dir / "summary_50hz_5s.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Mobilise-D TVS to 50 Hz, 5 s windows.")
    parser.add_argument("--root_path", default="./data/Mobilise-D-TVS")
    parser.add_argument("--output_dir", default="./data/Mobilise-D-TVS_50hz_5s")
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=0, help="0 means non-overlapping windows of window_size.")
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--default_source_rate", type=float, default=100.0)
    parser.add_argument("--sensor_systems", default="INDIP,SU")
    parser.add_argument("--body_parts", default="LowerBack,LeftWrist,RightWrist")
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--verbose_missing", action="store_true")
    return parser.parse_args()


def main():
    preprocess(parse_args())


if __name__ == "__main__":
    main()
