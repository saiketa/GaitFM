import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

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

from utils.regression_tools import (
    format_reg_metrics,
    regression_metrics,
)


COHORTS = ("HA", "PD", "MS", "PFF", "COPD", "CHF")
DEFAULT_TESTS = ("Test5", "Test6", "Test7")
FEATURE_COLUMNS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
TARGET_ALIASES = {
    "gait_speed": (
        ("walkingspeed",),
        ("gaitspeed",),
        ("speed",),
        ("velocity",),
    ),
    "step_length": (
        ("steplength",),
    ),
    "step_time": (
        ("steptime",),
        ("stepduration",),
    ),
}
CACHE_SCHEMA_VERSION = "mobilised_step_targets_v2"
TARGET_UNITS = {"gait_speed": "m/s", "step_length": "m", "step_time": "s"}


def import_loadmat():
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise RuntimeError("scipy is required to read Mobilise-D .mat files.") from exc
    return loadmat


def normalize_key(value):
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def is_mapping(value):
    return isinstance(value, dict)


def as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        return list(value.ravel())
    return [value]


def squeeze_numeric(value):
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == object:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    arr = np.squeeze(arr)
    if arr.size == 0:
        return None
    return arr.astype(np.float64)


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
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from iter_arrays(value, f"{prefix}/{idx}" if prefix else str(idx))
    else:
        arr = squeeze_numeric(obj)
        if arr is not None:
            yield prefix, arr


def looks_like_axis_matrix(arr):
    arr = squeeze_numeric(arr)
    return arr is not None and arr.ndim == 2 and 3 in arr.shape and max(arr.shape) >= 2


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
    return np.asarray(arr[:expected_len], dtype=np.float64), name


def load_mat(path):
    loadmat = import_loadmat()
    data = loadmat(path, simplify_cells=True)
    return {key: value for key, value in data.items() if not key.startswith("__")}


def extract_sensor_stream(mat_data, key_tokens, sensor_system, sensor_name, default_rate):
    roots = [mat_data.get("data"), mat_data] if is_mapping(mat_data) else [mat_data]
    block = None
    for root in roots:
        if root is None:
            continue
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
        return None, f"acc/gyro not found; acc={acc_name} gyro={gyro_name}"

    n = min(len(acc), len(gyro))
    acc = acc[:n]
    gyro = gyro[:n]
    time_vec, time_name = find_time_vector(block, n)
    if time_vec is not None:
        finite = np.isfinite(time_vec)
        if finite.sum() >= 2:
            time_sec = np.asarray(time_vec[finite], dtype=np.float64)
            values = np.concatenate([acc[finite], gyro[finite]], axis=1).astype(np.float32)
            order = np.argsort(time_sec)
            time_sec = time_sec[order]
            values = values[order]
            _, unique_idx = np.unique(time_sec, return_index=True)
            unique_idx = np.sort(unique_idx)
            time_sec = time_sec[unique_idx] - time_sec[unique_idx][0]
            values = values[unique_idx]
            return (time_sec, values), f"acc={acc_name};gyro={gyro_name};time={time_name}"
    time_sec = np.arange(n, dtype=np.float64) / float(default_rate)
    return (time_sec, np.concatenate([acc, gyro], axis=1).astype(np.float32)), f"acc={acc_name};gyro={gyro_name};time=synthetic"


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
    values = np.stack([np.interp(new_t, time_sec, data[:, ch]) for ch in range(data.shape[1])], axis=1)
    return new_t, values.astype(np.float32)


def iter_nodes(obj):
    yield obj
    if is_mapping(obj):
        for value in obj.values():
            yield from iter_nodes(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from iter_nodes(value)


def find_standard_block(mat_data, key_tokens):
    roots = [mat_data.get("data"), mat_data] if is_mapping(mat_data) else [mat_data]
    for root in roots:
        if root is None:
            continue
        block = find_path(root, list(key_tokens) + ["Standards"])
        if block is not None:
            return block
    return None


def numeric_by_name(node, aliases):
    if not is_mapping(node):
        return None, ""
    for key, value in node.items():
        lname = normalize_key(key)
        if any(all(term in lname for term in alias) for alias in aliases):
            arr = squeeze_numeric(value)
            if arr is not None:
                return arr, str(key)
    for name, arr in iter_arrays(node):
        lname = normalize_key(name)
        if any(all(term in lname for term in alias) for alias in aliases):
            return arr, name
    return None, ""


def infer_time_array(node):
    excluded = (
        "steptime",
        "stepduration",
        "stridetime",
        "strideduration",
        "gaitcycletime",
        "stancetime",
        "swingtime",
        "double supporttime",
    )
    candidates = []
    for name, raw in iter_arrays(node):
        full_name = normalize_key(name)
        if any(normalize_key(token) in full_name for token in excluded):
            continue
        leaf = normalize_key(str(name).rsplit("/", 1)[-1])
        if "timestamp" in leaf or "sampletime" in leaf:
            score = 0
        elif leaf in {"time", "times", "t"}:
            score = 1
        elif "initialcontactevent" in leaf or "footoffevent" in leaf:
            score = 2
        elif leaf in {"initialcontact", "footoff"}:
            score = 3
        elif "time" in leaf and ("start" in leaf or "end" in leaf):
            score = 4
        else:
            continue
        arr = np.asarray(raw, dtype=np.float64).ravel()
        if np.isfinite(arr).any():
            candidates.append((score, str(name), arr))
    if not candidates:
        return None, ""
    _, name, arr = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
    return arr, name


def event_times_from_node(node):
    candidates = []
    for name, arr in iter_arrays(node):
        lname = normalize_key(name)
        if any(term in lname for term in ("initialcontact", "footoff", "event")):
            arr = np.asarray(arr, dtype=np.float64).ravel()
            arr = arr[np.isfinite(arr)]
            if len(arr):
                candidates.append((len(arr), name, arr))
    if not candidates:
        return None, ""
    _, name, arr = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    return arr, name


def target_array_from_node(node, target):
    target_arr, target_name = numeric_by_name(node, TARGET_ALIASES[target])
    if target_arr is not None or target != "step_time":
        return target_arr, target_name
    frequency, frequency_name = numeric_by_name(node, (("cadence",), ("stepfrequency",)))
    if frequency is None:
        return None, ""
    frequency = np.asarray(frequency, dtype=np.float64)
    converted = np.full(frequency.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(frequency) & (frequency > 0)
    normalized_name = normalize_key(frequency_name)
    numerator = 1.0 if "hz" in normalized_name or "persecond" in normalized_name else 60.0
    converted[valid] = numerator / frequency[valid]
    return converted, f"derived_from_{frequency_name}"


def scalar_interval_from_node(node, target):
    if not is_mapping(node):
        return None
    target_arr, target_name = target_array_from_node(node, target)
    if target_arr is None:
        return None
    values = np.asarray(target_arr, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None

    event_times, event_name = event_times_from_node(node)
    time_arr, time_name = infer_time_array(node)
    time_base = event_times if event_times is not None else time_arr
    if time_base is None or len(time_base) == 0:
        return None
    start = float(np.nanmin(time_base))
    end = float(np.nanmax(time_base))
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None
    value = float(np.nanmean(values))
    return start, end, value, f"{target_name};time={event_name or time_name};mode=node_mean"


def point_intervals_from_node(node, target):
    if not is_mapping(node):
        return []
    target_arr, target_name = target_array_from_node(node, target)
    if target_arr is None:
        return []
    values = np.asarray(target_arr, dtype=np.float64).ravel()
    time_arr, time_name = infer_time_array(node)
    if time_arr is None:
        return []
    times = np.asarray(time_arr, dtype=np.float64).ravel()
    n = min(len(values), len(times))
    if n < 2:
        return []
    values = values[:n]
    times = times[:n]
    finite = np.isfinite(values) & np.isfinite(times)
    values = values[finite]
    times = times[finite]
    if len(values) < 2:
        return []
    order = np.argsort(times)
    values = values[order]
    times = times[order]
    intervals = []
    for i, value in enumerate(values):
        if i < len(times) - 1:
            start, end = float(times[i]), float(times[i + 1])
        else:
            delta = float(np.nanmedian(np.diff(times))) if len(times) > 1 else 0.0
            start, end = float(times[i]), float(times[i] + max(delta, 1e-3))
        if end > start:
            intervals.append((start, end, float(value), f"{target_name};time={time_name};mode=point"))
    return intervals


def reference_intervals(mat_data, key_tokens, target):
    standards = find_standard_block(mat_data, key_tokens)
    if standards is None:
        return []
    intervals = []
    seen = set()
    for node in iter_nodes(standards):
        for item in point_intervals_from_node(node, target):
            key = (round(item[0], 6), round(item[1], 6), round(item[2], 6), item[3])
            if key not in seen:
                seen.add(key)
                intervals.append(item)
        item = scalar_interval_from_node(node, target)
        if item is not None:
            key = (round(item[0], 6), round(item[1], 6), round(item[2], 6), item[3])
            if key not in seen:
                seen.add(key)
                intervals.append(item)
    intervals = [item for item in intervals if np.isfinite(item[2])]
    intervals.sort(key=lambda item: (item[0], item[1]))
    return intervals


def window_label(start_sec, end_sec, intervals, min_coverage, aggregation):
    weighted_sum = 0.0
    total_overlap = 0.0
    values = []
    for ref_start, ref_end, value, _ in intervals:
        overlap = max(0.0, min(end_sec, ref_end) - max(start_sec, ref_start))
        if overlap <= 0:
            continue
        weighted_sum += overlap * float(value)
        total_overlap += overlap
        values.append(float(value))
    if total_overlap < min_coverage * (end_sec - start_sec):
        return None
    if aggregation == "mean":
        return float(np.mean(values))
    return float(weighted_sum / max(total_overlap, 1e-12))


def read_test_list(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle).get("data", [])


def discover_records(args):
    root = Path(args.root_path)
    tests = {str(item) for item in args.tests}
    sensor_systems = {str(item) for item in args.sensor_systems}
    body_parts = {str(item) for item in args.body_parts}
    records = []
    for cohort_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name in COHORTS):
        cohort = cohort_dir.name
        for subject_dir in sorted(path for path in cohort_dir.iterdir() if path.is_dir()):
            lab_dir = subject_dir / "Laboratory"
            test_list = lab_dir / "test_list.json"
            data_path = lab_dir / "data.mat"
            if not test_list.exists() or not data_path.exists():
                continue
            for entry in read_test_list(test_list):
                key = [str(item) for item in entry.get("key", [])]
                if len(key) < 2 or key[1] not in tests:
                    continue
                available = entry.get("value", {}).get("available_sensors", [])
                for sensor_system, body_part in available:
                    if sensor_system not in sensor_systems or body_part not in body_parts:
                        continue
                    records.append(
                        {
                            "cohort": cohort,
                            "subject": f"{cohort}_{subject_dir.name}",
                            "subject_raw": subject_dir.name,
                            "key": key,
                            "test": key[1],
                            "trial": key[2] if len(key) > 2 else "",
                            "sensor_system": sensor_system,
                            "body_part": body_part,
                            "data_path": str(data_path),
                        }
                    )
    return records


def split_subjects(records, seed, val_ratio, test_ratio):
    subjects_by_cohort = defaultdict(list)
    for record in records:
        subjects_by_cohort[record["cohort"]].append(record["subject"])
    rng = np.random.default_rng(seed)
    split = {}
    for cohort, subjects in sorted(subjects_by_cohort.items()):
        unique = np.asarray(sorted(set(subjects)), dtype=object)
        rng.shuffle(unique)
        n = len(unique)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        if n >= 3:
            n_test = max(1, min(n_test, n - 2))
            n_val = max(1, min(n_val, n - n_test - 1))
        test_subjects = set(unique[:n_test])
        val_subjects = set(unique[n_test:n_test + n_val])
        for subject in unique:
            if subject in test_subjects:
                split[subject] = "TEST"
            elif subject in val_subjects:
                split[subject] = "VAL"
            else:
                split[subject] = "TRAIN"
    return split


def cache_scalar(samples, key):
    value = np.asarray(samples[key])
    return value.item() if value.size == 1 else value.tolist()


def validate_cache(samples, args, path):
    window_size = int(round(args.target_rate * args.window_seconds))
    resolved_stride = int(args.stride) if int(args.stride) > 0 else window_size
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_dataset": "Mobilise-D",
        "cache_target": args.target,
        "cache_target_unit": TARGET_UNITS[args.target],
        "cache_target_rate": float(args.target_rate),
        "cache_window_seconds": float(args.window_seconds),
        "cache_stride": resolved_stride,
        "cache_default_source_rate": float(args.default_source_rate),
        "cache_min_ref_coverage": float(args.min_ref_coverage),
        "cache_label_aggregation": args.label_aggregation,
        "cache_tests": "|".join(sorted(str(item) for item in args.tests)),
        "cache_sensor_systems": "|".join(sorted(str(item) for item in args.sensor_systems)),
        "cache_body_parts": "|".join(sorted(str(item) for item in args.body_parts)),
        "cache_source_root": str(Path(args.root_path).expanduser().resolve()),
        "cache_split_seed": int(args.split_seed),
        "cache_val_ratio": float(args.val_ratio),
        "cache_test_ratio": float(args.test_ratio),
        "cache_max_subjects": int(args.max_subjects),
        "cache_max_windows": int(args.max_windows),
    }
    missing = [key for key in expected if key not in samples]
    if missing:
        raise ValueError(f"Incompatible Mobilise-D cache {path}; missing {missing}. Rebuild with --rebuild_cache 1.")
    for key, wanted in expected.items():
        found = cache_scalar(samples, key)
        matches = np.isclose(float(found), float(wanted)) if isinstance(wanted, float) else found == wanted
        if not bool(matches):
            raise ValueError(
                f"Incompatible Mobilise-D cache {path}; {key}={found!r}, expected {wanted!r}. "
                "Rebuild with --rebuild_cache 1."
            )


def build_samples(args):
    cache_path = Path(args.cache_npz).expanduser() if args.cache_npz else None
    if cache_path and cache_path.exists() and not int(args.rebuild_cache):
        with np.load(cache_path, allow_pickle=True) as loaded:
            cached = {key: loaded[key] for key in loaded.files}
        validate_cache(cached, args, cache_path)
        return {
            "X": cached["X"],
            "y": cached["y"],
            "subjects": cached["subjects"].astype(str),
            "splits": cached["splits"].astype(str),
            "meta": cached["meta"].astype(object).tolist(),
        }

    records = discover_records(args)
    if args.max_subjects > 0:
        keep_subjects = set(sorted({r["subject"] for r in records})[: args.max_subjects])
        records = [r for r in records if r["subject"] in keep_subjects]
    split_by_subject = split_subjects(records, args.split_seed, args.val_ratio, args.test_ratio)
    window_size = int(round(args.target_rate * args.window_seconds))
    stride = args.stride if args.stride > 0 else window_size
    mat_cache = {}
    xs, ys, subjects, splits, meta = [], [], [], [], []
    stats = Counter()

    for idx, record in enumerate(records, start=1):
        data_path = record["data_path"]
        if data_path not in mat_cache:
            mat_cache[data_path] = load_mat(data_path)
        mat_data = mat_cache[data_path]
        parsed, extract_info = extract_sensor_stream(
            mat_data,
            record["key"],
            record["sensor_system"],
            record["body_part"],
            args.default_source_rate,
        )
        if parsed is None:
            stats["missing_sensor"] += 1
            continue
        intervals = reference_intervals(mat_data, record["key"], args.target)
        if not intervals:
            stats["missing_reference"] += 1
            continue
        resampled = resample_to_rate(parsed[0], parsed[1], args.target_rate)
        if resampled is None or len(resampled[1]) < window_size:
            stats["too_short"] += 1
            continue
        _, values = resampled
        produced = 0
        for start in range(0, len(values) - window_size + 1, stride):
            end = start + window_size
            start_sec = start / float(args.target_rate)
            end_sec = end / float(args.target_rate)
            label = window_label(start_sec, end_sec, intervals, args.min_ref_coverage, args.label_aggregation)
            if label is None or not np.isfinite(label):
                stats["unlabeled_windows"] += 1
                continue
            xs.append(values[start:end].astype(np.float32))
            ys.append(float(label))
            subjects.append(record["subject"])
            splits.append(split_by_subject[record["subject"]])
            meta.append(
                {
                    "subject": record["subject"],
                    "cohort": record["cohort"],
                    "test": record["test"],
                    "trial": record["trial"],
                    "sensor_system": record["sensor_system"],
                    "body_part": record["body_part"],
                    "window_start_sample": start,
                    "window_end_sample": end,
                    "extract_info": extract_info,
                }
            )
            produced += 1
            if args.max_windows > 0 and len(xs) >= args.max_windows:
                break
        stats["records_with_windows"] += int(produced > 0)
        stats["windows"] += produced
        if args.max_windows > 0 and len(xs) >= args.max_windows:
            break
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(records)} records | labeled_windows={len(xs)}")
    if not xs:
        raise RuntimeError(
            "No labeled Mobilise-D windows found. Use --body_parts/--sensor_systems or inspect Standards field names."
        )
    out = {
        "X": np.stack(xs, axis=0).astype(np.float32),
        "y": np.asarray(ys, dtype=np.float32),
        "subjects": np.asarray(subjects, dtype=object),
        "splits": np.asarray(splits, dtype=object),
        "meta": meta,
        "cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION),
        "cache_dataset": np.asarray("Mobilise-D"),
        "cache_target": np.asarray(args.target),
        "cache_target_unit": np.asarray(TARGET_UNITS[args.target]),
        "cache_target_rate": np.asarray(float(args.target_rate)),
        "cache_window_seconds": np.asarray(float(args.window_seconds)),
        "cache_stride": np.asarray(int(stride)),
        "cache_default_source_rate": np.asarray(float(args.default_source_rate)),
        "cache_min_ref_coverage": np.asarray(float(args.min_ref_coverage)),
        "cache_label_aggregation": np.asarray(args.label_aggregation),
        "cache_tests": np.asarray("|".join(sorted(str(item) for item in args.tests))),
        "cache_sensor_systems": np.asarray("|".join(sorted(str(item) for item in args.sensor_systems))),
        "cache_body_parts": np.asarray("|".join(sorted(str(item) for item in args.body_parts))),
        "cache_source_root": np.asarray(str(Path(args.root_path).expanduser().resolve())),
        "cache_split_seed": np.asarray(int(args.split_seed)),
        "cache_val_ratio": np.asarray(float(args.val_ratio)),
        "cache_test_ratio": np.asarray(float(args.test_ratio)),
        "cache_max_subjects": np.asarray(int(args.max_subjects)),
        "cache_max_windows": np.asarray(int(args.max_windows)),
    }
    print(f"Mobilise-D {args.target} reference extraction stats: {dict(stats)}")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **out)
        print(f"Saved cache: {cache_path}")
    return out


class WindowDataset(Dataset):
    def __init__(self, samples, split, mean=None, std=None):
        mask = samples["splits"] == split
        self.X = samples["X"][mask]
        self.y = samples["y"][mask]
        self.subjects = samples["subjects"][mask].astype(str)
        if len(self.X) == 0:
            raise ValueError(f"No Mobilise-D windows for split={split}")
        if mean is None or std is None:
            matrix = self.X.reshape(-1, self.X.shape[-1])
            mean = matrix.mean(axis=0).astype(np.float32)
            std = (matrix.std(axis=0) + np.finfo(np.float32).eps).astype(np.float32)
        self.mean = mean
        self.std = std
        self.X = ((self.X - self.mean) / self.std).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(float(self.y[idx]), dtype=torch.float32), self.subjects[idx]


def collate(batch):
    x = torch.stack([item[0] for item in batch], dim=0)
    y = torch.stack([item[1] for item in batch], dim=0)
    subjects = [item[2] for item in batch]
    return x, y, subjects


def aggregate_subject_regression(y_true, y_pred, subjects):
    out_true, out_pred = [], []
    subjects = np.asarray(subjects).astype(str)
    for subject in sorted(set(subjects.tolist())):
        idx = subjects == subject
        out_true.append(float(np.mean(y_true[idx])))
        out_pred.append(float(np.mean(y_pred[idx])))
    return np.asarray(out_true), np.asarray(out_pred)


def load_sensor_checkpoint(model, checkpoint):
    path = Path(checkpoint).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Sensor pretrain checkpoint not found: {path}")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    compatible = {k: v for k, v in state.items() if k.startswith("image_encoder.") or k.startswith("mae.")}
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded sensor-pretrained encoder from {path} | "
        f"tensors={len(compatible)} missing={len(missing)} unexpected={len(unexpected)}"
    )


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate)


def train_one_epoch(model, loader, criterion, optimizer, device, target_mean, target_std):
    model.train()
    losses = []
    for x, y, _ in loader:
        optimizer.zero_grad()
        pred = model(x.to(device))
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
    preds, targets, subjects = [], [], []
    for x, y, batch_subjects in loader:
        pred = model(x.to(device)).cpu()
        if target_std > 0:
            pred = pred * float(target_std) + float(target_mean)
        preds.append(pred)
        targets.append(y.float())
        subjects.extend(batch_subjects)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy(), np.asarray(subjects).astype(str)


@torch.no_grad()
def validation_loss(model, loader, criterion, device, target_mean, target_std):
    model.eval()
    losses = []
    for x, y, _ in loader:
        pred = model(x.to(device))
        yy = y.to(device)
        if target_std > 0:
            yy = (yy - float(target_mean)) / float(target_std)
        losses.append(float(criterion(pred, yy).item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, val_loader, test_loader, device, target_mean, target_std):
    val_y, val_pred, val_subjects = predict(model, val_loader, device, target_mean, target_std)
    test_y, test_pred, test_subjects = predict(model, test_loader, device, target_mean, target_std)
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


def format_eval(prefix, epoch, val_loss, result):
    return (
        f"{prefix}: Epoch {epoch} | val_loss={val_loss:.6f} "
        f"Sample-Val: {format_reg_metrics(result['val'])} "
        f"Sample-Test: {format_reg_metrics(result['test'])} "
        f"| Subject-Val(n={result['n_val_subjects']}): {format_reg_metrics(result['val_subject'])} "
        f"Subject-Test(n={result['n_test_subjects']}): {format_reg_metrics(result['test_subject'])}"
    )

