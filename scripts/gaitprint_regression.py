
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data_provider.data_loader import _split_three
from models.GaitFMFullFinetune import Model as MAEFinetuneModel
from utils.regression_tools import format_reg_metrics, json_default, regression_metrics, save_checkpoint
from mobilised_regression_data import load_sensor_checkpoint
from target_display_names import target_display_name


TARGETS = ("gait_speed", "step_length", "step_time")
CACHE_SCHEMA_VERSION = "gaitprint_step_targets_v2"
TARGET_UNITS = {"gait_speed": "m/s", "step_length": "m", "step_time": "s"}
PART_COLUMNS = {
    "Head": (
        ("Head Accel Sensor X (mG)", "Head Accel Sensor Y (mG)", "Head Accel Sensor Z (mG)"),
        (
            "Noraxon MyoMotion-Segments-Head-Gyroscope-x (deg/s)",
            "Noraxon MyoMotion-Segments-Head-Gyroscope-y (deg/s)",
            "Noraxon MyoMotion-Segments-Head-Gyroscope-z (deg/s)",
        ),
    ),
    "L_Hand": (
        ("Hand Accel Sensor X LT (mG)", "Hand Accel Sensor Y LT (mG)", "Hand Accel Sensor Z LT (mG)"),
        (
            "Noraxon MyoMotion-Segments-Hand LT-Gyroscope-x (deg/s)",
            "Noraxon MyoMotion-Segments-Hand LT-Gyroscope-y (deg/s)",
            "Noraxon MyoMotion-Segments-Hand LT-Gyroscope-z (deg/s)",
        ),
    ),
    "R_Hand": (
        ("Hand Accel Sensor X RT (mG)", "Hand Accel Sensor Y RT (mG)", "Hand Accel Sensor Z RT (mG)"),
        (
            "Noraxon MyoMotion-Segments-Hand RT-Gyroscope-x (deg/s)",
            "Noraxon MyoMotion-Segments-Hand RT-Gyroscope-y (deg/s)",
            "Noraxon MyoMotion-Segments-Hand RT-Gyroscope-z (deg/s)",
        ),
    ),
    "Pelvis": (
        ("Pelvis Accel Sensor X (mG)", "Pelvis Accel Sensor Y (mG)", "Pelvis Accel Sensor Z (mG)"),
        (
            "Noraxon MyoMotion-Segments-Pelvis-Gyroscope-x (deg/s)",
            "Noraxon MyoMotion-Segments-Pelvis-Gyroscope-y (deg/s)",
            "Noraxon MyoMotion-Segments-Pelvis-Gyroscope-z (deg/s)",
        ),
    ),
    "LowerSpine": (
        ("Lower spine Accel Sensor X (mG)", "Lower spine Accel Sensor Y (mG)", "Lower spine Accel Sensor Z (mG)"),
        (
            "Noraxon MyoMotion-Segments-Lower spine-Gyroscope-x (deg/s)",
            "Noraxon MyoMotion-Segments-Lower spine-Gyroscope-y (deg/s)",
            "Noraxon MyoMotion-Segments-Lower spine-Gyroscope-z (deg/s)",
        ),
    ),
    "L_Thigh": (
        ("Thigh Accel Sensor X LT (mG)", "Thigh Accel Sensor Y LT (mG)", "Thigh Accel Sensor Z LT (mG)"),
        (
            "Noraxon MyoMotion-Segments-Thigh LT-Gyroscope-x (deg/s)",
            "Noraxon MyoMotion-Segments-Thigh LT-Gyroscope-y (deg/s)",
            "Noraxon MyoMotion-Segments-Thigh LT-Gyroscope-z (deg/s)",
        ),
    ),
    "R_Thigh": (
        ("Thigh Accel Sensor X RT (mG)", "Thigh Accel Sensor Y RT (mG)", "Thigh Accel Sensor Z RT (mG)"),
        (
            "Noraxon MyoMotion-Segments-Thigh RT-Gyroscope-x (deg/s)",
            "Noraxon MyoMotion-Segments-Thigh RT-Gyroscope-y (deg/s)",
            "Noraxon MyoMotion-Segments-Thigh RT-Gyroscope-z (deg/s)",
        ),
    ),
    "L_Foot": (
        ("Foot Accel Sensor X LT (mG)", "Foot Accel Sensor Y LT (mG)", "Foot Accel Sensor Z LT (mG)"),
        (
            "Noraxon MyoMotion-Segments-Foot LT-Gyroscope-x (deg/s)",
            "Noraxon MyoMotion-Segments-Foot LT-Gyroscope-y (deg/s)",
            "Noraxon MyoMotion-Segments-Foot LT-Gyroscope-z (deg/s)",
        ),
    ),
    "R_Foot": (
        ("Foot Accel Sensor X RT (mG)", "Foot Accel Sensor Y RT (mG)", "Foot Accel Sensor Z RT (mG)"),
        (
            "Noraxon MyoMotion-Segments-Foot RT-Gyroscope-x (deg/s)",
            "Noraxon MyoMotion-Segments-Foot RT-Gyroscope-y (deg/s)",
            "Noraxon MyoMotion-Segments-Foot RT-Gyroscope-z (deg/s)",
        ),
    ),
}
TRIAL_RE = re.compile(r"(?P<id>S\d+)_G(?P<group>\d+)_D(?P<day>\d+)_B(?P<block>\d+)_T(?P<trial>\d+)\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(description="GaitPrint GaitFM gait-parameter regression.")
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--root_path", type=str, default="./data/GaitPrint")
    parser.add_argument("--output_dir", type=str, default="./results_gaitprint_reference_patch_resnet_sensorpretrain")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--mae_checkpoint_path", type=str, default="./checkpoints/backbones/mae_tiny_400e.pth.tar")
    parser.add_argument("--mae_model", type=str, default="mae_vit_tiny_patch16")
    parser.add_argument("--mae_tag", type=str, default="MAETiny_GaitFM_PatchResNet")
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--body_part", type=str, default="Pelvis", choices=sorted(PART_COLUMNS))
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--source_rate", type=float, default=200.0)
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


def _trial_key_from_path(path):
    match = TRIAL_RE.match(Path(path).name)
    if not match:
        return None
    return {
        "id": match.group("id"),
        "group": f"G{int(match.group('group')):02d}",
        "day": f"D{int(match.group('day')):02d}",
        "block": f"B{int(match.group('block')):02d}",
        "trial": f"T{int(match.group('trial')):02d}",
    }


def _truthy(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(int(value))


def read_trial_table(root_path, args):
    path = Path(root_path) / "subject_trial_characteristics" / "Gaitprint_trial_characteristics.csv"
    if not path.exists():
        raise FileNotFoundError(f"GaitPrint trial-characteristics file not found: {path}")
    table = pd.read_csv(path)
    table["id"] = table["id"].astype(str)
    for column in ("group", "day", "block", "trial"):
        table[column] = table[column].astype(str)
    mask = np.ones(len(table), dtype=bool)
    if int(args.require_full_trial) and "full_trial" in table:
        mask &= table["full_trial"].map(_truthy).to_numpy()
    if int(args.require_contacts) and "contacts" in table:
        mask &= table["contacts"].map(_truthy).to_numpy()
    if int(args.require_lower_body) and "lower_body" in table:
        mask &= table["lower_body"].map(_truthy).to_numpy()
    table = table[mask].copy()
    return {
        (row["id"], row["group"], row["day"], row["block"], row["trial"]): row.to_dict()
        for _, row in table.iterrows()
    }


def split_subjects(subjects, args):
    train, val, test = _split_three(sorted(subjects), args.split_seed, args.val_ratio, args.test_ratio)
    split = {subject: "TRAIN" for subject in train}
    split.update({subject: "VAL" for subject in val})
    split.update({subject: "TEST" for subject in test})
    return split


def numeric_frame(df, columns):
    return df[list(columns)].apply(pd.to_numeric, errors="coerce").interpolate(
        method="linear", limit_direction="both"
    ).fillna(0.0).to_numpy(dtype=np.float32)


def resample(values, timestamp, target_rate):
    timestamp = np.asarray(timestamp, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(timestamp)
    if valid.sum() < 2:
        return values
    timestamp, values = timestamp[valid], values[valid]
    order = np.argsort(timestamp)
    timestamp, values = timestamp[order], values[order]
    unique = np.concatenate([[True], np.diff(timestamp) > 0])
    timestamp, values = timestamp[unique], values[unique]
    duration = timestamp[-1] - timestamp[0]
    if duration <= 0:
        return values
    new_t = np.arange(0.0, duration + 1e-9, 1.0 / float(target_rate), dtype=np.float64) + timestamp[0]
    out = np.empty((len(new_t), values.shape[1]), dtype=np.float32)
    for col in range(values.shape[1]):
        out[:, col] = np.interp(new_t, timestamp, values[:, col])
    return out


def contact_starts(contact):
    contact = np.asarray(contact, dtype=np.float64)
    state = contact >= 500.0
    return np.where(state & np.concatenate([[True], ~state[:-1]]))[0]


def stride_events(time, contact, heel_xy):
    starts = contact_starts(contact)
    events = []
    for first, second in zip(starts[:-1], starts[1:]):
        dt = float(time[second] - time[first])
        if dt <= 0:
            continue
        distance = float(np.linalg.norm(heel_xy[second] - heel_xy[first]) / 1000.0)
        if not np.isfinite(distance) or distance <= 0:
            continue
        events.append(
            {
                "time": float((time[first] + time[second]) / 2.0),
                "length": distance,
                "speed": distance / dt,
            }
        )
    return starts, events


def step_events(time, left_starts, right_starts, left_heel, right_heel):
    contacts = [
        (int(index), "L", left_heel[int(index)]) for index in left_starts
    ] + [
        (int(index), "R", right_heel[int(index)]) for index in right_starts
    ]
    contacts.sort(key=lambda item: (item[0], item[1]))
    positions = np.asarray([item[2] for item in contacts], dtype=np.float64)
    if len(positions) < 2 or not np.all(np.isfinite(positions)):
        return []
    centered = positions - positions.mean(axis=0, keepdims=True)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if len(singular_values) == 0 or not np.isfinite(singular_values[0]) or singular_values[0] <= 0:
        return []
    progression_axis = axes[0]
    events = []
    for first, second in zip(contacts[:-1], contacts[1:]):
        first_index, first_side, first_position = first
        second_index, second_side, second_position = second
        if first_side == second_side:
            continue
        duration = float(time[second_index] - time[first_index])
        if not np.isfinite(duration) or duration <= 0:
            continue
        length = float(abs(np.dot(second_position - first_position, progression_axis)) / 1000.0)
        if not np.isfinite(length) or length <= 0:
            continue
        events.append(
            {
                "time": float((time[first_index] + time[second_index]) / 2.0),
                "length": length,
                "duration": duration,
            }
        )
    return events


def prepare_trial_events(df):
    time_col = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(time_col)
    if valid.sum() < 2:
        return None
    time_col = time_col[valid]
    time_col = time_col - time_col[0]
    left_contact = pd.to_numeric(df.loc[valid, "Contact LT"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    right_contact = pd.to_numeric(df.loc[valid, "Contact RT"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    left_heel = numeric_frame(df.loc[valid], [
        "Noraxon MyoMotion-Trajectories-Heel LT-x (mm)",
        "Noraxon MyoMotion-Trajectories-Heel LT-y (mm)",
    ])
    right_heel = numeric_frame(df.loc[valid], [
        "Noraxon MyoMotion-Trajectories-Heel RT-x (mm)",
        "Noraxon MyoMotion-Trajectories-Heel RT-y (mm)",
    ])
    left_starts, left_strides = stride_events(time_col, left_contact, left_heel)
    right_starts, right_strides = stride_events(time_col, right_contact, right_heel)
    stride_rows = left_strides + right_strides
    step_rows = step_events(time_col, left_starts, right_starts, left_heel, right_heel)
    return {
        "step_event_times": np.asarray([row["time"] for row in step_rows], dtype=np.float64),
        "step_lengths": np.asarray([row["length"] for row in step_rows], dtype=np.float64),
        "step_durations": np.asarray([row["duration"] for row in step_rows], dtype=np.float64),
        "stride_times": np.asarray([row["time"] for row in stride_rows], dtype=np.float64),
        "stride_speeds": np.asarray([row["speed"] for row in stride_rows], dtype=np.float64),
    }


def derive_window_targets(events, start_sec, end_sec, args):
    if events is None:
        return None
    step_mask = (events["step_event_times"] >= start_sec) & (events["step_event_times"] < end_sec)
    stride_mask = (events["stride_times"] >= start_sec) & (events["stride_times"] < end_sec)
    if int(np.sum(step_mask)) < int(args.min_steps_per_window) or int(np.sum(stride_mask)) < int(args.min_strides_per_window):
        return None
    return {
        "step_length": float(np.mean(events["step_lengths"][step_mask])),
        "step_time": float(np.mean(events["step_durations"][step_mask])),
        "gait_speed": float(np.mean(events["stride_speeds"][stride_mask])),
    }


def load_trial_window_data(path, args):
    df = pd.read_csv(path)
    acc_cols, gyro_cols = PART_COLUMNS[args.body_part]
    required = ["time", "Contact LT", "Contact RT"] + list(acc_cols) + list(gyro_cols)
    required += [
        "Noraxon MyoMotion-Trajectories-Heel LT-x (mm)",
        "Noraxon MyoMotion-Trajectories-Heel LT-y (mm)",
        "Noraxon MyoMotion-Trajectories-Heel RT-x (mm)",
        "Noraxon MyoMotion-Trajectories-Heel RT-y (mm)",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required GaitPrint columns: {missing}")
    timestamp = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=np.float64)
    features = numeric_frame(df, list(acc_cols) + list(gyro_cols))
    features[:, :3] = features[:, :3] / 1000.0
    return df, timestamp, resample(features, timestamp, args.target_rate)


def cache_scalar(samples, key):
    value = np.asarray(samples[key])
    return value.item() if value.size == 1 else value.tolist()


def validate_cache(samples, args, path):
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_dataset": "GaitPrint",
        "cache_target": args.target,
        "cache_target_unit": TARGET_UNITS[args.target],
        "cache_body_part": args.body_part,
        "cache_target_rate": float(args.target_rate),
        "cache_window_seconds": float(args.window_seconds),
        "cache_stride": int(args.stride),
        "cache_source_rate": float(args.source_rate),
        "cache_min_steps_per_window": int(args.min_steps_per_window),
        "cache_min_strides_per_window": int(args.min_strides_per_window),
        "cache_source_root": str(Path(args.root_path).expanduser().resolve()),
        "cache_split_seed": int(args.split_seed),
        "cache_val_ratio": float(args.val_ratio),
        "cache_test_ratio": float(args.test_ratio),
        "cache_max_subjects": int(args.max_subjects),
        "cache_max_windows": int(args.max_windows),
    }
    missing = [key for key in expected if key not in samples]
    if missing:
        raise ValueError(f"Incompatible GaitPrint cache {path}; missing {missing}. Rebuild with --rebuild_cache 1.")
    for key, wanted in expected.items():
        found = cache_scalar(samples, key)
        matches = np.isclose(float(found), float(wanted)) if isinstance(wanted, float) else found == wanted
        if not bool(matches):
            raise ValueError(
                f"Incompatible GaitPrint cache {path}; {key}={found!r}, expected {wanted!r}. "
                "Rebuild with --rebuild_cache 1."
            )


def build_samples(args):
    cache_path = Path(args.cache_npz).expanduser() if args.cache_npz else None
    if cache_path and cache_path.exists() and not int(args.rebuild_cache):
        with np.load(cache_path, allow_pickle=False) as data:
            samples = {key: data[key] for key in data.files}
        validate_cache(samples, args, cache_path)
        return samples

    root = Path(args.root_path)
    table = read_trial_table(root, args)
    csv_files = sorted(path for path in root.glob("S*/*.csv") if _trial_key_from_path(path) is not None)
    records = []
    for path in csv_files:
        key = _trial_key_from_path(path)
        table_key = (key["id"], key["group"], key["day"], key["block"], key["trial"])
        if table_key in table:
            records.append({"path": path, "subject": key["id"], **key})
    if int(args.max_subjects) > 0:
        keep = set(sorted({record["subject"] for record in records})[: int(args.max_subjects)])
        records = [record for record in records if record["subject"] in keep]
    if not records:
        raise ValueError(f"No valid GaitPrint records found under {root}")
    split_by_subject = split_subjects({record["subject"] for record in records}, args)
    window_size = int(round(args.target_rate * args.window_seconds))
    stride = int(args.stride)
    X, y, subjects, splits, files, starts = [], [], [], [], [], []
    for record in records:
        df, _, features = load_trial_window_data(record["path"], args)
        if len(features) < window_size:
            continue
        events = prepare_trial_events(df)
        for start in range(0, len(features) - window_size + 1, stride):
            start_sec = float(start / args.target_rate)
            end_sec = float((start + window_size) / args.target_rate)
            targets = derive_window_targets(events, start_sec, end_sec, args)
            if targets is None:
                continue
            value = targets[args.target]
            if not np.isfinite(value):
                continue
            X.append(features[start : start + window_size].astype(np.float32))
            y.append(float(value))
            subjects.append(record["subject"])
            splits.append(split_by_subject[record["subject"]])
            files.append(str(record["path"]))
            starts.append(int(start))
            if int(args.max_windows) > 0 and len(y) >= int(args.max_windows):
                break
        if int(args.max_windows) > 0 and len(y) >= int(args.max_windows):
            break
    if not y:
        raise ValueError(f"No GaitPrint windows with valid {args.target} labels.")
    samples = {
        "X": np.stack(X).astype(np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "subjects": np.asarray(subjects, dtype="U32"),
        "splits": np.asarray(splits, dtype="U8"),
        "files": np.asarray(files, dtype="U512"),
        "starts": np.asarray(starts, dtype=np.int64),
        "cache_schema_version": np.asarray(CACHE_SCHEMA_VERSION),
        "cache_dataset": np.asarray("GaitPrint"),
        "cache_target": np.asarray(args.target),
        "cache_target_unit": np.asarray(TARGET_UNITS[args.target]),
        "cache_body_part": np.asarray(args.body_part),
        "cache_target_rate": np.asarray(float(args.target_rate)),
        "cache_window_seconds": np.asarray(float(args.window_seconds)),
        "cache_stride": np.asarray(int(args.stride)),
        "cache_source_rate": np.asarray(float(args.source_rate)),
        "cache_min_steps_per_window": np.asarray(int(args.min_steps_per_window)),
        "cache_min_strides_per_window": np.asarray(int(args.min_strides_per_window)),
        "cache_source_root": np.asarray(str(Path(args.root_path).expanduser().resolve())),
        "cache_split_seed": np.asarray(int(args.split_seed)),
        "cache_val_ratio": np.asarray(float(args.val_ratio)),
        "cache_test_ratio": np.asarray(float(args.test_ratio)),
        "cache_max_subjects": np.asarray(int(args.max_subjects)),
        "cache_max_windows": np.asarray(int(args.max_windows)),
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **samples)
        print(f"Saved GaitPrint cache: {cache_path}")
    return samples


class WindowDataset(Dataset):
    def __init__(self, samples, split, mean=None, std=None):
        mask = samples["splits"].astype(str) == split
        self.X = samples["X"][mask]
        self.y = samples["y"][mask].astype(np.float32)
        self.subjects = samples["subjects"][mask].astype(str)
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
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(float(self.y[idx]), dtype=torch.float32), self.subjects[idx]


def collate(batch):
    return (
        torch.stack([item[0] for item in batch], dim=0),
        torch.stack([item[1] for item in batch], dim=0),
        [item[2] for item in batch],
    )


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate)


def aggregate_subject_regression(y_true, y_pred, subjects):
    subjects = np.asarray(subjects).astype(str)
    out_true, out_pred = [], []
    for subject in sorted(set(subjects.tolist())):
        idx = subjects == subject
        out_true.append(float(np.mean(y_true[idx])))
        out_pred.append(float(np.mean(y_pred[idx])))
    return np.asarray(out_true), np.asarray(out_pred)


class GaitPrintRegressorBase(nn.Module):
    def __init__(self, args):
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

    def encode_patch_tokens(self, x):
        x = self.encoder._fit_channels(x)
        x = self.encoder._apply_channel_mask(x)
        images = self.encoder.image_encoder(x)
        tokens = self.encoder.mae.patch_embed(images)
        tokens = tokens + self.encoder.mae.pos_embed[:, 1:, :]
        cls_token = self.encoder.mae.cls_token + self.encoder.mae.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        for block in self.encoder.mae.blocks:
            tokens = block(tokens)
        tokens = self.encoder.mae.norm(tokens)
        return tokens[:, 1:, :]

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


def main(model_class):
    args = parse_args()
    os.environ.setdefault("GAITFM_CHANNEL_MASK_MODE", "random_acc_only")
    os.environ.setdefault("GAITFM_CHANNEL_MASK_PROB", "0")
    os.environ.setdefault("GAITFM_CHANNEL_MASK_APPLY_EVAL", "0")
    if args.model_id is None:
        args.model_id = f"GaitPrint_{args.target}_{args.body_part}_{args.mae_tag}_PatchResNet"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    samples = build_samples(args)
    train_ds = WindowDataset(samples, "TRAIN")
    val_ds = WindowDataset(samples, "VAL", train_ds.mean, train_ds.std)
    test_ds = WindowDataset(samples, "TEST", train_ds.mean, train_ds.std)
    print(
        f"GaitPrint {target_display_name(args.target)} body_part={args.body_part} splits | "
        f"TRAIN windows={len(train_ds)} subjects={len(set(train_ds.subjects.tolist()))} "
        f"VAL windows={len(val_ds)} subjects={len(set(val_ds.subjects.tolist()))} "
        f"TEST windows={len(test_ds)} subjects={len(set(test_ds.subjects.tolist()))}"
    )
    print(
        f"Target stats | TRAIN mean={float(np.mean(train_ds.y)):.6f} std={float(np.std(train_ds.y)):.6f} "
        f"VAL mean={float(np.mean(val_ds.y)):.6f} TEST mean={float(np.mean(test_ds.y)):.6f}"
    )
    target_mean = float(np.mean(train_ds.y)) if int(args.standardize_target) else 0.0
    target_std = float(np.std(train_ds.y)) if int(args.standardize_target) else 1.0
    if not np.isfinite(target_std) or target_std <= 0:
        target_std = 1.0
    print(f"Target standardization: mean={target_mean:.6f}, std={target_std:.6f} (TRAIN only)")

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)
    model = model_class(args).to(device)
    head_param_ids = {id(param) for param in model.head.parameters()}
    encoder_params = [param for param in model.parameters() if id(param) not in head_param_ids and param.requires_grad]
    head_params = [param for param in model.head.parameters() if param.requires_grad]
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

    results = {}
    summary_path = out_dir / "summary.log"
    results_json_path = out_dir / "results.json"
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
            results[name] = {"epoch": record["epoch"], "val_loss": record["val_loss"], "result": result}
    with open(results_json_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=json_default)
    print(f"Saved summary: {summary_path}")
    print(f"Saved structured results: {results_json_path}")
