
import argparse
import csv
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np


SENSOR_LABELS = [
    "Pelvis",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
]

GROUP_LABELS = {
    "Healthy adults": 0,
    "Healthy adolescents": 0,
    "ACLD": 1,
    "ACLR": 1,
}

GROUP_NAMES = {
    "Healthy adults": "HealthyAdult",
    "Healthy adolescents": "HealthyAdolescent",
    "ACLD": "ACLD",
    "ACLR": "ACLR",
}


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def find_gait_dir(subject_dir):
    for child in subject_dir.iterdir():
        if child.is_dir() and child.name.lower() == "gait":
            return child
    return None


def speed_dirs(gait_dir):
    if gait_dir is None:
        return []
    out = []
    for child in gait_dir.iterdir():
        if child.is_dir() and child.name.lower() in {"slow", "normal", "fast"}:
            out.append(child)
    return sorted(out, key=lambda p: p.name.lower())


def parse_frame_rate(path):
    for _, elem in ET.iterparse(path, events=("start",)):
        if local_name(elem.tag) == "subject":
            return float(elem.attrib.get("frameRate", 100.0))
    return 100.0


def read_sensor_free_acceleration(path):
    rows = []
    in_normal_frame = False
    for event, elem in ET.iterparse(path, events=("start", "end")):
        name = local_name(elem.tag)
        if event == "start" and name == "frame":
            in_normal_frame = elem.attrib.get("type") == "normal"
        elif event == "end" and name == "sensorFreeAcceleration" and in_normal_frame:
            values = np.fromstring(elem.text or "", sep=" ", dtype=np.float32)
            if values.size >= len(SENSOR_LABELS) * 3:
                rows.append(values[: len(SENSOR_LABELS) * 3].reshape(len(SENSOR_LABELS), 3))
        elif event == "end" and name == "frame":
            in_normal_frame = False
            elem.clear()
    if not rows:
        return np.empty((0, len(SENSOR_LABELS), 3), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def resample_array(array, source_rate, target_rate):
    if array.size == 0 or source_rate == target_rate:
        return array.astype(np.float32, copy=False)
    old_len = array.shape[0]
    if old_len < 2:
        return array.astype(np.float32, copy=False)
    duration = (old_len - 1) / float(source_rate)
    new_len = int(round(duration * float(target_rate))) + 1
    if new_len <= 1:
        return np.empty((0,) + array.shape[1:], dtype=np.float32)
    old_t = np.arange(old_len, dtype=np.float32) / float(source_rate)
    new_t = np.arange(new_len, dtype=np.float32) / float(target_rate)
    new_t = np.minimum(new_t, old_t[-1])
    flat = array.reshape(old_len, -1)
    out = np.empty((new_len, flat.shape[1]), dtype=np.float32)
    for ch in range(flat.shape[1]):
        out[:, ch] = np.interp(new_t, old_t, flat[:, ch])
    return out.reshape((new_len,) + array.shape[1:]).astype(np.float32)


def write_shard(output_dir, shard_id, x_parts, y_parts):
    shard_name = f"windows_50hz_5s_shard_{shard_id:05d}.npz"
    path = output_dir / shard_name
    X = np.stack(x_parts, axis=0).astype(np.float32)
    y = np.asarray(y_parts, dtype=np.int64)
    np.savez(path, X=X, y=y)
    print(f"Wrote {shard_name}: X={X.shape}")
    return shard_name


def iter_trials(root_path):
    root = Path(root_path)
    for group_folder, label in GROUP_LABELS.items():
        group_dir = root / group_folder
        if not group_dir.exists():
            print(f"Warning: group folder missing: {group_dir}")
            continue
        for subject_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            gait_dir = find_gait_dir(subject_dir)
            for speed_dir in speed_dirs(gait_dir):
                speed = speed_dir.name.lower()
                for path in sorted(speed_dir.glob("*.mvnx")):
                    yield {
                        "path": path,
                        "group": group_folder,
                        "group_name": GROUP_NAMES[group_folder],
                        "label": label,
                        "subject": subject_dir.name,
                        "speed": speed,
                        "trial": path.stem,
                    }


def preprocess(args):
    root_path = Path(args.root_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_size = int(round(args.target_rate * args.window_seconds))
    stride = int(args.stride) if args.stride else window_size
    manifest_path = output_dir / "manifest_50hz_5s.csv"

    rows = []
    x_parts = []
    y_parts = []
    shard_id = 0
    shard_index = 0
    counters = Counter()

    for record_idx, trial in enumerate(iter_trials(root_path), start=1):
        path = trial["path"]
        try:
            source_rate = parse_frame_rate(path)
            acc = read_sensor_free_acceleration(path)
        except Exception as exc:
            counters["failed_files"] += 1
            print(f"Warning: failed to parse {path}: {exc}")
            continue
        if acc.shape[0] < window_size * source_rate / args.target_rate:
            counters["short_trials"] += 1
        acc = resample_array(acc, source_rate, args.target_rate)
        if acc.shape[0] < window_size:
            counters["too_short_after_resample"] += 1
            continue

        for start in range(0, acc.shape[0] - window_size + 1, stride):
            end = start + window_size
            for sensor_idx, sensor in enumerate(SENSOR_LABELS):
                x = np.zeros((window_size, 6), dtype=np.float32)
                x[:, :3] = acc[start:end, sensor_idx, :]
                x_parts.append(x)
                y_parts.append(trial["label"])
                rows.append({
                    "shard": "",
                    "shard_index": shard_index,
                    "subject": trial["subject"],
                    "label": trial["label"],
                    "label_name": "ACLAbnormal" if trial["label"] else "Healthy",
                    "group": trial["group_name"],
                    "speed": trial["speed"],
                    "trial": trial["trial"],
                    "body_part": sensor,
                    "source_file": str(path.relative_to(root_path)),
                    "start": start,
                    "end": end,
                    "source_rate": source_rate,
                    "target_rate": args.target_rate,
                })
                shard_index += 1
                counters["windows"] += 1
                if len(x_parts) >= args.shard_size:
                    shard_name = write_shard(output_dir, shard_id, x_parts, y_parts)
                    for row in rows[-len(x_parts):]:
                        row["shard"] = shard_name
                    shard_id += 1
                    shard_index = 0
                    x_parts = []
                    y_parts = []
        counters["trials"] += 1
        if args.max_trials and record_idx >= args.max_trials:
            break

    if x_parts:
        shard_name = write_shard(output_dir, shard_id, x_parts, y_parts)
        for row in rows[-len(x_parts):]:
            row["shard"] = shard_name

    if not rows:
        raise RuntimeError("No COMPWALK-ACL windows were produced.")

    fieldnames = [
        "shard", "shard_index", "subject", "label", "label_name", "group",
        "speed", "trial", "body_part", "source_file", "start", "end",
        "source_rate", "target_rate",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote manifest: {manifest_path} rows={len(rows)}")
    print("Summary:")
    for key, value in sorted(counters.items()):
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess COMPWALK-ACL into 50 Hz, 5 s gait windows.")
    parser.add_argument("--root_path", type=str, default="./data/COMPWALK-ACL")
    parser.add_argument("--output_dir", type=str, default="./data/COMPWALK-ACL_50hz_5s")
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--max_trials", type=int, default=0)
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
