
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List

import pandas as pd


STRICT_TASK = "SelfPace"
VARIANT_TASKS = {"SelfPace_mat", "SelfPace_matTURN", "SelfPace_doorpat"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract WearGait-PD SelfPace raw CSV files.")
    parser.add_argument(
        "--raw_root",
        required=True,
        help="Original WearGait-PD folder.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output folder for the SelfPace subset.",
    )
    parser.add_argument(
        "--include_variants",
        action="store_true",
        help="Also extract SelfPace_mat, SelfPace_matTURN, and SelfPace_doorpat.",
    )
    parser.add_argument(
        "--copy_subject_mat",
        action="store_true",
        help="Also copy each retained subject-level .mat file if present.",
    )
    parser.add_argument(
        "--keep_existing",
        action="store_true",
        help="Keep existing output folder contents instead of recreating output_root.",
    )
    return parser.parse_args()


def task_from_filename(path: Path) -> str | None:
    name = path.stem
    if "_" not in name:
        return None
    return name.split("_", 1)[1]


def subject_from_filename(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def center_from_subject(subject_id: str) -> str:
    if subject_id.startswith("NLS"):
        return "JHOC"
    if subject_id.startswith("HC"):
        return "JHM_Bayview"
    if subject_id.startswith(("WPD", "WHC")):
        return "VA_Seattle"
    return "unknown"


def prefix_from_subject(subject_id: str) -> str:
    match = re.match(r"^[A-Za-z]+", subject_id)
    return match.group(0) if match else "unknown"


def select_files(raw_root: Path, include_variants: bool) -> List[Path]:
    allowed = {STRICT_TASK}
    if include_variants:
        allowed |= VARIANT_TASKS
    selected: List[Path] = []
    for path in sorted(raw_root.glob("*_SelfPace*.csv")):
        task = task_from_filename(path)
        if task in allowed:
            selected.append(path)
    return selected


def recreate_output_root(output_root: Path, keep_existing: bool) -> None:
    if output_root.exists() and not keep_existing:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def copy_manifest_subset(raw_root: Path, output_root: Path, copied_names: set[str]) -> int:
    rows_written = 0
    for manifest_path in sorted(raw_root.glob("manifest*.csv")):
        try:
            manifest = pd.read_csv(manifest_path)
        except Exception:
            continue
        if "name" not in manifest.columns:
            continue
        subset = manifest[manifest["name"].astype(str).isin(copied_names)].copy()
        if subset.empty:
            continue
        subset.to_csv(output_root / manifest_path.name, index=False)
        rows_written += int(len(subset))
    return rows_written


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"WearGait-PD raw_root not found: {raw_root}")
    if not raw_root.is_dir():
        raise NotADirectoryError(f"WearGait-PD raw_root is not a directory: {raw_root}")

    selected = select_files(raw_root, args.include_variants)
    if not selected:
        raise ValueError(
            f"No WearGait-PD SelfPace files found in {raw_root}. "
            f"include_variants={args.include_variants}"
        )

    recreate_output_root(output_root, args.keep_existing)

    subject_rows: Dict[str, Dict[str, object]] = {}
    file_rows: List[Dict[str, object]] = []
    copied_names: set[str] = set()

    for src in selected:
        subject_id = subject_from_filename(src)
        task = task_from_filename(src) or ""
        dst = output_root / src.name
        shutil.copy2(src, dst)
        copied_names.add(src.name)

        subject_rows.setdefault(
            subject_id,
            {
                "subject_id": subject_id,
                "prefix": prefix_from_subject(subject_id),
                "center": center_from_subject(subject_id),
                "num_selfpace_files": 0,
                "copied_subject_mat": False,
            },
        )
        subject_rows[subject_id]["num_selfpace_files"] = int(subject_rows[subject_id]["num_selfpace_files"]) + 1
        file_rows.append(
            {
                "subject_id": subject_id,
                "task": task,
                "filename": src.name,
                "source_path": str(src),
                "copied_path": str(dst),
                "bytes": int(src.stat().st_size),
            }
        )

    if args.copy_subject_mat:
        for subject_id, row in subject_rows.items():
            mat_path = raw_root / f"{subject_id}.mat"
            if not mat_path.exists():
                continue
            shutil.copy2(mat_path, output_root / mat_path.name)
            copied_names.add(mat_path.name)
            row["copied_subject_mat"] = True

    manifest_rows = copy_manifest_subset(raw_root, output_root, copied_names)
    subjects_df = pd.DataFrame(sorted(subject_rows.values(), key=lambda item: str(item["subject_id"])))
    files_df = pd.DataFrame(file_rows)
    subjects_df.to_csv(output_root / "subjects.csv", index=False)
    files_df.to_csv(output_root / "files.csv", index=False)

    task_counts = files_df["task"].value_counts().to_dict()
    prefix_counts = subjects_df["prefix"].value_counts().to_dict()
    summary = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "strict_task": STRICT_TASK,
        "include_variants": bool(args.include_variants),
        "variant_tasks": sorted(VARIANT_TASKS) if args.include_variants else [],
        "raw_subset_only": True,
        "no_windowing_or_resampling": True,
        "no_normalization": True,
        "no_split": True,
        "copied_csv_files": int(len(selected)),
        "subjects": int(len(subject_rows)),
        "task_counts": {str(key): int(value) for key, value in task_counts.items()},
        "prefix_counts": {str(key): int(value) for key, value in prefix_counts.items()},
        "manifest_rows_written": int(manifest_rows),
        "notes": [
            "Default mode extracts only files named <subject>_SelfPace.csv.",
            "Use --include_variants to include SelfPace_mat, SelfPace_matTURN, and SelfPace_doorpat.",
            "CSV files are copied unchanged; downstream loaders should handle split, normalization, resampling, and windowing.",
        ],
    }
    with (output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
