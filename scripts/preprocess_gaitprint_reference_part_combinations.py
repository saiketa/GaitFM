
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
for path in (PROJECT_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gaitprint_part_combination_regression as combo


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_path", default="./data/GaitPrint")
    parser.add_argument(
        "--output_dir",
        default="./data/GaitPrint_reference_part_combinations_cache",
    )
    parser.add_argument("--parts", nargs="+", default=list(combo.DAILY_PARTS))
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--min_steps_per_window", type=int, default=3)
    parser.add_argument("--min_strides_per_window", type=int, default=1)
    parser.add_argument("--require_full_trial", type=int, default=1)
    parser.add_argument("--require_contacts", type=int, default=1)
    parser.add_argument("--require_lower_body", type=int, default=1)
    parser.add_argument("--max_subjects", type=int, default=0)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    part_tag = "_".join(cli.parts)
    rate_tag = int(cli.target_rate) if cli.target_rate.is_integer() else cli.target_rate
    window_tag = int(cli.window_seconds) if cli.window_seconds.is_integer() else cli.window_seconds
    cache_path = (
        Path(cli.output_dir)
        / (
            f"gaitprint_all_targets_parts-{part_tag}_sr{rate_tag}_"
            f"ws{window_tag}_stride{cli.stride}.npz"
        )
    )
    args = SimpleNamespace(
        **vars(cli),
        target="gait_speed",
        cache_npz=str(cache_path),
        rebuild_cache=int(cli.rebuild),
        attach_split=False,
    )
    samples = combo.build_moments(args)
    print(
        f"GaitPrint preprocessing complete: windows={len(samples['X'])}, "
        f"shape={samples['X'].shape}, cache={cache_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
