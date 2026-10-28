
import argparse

from preprocess_weargaitpd_50hz_5s import DEFAULT_BODY_PARTS, preprocess


WALKING_TASKS = [
    "SelfPace",
    "SP",
    "HurriedPace",
    "HP",
    "SelfPace_mat",
    "SPm",
    "HurriedPace_mat",
    "HPm",
    "SelfPace_doorpat",
    "SPdoorpat",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess WearGait-PD walking-only tasks to 50 Hz, 5 s windows."
    )
    parser.add_argument("--root_path", default="./data/WearGait-PD")
    parser.add_argument("--output_dir", default="./data/WearGait-PD_walking_50hz_5s")
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--source_rate", type=float, default=100.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=0, help="0 means non-overlapping windows of window_size.")
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=WALKING_TASKS,
        help="Task names parsed from file names. Defaults to walking-only tasks.",
    )
    parser.add_argument(
        "--body_parts",
        nargs="+",
        default=None,
        help="Optional body parts to extract. Defaults to all supported body parts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.body_parts is None:
        args.body_parts = DEFAULT_BODY_PARTS
    preprocess(args)


if __name__ == "__main__":
    main()
