
import argparse

from preprocess_mpower_walking_50hz_5s import DEVICE_MOTION_COLUMNS, preprocess


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess full mPower WalkingActivity deviceMotion to 50 Hz, 5 s windows.")
    parser.add_argument("--root_path", default="./data/mPower")
    parser.add_argument("--output_dir", default="./data/mPower_walking_50hz_5s")
    parser.add_argument("--target_rate", type=float, default=50.0)
    parser.add_argument("--window_seconds", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=0, help="0 means non-overlapping windows of window_size.")
    parser.add_argument("--shard_size", type=int, default=5000)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument("--tasks", nargs="+", default=DEVICE_MOTION_COLUMNS)
    parser.add_argument("--allow_sorted_fallback", action="store_true")
    parser.add_argument("--include_unlabeled", action="store_true", default=True)
    return parser.parse_args()


def main():
    preprocess(parse_args())


if __name__ == "__main__":
    main()
