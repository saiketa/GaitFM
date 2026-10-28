import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT / "scripts"
CATEGORIES = ("pretrain", "classification", "regression")


def discover_tasks():
    tasks = {}
    for category in CATEGORIES:
        for path in sorted((TASK_ROOT / category).glob("*.sh")):
            tasks[f"{category}/{path.stem}"] = path
    return tasks


def parse_assignments(values):
    assignments = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected KEY=VALUE, received: {value}")
        key, item = value.split("=", 1)
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            raise ValueError(f"Invalid environment variable name: {key}")
        assignments[key] = item
    return assignments


def list_tasks(args):
    for name in discover_tasks():
        if args.category is None or name.startswith(f"{args.category}/"):
            print(name)
    return 0


def run_task(args):
    tasks = discover_tasks()
    if args.task not in tasks:
        choices = "\n".join(sorted(tasks))
        raise ValueError(f"Unknown task: {args.task}\nAvailable tasks:\n{choices}")
    updates = parse_assignments(args.set)
    command = ["bash", str(tasks[args.task])]
    if args.dry_run:
        for key, value in sorted(updates.items()):
            print(f"{key}={value}")
        print(" ".join(command))
        return 0
    environment = os.environ.copy()
    environment.update(updates)
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


def build_parser():
    parser = argparse.ArgumentParser(prog="gaitfm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--category", choices=CATEGORIES)
    list_parser.set_defaults(handler=list_tasks)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("task")
    run_parser.add_argument("--set", action="append", default=[])
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(handler=run_task)

    return parser


def main():
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
