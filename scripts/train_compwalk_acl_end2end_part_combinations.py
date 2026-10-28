
import argparse
import itertools
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_loader import COMPWALKACLProcessedLoader
from scripts.train_weargaitpd_end2end_part_combinations import (
    EndToEndPartCombinationModel,
    RawCombinationDataset,
    build_examples as build_combo_examples,
    evaluate_by_combo,
    evaluate_overall,
    format_metrics,
    format_record,
    make_loader,
    save_checkpoint,
    train_one_epoch,
    validation_loss,
    weighted_sampler,
    json_default,
)


DEFAULT_PARTS = (
    "Pelvis",
    "RightUpperLeg",
    "RightLowerLeg",
    "RightFoot",
    "LeftUpperLeg",
    "LeftLowerLeg",
    "LeftFoot",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train COMPWALK-ACL end-to-end part-combination classifier.")
    parser.add_argument("--root_path", type=str, default="./data/COMPWALK-ACL_50hz_5s")
    parser.add_argument("--output_dir", type=str, default="./results_end2end_part_combinations")
    parser.add_argument("--model_id", type=str, default="COMPWALKACL_MAETiny_End2EndPartCombinations_epoch5_cls")
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--mae_checkpoint_path", type=str, default="./checkpoints/backbones/mae_tiny_400e.pth.tar")
    parser.add_argument("--mae_model", type=str, default="mae_vit_tiny_patch16")
    parser.add_argument("--mae_tag", type=str, default="MAETiny")
    parser.add_argument("--mae_finetune_pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--seq_len", type=int, default=250)
    parser.add_argument("--compwalk_val_ratio", type=float, default=0.2)
    parser.add_argument("--compwalk_test_ratio", type=float, default=0.2)
    parser.add_argument("--compwalk_split_seed", type=int, default=2021)
    parser.add_argument("--compwalk_body_parts", nargs="+", default=None)
    parser.add_argument("--max_combo_size", type=int, default=0, help="0 means enumerate all non-empty combinations.")
    parser.add_argument("--load_shards_to_memory", type=int, default=1)
    parser.add_argument("--shard_cache_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--classifier_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--gait_image_size", type=int, default=224)
    parser.add_argument("--gait_segment_stride", type=int, default=10)
    parser.add_argument("--gait_sum_to_one", type=int, default=1)
    parser.add_argument("--max_channels", type=int, default=6)
    return parser.parse_args()


def loader_args(args):
    return SimpleNamespace(
        seq_len=args.seq_len,
        compwalk_val_ratio=args.compwalk_val_ratio,
        compwalk_test_ratio=args.compwalk_test_ratio,
        compwalk_split_seed=args.compwalk_split_seed,
        load_shards_to_memory=args.load_shards_to_memory,
        shard_cache_size=args.shard_cache_size,
    )


def make_base_loader(args, flag):
    return COMPWALKACLProcessedLoader(loader_args(args), args.root_path, flag=flag)


def moment_key(row):
    required = ["subject", "speed", "trial", "start", "end"]
    missing = [column for column in required if column not in row.index]
    if missing:
        raise ValueError(f"COMPWALK-ACL manifest missing columns for same-time grouping: {missing}")
    return (
        str(row["subject"]),
        str(row["speed"]),
        str(row["trial"]),
        int(row["start"]),
        int(row["end"]),
    )


def load_window(base, index):
    if base.memory_x is not None:
        return base.memory_x[index].astype(np.float32), int(base.memory_y[index])
    window, label = base[index]
    return window.numpy().astype(np.float32), int(label.reshape(-1)[0].item())


def build_moments(base, part_order):
    part_set = set(part_order)
    part_rank = {part: idx for idx, part in enumerate(part_order)}
    grouped = {}
    for idx, row in base.rows.reset_index(drop=True).iterrows():
        part = str(row.get("body_part", ""))
        if part not in part_set:
            continue
        key = moment_key(row)
        window, label = load_window(base, idx)
        item = grouped.setdefault(
            key,
            {
                "subject": str(row["subject"]),
                "label": int(label),
                "parts": {},
            },
        )
        item["parts"][part] = window
    moments = []
    for key, item in grouped.items():
        ordered_parts = sorted(item["parts"], key=lambda p: part_rank[p])
        if ordered_parts:
            item["moment_id"] = "::".join(map(str, key))
            item["parts"] = {part: item["parts"][part] for part in ordered_parts}
            moments.append(item)
    return moments


def build_examples(moments, part_order, max_combo_size):
    examples = []
    max_size = len(part_order) if max_combo_size <= 0 else min(int(max_combo_size), len(part_order))
    for moment_index, moment in enumerate(moments):
        available = [part for part in part_order if part in moment["parts"]]
        for size in range(1, min(max_size, len(available)) + 1):
            for combo in itertools.combinations(available, size):
                examples.append(
                    {
                        "moment_index": moment_index,
                        "parts": combo,
                        "part_ids": [part_order.index(part) for part in combo],
                        "combo": "+".join(combo),
                        "subject": moment["subject"],
                        "label": moment["label"],
                    }
                )
    return examples


def train_loader(dataset, args):
    sampler = weighted_sampler(dataset.labels, args.seed)
    from torch.utils.data import DataLoader
    from scripts.train_weargaitpd_end2end_part_combinations import collate_raw_combinations

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_raw_combinations,
    )


def eval_loader(dataset, args):
    from torch.utils.data import DataLoader
    from scripts.train_weargaitpd_end2end_part_combinations import collate_raw_combinations

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_raw_combinations,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    base = {split: make_base_loader(args, split) for split in ("TRAIN", "VAL", "TEST")}
    found_parts = sorted(set().union(*(set(loader.rows["body_part"].astype(str).unique()) for loader in base.values())))
    requested = args.compwalk_body_parts or list(DEFAULT_PARTS)
    part_order = [part for part in requested if part in set(found_parts)]
    if not part_order:
        raise ValueError(f"No requested COMPWALK-ACL body parts found. Available parts: {found_parts}")
    if args.max_combo_size <= 0 and len(part_order) > 10:
        print(
            f"Warning: enumerating all combinations for {len(part_order)} body parts "
            f"creates {2 ** len(part_order) - 1} combinations per moment."
        )
    print(f"COMPWALK-ACL available body parts ({len(found_parts)}): {found_parts}")
    print(f"COMPWALK-ACL selected part order ({len(part_order)}): {part_order}")

    moments = {split: build_moments(base[split], part_order) for split in ("TRAIN", "VAL", "TEST")}
    examples = {split: build_examples(moments[split], part_order, args.max_combo_size) for split in ("TRAIN", "VAL", "TEST")}
    for split in ("TRAIN", "VAL", "TEST"):
        labels = np.asarray([item["label"] for item in examples[split]], dtype=np.int64)
        counts = dict(zip(*np.unique(labels, return_counts=True))) if len(labels) else {}
        print(f"{split}: moments={len(moments[split])} combinations={len(examples[split])} label_counts={counts}")
    if not examples["TRAIN"] or not examples["VAL"] or not examples["TEST"]:
        raise ValueError("TRAIN/VAL/TEST combinations must all be non-empty.")

    train_ds = RawCombinationDataset(moments["TRAIN"], examples["TRAIN"])
    val_ds = RawCombinationDataset(moments["VAL"], examples["VAL"])
    test_ds = RawCombinationDataset(moments["TEST"], examples["TEST"])
    train_dl = train_loader(train_ds, args)
    val_dl = eval_loader(val_ds, args)
    test_dl = eval_loader(test_ds, args)

    model = EndToEndPartCombinationModel(args, part_order).to(device)
    encoder_params = list(model.encoder.parameters())
    classifier_param_ids = {id(param) for param in model.classifier.parameters()}
    classifier_param_ids.update(id(param) for param in model.part_embedding.parameters())
    classifier_params = [param for param in model.parameters() if id(param) in classifier_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.learning_rate},
            {"params": classifier_params, "lr": args.classifier_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir) / args.model_id
    ckpt_dir = out_dir / "ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_records = {"val_loss": None}
    best_paths = {"val_loss": ckpt_dir / "val_loss.pth"}
    best_val_loss = float("inf")
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        import time

        start = time.time()
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val_loss = validation_loss(model, val_dl, criterion, device)
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
            if current is None or score > current["score"]:
                best_records[name] = {"epoch": epoch, "score": float(score), "val_loss": val_loss}
                save_checkpoint(model, best_paths[name])
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = float(val_loss)

    if best_records["val_loss"] is None:
        raise RuntimeError("No finite validation loss; no checkpoint selected")

    summary_path = out_dir / "summary.log"
    combo_json_path = out_dir / "combo_results.json"
    combo_results = {}
    with open(summary_path, "w", encoding="utf-8") as handle:
        for line in log_lines:
            handle.write(line + "\n")
        for name, record in best_records.items():
            if record is None:
                continue
            model.load_state_dict(torch.load(best_paths[name], map_location=device))
            overall = evaluate_overall(model, val_dl, test_dl, device)
            line = format_record(f"Best by {name}", record["epoch"], record["val_loss"], overall)
            print(line)
            handle.write(line + "\n")
            combo_results[name] = {
                "epoch": record["epoch"],
                "overall": overall,
                "part_order": part_order,
                "max_combo_size": int(args.max_combo_size),
                "combos": evaluate_by_combo(model, val_dl, test_dl, device),
            }
            handle.write(f"Per-combination results for {name}\n")
            for combo_record in combo_results[name]["combos"]:
                sample_line = (
                    f"  {combo_record['combo']} sample n={combo_record['n_test']} "
                    f"threshold={combo_record['threshold']:.2f} "
                    f"Val: {format_metrics(combo_record['val'])} "
                    f"Test: {format_metrics(combo_record['test'])}"
                )
                subject_line = (
                    f"  {combo_record['combo']} subject n={combo_record['n_test_subjects']} "
                    f"threshold={combo_record['subject_threshold']:.2f} "
                    f"Val: {format_metrics(combo_record['val_subject'])} "
                    f"Test: {format_metrics(combo_record['test_subject'])}"
                )
                print(sample_line)
                print(subject_line)
                handle.write(sample_line + "\n")
                handle.write(subject_line + "\n")
    with open(combo_json_path, "w", encoding="utf-8") as handle:
        import json

        json.dump(combo_results, handle, indent=2, default=json_default)
    print(f"Saved summary: {summary_path}")
    print(f"Saved structured combo results: {combo_json_path}")


if __name__ == "__main__":
    main()
