
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_loader import WearGaitPDShardedLoader
from models.GaitFMFullFinetune import Model as MAEFinetuneModel
from utils.tools import classification_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train WearGait-PD end-to-end part-combination classifier.")
    parser.add_argument("--root_path", type=str, default="./data/WearGait-PD_full_50hz_5s")
    parser.add_argument("--output_dir", type=str, default="./results_end2end_part_combinations")
    parser.add_argument("--model_id", type=str, default="WearGaitPD_MAETiny_End2EndPartCombinations_epoch5_cls")
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--mae_checkpoint_path", type=str, default="./checkpoints/backbones/mae_tiny_400e.pth.tar")
    parser.add_argument("--mae_model", type=str, default="mae_vit_tiny_patch16")
    parser.add_argument("--mae_tag", type=str, default="MAETiny")
    parser.add_argument("--mae_finetune_pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--seq_len", type=int, default=250)
    parser.add_argument("--weargait_val_ratio", type=float, default=0.2)
    parser.add_argument("--weargait_test_ratio", type=float, default=0.2)
    parser.add_argument("--weargait_split_seed", type=int, default=2021)
    parser.add_argument("--weargait_body_parts", nargs="+", default=None)
    parser.add_argument("--weargait_tasks", nargs="+", default=None)
    parser.add_argument("--max_combo_size", type=int, default=4, help="0 means enumerate all non-empty combinations.")
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


def format_metrics(metrics):
    return (
        "Acc: {acc:.6f} F1: {f1:.6f} BAcc: {balanced_acc:.6f} "
        "Sen: {sen:.6f} Spe: {spe:.6f} AUROC: {auroc:.6f} AUPRC: {auprc:.6f} "
        "TP: {tp} TN: {tn} FP: {fp} FN: {fn}"
    ).format(
        acc=metrics.get("acc", float("nan")),
        f1=metrics.get("f1", float("nan")),
        balanced_acc=metrics.get("balanced_acc", float("nan")),
        sen=metrics.get("sen", float("nan")),
        spe=metrics.get("spe", float("nan")),
        auroc=metrics.get("auroc", float("nan")),
        auprc=metrics.get("auprc", float("nan")),
        tp=metrics.get("tp", -1),
        tn=metrics.get("tn", -1),
        fp=metrics.get("fp", -1),
        fn=metrics.get("fn", -1),
    )


def load_sensor_checkpoint(model, checkpoint):
    if checkpoint is None or str(checkpoint).strip().lower() in {"", "none", "random", "scratch"}:
        print("Sensor pretrain checkpoint disabled; using MAE encoder without sensor-pretrained weights.")
        return
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
    compatible = {
        k: v
        for k, v in state.items()
        if k.startswith("image_encoder.") or k.startswith("mae.")
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(
        f"Loaded sensor-pretrained encoder from {path} | "
        f"tensors={len(compatible)} missing={len(missing)} unexpected={len(unexpected)}"
    )


def loader_args(args):
    return SimpleNamespace(
        seq_len=args.seq_len,
        weargait_val_ratio=args.weargait_val_ratio,
        weargait_test_ratio=args.weargait_test_ratio,
        weargait_split_seed=args.weargait_split_seed,
        weargait_body_parts=args.weargait_body_parts,
        weargait_tasks=args.weargait_tasks,
        load_shards_to_memory=args.load_shards_to_memory,
        shard_cache_size=args.shard_cache_size,
    )


def make_base_loader(args, flag):
    return WearGaitPDShardedLoader(loader_args(args), args.root_path, flag=flag)


def moment_key(row):
    required = ["subject", "task", "source_path", "window_start_sample", "window_end_sample"]
    missing = [column for column in required if column not in row.index]
    if missing:
        raise ValueError(f"WearGait-PD manifest missing columns for same-time grouping: {missing}")
    return (
        str(row["subject"]),
        str(row["task"]),
        str(row["source_path"]),
        int(row["window_start_sample"]),
        int(row["window_end_sample"]),
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
                "task": str(row["task"]),
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


class RawCombinationDataset(Dataset):
    def __init__(self, moments, examples):
        self.moments = moments
        self.examples = examples
        self.labels = np.asarray([item["label"] for item in examples], dtype=np.int64)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        moment = self.moments[item["moment_index"]]
        windows = np.stack([moment["parts"][part] for part in item["parts"]], axis=0)
        return {
            "windows": torch.from_numpy(windows),
            "part_ids": torch.tensor(item["part_ids"], dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "combo": item["combo"],
            "subject": item["subject"],
        }


def collate_raw_combinations(batch):
    max_parts = max(item["windows"].shape[0] for item in batch)
    seq_len = batch[0]["windows"].shape[1]
    channels = batch[0]["windows"].shape[2]
    x = torch.zeros(len(batch), max_parts, seq_len, channels, dtype=torch.float32)
    part_ids = torch.full((len(batch), max_parts), -1, dtype=torch.long)
    mask = torch.zeros(len(batch), max_parts, dtype=torch.bool)
    labels = torch.empty(len(batch), dtype=torch.long)
    combos, subjects = [], []
    for i, item in enumerate(batch):
        n = item["windows"].shape[0]
        x[i, :n] = item["windows"]
        part_ids[i, :n] = item["part_ids"]
        mask[i, :n] = True
        labels[i] = item["label"]
        combos.append(item["combo"])
        subjects.append(item["subject"])
    return x, part_ids, mask, labels, combos, subjects


class EndToEndPartCombinationModel(nn.Module):
    def __init__(self, args, part_order):
        super().__init__()
        configs = SimpleNamespace(
            gait_image_size=args.gait_image_size,
            gait_segment_stride=args.gait_segment_stride,
            gait_sum_to_one=args.gait_sum_to_one,
            gait_max_channels=args.max_channels,
            mae_model=args.mae_model,
            mae_checkpoint_path=args.mae_checkpoint_path,
            mae_norm_pix_loss=1,
            mae_finetune_pooling=args.mae_finetune_pooling,
            dropout=0.0,
            num_class=2,
        )
        self.encoder = MAEFinetuneModel(configs)
        load_sensor_checkpoint(self.encoder, args.pretrain_ckpt)
        self.part_order = tuple(part_order)
        self.num_parts = len(self.part_order)
        self.embedding_dim = int(self.encoder.mae.norm.normalized_shape[0])
        self.part_embedding = nn.Embedding(self.num_parts, self.embedding_dim)
        classifier_input = self.num_parts * self.embedding_dim + self.num_parts
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Linear(classifier_input, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, 2),
        )

    def encode_single_part(self, x):
        x = self.encoder._fit_channels(x)
        images = self.encoder.image_encoder(x)
        return self.encoder._encode_images(images)

    def forward(self, x, part_ids, mask):
        batch_size, max_parts, seq_len, channels = x.shape
        flat_x = x.reshape(batch_size * max_parts, seq_len, channels)
        flat_mask = mask.reshape(batch_size * max_parts)
        features = torch.zeros(batch_size * max_parts, self.embedding_dim, dtype=x.dtype, device=x.device)
        if flat_mask.any():
            features[flat_mask] = self.encode_single_part(flat_x[flat_mask]).to(dtype=x.dtype)
        features = features.reshape(batch_size, max_parts, self.embedding_dim)

        valid_part_ids = part_ids.clamp_min(0)
        features = features + self.part_embedding(valid_part_ids) * mask.unsqueeze(-1).to(dtype=features.dtype)
        slots = torch.zeros(batch_size, self.num_parts, self.embedding_dim, dtype=features.dtype, device=features.device)
        slot_mask = torch.zeros(batch_size, self.num_parts, dtype=features.dtype, device=features.device)
        rows = torch.arange(batch_size, device=features.device)
        for slot in range(max_parts):
            active = mask[:, slot]
            if not active.any():
                continue
            ids = part_ids[:, slot].clamp_min(0)
            slots[rows[active], ids[active]] = features[active, slot]
            slot_mask[rows[active], ids[active]] = 1.0
        classifier_input = torch.cat([slots.reshape(batch_size, -1), slot_mask], dim=-1)
        return self.classifier(classifier_input)


def weighted_sampler(labels, seed):
    labels = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return None
    class_weights = {int(cls): len(labels) / (len(classes) * int(count)) for cls, count in zip(classes, counts)}
    sample_weights = np.asarray([class_weights[int(label)] for label in labels], dtype=np.float64)
    summary = ", ".join(
        f"class_{int(cls)}=count:{int(count)}, weight:{class_weights[int(cls)]:.4f}"
        for cls, count in zip(classes, counts)
    )
    print(f"Using WeightedRandomSampler for TRAIN combinations: {summary}")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights), replacement=True, generator=generator)


def threshold_metrics(y_true, probs, threshold):
    pred = (probs[:, 1] >= float(threshold)).astype(np.int64)
    return classification_metrics(y_true, pred, probs)


def best_f1_threshold(y_true, probs):
    best_threshold = 0.5
    best_metrics = threshold_metrics(y_true, probs, best_threshold)
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = threshold_metrics(y_true, probs, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def aggregate_subjects(y_true, probs, subjects):
    subjects = np.asarray(subjects).astype(str)
    y_true = np.asarray(y_true).astype(np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    subject_true, subject_prob = [], []
    for subject in sorted(np.unique(subjects)):
        idx = np.where(subjects == subject)[0]
        values, counts = np.unique(y_true[idx], return_counts=True)
        subject_true.append(int(values[np.argmax(counts)]))
        subject_prob.append(probs[idx].mean(axis=0))
    return np.asarray(subject_true), np.asarray(subject_prob)


@torch.no_grad()
def predict(model, loader, device):
    logits_list, labels_list = [], []
    combos, subjects = [], []
    model.eval()
    for x, part_ids, mask, labels, batch_combos, batch_subjects in loader:
        logits = model(x.to(device), part_ids.to(device), mask.to(device))
        logits_list.append(logits.cpu())
        labels_list.append(labels)
        combos.extend(batch_combos)
        subjects.extend(batch_subjects)
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).numpy()
    probs = torch.softmax(logits, dim=1).numpy()
    return labels, probs, np.asarray(combos), np.asarray(subjects)


def evaluate_overall(model, val_loader, test_loader, device):
    val_y, val_probs, _, val_subjects = predict(model, val_loader, device)
    test_y, test_probs, _, test_subjects = predict(model, test_loader, device)
    threshold, val_metrics = best_f1_threshold(val_y, val_probs)
    test_metrics = threshold_metrics(test_y, test_probs, threshold)
    val_subject_y, val_subject_probs = aggregate_subjects(val_y, val_probs, val_subjects)
    test_subject_y, test_subject_probs = aggregate_subjects(test_y, test_probs, test_subjects)
    subject_threshold, val_subject_metrics = best_f1_threshold(val_subject_y, val_subject_probs)
    test_subject_metrics = threshold_metrics(test_subject_y, test_subject_probs, subject_threshold)
    return {
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "subject_threshold": subject_threshold,
        "val_subject_metrics": val_subject_metrics,
        "test_subject_metrics": test_subject_metrics,
        "n_val_subjects": int(len(val_subject_y)),
        "n_test_subjects": int(len(test_subject_y)),
    }


def evaluate_by_combo(model, val_loader, test_loader, device):
    val_y, val_probs, val_combos, val_subjects = predict(model, val_loader, device)
    test_y, test_probs, test_combos, test_subjects = predict(model, test_loader, device)
    results = []
    for combo in sorted(set(test_combos.tolist())):
        val_idx = np.where(val_combos == combo)[0]
        test_idx = np.where(test_combos == combo)[0]
        if len(val_idx) == 0 or len(test_idx) == 0:
            continue
        threshold, val_metrics = best_f1_threshold(val_y[val_idx], val_probs[val_idx])
        test_metrics = threshold_metrics(test_y[test_idx], test_probs[test_idx], threshold)
        val_subject_y, val_subject_probs = aggregate_subjects(val_y[val_idx], val_probs[val_idx], val_subjects[val_idx])
        test_subject_y, test_subject_probs = aggregate_subjects(test_y[test_idx], test_probs[test_idx], test_subjects[test_idx])
        subject_threshold, val_subject_metrics = best_f1_threshold(val_subject_y, val_subject_probs)
        test_subject_metrics = threshold_metrics(test_subject_y, test_subject_probs, subject_threshold)
        results.append(
            {
                "combo": combo,
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "threshold": threshold,
                "val": val_metrics,
                "test": test_metrics,
                "n_val_subjects": int(len(val_subject_y)),
                "n_test_subjects": int(len(test_subject_y)),
                "subject_threshold": subject_threshold,
                "val_subject": val_subject_metrics,
                "test_subject": test_subject_metrics,
            }
        )
    return results


def format_record(prefix, epoch, val_loss, overall):
    return (
        f"{prefix}: Epoch {epoch} | val_loss={val_loss:.6f} "
        f"sample_threshold={overall['threshold']:.2f} Sample-Val: {format_metrics(overall['val_metrics'])} "
        f"Sample-Test: {format_metrics(overall['test_metrics'])} "
        f"| subject_threshold={overall['subject_threshold']:.2f} "
        f"Subject-Val(n={overall['n_val_subjects']}): {format_metrics(overall['val_subject_metrics'])} "
        f"Subject-Test(n={overall['n_test_subjects']}): {format_metrics(overall['test_subject_metrics'])}"
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for x, part_ids, mask, labels, _, _ in loader:
        optimizer.zero_grad()
        logits = model(x.to(device), part_ids.to(device), mask.to(device))
        loss = criterion(logits, labels.to(device))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 4.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def validation_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    for x, part_ids, mask, labels, _, _ in loader:
        logits = model(x.to(device), part_ids.to(device), mask.to(device))
        losses.append(float(criterion(logits, labels.to(device)).item()))
    return float(np.mean(losses)) if losses else 0.0


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def make_loader(dataset, args, train=False):
    sampler = weighted_sampler(dataset.labels, args.seed) if train else None
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and train),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_raw_combinations,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    base = {split: make_base_loader(args, split) for split in ("TRAIN", "VAL", "TEST")}
    part_order = sorted(
        set().union(*(set(loader.rows["body_part"].astype(str).unique()) for loader in base.values()))
    )
    if args.weargait_body_parts:
        requested = [str(part) for part in args.weargait_body_parts]
        part_order = [part for part in requested if part in set(part_order)]
    if not part_order:
        raise ValueError("No WearGait-PD body parts found for combination training.")
    if args.max_combo_size <= 0 and len(part_order) > 10:
        print(
            f"Warning: enumerating all combinations for {len(part_order)} body parts "
            f"creates {2 ** len(part_order) - 1} combinations per moment."
        )
    print(f"WearGait-PD part order ({len(part_order)}): {part_order}")

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
    train_loader = make_loader(train_ds, args, train=True)
    val_loader = make_loader(val_ds, args, train=False)
    test_loader = make_loader(test_ds, args, train=False)

    model = EndToEndPartCombinationModel(args, part_order).to(device)
    encoder_params = list(model.encoder.parameters())
    classifier_param_ids = {id(param) for param in model.classifier.parameters()}
    classifier_param_ids.update(id(param) for param in model.part_embedding.parameters())
    other_params = [param for param in model.parameters() if id(param) in classifier_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.learning_rate},
            {"params": other_params, "lr": args.classifier_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir) / args.model_id
    ckpt_dir = out_dir / "ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_records = {"val_loss": None}
    best_paths = {"val_loss": ckpt_dir / "val_loss.pth"}
    best_val_loss = float("inf")
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validation_loss(model, val_loader, criterion, device)
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
    with open(summary_path, "w", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")
        for name, record in best_records.items():
            if record is None:
                continue
            model.load_state_dict(torch.load(best_paths[name], map_location=device))
            overall = evaluate_overall(model, val_loader, test_loader, device)
            line = format_record(f"Best by {name}", record["epoch"], record["val_loss"], overall)
            print(line)
            f.write(line + "\n")
            combo_results[name] = {
                "epoch": record["epoch"],
                "overall": overall,
                "part_order": part_order,
                "max_combo_size": int(args.max_combo_size),
                "combos": evaluate_by_combo(model, val_loader, test_loader, device),
            }
            f.write(f"Per-combination results for {name}\n")
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
                f.write(sample_line + "\n")
                f.write(subject_line + "\n")
    with open(combo_json_path, "w", encoding="utf-8") as f:
        json.dump(combo_results, f, indent=2, default=json_default)
    print(f"Saved summary: {summary_path}")
    print(f"Saved structured combo results: {combo_json_path}")


if __name__ == "__main__":
    main()
