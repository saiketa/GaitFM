
import argparse
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

from data_provider.data_loader import mPowerWalkingProcessedLoader
from models.GaitFMFullFinetune import Model as MAEFinetuneModel
from utils.tools import classification_metrics


PART_ORDER = ("POCKET",)


def parse_args():
    parser = argparse.ArgumentParser(description="Train mPower end-to-end single-part classifier.")
    parser.add_argument("--root_path", type=str, default="./data/mPower_full_50hz_5s")
    parser.add_argument("--output_dir", type=str, default="./results_end2end_single_part")
    parser.add_argument("--model_id", type=str, default="mPower_MAETiny_End2EndSinglePart_epoch5_cls")
    parser.add_argument("--pretrain_ckpt", type=str, required=True)
    parser.add_argument("--mae_checkpoint_path", type=str, default="./checkpoints/backbones/mae_tiny_400e.pth.tar")
    parser.add_argument("--mae_model", type=str, default="mae_vit_tiny_patch16")
    parser.add_argument("--mae_tag", type=str, default="MAETiny")
    parser.add_argument("--mae_finetune_pooling", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--seq_len", type=int, default=250)
    parser.add_argument("--mpower_val_ratio", type=float, default=0.2)
    parser.add_argument("--mpower_test_ratio", type=float, default=0.2)
    parser.add_argument("--mpower_split_seed", type=int, default=2021)
    parser.add_argument("--mpower_tasks", nargs="+", default=["outbound", "return"])
    parser.add_argument("--mpower_require_label", type=int, default=1)
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


class mPowerSinglePartDataset(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        self.labels = np.asarray(base_dataset.labels, dtype=np.int64)
        self.subjects = base_dataset.rows["subject"].astype(str).to_numpy()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        window, label = self.base[index]
        label_value = int(label.reshape(-1)[0].item())
        return {
            "windows": window.unsqueeze(0).float(),
            "part_ids": torch.tensor([0], dtype=torch.long),
            "label": torch.tensor(label_value, dtype=torch.long),
            "combo": PART_ORDER[0],
            "subject": self.subjects[index],
        }


def collate_single_part(batch):
    seq_len = batch[0]["windows"].shape[1]
    channels = batch[0]["windows"].shape[2]
    x = torch.zeros(len(batch), 1, seq_len, channels, dtype=torch.float32)
    part_ids = torch.zeros(len(batch), 1, dtype=torch.long)
    mask = torch.ones(len(batch), 1, dtype=torch.bool)
    labels = torch.empty(len(batch), dtype=torch.long)
    combos, subjects = [], []
    for i, item in enumerate(batch):
        x[i, 0] = item["windows"][0]
        labels[i] = item["label"]
        combos.append(item["combo"])
        subjects.append(item["subject"])
    return x, part_ids, mask, labels, combos, subjects


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


class EndToEndSinglePartModel(nn.Module):
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
            mae_finetune_pooling=args.mae_finetune_pooling,
            dropout=0.0,
            num_class=2,
        )
        self.encoder = MAEFinetuneModel(configs)
        load_sensor_checkpoint(self.encoder, args.pretrain_ckpt)
        self.num_parts = len(PART_ORDER)
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
        features = torch.zeros(
            batch_size * max_parts,
            self.embedding_dim,
            dtype=x.dtype,
            device=x.device,
        )
        if flat_mask.any():
            encoded = self.encode_single_part(flat_x[flat_mask])
            features[flat_mask] = encoded.to(dtype=x.dtype)
        features = features.reshape(batch_size, max_parts, self.embedding_dim)
        valid_part_ids = part_ids.clamp_min(0)
        features = features + self.part_embedding(valid_part_ids) * mask.unsqueeze(-1).to(dtype=features.dtype)

        slots = torch.zeros(batch_size, self.num_parts, self.embedding_dim, dtype=features.dtype, device=features.device)
        slot_mask = torch.zeros(batch_size, self.num_parts, dtype=features.dtype, device=features.device)
        for slot in range(max_parts):
            active = mask[:, slot]
            if not active.any():
                continue
            ids = part_ids[:, slot].clamp_min(0)
            rows = torch.arange(batch_size, device=features.device)
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
    print(f"Using WeightedRandomSampler for TRAIN single-part windows: {summary}")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


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
    overall = evaluate_overall(model, val_loader, test_loader, device)
    return [
        {
            "combo": PART_ORDER[0],
            "threshold": overall["threshold"],
            "val": overall["val_metrics"],
            "test": overall["test_metrics"],
            "n_val_subjects": overall["n_val_subjects"],
            "n_test_subjects": overall["n_test_subjects"],
            "subject_threshold": overall["subject_threshold"],
            "val_subject": overall["val_subject_metrics"],
            "test_subject": overall["test_subject_metrics"],
        }
    ]


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


def make_loader(args, flag):
    loader_args = SimpleNamespace(
        seq_len=args.seq_len,
        mpower_val_ratio=args.mpower_val_ratio,
        mpower_test_ratio=args.mpower_test_ratio,
        mpower_split_seed=args.mpower_split_seed,
        mpower_tasks=args.mpower_tasks,
        mpower_require_label=args.mpower_require_label,
        load_shards_to_memory=args.load_shards_to_memory,
        shard_cache_size=args.shard_cache_size,
    )
    base = mPowerWalkingProcessedLoader(loader_args, args.root_path, flag=flag)
    dataset = mPowerSinglePartDataset(base)
    sampler = weighted_sampler(dataset.labels, args.seed) if str(flag).upper() == "TRAIN" else None
    return dataset, DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and str(flag).upper() == "TRAIN"),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_single_part,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.no_gpu else "cpu")

    train_ds, train_loader = make_loader(args, "TRAIN")
    val_ds, val_loader = make_loader(args, "VAL")
    test_ds, test_loader = make_loader(args, "TEST")
    for name, ds in (("TRAIN", train_ds), ("VAL", val_ds), ("TEST", test_ds)):
        labels, counts = np.unique(ds.labels, return_counts=True)
        label_counts = {int(label): int(count) for label, count in zip(labels, counts)}
        print(f"{name}: windows={len(ds)} subjects={len(set(ds.subjects.tolist()))} label_counts={label_counts}")
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise ValueError("TRAIN/VAL/TEST must all be non-empty.")

    model = EndToEndSinglePartModel(args).to(device)
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
    results_json_path = out_dir / "combo_results.json"
    results = {}
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
            results[name] = {
                "epoch": record["epoch"],
                "overall": overall,
                "combos": evaluate_by_combo(model, val_loader, test_loader, device),
            }
            f.write(f"Single-part result for {name}\n")
            for combo_record in results[name]["combos"]:
                sample_line = (
                    f"  {combo_record['combo']} sample "
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
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=json_default)
    print(f"Saved summary: {summary_path}")
    print(f"Saved structured single-part results: {results_json_path}")


if __name__ == "__main__":
    main()
