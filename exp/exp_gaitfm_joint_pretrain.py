import copy
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader

from data_provider.data_loader import (
    ClinicalGaitLoader,
    COMPWALKACLProcessedLoader,
    FoGContestWindowLoader,
    GSTRIDELoader,
    MobiliseDTVSProcessedLoader,
    WearGaitPDShardedLoader,
    mPowerWalkingProcessedLoader,
)
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate


class Exp_GaitFM_JointPretrain(Exp_Basic):

    CLINICAL_DISEASES = ("ACL", "CIPN", "CVA", "HOA", "KOA", "PD", "RIL")

    def __init__(self, args):
        super().__init__(args)

    def _build_model(self):
        raise NotImplementedError("Subclasses must build a concrete fixed-channel pretraining model.")

    def _select_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _copy_args(self, **updates):
        args = copy.copy(self.args)
        for key, value in updates.items():
            setattr(args, key, value)
        return args

    def _dataset_specs(self):
        specs = []
        if getattr(self.args, "joint_use_clinical", 1):
            diseases = getattr(self.args, "joint_clinical_diseases", None) or self.CLINICAL_DISEASES
            for disease in diseases:
                specs.append(
                    (
                        f"ClinicalGait_{disease}",
                        ClinicalGaitLoader,
                        self.args.clinical_root,
                        self._copy_args(
                            data="ClinicalGait",
                            root_path=self.args.clinical_root,
                            clinical_disease=disease,
                            clinical_source_rate=100,
                            clinical_target_rate=50,
                            clinical_window_seconds=5,
                            seq_len=self.args.seq_len,
                        ),
                    )
                )
        if getattr(self.args, "joint_use_fog", 1):
            specs.append(
                (
                    "FoGContest",
                    FoGContestWindowLoader,
                    self.args.fog_root,
                    self._copy_args(
                        data="FoGContest",
                        root_path=self.args.fog_root,
                        fog_types=["tdcsfog", "defog", "notype"],
                        fog_target_rate=50,
                        fog_stride=self.args.seq_len,
                        fog_filter_valid_task=1,
                        seq_len=self.args.seq_len,
                    ),
                )
            )
        if getattr(self.args, "joint_use_mpower", 1):
            specs.append(
                (
                    "mPower",
                    mPowerWalkingProcessedLoader,
                    self.args.mpower_root,
                    self._copy_args(
                        data="mPowerWalkingProcessed",
                        root_path=self.args.mpower_root,
                        mpower_target_rate=50,
                        mpower_window_seconds=5,
                        mpower_stride=self.args.seq_len,
                        mpower_tasks=None,
                        mpower_require_label=0,
                        seq_len=self.args.seq_len,
                    ),
                )
            )
        if getattr(self.args, "joint_use_weargait", 1):
            specs.append(
                (
                    "WearGait-PD",
                    WearGaitPDShardedLoader,
                    self.args.weargait_root,
                    self._copy_args(
                        data="WearGaitPDSharded",
                        root_path=self.args.weargait_root,
                        weargait_source_rate=100,
                        weargait_target_rate=50,
                        weargait_window_seconds=5,
                        weargait_stride=self.args.seq_len,
                        weargait_tasks=None,
                        seq_len=self.args.seq_len,
                    ),
                )
            )
        if getattr(self.args, "joint_use_compwalk", 1):
            specs.append(
                (
                    "COMPWALK-ACL",
                    COMPWALKACLProcessedLoader,
                    self.args.compwalk_root,
                    self._copy_args(
                        data="COMPWALKACLProcessed",
                        root_path=self.args.compwalk_root,
                        compwalk_val_ratio=getattr(self.args, "compwalk_val_ratio", 0.2),
                        compwalk_test_ratio=getattr(self.args, "compwalk_test_ratio", 0.2),
                        compwalk_split_seed=getattr(self.args, "compwalk_split_seed", 2021),
                        seq_len=self.args.seq_len,
                    ),
                )
            )
        if getattr(self.args, "joint_use_gstride", 1):
            specs.append(
                (
                    "GSTRIDE",
                    GSTRIDELoader,
                    self.args.gstride_root,
                    self._copy_args(
                        data="GSTRIDE",
                        root_path=self.args.gstride_root,
                        gstride_task=getattr(self.args, "gstride_task", "pretrain"),
                        gstride_target_rate=50,
                        gstride_window_seconds=5,
                        gstride_stride=self.args.seq_len,
                        gstride_val_ratio=getattr(self.args, "gstride_val_ratio", 0.2),
                        gstride_test_ratio=getattr(self.args, "gstride_test_ratio", 0.2),
                        gstride_split_seed=getattr(self.args, "gstride_split_seed", 2021),
                        seq_len=self.args.seq_len,
                    ),
                )
            )
        if getattr(self.args, "joint_use_mobilised", 1):
            specs.append(
                (
                    "Mobilise-D-TVS",
                    MobiliseDTVSProcessedLoader,
                    self.args.mobilised_root,
                    self._copy_args(
                        data="MobiliseDTVSProcessed",
                        root_path=self.args.mobilised_root,
                        mobilised_subset="pretrain",
                        mobilised_disease=None,
                        mobilised_val_ratio=getattr(self.args, "mobilised_val_ratio", 0.2),
                        mobilised_test_ratio=getattr(self.args, "mobilised_test_ratio", 0.2),
                        mobilised_split_seed=getattr(self.args, "mobilised_split_seed", 2021),
                        seq_len=self.args.seq_len,
                    ),
                )
            )
        if not specs:
            raise ValueError("No datasets selected for joint pretraining.")
        return specs

    def _make_loaders(self, flag):
        datasets = []
        source_sizes = []
        num_workers = int(getattr(self.args, "joint_num_workers", min(int(getattr(self.args, "num_workers", 0)), 2)))
        pin_memory = bool(getattr(self.args, "use_gpu", False) and torch.cuda.is_available())
        for name, dataset_cls, root_path, dataset_args in self._dataset_specs():
            dataset = dataset_cls(args=dataset_args, root_path=root_path, flag=flag)
            datasets.append(dataset)
            source_sizes.append((name, len(dataset)))
        joint_dataset = ConcatDataset(datasets)
        shuffle = str(flag).upper() == "TRAIN"
        kwargs = {
            "batch_size": self.args.batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "drop_last": False,
            "pin_memory": pin_memory,
            "collate_fn": lambda batch, max_len=self.args.seq_len: self._joint_collate(batch, max_len=max_len),
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = int(getattr(self.args, "joint_prefetch_factor", 2))
        loader = DataLoader(joint_dataset, **kwargs)
        return [("JointFM", joint_dataset, loader, source_sizes)]

    def _joint_collate(self, batch, max_len):
        features = []
        labels = []
        lengths = []
        channels = []
        for item in batch:
            x, y = item[:2] if isinstance(item, (tuple, list)) else (item, torch.zeros(1, dtype=torch.long))
            if not torch.is_tensor(x):
                x = torch.as_tensor(x)
            x = x.float()
            features.append(x)
            labels.append(y if torch.is_tensor(y) else torch.as_tensor(y))
            lengths.append(min(int(x.shape[0]), int(max_len)))
            channels.append(int(x.shape[-1]))
        batch_size = len(features)
        max_channels = max(channels)
        x_out = torch.zeros(batch_size, int(max_len), max_channels, dtype=torch.float32)
        channel_valid = torch.zeros(batch_size, max_channels, dtype=torch.bool)
        for idx, x in enumerate(features):
            length = lengths[idx]
            ch = channels[idx]
            x_out[idx, :length, :ch] = x[:length, :ch]
            channel_valid[idx, :ch] = True
        return {
            "x": x_out,
            "labels": torch.stack(labels, dim=0),
            "channel_valid": channel_valid,
        }

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch["x"], batch.get("channel_valid")
        if isinstance(batch, (list, tuple)):
            return batch[0], None
        return batch, None

    def _run_batch(self, batch, optimizer=None):
        train = optimizer is not None
        batch_x, channel_valid = self._unpack_batch(batch)
        batch_x = batch_x.float().to(self.device, non_blocking=True)
        if channel_valid is not None:
            channel_valid = channel_valid.to(self.device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        out = self.model.reconstruction_loss(batch_x, channel_valid_mask=channel_valid)
        loss = out["loss"]
        if train:
            loss.backward()
            if getattr(self.args, "gait_grad_clip", 0.0) > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gait_grad_clip)
            optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "mse": float(out["mse"].detach().cpu()),
            "mae": float(out["mae"].detach().cpu()),
            "mask_ratio": float(out["mask_ratio"].detach().cpu()),
        }

    def _run_epoch(self, named_loaders, optimizer=None):
        train = optimizer is not None
        self.model.train(train)
        max_steps = int(getattr(self.args, "joint_epoch_steps", 0) or 0)
        log_every = int(getattr(self.args, "joint_log_every", 0) or 0)
        start_time = time.time()
        active = [(item[0], iter(item[2])) for item in named_loaders]
        metrics = []
        per_source = {name: [] for name, *_ in named_loaders}
        steps = 0
        while active:
            next_active = []
            for name, iterator in active:
                if max_steps > 0 and steps >= max_steps:
                    break
                try:
                    batch = next(iterator)
                except StopIteration:
                    continue
                batch_metrics = self._run_batch(batch, optimizer=optimizer)
                metrics.append(batch_metrics)
                per_source[name].append(batch_metrics)
                steps += 1
                if log_every > 0 and steps % log_every == 0:
                    mode = "train" if train else "val"
                    print(
                        f"    joint {mode} step {steps} | source={name} | "
                        f"loss={batch_metrics['loss']:.6f} | elapsed={time.time() - start_time:.1f}s",
                        flush=True,
                    )
                next_active.append((name, iterator))
            if max_steps > 0 and steps >= max_steps:
                break
            active = next_active

        def average(items, key):
            return float(np.average([item[key] for item in items])) if items else 0.0

        summary = {
            "loss": average(metrics, "loss"),
            "mse": average(metrics, "mse"),
            "mae": average(metrics, "mae"),
            "mask_ratio": average(metrics, "mask_ratio"),
            "steps": steps,
            "per_source": {},
        }
        for name, values in per_source.items():
            summary["per_source"][name] = {
                "steps": len(values),
                "loss": average(values, "loss"),
                "mse": average(values, "mse"),
                "mae": average(values, "mae"),
            }
        return summary

    def _source_line(self, metrics):
        parts = []
        for name, values in metrics["per_source"].items():
            parts.append(f"{name}:steps={values['steps']},loss={values['loss']:.5f}")
        return " | ".join(parts)

    def train(self, setting):
        train_loaders = self._make_loaders(flag="TRAIN")
        vali_loaders = self._make_loaders(flag="VAL")
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        optimizer = self._select_optimizer()
        best_val_loss = float("inf")
        best_model_path = os.path.join(path, "checkpoint.pth")

        print("GaitFM joint IMU pretraining sources:")
        for _, dataset, _, source_sizes in train_loaders:
            print(f"  TRAIN JointFM total: {len(dataset)} windows")
            for source_name, source_len in source_sizes:
                print(f"    TRAIN {source_name}: {source_len} windows")
        for _, dataset, _, source_sizes in vali_loaders:
            print(f"  VAL   JointFM total: {len(dataset)} windows")
            for source_name, source_len in source_sizes:
                print(f"    VAL   {source_name}: {source_len} windows")
        print("Labels are ignored for pretraining; best model is selected by joint validation loss.")

        for epoch in range(self.args.train_epochs):
            epoch_start = time.time()
            train_metrics = self._run_epoch(train_loaders, optimizer=optimizer)
            with torch.no_grad():
                vali_metrics = self._run_epoch(vali_loaders, optimizer=None)
            print(
                "Epoch: {0}, Time: {1:.2f}s | "
                "Train Loss: {2:.7f} MSE: {3:.7f} MAE: {4:.7f} Mask: {5:.3f} Steps: {6} | "
                "Vali Loss: {7:.7f} MSE: {8:.7f} MAE: {9:.7f} Mask: {10:.3f} Steps: {11}".format(
                    epoch + 1,
                    time.time() - epoch_start,
                    train_metrics["loss"],
                    train_metrics["mse"],
                    train_metrics["mae"],
                    train_metrics["mask_ratio"],
                    train_metrics["steps"],
                    vali_metrics["loss"],
                    vali_metrics["mse"],
                    vali_metrics["mae"],
                    vali_metrics["mask_ratio"],
                    vali_metrics["steps"],
                )
            )
            print(f"  Train per-source | {self._source_line(train_metrics)}")
            print(f"  Vali  per-source | {self._source_line(vali_metrics)}")
            if np.isfinite(vali_metrics["loss"]) and vali_metrics["loss"] < best_val_loss:
                best_val_loss = float(vali_metrics["loss"])
                torch.save(self.model.state_dict(), best_model_path)
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        if not os.path.exists(best_model_path):
            raise RuntimeError("No finite validation loss; no checkpoint selected")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model
