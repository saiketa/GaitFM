import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"mse": np.nan, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson": np.nan}
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else np.nan
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = np.nan
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2, "pearson": pearson}


def format_reg_metrics(metrics):
    return (
        f"MSE: {metrics['mse']:.6f} MAE: {metrics['mae']:.6f} "
        f"RMSE: {metrics['rmse']:.6f} R2: {metrics['r2']:.6f} "
        f"Pearson: {metrics['pearson']:.6f}"
    )


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


def classification_threshold_metrics(y_true, probs, threshold):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    probs = np.asarray(probs)
    if probs.ndim == 2:
        pos_probs = probs[:, 1]
    else:
        pos_probs = probs.reshape(-1)
    pred = (pos_probs >= float(threshold)).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    acc = float((tp + tn) / max(len(y_true), 1))
    sen = float(tp / max(tp + fn, 1))
    spe = float(tn / max(tn + fp, 1))
    f1_pos = float(2 * tp / max(2 * tp + fp + fn, 1))
    f1_neg = float(2 * tn / max(2 * tn + fn + fp, 1))
    f1 = (f1_pos + f1_neg) / 2.0
    balanced_acc = (sen + spe) / 2.0
    try:
        auroc = float(roc_auc_score(y_true, pos_probs))
    except Exception:
        auroc = np.nan
    try:
        auprc = float(average_precision_score(y_true, pos_probs))
    except Exception:
        auprc = np.nan
    return {
        "acc": acc,
        "f1": f1,
        "sen": sen,
        "spe": spe,
        "balanced_acc": balanced_acc,
        "auroc": auroc,
        "auprc": auprc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def best_f1_threshold(y_true, probs):
    best_threshold, best_metrics, best_f1 = 0.5, None, -np.inf
    for threshold in np.linspace(0.01, 0.99, 99):
        metrics = classification_threshold_metrics(y_true, probs, threshold)
        score = metrics["f1"]
        if np.isfinite(score) and score > best_f1:
            best_threshold, best_metrics, best_f1 = float(threshold), metrics, float(score)
    if best_metrics is None:
        best_metrics = classification_threshold_metrics(y_true, probs, 0.5)
    return best_threshold, best_metrics


def format_cls_metrics(metrics):
    return (
        f"Acc: {metrics['acc']:.6f} F1: {metrics['f1']:.6f} "
        f"BAcc: {metrics['balanced_acc']:.6f} Sen: {metrics['sen']:.6f} "
        f"Spe: {metrics['spe']:.6f} AUROC: {metrics['auroc']:.6f} "
        f"AUPRC: {metrics['auprc']:.6f} TP: {metrics['tp']} TN: {metrics['tn']} "
        f"FP: {metrics['fp']} FN: {metrics['fn']}"
    )


def aggregate_subject_predictions(y_true, preds, subjects, task_type):
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)
    subjects = np.asarray(subjects).astype(str)
    out_y, out_pred = [], []
    for subject in sorted(set(subjects.tolist())):
        idx = subjects == subject
        if "regression" in str(task_type):
            out_y.append(float(np.mean(y_true[idx])))
            out_pred.append(float(np.mean(preds[idx])))
        else:
            labels = y_true[idx].astype(int)
            values, counts = np.unique(labels, return_counts=True)
            out_y.append(int(values[np.argmax(counts)]))
            if preds.ndim == 2:
                out_pred.append(np.mean(preds[idx], axis=0))
            else:
                out_pred.append(float(np.mean(preds[idx])))
    return np.asarray(out_y), np.asarray(out_pred)
