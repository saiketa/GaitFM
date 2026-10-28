import numpy as np
import matplotlib.pyplot as plt
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    if args.lr_schedule == "constant":
        return
    
    if args.lr_schedule == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lr_schedule == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lr_schedule == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lr_schedule == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    plt.figure()
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.plot(true, label='GroundTruth', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, -1, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def _binary_auc(y_true_binary, y_score):
    y_true_binary = np.asarray(y_true_binary).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true_binary == 1
    neg = y_true_binary == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    sum_pos_ranks = ranks[pos].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _binary_average_precision(y_true_binary, y_score):
    y_true_binary = np.asarray(y_true_binary).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true_binary == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    sorted_true = y_true_binary[order]
    tp_cumsum = np.cumsum(sorted_true == 1)
    ranks = np.arange(1, len(sorted_true) + 1, dtype=np.float64)
    precision = tp_cumsum / ranks
    return float(precision[sorted_true == 1].sum() / n_pos)


def _macro_f1(y_true, y_pred):
    labels = np.union1d(np.unique(y_true), np.unique(y_pred))
    f1_scores = []
    for label in labels:
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(f1_scores)) if f1_scores else float("nan")


def _binary_confusion_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sen = tp / max(tp + fn, 1)
    spe = tn / max(tn + fp, 1)
    pos_precision = tp / max(tp + fp, 1)
    pos_f1 = 2 * pos_precision * sen / max(pos_precision + sen, 1e-12)
    neg_precision = tn / max(tn + fn, 1)
    neg_recall = spe
    neg_f1 = 2 * neg_precision * neg_recall / max(neg_precision + neg_recall, 1e-12)
    f1 = 0.5 * (pos_f1 + neg_f1)
    return {
        "acc": float(acc), "f1": float(f1), "sen": float(sen), "spe": float(spe),
        "balanced_acc": float(0.5 * (sen + spe)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _multiclass_confusion_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    labels = np.arange(int(num_classes))
    tp_total = tn_total = fp_total = fn_total = 0
    sensitivities = []
    specificities = []
    for label in labels:
        true_pos = y_true == label
        pred_pos = y_pred == label
        tp = int(np.sum(true_pos & pred_pos))
        tn = int(np.sum((~true_pos) & (~pred_pos)))
        fp = int(np.sum((~true_pos) & pred_pos))
        fn = int(np.sum(true_pos & (~pred_pos)))
        tp_total += tp
        tn_total += tn
        fp_total += fp
        fn_total += fn
        if tp + fn > 0:
            sensitivities.append(tp / (tp + fn))
        if tn + fp > 0:
            specificities.append(tn / (tn + fp))
    sen = float(np.mean(sensitivities)) if sensitivities else float("nan")
    spe = float(np.mean(specificities)) if specificities else float("nan")
    if np.isnan(sen) or np.isnan(spe):
        balanced_acc = float("nan")
    else:
        balanced_acc = 0.5 * (sen + spe)
    return {
        "acc": float(cal_accuracy(y_pred, y_true)),
        "f1": float(_macro_f1(y_true, y_pred)),
        "sen": sen,
        "spe": spe,
        "balanced_acc": float(balanced_acc),
        "tp": tp_total,
        "tn": tn_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def classification_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_prob.ndim == 1 or y_prob.shape[1] == 1:
        positive_score = y_prob.reshape(-1)
        metrics = _binary_confusion_metrics(y_true, y_pred)
        metrics["auroc"] = _binary_auc(y_true, positive_score)
        metrics["auprc"] = _binary_average_precision(y_true, positive_score)
    elif y_prob.shape[1] == 2:
        positive_score = y_prob[:, 1]
        metrics = _binary_confusion_metrics(y_true, y_pred)
        metrics["auroc"] = _binary_auc(y_true, positive_score)
        metrics["auprc"] = _binary_average_precision(y_true, positive_score)
    else:
        metrics = _multiclass_confusion_metrics(y_true, y_pred, y_prob.shape[1])
        aucs = []
        auprcs = []
        for class_idx in range(y_prob.shape[1]):
            y_true_binary = (y_true == class_idx).astype(np.int64)
            auc = _binary_auc(y_true_binary, y_prob[:, class_idx])
            if not np.isnan(auc):
                aucs.append(auc)
            auprc = _binary_average_precision(y_true_binary, y_prob[:, class_idx])
            if not np.isnan(auprc):
                auprcs.append(auprc)
        metrics["auroc"] = float(np.mean(aucs)) if aucs else float("nan")
        metrics["auprc"] = float(np.mean(auprcs)) if auprcs else float("nan")

    return metrics
