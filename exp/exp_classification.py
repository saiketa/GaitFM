from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import classification_metrics
import torch
import torch.nn as nn
from torch import optim
import os
import time
import numpy as np
import pdb



class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    @staticmethod
    def _infer_feature_dim(data_set):
        if hasattr(data_set, "feature_df"):
            return int(data_set.feature_df.shape[1])
        if hasattr(data_set, "feature_names"):
            return int(len(data_set.feature_names))
        if hasattr(data_set, "FEATURE_COLUMNS"):
            return int(len(data_set.FEATURE_COLUMNS))
        if hasattr(data_set, "SENSOR_COLUMNS"):
            return int(len(data_set.SENSOR_COLUMNS))
        if hasattr(data_set, "feature_columns"):
            return int(len(data_set.feature_columns))
        sample = data_set[0][0]
        if hasattr(sample, "shape") and len(sample.shape) >= 2:
            return int(sample.shape[-1])
        raise AttributeError("Unable to infer classification input feature dimension from dataset.")

    @staticmethod
    def _infer_num_class(data_set):
        if hasattr(data_set, "class_names"):
            return int(len(data_set.class_names))
        if hasattr(data_set, "labels"):
            return int(len(np.unique(data_set.labels)))
        if hasattr(data_set, "labels_df"):
            return int(len(np.unique(np.asarray(data_set.labels_df).reshape(-1))))
        return 2

    @staticmethod
    def _format_metrics(metrics):
        def fmt_value(value):
            if value is None:
                return "NA"
            try:
                if np.isnan(value):
                    return "NA"
            except TypeError:
                pass
            return f"{float(value):.6f}"

        auroc = metrics.get("auroc", float("nan"))
        auprc = metrics.get("auprc", float("nan"))
        balanced_acc = metrics.get("balanced_acc", float("nan"))
        return (
            "Acc: {acc} F1: {f1} BAcc: {balanced_acc} "
            "Sen: {sen} Spe: {spe} "
            "AUROC: {auroc} AUPRC: {auprc} "
            "TP: {tp} TN: {tn} FP: {fp} FN: {fn}"
        ).format(
            acc=fmt_value(metrics.get("acc", float("nan"))),
            f1=fmt_value(metrics.get("f1", float("nan"))),
            balanced_acc=fmt_value(balanced_acc),
            sen=fmt_value(metrics.get("sen", float("nan"))),
            spe=fmt_value(metrics.get("spe", float("nan"))),
            auroc=fmt_value(auroc),
            auprc=fmt_value(auprc),
            tp=metrics.get("tp", -1),
            tn=metrics.get("tn", -1),
            fp=metrics.get("fp", -1),
            fn=metrics.get("fn", -1),
        )

    def _build_model(self):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, vali_data.max_seq_len, test_data.max_seq_len)
        self.args.num_class = self._infer_num_class(train_data)
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        classifier_parameters = list(model.classifier.parameters()) if hasattr(model, "classifier") else []
        classifier_ids = {id(parameter) for parameter in classifier_parameters}
        encoder_parameters = [
            parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in classifier_ids
        ]
        groups = []
        if encoder_parameters:
            groups.append({"params": encoder_parameters, "lr": self.args.learning_rate})
        if classifier_parameters:
            groups.append({
                "params": [parameter for parameter in classifier_parameters if parameter.requires_grad],
                "lr": getattr(self.args, "classifier_learning_rate", 1e-4),
            })
        return optim.AdamW(groups, weight_decay=getattr(self.args, "weight_decay", 0.01))

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _uses_val_f1_threshold(self):
        return int(getattr(self.args, "num_class", 0)) == 2

    def _threshold_metrics(self, trues, probs, threshold):
        predictions = (probs[:, 1] >= float(threshold)).astype(np.int64)
        return classification_metrics(trues, predictions, probs)

    def _best_f1_threshold(self, trues, probs):
        best_threshold = 0.5
        best_metrics = self._threshold_metrics(trues, probs, best_threshold)
        for threshold in np.linspace(0.01, 0.99, 99):
            metrics = self._threshold_metrics(trues, probs, threshold)
            if metrics["f1"] > best_metrics["f1"]:
                best_threshold = float(threshold)
                best_metrics = metrics
        return best_threshold, best_metrics

    def _subject_ids_for_dataset(self, data_set, expected_len):
        if hasattr(data_set, "rows"):
            rows = getattr(data_set, "rows")
            for column in ("subject", "subject_id", "record_id"):
                if hasattr(rows, "columns") and column in rows.columns:
                    values = rows[column].astype(str).to_numpy()
                    if len(values) == expected_len:
                        return values
        if hasattr(data_set, "sample_subjects"):
            values = np.asarray(getattr(data_set, "sample_subjects")).astype(str)
            if len(values) == expected_len:
                return values
        return None

    def _aggregate_subject_outputs(self, trues, probs, subjects):
        trues = np.asarray(trues).reshape(-1).astype(np.int64)
        probs = np.asarray(probs, dtype=np.float64)
        subjects = np.asarray(subjects).reshape(-1)
        if len(subjects) != len(trues):
            return None, None, None

        subject_true = []
        subject_prob = []
        for subject in sorted(np.unique(subjects).astype(str)):
            indices = np.where(subjects.astype(str) == subject)[0]
            labels = trues[indices]
            values, counts = np.unique(labels, return_counts=True)
            subject_true.append(int(values[np.argmax(counts)]))
            subject_prob.append(probs[indices].mean(axis=0))
        return (
            np.asarray(subject_true, dtype=np.int64),
            np.asarray(subject_prob, dtype=np.float64),
            len(subject_true),
        )

    def _subject_metrics(self, trues, probs, subjects, threshold=None):
        subject_trues, subject_probs, n_subjects = self._aggregate_subject_outputs(trues, probs, subjects)
        if subject_trues is None or n_subjects == 0:
            return None, 0
        if subject_probs.ndim > 1 and subject_probs.shape[1] == 2 and threshold is not None:
            subject_predictions = (subject_probs[:, 1] >= float(threshold)).astype(np.int64)
        else:
            subject_predictions = np.argmax(subject_probs, axis=1)
        return classification_metrics(subject_trues, subject_predictions, subject_probs), n_subjects

    def _best_subject_f1_threshold(self, trues, probs, subjects):
        subject_trues, subject_probs, n_subjects = self._aggregate_subject_outputs(trues, probs, subjects)
        if subject_trues is None or n_subjects == 0:
            return None, None, 0
        best_threshold = 0.5
        best_predictions = (subject_probs[:, 1] >= best_threshold).astype(np.int64)
        best_metrics = classification_metrics(subject_trues, best_predictions, subject_probs)
        for threshold in np.linspace(0.01, 0.99, 99):
            predictions = (subject_probs[:, 1] >= float(threshold)).astype(np.int64)
            metrics = classification_metrics(subject_trues, predictions, subject_probs)
            if metrics["f1"] > best_metrics["f1"]:
                best_threshold = float(threshold)
                best_metrics = metrics
        return best_threshold, best_metrics, n_subjects

    def vali(self, vali_data, vali_loader, criterion, return_outputs=False):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze(-1))
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        probs = probs.cpu().numpy()
        metrics = classification_metrics(trues, predictions, probs)

        self.model.train()
        if return_outputs:
            return total_loss, metrics, trues, probs
        return total_loss, metrics

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        time_now = time.time()
        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        best_val_loss = np.inf
        best_model_path = os.path.join(path, 'checkpoint.pth')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_metrics, val_trues, val_probs = self.vali(
                vali_data, vali_loader, criterion, return_outputs=True
            )
            if self._uses_val_f1_threshold():
                best_threshold, val_threshold_metrics = self._best_f1_threshold(val_trues, val_probs)
                metric_text = "threshold: {0:.2f} | {1}".format(
                    best_threshold,
                    self._format_metrics(val_threshold_metrics),
                )
            else:
                metric_text = self._format_metrics(val_metrics)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} {4}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, metric_text)
            )
            if np.isfinite(vali_loss) and vali_loss < best_val_loss:
                best_val_loss = float(vali_loss)
                torch.save(self.model.state_dict(), best_model_path)
        if not os.path.exists(best_model_path):
            raise RuntimeError("No finite validation loss; no checkpoint selected")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        criterion = self._select_criterion()
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        total_loss = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        probs = probs.cpu().numpy()
        metrics = classification_metrics(trues, predictions, probs)
        test_loss = float(np.average(total_loss)) if total_loss else 0.0
        result_lines = []
        test_subjects = self._subject_ids_for_dataset(test_data, len(trues))
        if self._uses_val_f1_threshold():
            vali_data, vali_loader = self._get_data(flag='VAL')
            _, _, val_trues, val_probs = self.vali(vali_data, vali_loader, criterion, return_outputs=True)
            best_threshold, val_threshold_metrics = self._best_f1_threshold(val_trues, val_probs)
            test_threshold_metrics = self._threshold_metrics(trues, probs, best_threshold)
            result_lines.append(
                'val_loss_best_test sample_level threshold:{:.2f} loss:{} val:{} test:{}'
                .format(
                    best_threshold,
                    test_loss,
                    self._format_metrics(val_threshold_metrics),
                    self._format_metrics(test_threshold_metrics),
                )
            )
            val_subjects = self._subject_ids_for_dataset(vali_data, len(val_trues))
            if val_subjects is not None and test_subjects is not None:
                subject_threshold, subject_val_metrics, n_val_subjects = self._best_subject_f1_threshold(
                    val_trues, val_probs, val_subjects
                )
                subject_test_metrics, n_test_subjects = self._subject_metrics(
                    trues, probs, test_subjects, threshold=subject_threshold
                )
                if subject_test_metrics is not None:
                    result_lines.append(
                        'val_loss_best_test subject_level threshold:{:.2f} '
                        'val_subjects:{} test_subjects:{} val:{} test:{}'
                        .format(
                            subject_threshold,
                            n_val_subjects,
                            n_test_subjects,
                            self._format_metrics(subject_val_metrics),
                            self._format_metrics(subject_test_metrics),
                        )
                    )
            else:
                result_lines.append('val_loss_best_test subject_level unavailable: subject ids not found')
        else:
            result_lines.append(
                'val_loss_best_test sample_level loss:{} {}'.format(test_loss, self._format_metrics(metrics))
            )
            if test_subjects is not None:
                subject_test_metrics, n_test_subjects = self._subject_metrics(trues, probs, test_subjects)
                if subject_test_metrics is not None:
                    result_lines.append(
                        'val_loss_best_test subject_level test_subjects:{} test:{}'
                        .format(n_test_subjects, self._format_metrics(subject_test_metrics))
                    )
            else:
                result_lines.append('val_loss_best_test subject_level unavailable: subject ids not found')

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for line in result_lines:
            print(line)
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        for line in result_lines:
            f.write(line)
            f.write('\n')
        f.write('\n')
        f.write('\n')
        f.close()
        return
