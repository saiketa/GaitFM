import os
import json
import numpy as np
import pandas as pd
import glob
import re
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings


VAL_FLAGS = {'VAL', 'VALID', 'VALIDATION'}
TEST_FLAGS = {'TEST'}


def _split_flag(flag):
    name = str(flag).upper()
    if name in VAL_FLAGS:
        return 'val'
    if name in TEST_FLAGS:
        return 'test'
    return 'train'


def _split_items_6_2_2(items, seed):
    items = list(items)
    if len(items) == 0:
        return [], [], []
    if len(items) == 1:
        return items, [], []
    if len(items) == 2:
        return [items[0]], [items[1]], []

    rng = np.random.default_rng(seed)
    order = np.arange(len(items))
    rng.shuffle(order)

    val_count = max(1, int(round(len(items) * 0.2)))
    test_count = max(1, int(round(len(items) * 0.2)))
    if val_count + test_count >= len(items):
        overflow = val_count + test_count - len(items) + 1
        test_count = max(0, test_count - overflow)
    val_indices = set(order[:val_count])
    test_indices = set(order[val_count:val_count + test_count])

    train = [item for idx, item in enumerate(items) if idx not in val_indices and idx not in test_indices]
    val = [item for idx, item in enumerate(items) if idx in val_indices]
    test = [item for idx, item in enumerate(items) if idx in test_indices]
    return train, val, test


def _is_val_flag(flag):
    return str(flag).upper() in {'VAL', 'VALID', 'VALIDATION'}


def _is_test_flag(flag):
    return str(flag).upper() == 'TEST'


def _split_three(items, seed, val_ratio=0.2, test_ratio=0.2):
    items = list(items)
    rng = np.random.default_rng(seed)
    order = np.arange(len(items))
    rng.shuffle(order)
    if len(items) <= 1:
        return items, [], []

    val_count = max(1, int(round(len(items) * val_ratio))) if val_ratio > 0 else 0
    test_count = max(1, int(round(len(items) * test_ratio))) if test_ratio > 0 else 0
    if val_count + test_count >= len(items):
        overflow = val_count + test_count - (len(items) - 1)
        while overflow > 0 and test_count > 0:
            test_count -= 1
            overflow -= 1
        while overflow > 0 and val_count > 0:
            val_count -= 1
            overflow -= 1

    val_indices = set(order[:val_count])
    test_indices = set(order[val_count:val_count + test_count])
    train, val, test = [], [], []
    for idx, item in enumerate(items):
        if idx in val_indices:
            val.append(item)
        elif idx in test_indices:
            test.append(item)
        else:
            train.append(item)
    return train, val, test


def _select_three_way_split(train_items, val_items, test_items, flag):
    if _is_test_flag(flag):
        return test_items
    if _is_val_flag(flag):
        return val_items
    return train_items


class ClinicalGaitLoader(Dataset):

    FEATURE_COLUMNS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z']
    DISEASE_ROOTS = {
        'ACL': ('ortho', 'ACL'),
        'CIPN': ('neuro', 'CIPN'),
        'CVA': ('neuro', 'CVA'),
        'HOA': ('ortho', 'HOA'),
        'KOA': ('ortho', 'KOA'),
        'PD': ('neuro', 'PD'),
        'RIL': ('neuro', 'RIL'),
    }

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag or 'TRAIN'
        self.source_rate = getattr(args, 'clinical_source_rate', 100)
        self.target_rate = getattr(args, 'clinical_target_rate', 50)
        self.window_seconds = getattr(args, 'clinical_window_seconds', 1.0)
        self.window_size = int(round(self.target_rate * self.window_seconds))
        self.val_ratio = getattr(args, 'clinical_val_ratio', 0.2)
        self.test_ratio = getattr(args, 'clinical_test_ratio', 0.2)
        self.split_seed = getattr(args, 'clinical_split_seed', 2021)
        self.disease = getattr(args, 'clinical_disease', None)
        if not self.disease:
            raise ValueError('ClinicalGait requires --clinical_disease, e.g. KOA, PD, CVA.')
        self.disease = self.disease.upper()
        if self.disease not in self.DISEASE_ROOTS:
            valid = ', '.join(sorted(self.DISEASE_ROOTS))
            raise ValueError(f'Unsupported ClinicalGait disease "{self.disease}". Choose from: {valid}.')
        if self.window_size <= 0:
            raise ValueError('ClinicalGait window size must be positive.')

        self.class_names = np.array(['healthy', self.disease])
        self.max_seq_len = self.window_size
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=self.flag)
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df
        print(f'ClinicalGait {self.disease} {self.flag}: {len(self.all_IDs)} samples')

    def load_all(self, root_path, file_list=None, flag=None):
        samples = self._build_split(root_path, flag)
        if not samples:
            raise ValueError(f'No ClinicalGait samples found for disease {self.disease} and flag {flag}.')

        frames = []
        labels = []
        subjects = []
        for sample_id, (window, label, subject) in enumerate(samples):
            frames.append(pd.DataFrame(window, columns=self.FEATURE_COLUMNS,
                                       index=pd.Series([sample_id] * len(window))))
            labels.append(label)
            subjects.append(subject)

        all_df = pd.concat(frames, axis=0)
        labels_df = pd.DataFrame(labels, dtype=np.int64)
        self.sample_subjects = np.asarray(subjects, dtype=object)
        return all_df, labels_df

    def _build_split(self, root_path, flag):
        train_subjects, val_subjects, test_subjects = self._split_subjects(root_path)
        subjects = _select_three_way_split(train_subjects, val_subjects, test_subjects, flag)
        samples = self._load_samples(subjects)

        train_samples = self._load_samples(train_subjects)
        train_matrix = np.concatenate([sample for sample, _, _ in train_samples], axis=0)
        mean = train_matrix.mean(axis=0)
        std = train_matrix.std(axis=0) + np.finfo(np.float32).eps
        return [((sample - mean) / std, label, subject) for sample, label, subject in samples]

    def _split_subjects(self, root_path):
        healthy_root = os.path.join(root_path, 'healthy', 'HS')
        disease_top, disease_name = self.DISEASE_ROOTS[self.disease]
        disease_root = os.path.join(root_path, disease_top, disease_name)
        healthy_subjects = self._subject_dirs(healthy_root, label=0)
        disease_subjects = self._subject_dirs(disease_root, label=1)
        train_h, val_h, test_h = self._split_class_subjects(healthy_subjects, self.split_seed)
        train_d, val_d, test_d = self._split_class_subjects(disease_subjects, self.split_seed + 1)
        return train_h + train_d, val_h + val_d, test_h + test_d

    def _subject_dirs(self, cohort_root, label):
        if not os.path.isdir(cohort_root):
            raise FileNotFoundError(f'ClinicalGait cohort path not found: {cohort_root}')
        subjects = []
        for name in sorted(os.listdir(cohort_root)):
            path = os.path.join(cohort_root, name)
            if os.path.isdir(path) and not name.startswith('.'):
                subjects.append((path, label))
        if not subjects:
            raise ValueError(f'No subject directories found under {cohort_root}')
        return subjects

    def _split_class_subjects(self, subjects, seed):
        return _split_three(subjects, seed, self.val_ratio, self.test_ratio)

    def _load_samples(self, subjects):
        samples = []
        for subject_path, label in subjects:
            subject_id = os.path.basename(subject_path)
            pattern = os.path.join(subject_path, '*', '*_raw_data_*.txt')
            for file_path in sorted(glob.glob(pattern)):
                data = self._read_raw_file(file_path)
                for window in self._window_recording(data):
                    samples.append((window, label, subject_id))
        return samples

    def _read_raw_file(self, file_path):
        df = pd.read_csv(file_path, sep='\t')
        missing = [col for col in self.FEATURE_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f'{file_path} is missing columns: {missing}')
        data = df[self.FEATURE_COLUMNS].apply(pd.to_numeric, errors='coerce')
        data = data.interpolate(method='linear', limit_direction='both').fillna(0.0).to_numpy(dtype=np.float32)
        return self._resample(data)

    def _resample(self, data):
        if self.source_rate == self.target_rate:
            return data
        if self.source_rate % self.target_rate == 0:
            return data[::self.source_rate // self.target_rate]
        duration = data.shape[0] / float(self.source_rate)
        old_t = np.arange(data.shape[0], dtype=np.float32) / float(self.source_rate)
        new_len = int(np.floor(duration * self.target_rate))
        new_t = np.arange(new_len, dtype=np.float32) / float(self.target_rate)
        resampled = np.empty((new_len, data.shape[1]), dtype=np.float32)
        for col in range(data.shape[1]):
            resampled[:, col] = np.interp(new_t, old_t, data[:, col])
        return resampled

    def _window_recording(self, data):
        usable_len = (len(data) // self.window_size) * self.window_size
        if usable_len < self.window_size:
            return []
        data = data[:usable_len]
        return data.reshape(-1, self.window_size, data.shape[1])

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values.astype(np.float32)
        labels = self.labels_df.loc[self.all_IDs[ind]].values.astype(np.int64)
        return torch.from_numpy(batch_x), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class FoGContestWindowLoader(Dataset):

    SENSOR_COLUMNS = ['AccV', 'AccML', 'AccAP']
    FOG_COLUMNS = ['StartHesitation', 'Turn', 'Walking']
    VALID_TYPES = {'tdcsfog', 'defog', 'notype'}
    TYPE_SOURCE_RATES = {'tdcsfog': 128.0, 'defog': 100.0, 'notype': 100.0}

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag or 'TRAIN'
        self.window_size = int(getattr(args, 'seq_len', 256))
        stride = getattr(args, 'fog_stride', None)
        self.stride = self.window_size if stride is None else int(stride)
        self.target_rate = float(getattr(args, 'fog_target_rate', 50))
        self.val_ratio = getattr(args, 'fog_val_ratio', 0.2)
        self.test_ratio = getattr(args, 'fog_test_ratio', 0.2)
        self.split_seed = getattr(args, 'fog_split_seed', 2021)
        self.filter_valid_task = bool(getattr(args, 'fog_filter_valid_task', 1))
        self.types = [item.lower() for item in getattr(args, 'fog_types', ['tdcsfog', 'defog', 'notype'])]
        invalid = [item for item in self.types if item not in self.VALID_TYPES]
        if invalid:
            valid = ', '.join(sorted(self.VALID_TYPES))
            raise ValueError(f'Unsupported FoGContest type(s) {invalid}. Choose from: {valid}.')
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError('FoGContest window size and stride must be positive.')

        self.class_names = np.array(['NonFoG', 'FoG'])
        self.max_seq_len = self.window_size
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=self.flag)
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df
        print(f'FoGContest {self.flag}: {len(self.all_IDs)} windows, types={self.types}, stride={self.stride}')

    def load_all(self, root_path, file_list=None, flag=None):
        samples = self._build_split(root_path, flag)
        if not samples:
            raise ValueError(f'No FoGContest windows found for flag {flag}.')

        frames = []
        labels = []
        subjects = []
        for sample_id, (window, label, subject) in enumerate(samples):
            frames.append(pd.DataFrame(window, columns=self.SENSOR_COLUMNS,
                                       index=pd.Series([sample_id] * len(window))))
            labels.append(label)
            subjects.append(subject)
        self.sample_subjects = np.asarray(subjects, dtype=object)
        return pd.concat(frames, axis=0), pd.DataFrame(labels, dtype=np.int64)

    def _build_split(self, root_path, flag):
        train_records, val_records, test_records = self._split_records(root_path)
        records = _select_three_way_split(train_records, val_records, test_records, flag)
        samples = self._load_windows(records)

        train_samples = self._load_windows(train_records)
        train_matrix = np.concatenate([sample for sample, _, _ in train_samples], axis=0)
        mean = train_matrix.mean(axis=0)
        std = train_matrix.std(axis=0) + np.finfo(np.float32).eps
        return [((sample - mean) / std, label, subject) for sample, label, subject in samples]

    def _split_records(self, root_path):
        records = self._recording_files(root_path)
        grouped = {}
        for record in records:
            grouped.setdefault(record['subject'], []).append(record)

        subjects = sorted(grouped)
        train_subjects, val_subjects, test_subjects = _split_three(
            subjects, self.split_seed, self.val_ratio, self.test_ratio
        )
        train_subjects = set(train_subjects)
        val_subjects = set(val_subjects)
        test_subjects = set(test_subjects)

        train_records = []
        val_records = []
        test_records = []
        for subject in subjects:
            if subject in val_subjects:
                val_records.extend(grouped[subject])
            elif subject in test_subjects:
                test_records.extend(grouped[subject])
            else:
                train_records.extend(grouped[subject])
        return train_records, val_records, test_records

    def _recording_files(self, root_path):
        records = []
        metadata = self._read_metadata(root_path)
        for fog_type in self.types:
            folder = os.path.join(root_path, 'train', fog_type)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f'FoGContest path not found: {folder}')
            meta = metadata['tdcsfog'] if fog_type == 'tdcsfog' else metadata['defog']
            for path in sorted(glob.glob(os.path.join(folder, '*.csv'))):
                recording_id = os.path.splitext(os.path.basename(path))[0]
                subject = self._subject_for_recording(meta, recording_id) or recording_id
                records.append({
                    'path': path,
                    'id': recording_id,
                    'type': fog_type,
                    'subject': subject,
                })
        if not records:
            raise ValueError(f'No FoGContest training files found under {root_path}.')
        return records

    def _read_metadata(self, root_path):
        metadata = {}
        for name, file_name in [('tdcsfog', 'tdcsfog_metadata.csv'), ('defog', 'defog_metadata.csv')]:
            path = os.path.join(root_path, file_name)
            metadata[name] = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
        return metadata

    def _subject_for_recording(self, meta, recording_id):
        if meta.empty or 'Id' not in meta or 'Subject' not in meta:
            return None
        row = meta[meta['Id'] == recording_id]
        if row.empty:
            return None
        return str(row.iloc[0]['Subject'])

    def _load_windows(self, records):
        samples = []
        for record in records:
            df = pd.read_csv(record['path'])
            df = self._prepare_frame(df, record['type'])
            if len(df) < self.window_size:
                continue
            x = df[self.SENSOR_COLUMNS].to_numpy(dtype=np.float32)
            y = df['FoG'].to_numpy(dtype=np.int8)
            x, y = self._resample_record(x, y, record['type'])
            for start in range(0, len(x) - self.window_size + 1, self.stride):
                end = start + self.window_size
                samples.append((x[start:end], int(y[start:end].max()), record['subject']))
        return samples

    def _resample_record(self, x, y, fog_type):
        source_rate = self.TYPE_SOURCE_RATES.get(fog_type, self.target_rate)
        if source_rate == self.target_rate:
            return x, y
        if len(x) < 2:
            return x, y
        duration = (len(x) - 1) / float(source_rate)
        new_len = int(np.floor(duration * self.target_rate)) + 1
        if new_len < 2:
            return x[:new_len], y[:new_len]
        old_t = np.arange(len(x), dtype=np.float64) / float(source_rate)
        new_t = np.arange(new_len, dtype=np.float64) / float(self.target_rate)
        x_new = np.stack([np.interp(new_t, old_t, x[:, ch]) for ch in range(x.shape[1])], axis=1)
        y_new = np.interp(new_t, old_t, y.astype(np.float32)) > 0.5
        return x_new.astype(np.float32), y_new.astype(np.int8)

    def _prepare_frame(self, df, fog_type):
        missing = [col for col in self.SENSOR_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f'FoGContest file is missing columns: {missing}')
        if self.filter_valid_task and fog_type in {'defog', 'notype'} and {'Valid', 'Task'}.issubset(df.columns):
            df = df[df['Valid'].astype(bool) & df['Task'].astype(bool)].reset_index(drop=True)
        data = df[self.SENSOR_COLUMNS].apply(pd.to_numeric, errors='coerce')
        df.loc[:, self.SENSOR_COLUMNS] = data.interpolate(method='linear', limit_direction='both').fillna(0.0)

        if all(col in df.columns for col in self.FOG_COLUMNS):
            df['FoG'] = df[self.FOG_COLUMNS].max(axis=1).astype(np.int8)
        elif 'Event' in df.columns:
            df['FoG'] = df['Event'].astype(np.int8)
        else:
            raise ValueError('FoGContest training files require FoG columns or Event.')
        return df

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values.astype(np.float32)
        labels = self.labels_df.loc[self.all_IDs[ind]].values.astype(np.int64)
        return torch.from_numpy(batch_x), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class GSTRIDELoader(Dataset):

    FEATURE_COLUMNS = [
        'accX[m/s^2]', 'accY[m/s^2]', 'accZ[m/s^2]',
        'gyrX[rad/s]', 'gyrY[rad/s]', 'gyrZ[rad/s]',
    ]

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag or 'TRAIN'
        self.task = str(getattr(args, 'gstride_task', 'frailty')).lower()
        self.target_rate = float(getattr(args, 'gstride_target_rate', 50))
        self.window_seconds = float(getattr(args, 'gstride_window_seconds', 5.0))
        self.window_size = int(round(self.target_rate * self.window_seconds))
        stride = getattr(args, 'gstride_stride', None)
        self.stride = self.window_size if stride is None else int(stride)
        self.val_ratio = float(getattr(args, 'gstride_val_ratio', 0.2))
        self.test_ratio = float(getattr(args, 'gstride_test_ratio', 0.2))
        self.split_seed = int(getattr(args, 'gstride_split_seed', 2021))
        self.followup_labels = str(getattr(args, 'gstride_followup_labels', '') or '')
        self.followup_subject_column = str(getattr(args, 'gstride_followup_subject_column', 'subject_id'))
        self.followup_outcome_column = str(getattr(args, 'gstride_followup_outcome_column', 'subsequent_year_fall'))
        if self.task not in {'frailty', 'subsequent_year_fall', 'pretrain'}:
            raise ValueError('GSTRIDE requires --gstride_task frailty, subsequent_year_fall, or pretrain.')
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError('GSTRIDE window size and stride must be positive.')

        if self.task == 'frailty':
            self.class_names = np.array(['Non-frail', 'Pre-frail', 'Frail'])
        elif self.task == 'subsequent_year_fall':
            self.class_names = np.array(['NoFall', 'Fall'])
        else:
            self.class_names = np.array(['Unlabeled'])
        self.max_seq_len = self.window_size
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=self.flag)
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df
        print(f'GSTRIDE {self.task} {self.flag}: {len(self.all_IDs)} windows, stride={self.stride}')

    def load_all(self, root_path, file_list=None, flag=None):
        samples = self._build_split(root_path, flag)
        if not samples:
            raise ValueError(f'No GSTRIDE windows found for task {self.task} and flag {flag}.')

        frames = []
        labels = []
        subjects = []
        for sample_id, (window, label, subject) in enumerate(samples):
            frames.append(pd.DataFrame(window, columns=self.FEATURE_COLUMNS,
                                       index=pd.Series([sample_id] * len(window))))
            labels.append(label)
            subjects.append(subject)
        self.sample_subjects = np.asarray(subjects, dtype=object)
        return pd.concat(frames, axis=0), pd.DataFrame(labels, dtype=np.int64)

    def _build_split(self, root_path, flag):
        train_records, val_records, test_records = self._split_records(root_path)
        records = _select_three_way_split(train_records, val_records, test_records, flag)
        samples = self._load_windows(records)

        train_samples = self._load_windows(train_records)
        if not train_samples:
            raise ValueError('No GSTRIDE training windows available for normalization.')
        train_matrix = np.concatenate([sample for sample, _, _ in train_samples], axis=0)
        mean = train_matrix.mean(axis=0)
        std = train_matrix.std(axis=0) + np.finfo(np.float32).eps
        return [((sample - mean) / std, label, subject) for sample, label, subject in samples]

    def _split_records(self, root_path):
        all_records = self._all_task_records(root_path)
        if self.task == 'pretrain':
            records = [
                {'subject': record['subject'], 'path': record['path'], 'label': 0}
                for record in all_records
            ]
        else:
            records = [
                {
                    'subject': record['subject'],
                    'path': record['path'],
                    'label': int(record[f'{self.task}_label']),
                }
                for record in all_records
                if record.get(f'{self.task}_label') is not None
            ]
        if not records:
            raise ValueError(f'No GSTRIDE records with {self.task} labels and IMU files found under {root_path}.')

        split_by_subject = self._joint_subject_split(all_records)
        train_records, val_records, test_records = [], [], []
        for record in records:
            split = split_by_subject.get(record['subject'])
            if split == 'val':
                val_records.append(record)
            elif split == 'test':
                test_records.append(record)
            else:
                train_records.append(record)
        return train_records, val_records, test_records

    def _joint_subject_split(self, records):
        train_records, val_records, test_records = _split_three(
            sorted(records, key=lambda item: item['subject']),
            self.split_seed,
            self.val_ratio,
            self.test_ratio,
        )
        split_by_subject = {}
        for item in train_records:
            split_by_subject[item['subject']] = 'train'
        for item in val_records:
            split_by_subject[item['subject']] = 'val'
        for item in test_records:
            split_by_subject[item['subject']] = 'test'
        return split_by_subject

    def _all_task_records(self, root_path):
        base = self._base_dir(root_path)
        table = self._read_table(root_path)
        followup = self._read_followup_labels(root_path) if self.task == 'subsequent_year_fall' else {}
        imu_dir = os.path.join(base, '_IMU')
        if not os.path.isdir(imu_dir):
            raise FileNotFoundError(f'GSTRIDE IMU path not found: {imu_dir}')

        grouped = {}
        for _, row in table.iterrows():
            subject = str(row['ID (VNNN)']).strip()
            path = os.path.join(imu_dir, f'{subject}.csv')
            if not os.path.exists(path):
                continue
            frailty_label = self._label_from_row(row, task='frailty')
            subsequent_year_fall_label = followup.get(subject)
            grouped[subject] = {
                'subject': subject,
                'path': path,
                'frailty_label': None if frailty_label is None else int(frailty_label),
                'subsequent_year_fall_label': None if subsequent_year_fall_label is None else int(subsequent_year_fall_label),
            }
        records = list(grouped.values())
        if not records:
            raise ValueError(f'No GSTRIDE records with labels and IMU files found under {root_path}.')
        return records

    def _records(self, root_path):
        records = []
        for record in self._all_task_records(root_path):
            label = record.get(f'{self.task}_label')
            if label is None:
                continue
            records.append({'subject': record['subject'], 'path': record['path'], 'label': int(label)})
        if not records:
            raise ValueError(f'No GSTRIDE records with labels and IMU files found under {root_path}.')
        return records

    def _base_dir(self, root_path):
        if os.path.isdir(os.path.join(root_path, 'GSTRIDE_database')):
            return os.path.join(root_path, 'GSTRIDE_database')
        return root_path

    def _read_table(self, root_path):
        candidates = [
            os.path.join(root_path, 'GSTRIDE_DDBB.xlsx'),
            os.path.join(os.path.dirname(root_path), 'GSTRIDE_DDBB.xlsx'),
        ]
        table_path = next((path for path in candidates if os.path.exists(path)), None)
        if table_path is None:
            raise FileNotFoundError(f'GSTRIDE_DDBB.xlsx not found under {root_path}')
        df = pd.read_excel(table_path, header=1)
        df = df[pd.to_numeric(df['Test index'], errors='coerce').notna()].copy()
        return df

    def _read_followup_labels(self, root_path):
        candidates = []
        if self.followup_labels:
            candidates.append(self.followup_labels)
        candidates.extend([
            os.path.join(root_path, 'GSTRIDE_subsequent_year_falls.csv'),
            os.path.join(self._base_dir(root_path), 'GSTRIDE_subsequent_year_falls.csv'),
        ])
        path = next((item for item in candidates if os.path.isfile(item)), None)
        if path is None:
            raise FileNotFoundError(
                'GSTRIDE subsequent-year fall labels are required. Provide '
                '--gstride_followup_labels with a CSV containing subject_id and subsequent_year_fall.'
            )
        table = pd.read_csv(path)
        missing = [
            column for column in (self.followup_subject_column, self.followup_outcome_column)
            if column not in table.columns
        ]
        if missing:
            raise ValueError(f'{path} is missing follow-up label columns: {missing}')
        labels = {}
        for _, row in table.iterrows():
            subject = str(row[self.followup_subject_column]).strip()
            value = row[self.followup_outcome_column]
            if not subject or subject.lower() == 'nan' or pd.isna(value):
                continue
            numeric = pd.to_numeric(value, errors='coerce')
            label = None
            if not pd.isna(numeric) and np.isfinite(float(numeric)) and float(numeric) in {0.0, 1.0}:
                label = int(float(numeric))
            else:
                normalized = str(value).strip().upper()
                if normalized in {'YES', 'Y', 'TRUE', 'FALL'}:
                    label = 1
                elif normalized in {'NO', 'N', 'FALSE', 'NOFALL', 'NO_FALL'}:
                    label = 0
            if label is None:
                continue
            if subject in labels and labels[subject] != label:
                raise ValueError(f'Conflicting subsequent-year fall labels for subject {subject} in {path}')
            labels[subject] = label
        if not labels:
            raise ValueError(f'No valid subsequent-year fall labels found in {path}')
        return labels

    def _label_from_row(self, row, task=None):
        task = self.task if task is None else str(task).lower()
        if task != 'frailty':
            raise ValueError(f'Unsupported GSTRIDE label task: {task}')
        value = pd.to_numeric(row['Frailty assessment (index)'], errors='coerce')
        if pd.isna(value):
            return None
        value = int(value)
        if value == 0:
            return 0
        if value in {1, 2}:
            return 1
        if value in {3, 4, 5}:
            return 2
        return None

    def _load_windows(self, records):
        samples = []
        for record in records:
            data = self._read_imu(record['path'])
            if len(data) < self.window_size:
                continue
            for start in range(0, len(data) - self.window_size + 1, self.stride):
                samples.append((data[start:start + self.window_size], record['label'], record['subject']))
        return samples

    def _read_imu(self, path):
        df = pd.read_csv(path)
        missing = [col for col in self.FEATURE_COLUMNS + ['timestamp[s]'] if col not in df.columns]
        if missing:
            raise ValueError(f'{path} is missing columns: {missing}')
        values = df[self.FEATURE_COLUMNS].apply(pd.to_numeric, errors='coerce')
        values = values.interpolate(method='linear', limit_direction='both').fillna(0.0).to_numpy(dtype=np.float32)
        timestamp = pd.to_numeric(df['timestamp[s]'], errors='coerce').to_numpy(dtype=np.float64)
        return self._resample(values, timestamp)

    def _resample(self, values, timestamp):
        valid = np.isfinite(timestamp)
        if valid.sum() < 2:
            return values
        timestamp = timestamp[valid]
        values = values[valid]
        order = np.argsort(timestamp)
        timestamp = timestamp[order]
        values = values[order]
        unique = np.concatenate([[True], np.diff(timestamp) > 0])
        timestamp = timestamp[unique]
        values = values[unique]
        if len(timestamp) < 2:
            return values.astype(np.float32)
        start, end = float(timestamp[0]), float(timestamp[-1])
        duration = end - start
        if duration <= 0:
            return values.astype(np.float32)
        new_len = int(np.floor(duration * self.target_rate)) + 1
        new_t = start + np.arange(new_len, dtype=np.float64) / float(self.target_rate)
        new_t = new_t[new_t <= end]
        resampled = np.stack([
            np.interp(new_t, timestamp, values[:, ch]) for ch in range(values.shape[1])
        ], axis=1)
        return resampled.astype(np.float32)

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values.astype(np.float32)
        labels = self.labels_df.loc[self.all_IDs[ind]].values.astype(np.int64)
        return torch.from_numpy(batch_x), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class WearGaitPDProcessedLoader(Dataset):

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag or 'TRAIN'
        self.test_ratio = getattr(args, 'weargait_test_ratio', 0.2)
        self.split_seed = getattr(args, 'weargait_split_seed', 2021)
        self.body_parts = getattr(args, 'weargait_body_parts', None)
        self.tasks = getattr(args, 'weargait_tasks', None)

        self.class_names = np.array(['Control', 'PD'])
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=self.flag)
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df
        self.max_seq_len = int(self.feature_df.groupby(level=0).size().max())
        print(f'WearGait-PD processed {self.flag}: {len(self.all_IDs)} windows')

    def load_all(self, root_path, file_list=None, flag=None):
        npz_path = os.path.join(root_path, 'windows.npz')
        manifest_path = os.path.join(root_path, 'manifest.csv')
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f'Processed WearGait-PD windows not found: {npz_path}')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f'Processed WearGait-PD manifest not found: {manifest_path}')

        data = np.load(npz_path, allow_pickle=True)
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.int64)
        manifest = pd.read_csv(manifest_path)

        if len(X) != len(y) or len(X) != len(manifest):
            raise ValueError(
                f'Inconsistent processed WearGait-PD shapes: X={len(X)}, y={len(y)}, manifest={len(manifest)}'
            )
        if X.ndim != 3:
            raise ValueError(f'WearGait-PD X must be [N, T, C], got shape {X.shape}')
        if 'subject_id' not in manifest.columns:
            raise ValueError('WearGait-PD manifest requires subject_id for subject-level split.')

        candidate_indices = self._filtered_indices(manifest)
        if len(candidate_indices) == 0:
            raise ValueError('No WearGait-PD windows remain after task/body-part filtering.')

        selected, train_indices = self._select_split(manifest, candidate_indices, flag)
        if len(train_indices) == 0:
            raise ValueError('No train windows found for processed WearGait-PD.')
        if len(selected) == 0:
            raise ValueError(f'No {flag} windows found for processed WearGait-PD.')

        mean = X[train_indices].reshape(-1, X.shape[-1]).mean(axis=0)
        std = X[train_indices].reshape(-1, X.shape[-1]).std(axis=0) + np.finfo(np.float32).eps
        X = (X - mean) / std

        feature_names = self._feature_names(data, X.shape[-1])
        frames = []
        labels = []
        for sample_id, source_idx in enumerate(selected):
            frames.append(pd.DataFrame(X[source_idx], columns=feature_names,
                                       index=pd.Series([sample_id] * X.shape[1])))
            labels.append(int(y[source_idx]))

        return pd.concat(frames, axis=0), pd.DataFrame(labels, dtype=np.int64)

    def _filtered_indices(self, manifest):
        mask = pd.Series(True, index=manifest.index)
        if self.body_parts:
            if 'body_part' not in manifest.columns:
                raise ValueError('--weargait_body_parts was set, but manifest has no body_part column.')
            allowed = {str(item) for item in self.body_parts}
            mask &= manifest['body_part'].astype(str).isin(allowed)
        if self.tasks:
            if 'task' not in manifest.columns:
                raise ValueError('--weargait_tasks was set, but manifest has no task column.')
            allowed = {str(item) for item in self.tasks}
            mask &= manifest['task'].astype(str).isin(allowed)
        return manifest.index[mask].to_numpy()

    def _select_split(self, manifest, candidate_indices, flag):
        use_test = str(flag).upper() in {'TEST', 'VAL', 'VALID', 'VALIDATION'}
        if 'split' in manifest.columns:
            split = 'test' if use_test else 'train'
            split_mask = manifest['split'].astype(str).str.lower() == split
            selected = np.intersect1d(candidate_indices, manifest.index[split_mask].to_numpy())
            train_mask = manifest['split'].astype(str).str.lower() == 'train'
            train_indices = np.intersect1d(candidate_indices, manifest.index[train_mask].to_numpy())
            return selected, train_indices

        train_subjects, test_subjects = self._split_subjects(manifest.loc[candidate_indices])
        selected_subjects = test_subjects if use_test else train_subjects
        selected = manifest.index[
            manifest.index.isin(candidate_indices)
            & manifest['subject_id'].astype(str).isin(selected_subjects)
        ].to_numpy()
        train_indices = manifest.index[
            manifest.index.isin(candidate_indices)
            & manifest['subject_id'].astype(str).isin(train_subjects)
        ].to_numpy()
        return selected, train_indices

    def _split_subjects(self, filtered_manifest):
        subject_labels = filtered_manifest[['subject_id', 'label']].drop_duplicates()
        if subject_labels['subject_id'].duplicated().any():
            raise ValueError('A WearGait-PD subject has multiple labels in manifest.')

        train_subjects = []
        test_subjects = []
        rng = np.random.default_rng(self.split_seed)
        for label in sorted(subject_labels['label'].unique()):
            subjects = subject_labels[subject_labels['label'] == label]['subject_id'].astype(str).to_numpy()
            subjects = np.sort(subjects)
            order = np.arange(len(subjects))
            rng.shuffle(order)
            test_count = max(1, int(round(len(subjects) * self.test_ratio))) if len(subjects) > 1 else 0
            test_idx = set(order[:test_count])
            for idx, subject in enumerate(subjects):
                if idx in test_idx:
                    test_subjects.append(subject)
                else:
                    train_subjects.append(subject)
        if not train_subjects and test_subjects:
            train_subjects, test_subjects = test_subjects, []
        return set(train_subjects), set(test_subjects)

    def _feature_names(self, npz_data, n_channels):
        if 'channels' in npz_data:
            channels = [str(item) for item in npz_data['channels']]
            if len(channels) == n_channels:
                return channels
        return [f'ch_{idx}' for idx in range(n_channels)]

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values.astype(np.float32)
        labels = self.labels_df.loc[self.all_IDs[ind]].values.astype(np.int64)
        return torch.from_numpy(batch_x), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class WearGaitPDShardedLoader(Dataset):

    FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = os.path.abspath(root_path)
        self.flag = flag or 'TRAIN'
        self.window_size = int(getattr(args, 'seq_len', 250))
        self.val_ratio = float(getattr(args, 'weargait_val_ratio', 0.2))
        self.test_ratio = float(getattr(args, 'weargait_test_ratio', 0.2))
        self.split_seed = int(getattr(args, 'weargait_split_seed', 2021))
        self.body_parts = getattr(args, 'weargait_body_parts', None)
        self.tasks = getattr(args, 'weargait_tasks', None)
        self.class_names = np.array(['Control', 'PD'])
        self.max_seq_len = self.window_size
        self.feature_names = pd.Index(self.FEATURE_COLUMNS)
        self.load_shards_to_memory = bool(int(getattr(args, 'load_shards_to_memory', 0)))
        self.prefer_sequential = not self.load_shards_to_memory
        self.shard_cache_size = int(getattr(args, 'shard_cache_size', 2))
        self._shard_cache = OrderedDict()
        self.memory_x = None
        self.memory_y = None

        full_manifest = self._read_manifest()
        split_manifest = self._split_manifest(full_manifest)
        manifest = self._filter_manifest(full_manifest)
        if manifest.empty:
            raise ValueError('No sharded WearGait-PD windows remain after filtering.')
        selected = self._select_split(manifest, split_manifest)
        if selected.empty:
            raise ValueError(f'No sharded WearGait-PD windows found for flag {flag}.')
        self.mean, self.std = self._load_or_compute_train_stats(manifest, split_manifest)
        self.rows = selected.reset_index(drop=True)
        self.labels = self.rows['label'].astype(np.int64).to_numpy()
        self.labels_df = pd.DataFrame(self.labels, dtype=np.int64)
        self.all_IDs = np.arange(len(self.rows))
        if self.load_shards_to_memory:
            self.memory_x, self.memory_y, self.rows = self._load_rows_to_memory(self.rows)
            self.labels = self.memory_y.astype(np.int64)
            self.labels_df = pd.DataFrame(self.labels, dtype=np.int64)
            self.all_IDs = np.arange(len(self.memory_y))
            self._shard_cache.clear()
            print(f'WearGait-PD sharded {self.flag}: loaded {len(self.memory_y)} windows into memory')
        print(f'WearGait-PD sharded {self.flag}: {len(self.rows)} windows from {self.root_path}')

    def _read_manifest(self):
        candidates = [
            os.path.join(self.root_path, 'manifest_50hz_5s.csv'),
            os.path.join(self.root_path, 'manifest_50hz_10s.csv'),
        ]
        manifest_path = next((path for path in candidates if os.path.exists(path)), None)
        if manifest_path is None:
            raise FileNotFoundError(f'WearGait-PD sharded manifest not found under: {self.root_path}')
        manifest = pd.read_csv(manifest_path)
        required = {'shard', 'shard_index', 'subject', 'label'}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f'WearGait-PD sharded manifest missing columns: {missing}')
        manifest['subject'] = manifest['subject'].astype(str)
        manifest['label'] = manifest['label'].astype(int)
        manifest['shard_index'] = manifest['shard_index'].astype(int)
        return manifest

    def _filter_manifest(self, manifest):
        mask = pd.Series(True, index=manifest.index)
        if self.body_parts:
            if 'body_part' not in manifest.columns:
                raise ValueError('--weargait_body_parts was set, but manifest has no body_part column.')
            allowed = {str(item) for item in self.body_parts}
            mask &= manifest['body_part'].astype(str).isin(allowed)
        if self.tasks:
            if 'task' not in manifest.columns:
                raise ValueError('--weargait_tasks was set, but manifest has no task column.')
            allowed = {str(item) for item in self.tasks}
            mask &= manifest['task'].astype(str).isin(allowed)
        return manifest[mask].copy()

    def _split_manifest(self, manifest):
        return manifest.copy()

    def _split_subject_sets(self, manifest):
        subject_labels = manifest[['subject', 'label']].drop_duplicates()
        if subject_labels['subject'].duplicated().any():
            raise ValueError('A sharded WearGait-PD subject has multiple labels in manifest.')
        train_subjects, val_subjects, test_subjects = [], [], []
        for label in sorted(subject_labels['label'].unique()):
            subjects = sorted(subject_labels[subject_labels['label'] == label]['subject'].tolist())
            tr, va, te = _split_three(subjects, self.split_seed + int(label), self.val_ratio, self.test_ratio)
            train_subjects.extend(tr)
            val_subjects.extend(va)
            test_subjects.extend(te)
        return set(train_subjects), set(val_subjects), set(test_subjects)

    def _select_split(self, manifest, split_manifest=None):
        split_manifest = manifest if split_manifest is None else split_manifest
        train_subjects, val_subjects, test_subjects = self._split_subject_sets(split_manifest)
        selected_subjects = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects,
        }[_split_flag(self.flag)]
        return manifest[manifest['subject'].isin(selected_subjects)].copy()

    def _train_rows(self, manifest, split_manifest=None):
        split_manifest = manifest if split_manifest is None else split_manifest
        train_subjects, _, _ = self._split_subject_sets(split_manifest)
        return manifest[manifest['subject'].isin(train_subjects)].copy()

    def _stats_path(self):
        suffix = f'sl{self.window_size}_seed{self.split_seed}_global_subject_split'
        if self.body_parts:
            suffix += '_bp' + '-'.join(str(x) for x in self.body_parts)
        if self.tasks:
            suffix += '_task' + '-'.join(str(x) for x in self.tasks)
        return os.path.join(self.root_path, f'train_stats_{suffix}.npz')

    def _load_or_compute_train_stats(self, manifest, split_manifest=None):
        path = self._stats_path()
        if os.path.exists(path):
            stats = np.load(path)
            return stats['mean'].astype(np.float32), stats['std'].astype(np.float32)
        rows = self._train_rows(manifest, split_manifest)
        if rows.empty:
            raise ValueError('No train windows found for sharded WearGait-PD.')
        total_count = 0
        total_sum = np.zeros(len(self.FEATURE_COLUMNS), dtype=np.float64)
        total_sq = np.zeros(len(self.FEATURE_COLUMNS), dtype=np.float64)
        for shard_name, group in rows.groupby('shard'):
            shard = self._load_shard(shard_name)
            x = shard['X'][group['shard_index'].astype(int).to_numpy()]
            if x.shape[1] != self.window_size:
                x = self._resize_windows(x, self.window_size)
            flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
            total_count += flat.shape[0]
            total_sum += flat.sum(axis=0)
            total_sq += np.square(flat).sum(axis=0)
        mean = total_sum / max(total_count, 1)
        var = total_sq / max(total_count, 1) - np.square(mean)
        std = np.sqrt(np.maximum(var, 0.0)) + np.finfo(np.float32).eps
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        np.savez(path, mean=mean, std=std)
        print(f'WearGait-PD sharded train stats saved: {path}')
        return mean, std

    def _load_shard(self, shard_name):
        if shard_name in self._shard_cache:
            shard = self._shard_cache.pop(shard_name)
            self._shard_cache[shard_name] = shard
            return shard
        path = shard_name if os.path.isabs(shard_name) else os.path.join(self.root_path, shard_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'WearGait-PD sharded shard not found: {path}')
        with np.load(path) as data:
            shard = {key: data[key] for key in data.files}
        self._shard_cache[shard_name] = shard
        while len(self._shard_cache) > max(1, self.shard_cache_size):
            self._shard_cache.popitem(last=False)
        return shard

    def _resize_windows(self, x, target_len):
        old_len = x.shape[1]
        if old_len == target_len:
            return x
        old_t = np.linspace(0.0, 1.0, old_len, dtype=np.float32)
        new_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
        out = np.empty((x.shape[0], target_len, x.shape[2]), dtype=np.float32)
        for i in range(x.shape[0]):
            for ch in range(x.shape[2]):
                out[i, :, ch] = np.interp(new_t, old_t, x[i, :, ch])
        return out

    def _load_rows_to_memory(self, rows):
        x_parts = []
        y_parts = []
        row_parts = []
        for shard_name, group in rows.groupby('shard', sort=False):
            shard = self._load_shard(shard_name)
            indices = group['shard_index'].astype(int).to_numpy()
            x = shard['X'][indices].astype(np.float32)
            if x.shape[1] != self.window_size:
                x = self._resize_windows(x, self.window_size)
            x = (x - self.mean) / self.std
            x_parts.append(x.astype(np.float32))
            y_parts.append(group['label'].astype(np.int64).to_numpy())
            row_parts.append(group.copy())
        if not x_parts:
            raise ValueError(f'No rows to load into memory for WearGait-PD {self.flag}.')
        return (
            np.concatenate(x_parts, axis=0),
            np.concatenate(y_parts, axis=0),
            pd.concat(row_parts, axis=0).reset_index(drop=True),
        )

    def __getitem__(self, ind):
        if self.memory_x is not None:
            label = np.asarray([int(self.memory_y[ind])], dtype=np.int64)
            return torch.from_numpy(self.memory_x[ind]), torch.from_numpy(label)
        row = self.rows.iloc[ind]
        shard = self._load_shard(row['shard'])
        window = shard['X'][int(row['shard_index'])].astype(np.float32)
        if window.shape[0] != self.window_size:
            window = self._resize_windows(window[None, ...], self.window_size)[0]
        window = (window - self.mean) / self.std
        label = np.asarray([int(row['label'])], dtype=np.int64)
        return torch.from_numpy(window.astype(np.float32)), torch.from_numpy(label)

    def __len__(self):
        return len(self.rows)


class WearGaitPDSelfPaceLoader(Dataset):

    DEFAULT_BODY_PARTS = [
        'LowerBack', 'R_Wrist', 'L_Wrist', 'R_MidLatThigh', 'L_MidLatThigh',
        'R_LatShank', 'L_LatShank', 'R_DorsalFoot', 'L_DorsalFoot',
        'R_Ankle', 'L_Ankle', 'Xiphoid', 'Forehead'
    ]
    AXES = ['X', 'Y', 'Z']

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag or 'TRAIN'
        self.target_rate = float(getattr(args, 'weargait_target_rate', 50))
        self.window_seconds = float(getattr(args, 'weargait_window_seconds', 2.0))
        self.window_size = int(getattr(args, 'seq_len', round(self.target_rate * self.window_seconds)))
        stride = getattr(args, 'weargait_stride', None)
        self.stride = max(1, self.window_size // 2) if stride is None else int(stride)
        self.val_ratio = float(getattr(args, 'weargait_val_ratio', 0.2))
        self.test_ratio = float(getattr(args, 'weargait_test_ratio', 0.2))
        self.split_seed = int(getattr(args, 'weargait_split_seed', 2021))
        self.body_parts = getattr(args, 'weargait_body_parts', None) or self.DEFAULT_BODY_PARTS
        self.tasks = getattr(args, 'weargait_tasks', None)
        if self.tasks:
            self.tasks = {str(task) for task in self.tasks}
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError('WearGait-PD window size and stride must be positive.')

        self.class_names = np.array(['Control', 'PD'])
        self.max_seq_len = self.window_size
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=self.flag)
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df
        task_msg = 'all tasks' if not self.tasks else ','.join(sorted(self.tasks))
        print(f'WearGait-PD raw {self.flag}: {len(self.all_IDs)} windows, tasks={task_msg}')

    def load_all(self, root_path, file_list=None, flag=None):
        root = os.path.abspath(root_path)
        files = self._raw_files(root)
        subject_labels = {self._subject_from_path(path): self._label_from_subject(self._subject_from_path(path))
                          for path in files}
        selected_subjects, train_subjects = self._select_subjects(subject_labels, flag)
        selected_files = [path for path in files if self._subject_from_path(path) in selected_subjects]
        if not selected_files:
            raise ValueError(f'No WearGait-PD files selected for {flag}.')

        train_windows = self._collect_windows([path for path in files if self._subject_from_path(path) in train_subjects])
        if not train_windows:
            raise ValueError('No train windows found for WearGait-PD.')
        mean = np.concatenate([item[0] for item in train_windows], axis=0).mean(axis=0)
        std = np.concatenate([item[0] for item in train_windows], axis=0).std(axis=0) + np.finfo(np.float32).eps

        selected_windows = self._collect_windows(selected_files)
        if not selected_windows:
            raise ValueError(f'No WearGait-PD windows created for {flag}.')

        frames = []
        labels = []
        columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
        for sample_id, (window, label, _, _) in enumerate(selected_windows):
            window = (window - mean) / std
            frames.append(pd.DataFrame(window.astype(np.float32), columns=columns,
                                       index=pd.Series([sample_id] * self.window_size)))
            labels.append(int(label))
        return pd.concat(frames, axis=0), pd.DataFrame(labels, dtype=np.int64)

    def _raw_files(self, root):
        files = []
        for path in sorted(glob.glob(os.path.join(root, '**', '*.csv'), recursive=True)):
            name = os.path.basename(path)
            if name.lower() in {'subjects.csv', 'files.csv'} or name.startswith('manifest'):
                continue
            task = self._task_from_path(path)
            if not self.tasks or task in self.tasks:
                files.append(path)
        if not files:
            raise FileNotFoundError(f'No WearGait-PD CSV files found in {root}.')
        return files

    def _task_from_path(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem.split('_', 1)[1] if '_' in stem else ''

    def _subject_from_path(self, path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return stem.split('_', 1)[0]

    def _label_from_subject(self, subject_id):
        subject = str(subject_id)
        if '(control)' in subject.lower() or subject.startswith(('HC', 'WHC')):
            return 0
        if subject.startswith(('WPD', 'NLS')):
            return 1
        raise ValueError(f'Cannot infer WearGait-PD label from subject id: {subject_id}')

    def _select_subjects(self, subject_labels, flag):
        split = _split_flag(flag)
        train, val, test = [], [], []
        for label in sorted(set(subject_labels.values())):
            subjects = sorted([sid for sid, y in subject_labels.items() if y == label])
            tr, va, te = _split_three(subjects, self.split_seed + int(label), self.val_ratio, self.test_ratio)
            train.extend(tr)
            val.extend(va)
            test.extend(te)
        split_map = {'train': set(train), 'val': set(val), 'test': set(test)}
        return split_map[split], set(train)

    def _collect_windows(self, files):
        windows = []
        for path in files:
            subject_id = self._subject_from_path(path)
            label = self._label_from_subject(subject_id)
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                warnings.warn(f'Failed to read WearGait-PD file {path}: {exc}')
                continue
            time_sec = self._time_seconds(df)
            for body_part in self.body_parts:
                array = self._body_part_array(df, body_part)
                if array is None:
                    continue
                resampled = self._resample(array, time_sec)
                for start in range(0, len(resampled) - self.window_size + 1, self.stride):
                    windows.append((resampled[start:start + self.window_size], label, subject_id, body_part))
        return windows

    def _time_seconds(self, df):
        if 'Time' not in df.columns:
            return None
        values = df['Time'].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
        values = pd.to_numeric(values, errors='coerce').to_numpy(dtype=np.float64)
        if np.isfinite(values).sum() < 2:
            return None
        return values

    def _body_part_array(self, df, body_part):
        columns = [f'{body_part}_Acc_{axis}' for axis in self.AXES] + [f'{body_part}_Gyr_{axis}' for axis in self.AXES]
        if not all(column in df.columns for column in columns):
            return None
        values = df[columns].apply(pd.to_numeric, errors='coerce')
        values = values.interpolate(limit_direction='both').fillna(0.0).to_numpy(dtype=np.float32)
        if len(values) < 2:
            return None
        return values

    def _resample(self, array, time_sec):
        if time_sec is None:
            source_rate = float(getattr(self.args, 'weargait_source_rate', 100))
            duration = (len(array) - 1) / source_rate
            old_t = np.linspace(0.0, duration, len(array), dtype=np.float64)
        else:
            old_t = np.asarray(time_sec, dtype=np.float64)
            finite = np.isfinite(old_t)
            array = array[finite]
            old_t = old_t[finite]
            if len(old_t) < 2:
                return array.astype(np.float32)
            old_t = old_t - old_t[0]
        duration = old_t[-1]
        if duration <= 0:
            return array.astype(np.float32)
        new_len = int(np.floor(duration * self.target_rate)) + 1
        new_t = np.arange(new_len, dtype=np.float64) / self.target_rate
        resampled = np.stack([np.interp(new_t, old_t, array[:, ch]) for ch in range(array.shape[1])], axis=1)
        return resampled.astype(np.float32)

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values.astype(np.float32)
        labels = self.labels_df.loc[self.all_IDs[ind]].values.astype(np.int64)
        return torch.from_numpy(batch_x), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class LTMMLabWalksLoader(Dataset):

    ACCEL_COLUMNS = ['v_acc', 'ml_acc', 'ap_acc']
    ACCEL_DESCRIPTIONS = ['v-acceleration', 'ml-acceleration', 'ap-acceleration']

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag or 'TRAIN'
        self.window_size = int(getattr(args, 'seq_len', 200))
        self.target_rate = float(getattr(args, 'ltmm_target_rate', 50))
        stride = getattr(args, 'ltmm_stride', None)
        self.stride = self.window_size if stride is None else int(stride)
        self.val_ratio = getattr(args, 'ltmm_val_ratio', 0.2)
        self.test_ratio = getattr(args, 'ltmm_test_ratio', 0.2)
        self.split_seed = getattr(args, 'ltmm_split_seed', 2021)
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError('LTMM LabWalks window size and stride must be positive.')

        self.class_names = np.array(['NonFaller', 'Faller'])
        self.max_seq_len = self.window_size
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=self.flag)
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df
        print(f'LTMM LabWalks {self.flag}: {len(self.all_IDs)} windows, stride={self.stride}')

    def load_all(self, root_path, file_list=None, flag=None):
        train_records, val_records, test_records = self._split_records(root_path)
        records = _select_three_way_split(train_records, val_records, test_records, flag)
        samples = self._load_windows(records)

        train_samples = self._load_windows(train_records)
        if not train_samples:
            raise ValueError('No LTMM LabWalks train windows found.')
        train_matrix = np.concatenate([sample for sample, _, _ in train_samples], axis=0)
        mean = train_matrix.mean(axis=0)
        std = train_matrix.std(axis=0) + np.finfo(np.float32).eps
        samples = [((sample - mean) / std, label, subject) for sample, label, subject in samples]

        if not samples:
            raise ValueError(f'No LTMM LabWalks windows found for flag {flag}.')

        frames = []
        labels = []
        subjects = []
        for sample_id, (window, label, subject) in enumerate(samples):
            frames.append(pd.DataFrame(window, columns=self.ACCEL_COLUMNS,
                                       index=pd.Series([sample_id] * len(window))))
            labels.append(label)
            subjects.append(subject)
        self.sample_subjects = np.asarray(subjects, dtype=object)
        return pd.concat(frames, axis=0), pd.DataFrame(labels, dtype=np.int64)

    def _split_records(self, root_path):
        records = self._recording_files(root_path)
        by_label = {0: [], 1: []}
        for record in records:
            by_label[record['label']].append(record)

        rng = np.random.default_rng(self.split_seed)
        train_records = []
        val_records = []
        test_records = []
        for label, label_records in by_label.items():
            label_records = sorted(label_records, key=lambda item: item['record_id'])
            train, val, test = _split_three(label_records, self.split_seed + label, self.val_ratio, self.test_ratio)
            train_records.extend(train)
            val_records.extend(val)
            test_records.extend(test)
        return train_records, val_records, test_records

    def _recording_files(self, root_path):
        root_path = os.path.abspath(root_path)
        records = []
        for hea_path in sorted(glob.glob(os.path.join(root_path, '*.hea'))):
            record_id = os.path.splitext(os.path.basename(hea_path))[0]
            prefix = record_id[:2].lower()
            if prefix == 'co':
                label = 0
            elif prefix in {'fl', 'fo'}:
                label = 1
            else:
                continue
            dat_path = os.path.join(root_path, f'{record_id}.dat')
            if not os.path.exists(dat_path):
                raise FileNotFoundError(f'LTMM LabWalks data file not found for {record_id}: {dat_path}')
            records.append({
                'record_id': record_id,
                'hea_path': hea_path,
                'dat_path': dat_path,
                'label': label,
            })
        if not records:
            raise ValueError(f'No LTMM LabWalks .hea/.dat records found under {root_path}.')
        return records

    def _load_windows(self, records):
        samples = []
        for record in records:
            data = self._read_record(record['hea_path'], record['dat_path'])
            if len(data) < self.window_size:
                continue
            for start in range(0, len(data) - self.window_size + 1, self.stride):
                end = start + self.window_size
                samples.append((data[start:end], record['label'], record['record_id']))
        return samples

    def _read_record(self, hea_path, dat_path):
        header = self._parse_header(hea_path)
        n_sig = header['n_sig']
        n_samples = header['n_samples']
        signal_specs = header['signals']
        if n_sig < 3:
            raise ValueError(f'LTMM record has fewer than 3 channels: {hea_path}')

        raw = np.fromfile(dat_path, dtype='<i2')
        expected = n_samples * n_sig
        if raw.size < expected:
            raise ValueError(f'LTMM data file is shorter than header expects: {dat_path}')
        raw = raw[:expected].reshape(n_samples, n_sig).astype(np.float32)

        channels = []
        for col_idx, spec in enumerate(signal_specs[:3]):
            gain = spec['gain']
            baseline = spec['baseline']
            if not np.isfinite(gain) or gain == 0:
                gain = 1.0
            channels.append((raw[:, col_idx] - baseline) / gain)
        data = np.stack(channels, axis=1).astype(np.float32)
        return self._resample(data, header['fs'])

    def _resample(self, data, source_rate):
        if float(source_rate) == self.target_rate:
            return data
        if len(data) < 2:
            return data
        duration = (len(data) - 1) / float(source_rate)
        new_len = int(np.floor(duration * self.target_rate)) + 1
        if new_len < 2:
            return data[:new_len]
        old_t = np.arange(len(data), dtype=np.float64) / float(source_rate)
        new_t = np.arange(new_len, dtype=np.float64) / float(self.target_rate)
        resampled = np.stack([np.interp(new_t, old_t, data[:, ch]) for ch in range(data.shape[1])], axis=1)
        return resampled.astype(np.float32)

    def _parse_header(self, hea_path):
        with open(hea_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        first = lines[0].split()
        if len(first) < 4:
            raise ValueError(f'Invalid LTMM header first line: {hea_path}')
        n_sig = int(first[1])
        fs = float(first[2])
        n_samples = int(first[3])
        if int(round(fs)) != 100:
            warnings.warn(f'Unexpected LTMM sampling rate {fs} Hz in {hea_path}')

        signal_specs = []
        for line in lines[1:1 + n_sig]:
            parts = line.split()
            if len(parts) < 9:
                raise ValueError(f'Invalid LTMM signal line in {hea_path}: {line}')
            gain, baseline, unit = self._parse_gain_baseline_unit(parts[2])
            signal_specs.append({
                'fmt': parts[1],
                'gain': gain,
                'baseline': baseline,
                'unit': unit,
                'description': ' '.join(parts[8:]),
            })
        return {
            'n_sig': n_sig,
            'fs': fs,
            'n_samples': n_samples,
            'signals': signal_specs,
        }

    def _parse_gain_baseline_unit(self, token):
        match = re.match(r'([+-]?\d+(?:\.\d+)?)(?:\(([+-]?\d+)\))?(?:/(.*))?$', token)
        if not match:
            return 1.0, 0.0, ''
        gain = float(match.group(1))
        baseline = float(match.group(2)) if match.group(2) is not None else 0.0
        unit = match.group(3) or ''
        return gain, baseline, unit

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values.astype(np.float32)
        labels = self.labels_df.loc[self.all_IDs[ind]].values.astype(np.int64)
        return torch.from_numpy(batch_x), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class mPowerWalkingLoader(Dataset):

    FEATURE_COLUMNS = [
        'user_acc_x', 'user_acc_y', 'user_acc_z',
        'gyro_x', 'gyro_y', 'gyro_z'
    ]
    DEVICE_MOTION_COLUMNS = [
        'deviceMotion_walking_outbound.json.items',
        'deviceMotion_walking_return.json.items',
        'deviceMotion_walking_rest.json.items',
    ]

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = os.path.abspath(root_path)
        self.flag = flag or 'TRAIN'
        self.target_rate = float(getattr(args, 'mpower_target_rate', 50))
        self.window_seconds = float(getattr(args, 'mpower_window_seconds', 5.0))
        self.window_size = int(getattr(args, 'seq_len', round(self.target_rate * self.window_seconds)))
        stride = getattr(args, 'mpower_stride', None)
        self.stride = self.window_size if stride is None else int(stride)
        self.val_ratio = float(getattr(args, 'mpower_val_ratio', 0.2))
        self.test_ratio = float(getattr(args, 'mpower_test_ratio', 0.2))
        self.split_seed = int(getattr(args, 'mpower_split_seed', 2021))
        self.max_records = int(getattr(args, 'mpower_max_records', 0) or 0)
        tasks = getattr(args, 'mpower_tasks', None)
        self.device_motion_columns = list(tasks) if tasks else self.DEVICE_MOTION_COLUMNS
        if self.window_size <= 0 or self.stride <= 0:
            raise ValueError('mPower window size and stride must be positive.')

        self.class_names = np.array(['NonPD', 'PD'])
        self.max_seq_len = self.window_size
        self.all_df, self.labels_df = self.load_all(self.root_path, file_list=file_list, flag=self.flag)
        self.all_IDs = self.all_df.index.unique()
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df
        print(f'mPower WalkingActivity {self.flag}: {len(self.all_IDs)} windows, stride={self.stride}')

    def load_all(self, root_path, file_list=None, flag=None):
        walking_meta = self._read_walking_metadata(root_path)
        label_map = self._read_demographic_labels(root_path)
        records = self._recording_files(root_path, walking_meta, label_map)
        train_records, val_records, test_records = self._split_records(records)
        selected_records = _select_three_way_split(train_records, val_records, test_records, flag)

        train_windows = self._load_windows(train_records)
        if not train_windows:
            raise ValueError('No train windows found for mPower WalkingActivity.')
        train_matrix = np.concatenate([window for window, _ in train_windows], axis=0)
        mean = train_matrix.mean(axis=0)
        std = train_matrix.std(axis=0) + np.finfo(np.float32).eps

        selected_windows = self._load_windows(selected_records)
        if not selected_windows:
            raise ValueError(f'No mPower windows found for flag {flag}.')

        frames = []
        labels = []
        for sample_id, (window, label) in enumerate(selected_windows):
            window = (window - mean) / std
            frames.append(pd.DataFrame(window.astype(np.float32), columns=self.FEATURE_COLUMNS,
                                       index=pd.Series([sample_id] * self.window_size)))
            labels.append(int(label))
        return pd.concat(frames, axis=0), pd.DataFrame(labels, dtype=np.int64)

    def _read_walking_metadata(self, root_path):
        path = os.path.join(root_path, 'WalkingActivity', 'WalkingActivity_metadata.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f'mPower WalkingActivity metadata not found: {path}')
        return pd.read_csv(path)

    def _read_demographic_labels(self, root_path):
        path = os.path.join(root_path, 'Demographics', 'Demographics_metadata.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(f'mPower Demographics metadata not found: {path}')
        df = pd.read_csv(path)
        required = {'healthCode', 'professional-diagnosis'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'mPower Demographics missing columns: {missing}')
        df = df[df['professional-diagnosis'].notna()].copy()
        df['healthCode'] = df['healthCode'].astype(str)
        df['label'] = df['professional-diagnosis'].apply(self._parse_pd_label)
        label_df = df[['healthCode', 'label']].drop_duplicates()
        if label_df['healthCode'].duplicated().any():
            label_df = label_df.groupby('healthCode', as_index=False)['label'].max()
        return dict(zip(label_df['healthCode'], label_df['label']))

    def _parse_pd_label(self, value):
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        text = str(value).strip().lower()
        if text in {'true', '1', 'yes', 'y'}:
            return 1
        if text in {'false', '0', 'no', 'n'}:
            return 0
        raise ValueError(f'Cannot parse mPower professional-diagnosis value: {value}')

    def _recording_files(self, root_path, walking_meta, label_map):
        walking_dir = os.path.join(root_path, 'WalkingActivity')
        file_map = self._load_file_map(root_path, walking_dir)
        fallback_maps = None
        if not file_map:
            fallback_maps = self._fallback_column_file_maps(walking_dir, walking_meta)
            warnings.warn(
                'mPower explicit file map not found. Pairing downloaded deviceMotion files to metadata rows by sorted order. '
                'For strict reproducibility, save a Synapse file map with fileHandleId/path columns.'
            )

        records = []
        for _, row in walking_meta.iterrows():
            health_code = str(row.get('healthCode', ''))
            if health_code not in label_map:
                continue
            label = int(label_map[health_code])
            for column in self.device_motion_columns:
                if column not in walking_meta.columns or pd.isna(row.get(column)):
                    continue
                handle = self._handle_to_str(row[column])
                path = file_map.get(handle)
                if path is None and fallback_maps is not None:
                    path = fallback_maps.get(column, {}).get(handle)
                if path is None or not os.path.exists(path):
                    continue
                records.append({
                    'path': path,
                    'subject': health_code,
                    'record_id': str(row.get('recordId', '')),
                    'task': column.replace('deviceMotion_walking_', '').replace('.json.items', ''),
                    'label': label,
                })
                if self.max_records > 0 and len(records) >= self.max_records:
                    break
            if self.max_records > 0 and len(records) >= self.max_records:
                break
        if not records:
            raise ValueError('No mPower deviceMotion records could be matched to demographic labels.')
        print(f'mPower matched deviceMotion records: {len(records)} subjects={len(set(r["subject"] for r in records))}')
        return records

    def _handle_to_str(self, value):
        if pd.isna(value):
            return ''
        try:
            f = float(value)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
        return str(value).strip()

    def _load_file_map(self, root_path, walking_dir):
        candidates = []
        for folder in [root_path, walking_dir]:
            for name in ['WalkingActivity_file_map.csv', 'file_map.csv', 'synapse_file_map.csv', 'download_file_map.csv']:
                candidates.append(os.path.join(folder, name))
        for path in candidates:
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            lower = {c.lower(): c for c in df.columns}
            handle_col = None
            for key in ['filehandleid', 'file_handle_id', 'handle_id', 'id']:
                if key in lower:
                    handle_col = lower[key]
                    break
            path_col = None
            for key in ['path', 'file_path', 'filepath', 'filename', 'name']:
                if key in lower:
                    path_col = lower[key]
                    break
            if handle_col is None or path_col is None:
                continue
            out = {}
            for _, row in df.iterrows():
                handle = self._handle_to_str(row[handle_col])
                value = str(row[path_col])
                full_path = value if os.path.isabs(value) else os.path.join(walking_dir, value)
                if os.path.exists(full_path):
                    out[handle] = full_path
            if out:
                print(f'mPower loaded file map: {path} entries={len(out)}')
                return out
        return {}

    def _fallback_column_file_maps(self, walking_dir, walking_meta):
        maps = {}
        for column in self.device_motion_columns:
            handles = [self._handle_to_str(v) for v in walking_meta[column].dropna()] if column in walking_meta.columns else []
            files = sorted(glob.glob(os.path.join(walking_dir, f'{column}-*.tmp')))
            count = min(len(handles), len(files))
            maps[column] = {handles[idx]: files[idx] for idx in range(count)}
            print(f'mPower fallback map {column}: handles={len(handles)} files={len(files)} paired={count}')
        return maps

    def _split_records(self, records):
        subject_labels = {}
        for record in records:
            subject_labels[record['subject']] = record['label']
        train_subjects, val_subjects, test_subjects = [], [], []
        for label in sorted(set(subject_labels.values())):
            subjects = sorted([sid for sid, y in subject_labels.items() if y == label])
            tr, va, te = _split_three(subjects, self.split_seed + int(label), self.val_ratio, self.test_ratio)
            train_subjects.extend(tr)
            val_subjects.extend(va)
            test_subjects.extend(te)
        split = _split_flag(self.flag)
        selected = {
            'train': set(train_subjects),
            'val': set(val_subjects),
            'test': set(test_subjects),
        }
        return (
            [r for r in records if r['subject'] in set(train_subjects)],
            [r for r in records if r['subject'] in set(val_subjects)],
            [r for r in records if r['subject'] in set(test_subjects)],
        )

    def _load_windows(self, records):
        windows = []
        for record in records:
            array = self._read_device_motion(record['path'])
            if array is None or len(array) < self.window_size:
                continue
            for start in range(0, len(array) - self.window_size + 1, self.stride):
                end = start + self.window_size
                windows.append((array[start:end], record['label']))
        return windows

    def _read_device_motion(self, path):
        try:
            with open(path, 'r') as f:
                items = json.load(f)
        except Exception as exc:
            warnings.warn(f'Failed to read mPower deviceMotion file {path}: {exc}')
            return None
        if items is None:
            warnings.warn(f'Skipping empty mPower deviceMotion file {path}: JSON content is null.')
            return None
        if isinstance(items, dict):
            for key in ('items', 'data', 'records'):
                if isinstance(items.get(key), list):
                    items = items[key]
                    break
        if not isinstance(items, list):
            warnings.warn(f'Skipping mPower deviceMotion file {path}: expected a JSON list, got {type(items).__name__}.')
            return None
        rows = []
        times = []
        for item in items:
            if not isinstance(item, dict):
                continue
            ts = item.get('timestamp')
            ua = item.get('userAcceleration') or {}
            rr = item.get('rotationRate') or {}
            values = [ua.get(axis) for axis in ['x', 'y', 'z']] + [rr.get(axis) for axis in ['x', 'y', 'z']]
            if ts is None or any(v is None for v in values):
                continue
            try:
                times.append(float(ts))
                rows.append([float(v) for v in values])
            except Exception:
                continue
        if len(rows) < 2:
            return None
        time_sec = np.asarray(times, dtype=np.float64)
        data = np.asarray(rows, dtype=np.float32)
        order = np.argsort(time_sec)
        time_sec = time_sec[order]
        data = data[order]
        _, unique_idx = np.unique(time_sec, return_index=True)
        unique_idx = np.sort(unique_idx)
        time_sec = time_sec[unique_idx]
        data = data[unique_idx]
        time_sec = time_sec - time_sec[0]
        duration = time_sec[-1]
        if duration <= 0:
            return None
        new_len = int(np.floor(duration * self.target_rate)) + 1
        if new_len < self.window_size:
            return None
        new_t = np.arange(new_len, dtype=np.float64) / self.target_rate
        resampled = np.stack([np.interp(new_t, time_sec, data[:, ch]) for ch in range(data.shape[1])], axis=1)
        return resampled.astype(np.float32)

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values.astype(np.float32)
        labels = self.labels_df.loc[self.all_IDs[ind]].values.astype(np.int64)
        return torch.from_numpy(batch_x), torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)


class mPowerWalkingProcessedLoader(Dataset):

    FEATURE_COLUMNS = [
        'user_acc_x', 'user_acc_y', 'user_acc_z',
        'gyro_x', 'gyro_y', 'gyro_z'
    ]

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = os.path.abspath(root_path)
        self.flag = flag or 'TRAIN'
        self.window_size = int(getattr(args, 'seq_len', 250))
        self.val_ratio = float(getattr(args, 'mpower_val_ratio', 0.2))
        self.test_ratio = float(getattr(args, 'mpower_test_ratio', 0.2))
        self.split_seed = int(getattr(args, 'mpower_split_seed', 2021))
        self.tasks = getattr(args, 'mpower_tasks', None)
        self.require_label = bool(int(getattr(args, 'mpower_require_label', 0)))
        self.class_names = np.array(['NonPD', 'PD'])
        self.max_seq_len = self.window_size
        self.feature_names = pd.Index(self.FEATURE_COLUMNS)
        self.load_shards_to_memory = bool(int(getattr(args, 'load_shards_to_memory', 0)))
        self.prefer_sequential = not self.load_shards_to_memory
        self.shard_cache_size = int(getattr(args, 'shard_cache_size', 2))
        self._shard_cache = OrderedDict()
        self.memory_x = None
        self.memory_y = None

        full_manifest = self._read_manifest()
        split_manifest = self._split_manifest(full_manifest)
        manifest = self._filter_manifest(full_manifest)
        if manifest.empty:
            raise ValueError('No preprocessed mPower windows remain after filtering.')
        selected = self._select_split(manifest, split_manifest)
        if selected.empty:
            raise ValueError(f'No preprocessed mPower windows found for flag {flag}.')
        self.mean, self.std = self._load_or_compute_train_stats(manifest, split_manifest)
        self.rows = selected.reset_index(drop=True)
        self.labels = self.rows['label'].astype(np.int64).to_numpy()
        self.labels_df = pd.DataFrame(self.labels, dtype=np.int64)
        self.all_IDs = np.arange(len(self.rows))
        if self.load_shards_to_memory:
            self.memory_x, self.memory_y, self.rows = self._load_rows_to_memory(self.rows)
            self.labels = self.memory_y.astype(np.int64)
            self.labels_df = pd.DataFrame(self.labels, dtype=np.int64)
            self.all_IDs = np.arange(len(self.memory_y))
            self._shard_cache.clear()
            print(f'mPower processed {self.flag}: loaded {len(self.memory_y)} windows into memory')
        print(f'mPower processed {self.flag}: {len(self.rows)} windows from {self.root_path}')

    def _read_manifest(self):
        candidates = [
            os.path.join(self.root_path, 'manifest_50hz_5s.csv'),
            os.path.join(self.root_path, 'manifest_50hz_10s.csv'),
        ]
        manifest_path = next((path for path in candidates if os.path.exists(path)), None)
        if manifest_path is None:
            raise FileNotFoundError(f'mPower processed manifest not found under: {self.root_path}')
        manifest = pd.read_csv(manifest_path)
        required = {'shard', 'shard_index', 'subject', 'label'}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f'mPower processed manifest missing columns: {missing}')
        manifest['subject'] = manifest['subject'].astype(str)
        manifest['label'] = manifest['label'].astype(int)
        manifest['shard_index'] = manifest['shard_index'].astype(int)
        return manifest

    def _filter_manifest(self, manifest):
        mask = pd.Series(True, index=manifest.index)
        if self.tasks:
            if 'task' not in manifest.columns:
                raise ValueError('--mpower_tasks was set, but manifest has no task column.')
            allowed = {str(item) for item in self.tasks}
            mask &= manifest['task'].astype(str).isin(allowed)
        if self.require_label:
            mask &= manifest['label'].astype(int) >= 0
        return manifest[mask].copy()

    def _split_manifest(self, manifest):
        mask = pd.Series(True, index=manifest.index)
        if self.require_label:
            mask &= manifest['label'].astype(int) >= 0
        return manifest[mask].copy()

    def _select_split(self, manifest, split_manifest=None):
        split_manifest = manifest if split_manifest is None else split_manifest
        subject_labels = split_manifest[['subject', 'label']].drop_duplicates()
        if subject_labels['subject'].duplicated().any():
            raise ValueError('A mPower processed subject has multiple labels in split manifest.')
        train_subjects, val_subjects, test_subjects = self._split_subject_sets(subject_labels)
        split = _split_flag(self.flag)
        selected_subjects = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects,
        }[split]
        return manifest[manifest['subject'].isin(selected_subjects)].copy()

    def _split_subject_sets(self, subject_labels):
        subject_labels = subject_labels[['subject', 'label']].drop_duplicates()
        if subject_labels['subject'].duplicated().any():
            raise ValueError('A mPower processed subject has multiple labels in manifest.')
        train_subjects, val_subjects, test_subjects = [], [], []
        for label in sorted(subject_labels['label'].unique()):
            subjects = sorted(subject_labels[subject_labels['label'] == label]['subject'].tolist())
            tr, va, te = _split_three(subjects, self.split_seed + int(label), self.val_ratio, self.test_ratio)
            train_subjects.extend(tr)
            val_subjects.extend(va)
            test_subjects.extend(te)
        return set(train_subjects), set(val_subjects), set(test_subjects)

    def _train_rows(self, manifest, split_manifest=None):
        split_manifest = manifest if split_manifest is None else split_manifest
        subject_labels = split_manifest[['subject', 'label']].drop_duplicates()
        train_subjects, _, _ = self._split_subject_sets(subject_labels)
        return manifest[manifest['subject'].isin(set(train_subjects))].copy()

    def _stats_path(self):
        suffix = f'sl{self.window_size}_seed{self.split_seed}_global_subject_split'
        if self.tasks:
            suffix += '_task' + '-'.join(str(x) for x in self.tasks)
        if self.require_label:
            suffix += '_labeled'
        return os.path.join(self.root_path, f'train_stats_{suffix}.npz')

    def _load_or_compute_train_stats(self, manifest, split_manifest=None):
        path = self._stats_path()
        if os.path.exists(path):
            stats = np.load(path)
            return stats['mean'].astype(np.float32), stats['std'].astype(np.float32)
        rows = self._train_rows(manifest, split_manifest)
        if rows.empty:
            raise ValueError('No train windows found for preprocessed mPower.')
        total_count = 0
        total_sum = np.zeros(len(self.FEATURE_COLUMNS), dtype=np.float64)
        total_sq = np.zeros(len(self.FEATURE_COLUMNS), dtype=np.float64)
        for shard_name, group in rows.groupby('shard'):
            shard = self._load_shard(shard_name)
            x = shard['X'][group['shard_index'].astype(int).to_numpy()]
            if x.shape[1] != self.window_size:
                x = self._resize_windows(x, self.window_size)
            flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
            total_count += flat.shape[0]
            total_sum += flat.sum(axis=0)
            total_sq += np.square(flat).sum(axis=0)
        mean = total_sum / max(total_count, 1)
        var = total_sq / max(total_count, 1) - np.square(mean)
        std = np.sqrt(np.maximum(var, 0.0)) + np.finfo(np.float32).eps
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        np.savez(path, mean=mean, std=std)
        print(f'mPower processed train stats saved: {path}')
        return mean, std

    def _load_shard(self, shard_name):
        if shard_name in self._shard_cache:
            shard = self._shard_cache.pop(shard_name)
            self._shard_cache[shard_name] = shard
            return shard
        path = shard_name if os.path.isabs(shard_name) else os.path.join(self.root_path, shard_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'mPower processed shard not found: {path}')
        with np.load(path) as data:
            shard = {key: data[key] for key in data.files}
        self._shard_cache[shard_name] = shard
        while len(self._shard_cache) > max(1, self.shard_cache_size):
            self._shard_cache.popitem(last=False)
        return shard

    def _resize_windows(self, x, target_len):
        old_len = x.shape[1]
        if old_len == target_len:
            return x
        old_t = np.linspace(0.0, 1.0, old_len, dtype=np.float32)
        new_t = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
        out = np.empty((x.shape[0], target_len, x.shape[2]), dtype=np.float32)
        for i in range(x.shape[0]):
            for ch in range(x.shape[2]):
                out[i, :, ch] = np.interp(new_t, old_t, x[i, :, ch])
        return out

    def _load_rows_to_memory(self, rows):
        x_parts = []
        y_parts = []
        row_parts = []
        for shard_name, group in rows.groupby('shard', sort=False):
            shard = self._load_shard(shard_name)
            indices = group['shard_index'].astype(int).to_numpy()
            x = shard['X'][indices].astype(np.float32)
            if x.shape[1] != self.window_size:
                x = self._resize_windows(x, self.window_size)
            x = (x - self.mean) / self.std
            x_parts.append(x.astype(np.float32))
            y_parts.append(group['label'].astype(np.int64).to_numpy())
            row_parts.append(group.copy())
        if not x_parts:
            raise ValueError(f'No rows to load into memory for mPower {self.flag}.')
        return (
            np.concatenate(x_parts, axis=0),
            np.concatenate(y_parts, axis=0),
            pd.concat(row_parts, axis=0).reset_index(drop=True),
        )

    def __getitem__(self, ind):
        if self.memory_x is not None:
            label = np.asarray([int(self.memory_y[ind])], dtype=np.int64)
            return torch.from_numpy(self.memory_x[ind]), torch.from_numpy(label)
        row = self.rows.iloc[ind]
        shard = self._load_shard(row['shard'])
        window = shard['X'][int(row['shard_index'])].astype(np.float32)
        if window.shape[0] != self.window_size:
            window = self._resize_windows(window[None, ...], self.window_size)[0]
        window = (window - self.mean) / self.std
        label = np.asarray([int(row['label'])], dtype=np.int64)
        return torch.from_numpy(window.astype(np.float32)), torch.from_numpy(label)

    def __len__(self):
        return len(self.rows)


class COMPWALKACLProcessedLoader(mPowerWalkingProcessedLoader):

    FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        setattr(args, 'mpower_val_ratio', float(getattr(args, 'compwalk_val_ratio', 0.2)))
        setattr(args, 'mpower_test_ratio', float(getattr(args, 'compwalk_test_ratio', 0.2)))
        setattr(args, 'mpower_split_seed', int(getattr(args, 'compwalk_split_seed', 2021)))
        super().__init__(args=args, root_path=root_path, file_list=file_list, limit_size=limit_size, flag=flag)
        self.class_names = np.array(['Healthy', 'ACLAbnormal'])
        print(f'COMPWALK-ACL processed {self.flag}: {len(self.rows)} windows from {self.root_path}')

    def _read_manifest(self):
        candidates = [
            os.path.join(self.root_path, 'manifest_50hz_5s.csv'),
            os.path.join(self.root_path, 'manifest_50hz_10s.csv'),
        ]
        manifest_path = next((path for path in candidates if os.path.exists(path)), None)
        if manifest_path is None:
            raise FileNotFoundError(f'COMPWALK-ACL processed manifest not found under: {self.root_path}')
        manifest = pd.read_csv(manifest_path)
        required = {'shard', 'shard_index', 'subject', 'label'}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f'COMPWALK-ACL processed manifest missing columns: {missing}')
        manifest['subject'] = manifest['subject'].astype(str)
        manifest['label'] = manifest['label'].astype(int)
        manifest['shard_index'] = manifest['shard_index'].astype(int)
        return manifest

    def _stats_path(self):
        return os.path.join(self.root_path, f'train_stats_sl{self.window_size}_seed{self.split_seed}.npz')


class MobiliseDTVSProcessedLoader(mPowerWalkingProcessedLoader):

    FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    COHORTS = ('HA', 'PD', 'MS', 'PFF', 'COPD', 'CHF')
    DOWNSTREAM_DISEASES = ('PD', 'MS', 'COPD', 'CHF')

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.subset = str(getattr(args, 'mobilised_subset', 'finetune')).lower()
        self.disease = getattr(args, 'mobilised_disease', None)
        if self.disease is not None:
            self.disease = str(self.disease).upper()
        setattr(args, 'mpower_val_ratio', float(getattr(args, 'mobilised_val_ratio', 0.2)))
        setattr(args, 'mpower_test_ratio', float(getattr(args, 'mobilised_test_ratio', 0.2)))
        setattr(args, 'mpower_split_seed', int(getattr(args, 'mobilised_split_seed', 2021)))
        super().__init__(args=args, root_path=root_path, file_list=file_list, limit_size=limit_size, flag=flag)
        if self.disease:
            self.class_names = np.array(['HA', self.disease])
        else:
            self.class_names = np.array(self.COHORTS)
        print(
            f'Mobilise-D TVS {self.subset} {self.disease or "all"} {self.flag}: '
            f'{len(self.rows)} windows from {self.root_path}'
        )

    def _read_manifest(self):
        candidates = [
            os.path.join(self.root_path, 'manifest_50hz_5s.csv'),
            os.path.join(self.root_path, 'manifest_50hz_10s.csv'),
        ]
        manifest_path = next((path for path in candidates if os.path.exists(path)), None)
        if manifest_path is None:
            raise FileNotFoundError(f'Mobilise-D TVS processed manifest not found under: {self.root_path}')
        manifest = pd.read_csv(manifest_path)
        required = {'shard', 'shard_index', 'subject', 'label', 'cohort'}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f'Mobilise-D TVS processed manifest missing columns: {missing}')
        manifest['subject'] = manifest['subject'].astype(str)
        manifest['cohort'] = manifest['cohort'].astype(str).str.upper()
        manifest['label'] = manifest['label'].astype(int)
        manifest['shard_index'] = manifest['shard_index'].astype(int)
        return manifest

    def _filter_manifest(self, manifest):
        mask = pd.Series(True, index=manifest.index)
        if self.subset == 'pretrain':
            if 'use_for_pretrain' not in manifest.columns:
                raise ValueError('Mobilise-D manifest has no use_for_pretrain column.')
            mask &= manifest['use_for_pretrain'].astype(int) == 1
        elif self.subset == 'finetune':
            if 'use_for_finetune' not in manifest.columns:
                raise ValueError('Mobilise-D manifest has no use_for_finetune column.')
            mask &= manifest['use_for_finetune'].astype(int) == 1
        elif self.subset in {'all', 'any'}:
            pass
        else:
            raise ValueError('--mobilised_subset must be pretrain, finetune, or all.')

        if self.disease:
            if self.disease not in self.DOWNSTREAM_DISEASES:
                raise ValueError(f'Unsupported Mobilise-D disease cohort: {self.disease}')
            mask &= manifest['cohort'].isin(['HA', self.disease])
        out = manifest[mask].copy()
        if self.disease:
            out['label'] = (out['cohort'] == self.disease).astype(int)
        return out

    def _select_split(self, manifest, split_manifest=None):
        split_manifest = manifest if split_manifest is None else split_manifest
        split_by_subject = self._cohort_subject_split(split_manifest)
        split = _split_flag(self.flag)
        selected_subjects = {
            subject for subject, subject_split in split_by_subject.items()
            if subject_split == split
        }
        return manifest[manifest['subject'].isin(selected_subjects)].copy()

    def _train_rows(self, manifest, split_manifest=None):
        split_manifest = manifest if split_manifest is None else split_manifest
        split_by_subject = self._cohort_subject_split(split_manifest)
        train_subjects = {
            subject for subject, subject_split in split_by_subject.items()
            if subject_split == 'train'
        }
        return manifest[manifest['subject'].isin(train_subjects)].copy()

    def _cohort_subject_split(self, manifest):
        if self.disease:
            subject_cohorts = manifest[
                manifest['cohort'].isin(['HA', self.disease])
            ][['subject', 'cohort']].drop_duplicates()
        else:
            subject_cohorts = manifest[['subject', 'cohort']].drop_duplicates()
        if subject_cohorts['subject'].duplicated().any():
            raise ValueError('A Mobilise-D subject appears with multiple cohorts in manifest.')
        split_by_subject = {}
        for cohort in sorted(subject_cohorts['cohort'].unique()):
            subjects = sorted(subject_cohorts[subject_cohorts['cohort'] == cohort]['subject'].tolist())
            seed_offset = sum(ord(ch) for ch in str(cohort))
            train, val, test = _split_three(
                subjects,
                self.split_seed + seed_offset,
                self.val_ratio,
                self.test_ratio,
            )
            for subject in train:
                split_by_subject[subject] = 'train'
            for subject in val:
                split_by_subject[subject] = 'val'
            for subject in test:
                split_by_subject[subject] = 'test'
        return split_by_subject

    def _stats_path(self):
        suffix = f'sl{self.window_size}_seed{self.split_seed}_global_subject_split_{self.subset}'
        if self.disease:
            suffix += f'_{self.disease}'
        return os.path.join(self.root_path, f'train_stats_{suffix}.npz')
