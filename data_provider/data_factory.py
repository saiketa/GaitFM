from data_provider.data_loader import ClinicalGaitLoader, FoGContestWindowLoader, WearGaitPDShardedLoader, \
    LTMMLabWalksLoader, mPowerWalkingProcessedLoader, GSTRIDELoader, COMPWALKACLProcessedLoader, \
    MobiliseDTVSProcessedLoader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

data_dict = {
    'ClinicalGait': ClinicalGaitLoader,
    'FoGContest': FoGContestWindowLoader,
    'WearGaitPDSharded': WearGaitPDShardedLoader,
    'LTMMLabWalks': LTMMLabWalksLoader,
    'mPowerWalkingProcessed': mPowerWalkingProcessedLoader,
    'GSTRIDE': GSTRIDELoader,
    'COMPWALKACLProcessed': COMPWALKACLProcessedLoader,
    'MobiliseDTVSProcessed': MobiliseDTVSProcessedLoader,
}


def _classification_labels(data_set):
    if hasattr(data_set, 'labels_df'):
        labels = data_set.labels_df
        if hasattr(labels, 'iloc'):
            labels = labels.iloc[:, 0].to_numpy()
        labels = np.asarray(labels).reshape(-1).astype(np.int64)
        return labels
    if hasattr(data_set, 'labels'):
        labels = np.asarray(data_set.labels).reshape(-1).astype(np.int64)
        return labels
    return None


def _weighted_sampler(data_set, flag):
    if str(flag).upper() != 'TRAIN':
        return None
    labels = _classification_labels(data_set)
    if labels is None or len(labels) == 0:
        return None
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        return None
    class_weights = {cls: len(labels) / (len(classes) * count) for cls, count in zip(classes, counts)}
    sample_weights = np.asarray([class_weights[label] for label in labels], dtype=np.float64)
    summary = ', '.join(
        f'class_{int(cls)}=count:{int(count)}, weight:{class_weights[cls]:.4f}'
        for cls, count in zip(classes, counts)
    )
    print(f'Using WeightedRandomSampler for TRAIN: {summary}')
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def data_provider(args, flag):
    Data = data_dict[args.data]
    shuffle_flag = False if str(flag).upper() in {'TEST', 'VAL', 'VALID', 'VALIDATION'} else True
    drop_last = False
    batch_size = args.batch_size
    if args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )
        sampler = _weighted_sampler(data_set, flag)

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if sampler is not None else shuffle_flag,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    raise ValueError(f"Unsupported task_name for this simplified FM project: {args.task_name}")
