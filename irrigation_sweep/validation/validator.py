from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from irrigation_sweep.config import settings
from irrigation_sweep.preprocessing.frame import load_training_frame


@dataclass(frozen=True)
class DatasetPartition:
    X: pd.DataFrame
    y: pd.Series
    indices: pd.Index

    def __len__(self) -> int:
        return len(self.X)


def _make_partition(X: pd.DataFrame, y: pd.Series, indices) -> DatasetPartition:
    selected_indices = pd.Index(indices)
    return DatasetPartition(
        X=X.loc[selected_indices].copy(),
        y=y.loc[selected_indices].copy(),
        indices=selected_indices.copy(),
    )


def _stratified_indices(
    X: pd.DataFrame,
    y: pd.Series,
    train_size=None,
    test_size=None,
    seed: int = settings.SEED,
) -> tuple[pd.Index, pd.Index]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        test_size=test_size,
        random_state=seed,
    )
    train_idx, test_idx = next(splitter.split(X, y))
    train_indices = X.index[train_idx]
    test_indices = X.index[test_idx]
    return pd.Index(train_indices), pd.Index(test_indices)


def class_proportions(y: pd.Series) -> dict[str, float]:
    normalized = y.value_counts(normalize=True)
    return {label: float(normalized.get(label, 0.0)) for label in settings.TARGET_LABELS}


def sample_partition(
    X: pd.DataFrame,
    y: pd.Series,
    sample_size: int,
    seed: int = settings.SEED,
) -> DatasetPartition:
    if sample_size >= len(X):
        return DatasetPartition(X=X.copy(), y=y.copy(), indices=X.index.copy())
    sampled_idx, _ = _stratified_indices(X, y, train_size=sample_size, seed=seed)
    return _make_partition(X, y, sampled_idx)


def global_split_from_frame(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = settings.SEED,
) -> tuple[DatasetPartition, DatasetPartition]:
    dev_idx, holdout_idx = _stratified_indices(
        X,
        y,
        test_size=settings.GLOBAL_HOLDOUT_SIZE,
        seed=seed,
    )
    return _make_partition(X, y, dev_idx), _make_partition(X, y, holdout_idx)


def global_split(
    seed: int = settings.SEED,
    sample_size: int | None = None,
) -> tuple[DatasetPartition, DatasetPartition]:
    X, y = load_training_frame()
    if sample_size is not None:
        sampled = sample_partition(X, y, sample_size=sample_size, seed=seed)
        return global_split_from_frame(sampled.X, sampled.y, seed=seed)
    return global_split_from_frame(X, y, seed=seed)


def phase1_split(
    dev_set: DatasetPartition,
    sample_size: int = settings.PHASE1_SAMPLE_SIZE,
    seed: int = settings.SEED,
) -> tuple[DatasetPartition, DatasetPartition]:
    if sample_size >= len(dev_set):
        sampled = dev_set
    else:
        sampled_idx, _ = _stratified_indices(
            dev_set.X,
            dev_set.y,
            train_size=sample_size,
            seed=seed,
        )
        sampled = _make_partition(dev_set.X, dev_set.y, sampled_idx)

    train_idx, val_idx = _stratified_indices(
        sampled.X,
        sampled.y,
        test_size=settings.PHASE1_VAL_SIZE,
        seed=seed,
    )
    return _make_partition(sampled.X, sampled.y, train_idx), _make_partition(sampled.X, sampled.y, val_idx)


def phase2_split(
    dev_set: DatasetPartition,
    seed: int = settings.SEED,
) -> tuple[DatasetPartition, DatasetPartition]:
    train_idx, val_idx = _stratified_indices(
        dev_set.X,
        dev_set.y,
        test_size=settings.PHASE2_VAL_SIZE,
        seed=seed,
    )
    return _make_partition(dev_set.X, dev_set.y, train_idx), _make_partition(dev_set.X, dev_set.y, val_idx)


__all__ = [
    "DatasetPartition",
    "class_proportions",
    "global_split",
    "global_split_from_frame",
    "phase1_split",
    "phase2_split",
    "sample_partition",
]
