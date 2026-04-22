from __future__ import annotations

import pandas as pd

from irrigation_sweep.config import settings


def _cast_unordered_categories(frame: pd.DataFrame) -> pd.DataFrame:
    for column in settings.UNORDERED_CATEGORICAL_COLS:
        frame[column] = pd.Categorical(
            frame[column],
            categories=settings.ALLOWED_CATEGORIES[column],
        )
    return frame


def _map_ordinals(frame: pd.DataFrame) -> pd.DataFrame:
    for column, mapping in settings.ORDINAL_MAPPINGS.items():
        frame[column] = frame[column].map(mapping).astype("int64")
    return frame


def _map_binaries(frame: pd.DataFrame) -> pd.DataFrame:
    for column, mapping in settings.BINARY_MAPPINGS.items():
        frame[column] = frame[column].map(mapping).astype("int64")
    return frame


def prepare_frame(df: pd.DataFrame, include_target: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
    working = df.copy(deep=True)
    working = working.drop(columns=settings.DROP_COLS, errors="ignore")
    working = _map_binaries(working)
    working = _map_ordinals(working)
    working = _cast_unordered_categories(working)

    y = None
    if include_target:
        y = working[settings.TARGET_COL].copy()

    X = working[settings.FEATURE_COLS].copy()
    return X, y


def load_training_frame(path: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    csv_path = path or str(settings.TRAIN_PATH)
    df = pd.read_csv(csv_path)
    X, y = prepare_frame(df, include_target=True)
    assert y is not None
    return X, y


def load_test_frame(path: str | None = None) -> pd.DataFrame:
    csv_path = path or str(settings.TEST_PATH)
    df = pd.read_csv(csv_path)
    X, _ = prepare_frame(df, include_target=False)
    return X


__all__ = [
    "load_test_frame",
    "load_training_frame",
    "prepare_frame",
]
