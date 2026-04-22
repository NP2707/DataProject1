from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from irrigation_sweep.config import settings


class ConfigValidationError(ValueError):
    """Raised when config/settings.py does not match the dataset contract."""


@dataclass(frozen=True)
class DatasetSchemaSummary:
    columns: list[str]
    categorical_values: dict[str, list[str]]


def _ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ConfigValidationError(
            "Required dataset files are missing: " + ", ".join(sorted(missing))
        )


def _read_dataset_summary() -> DatasetSchemaSummary:
    schema_df = pd.read_csv(settings.TRAIN_PATH, nrows=0)
    categorical_cols = list(settings.ALLOWED_CATEGORIES.keys())
    cat_df = pd.read_csv(settings.TRAIN_PATH, usecols=categorical_cols)
    categorical_values = {
        col: sorted(cat_df[col].dropna().astype(str).unique().tolist())
        for col in categorical_cols
    }
    return DatasetSchemaSummary(
        columns=schema_df.columns.tolist(),
        categorical_values=categorical_values,
    )


def _validate_columns(columns: list[str]) -> None:
    expected = set(settings.FEATURE_COLS + settings.DROP_COLS + [settings.TARGET_COL])
    observed = set(columns)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if unexpected:
            parts.append(f"unexpected columns: {unexpected}")
        raise ConfigValidationError("Dataset schema mismatch in train.csv: " + "; ".join(parts))


def _validate_test_columns() -> None:
    test_columns = pd.read_csv(settings.TEST_PATH, nrows=0).columns.tolist()
    expected = set(settings.FEATURE_COLS + settings.DROP_COLS)
    observed = set(test_columns)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if unexpected:
            parts.append(f"unexpected columns: {unexpected}")
        raise ConfigValidationError("Dataset schema mismatch in test.csv: " + "; ".join(parts))


def _validate_category_contract(column: str, observed: list[str], expected: list[str]) -> None:
    observed_set = set(observed)
    expected_set = set(expected)
    missing_from_data = sorted(expected_set - observed_set)
    unexpected_in_data = sorted(observed_set - expected_set)
    if missing_from_data or unexpected_in_data:
        parts = []
        if missing_from_data:
            parts.append(f"configured but not present in data: {missing_from_data}")
        if unexpected_in_data:
            parts.append(f"present in data but not configured: {unexpected_in_data}")
        raise ConfigValidationError(
            f"Category contract mismatch for '{column}' in settings.ALLOWED_CATEGORIES: "
            + "; ".join(parts)
        )


def _validate_mapping_contract(mapping_name: str, mappings: dict[str, dict[str, int]]) -> None:
    for column, mapping in mappings.items():
        observed = set(_DATASET_SUMMARY.categorical_values[column])
        configured = set(mapping.keys())
        missing_from_data = sorted(configured - observed)
        missing_from_config = sorted(observed - configured)
        if missing_from_data or missing_from_config:
            parts = []
            if missing_from_data:
                parts.append(f"configured values absent from data: {missing_from_data}")
            if missing_from_config:
                parts.append(f"data values absent from config: {missing_from_config}")
            raise ConfigValidationError(
                f"Mismatch for settings.{mapping_name}['{column}']: " + "; ".join(parts)
            )


def _validate_mapping_values() -> None:
    for column, mapping in settings.ORDINAL_MAPPINGS.items():
        ordered_categories = settings.ALLOWED_CATEGORIES[column]
        ordered_values = [mapping[category] for category in ordered_categories]
        if ordered_values != sorted(ordered_values) or len(set(ordered_values)) != len(ordered_values):
            raise ConfigValidationError(
                f"settings.ORDINAL_MAPPINGS['{column}'] must assign unique, strictly "
                "increasing integer codes in dataset order."
            )

    for column, mapping in settings.BINARY_MAPPINGS.items():
        values = sorted(mapping.values())
        if values != [0, 1]:
            raise ConfigValidationError(
                f"settings.BINARY_MAPPINGS['{column}'] must map to integer values [0, 1]."
            )


def _validate_feature_groups() -> None:
    feature_set = set(settings.FEATURE_COLS)
    grouped = set(
        settings.NUMERIC_COLS
        + settings.UNORDERED_CATEGORICAL_COLS
        + settings.ORDINAL_COLS
        + settings.BINARY_COLS
    )
    if feature_set != grouped:
        raise ConfigValidationError(
            "settings.FEATURE_COLS must exactly match the union of numeric, unordered "
            "categorical, ordinal, and binary columns."
        )


def _validate_target_labels() -> None:
    observed = _DATASET_SUMMARY.categorical_values[settings.TARGET_COL]
    if observed != sorted(settings.TARGET_LABELS):
        raise ConfigValidationError(
            "settings.TARGET_LABELS must match the labels present in train.csv exactly. "
            f"Observed {observed}, configured {sorted(settings.TARGET_LABELS)}."
        )


def startup_validate() -> DatasetSchemaSummary:
    _ensure_paths_exist(
        [
            settings.TRAIN_PATH,
            settings.TEST_PATH,
            settings.SAMPLE_SUBMISSION_PATH,
        ]
    )
    summary = _read_dataset_summary()
    global _DATASET_SUMMARY
    _DATASET_SUMMARY = summary
    _validate_columns(summary.columns)
    _validate_test_columns()
    _validate_feature_groups()
    for column, expected in settings.ALLOWED_CATEGORIES.items():
        _validate_category_contract(column, summary.categorical_values[column], expected)
    _validate_mapping_contract("ORDINAL_MAPPINGS", settings.ORDINAL_MAPPINGS)
    _validate_mapping_contract("BINARY_MAPPINGS", settings.BINARY_MAPPINGS)
    _validate_mapping_values()
    _validate_target_labels()
    return summary


__all__ = [
    "ConfigValidationError",
    "DatasetSchemaSummary",
    "startup_validate",
]
