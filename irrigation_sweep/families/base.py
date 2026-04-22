from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from irrigation_sweep.config import settings


class PreprocessedEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        preprocessor: ColumnTransformer,
        estimator: Any,
        eval_set_builder: Callable[[np.ndarray, np.ndarray], dict[str, Any]] | None = None,
        encode_target: bool = False,
    ) -> None:
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.eval_set_builder = eval_set_builder
        self.encode_target = encode_target

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_eval: pd.DataFrame | None = None,
        y_eval: pd.Series | None = None,
    ) -> "PreprocessedEstimator":
        X_train = self.preprocessor.fit_transform(X, y)
        y_train = y
        y_eval_transformed = y_eval
        if self.encode_target:
            self.label_encoder_ = LabelEncoder()
            y_train = self.label_encoder_.fit_transform(y)
            if y_eval is not None:
                y_eval_transformed = self.label_encoder_.transform(y_eval)
            self.classes_ = self.label_encoder_.classes_
        else:
            self.classes_ = np.asarray(pd.Index(y).unique())

        fit_kwargs: dict[str, Any] = {}
        if X_eval is not None and y_eval is not None and self.eval_set_builder is not None:
            X_eval_transformed = self.preprocessor.transform(X_eval)
            fit_kwargs.update(self.eval_set_builder(X_eval_transformed, y_eval_transformed))
        self.estimator.fit(X_train, y_train, **fit_kwargs)
        if not self.encode_target:
            self.classes_ = getattr(self.estimator, "classes_", self.classes_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(X)
        preds = np.asarray(self.estimator.predict(transformed))
        if self.encode_target:
            return self.label_encoder_.inverse_transform(preds.astype(int))
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.preprocessor.transform(X)
        return np.asarray(self.estimator.predict_proba(transformed))


class BaseFamily(ABC):
    family_name = "base"

    def __init__(self) -> None:
        self.available = True
        self.unavailable_reason: str | None = None

    def name(self) -> str:
        return self.family_name

    def is_available(self) -> bool:
        return self.available

    def set_unavailable(self, reason: str) -> None:
        self.available = False
        self.unavailable_reason = reason

    @abstractmethod
    def build(self, config: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_search_space(self, trial: Any) -> dict[str, Any]:
        raise NotImplementedError

    def fit(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame | None = None,
        y_eval: pd.Series | None = None,
    ) -> Any:
        del X_eval, y_eval
        model.fit(X_train, y_train)
        return model

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(model.predict(X))

    def predict_proba(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        proba = np.asarray(model.predict_proba(X))
        model_classes = list(getattr(model, "classes_", settings.TARGET_LABELS))
        if set(model_classes) == set(settings.TARGET_LABELS):
            ordered = [model_classes.index(label) for label in settings.TARGET_LABELS]
            proba = proba[:, ordered]
        return proba

    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        preds = self.predict(model, X)
        per_class = recall_score(
            y,
            preds,
            labels=settings.TARGET_LABELS,
            average=None,
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "per_class_recall": {
                label: float(value)
                for label, value in zip(settings.TARGET_LABELS, per_class, strict=True)
            },
        }

    def merge_with_defaults(
        self,
        defaults: dict[str, Any],
        config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(defaults)
        if config:
            merged.update(config)
        return merged

    def sample_space(
        self,
        trial: Any,
        search_space: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        sampled: dict[str, Any] = {}
        for param, spec in search_space.items():
            kind = spec["type"]
            if kind == "categorical":
                sampled[param] = trial.suggest_categorical(param, spec["choices"])
            elif kind == "int":
                sampled[param] = trial.suggest_int(param, spec["low"], spec["high"])
            elif kind == "float":
                sampled[param] = trial.suggest_float(
                    param,
                    spec["low"],
                    spec["high"],
                    log=spec.get("log", False),
                )
            else:
                raise ValueError(f"Unsupported search space type '{kind}' for '{param}'.")
        return sampled

    def build_one_hot_preprocessor(self, scale_numeric: bool = False) -> ColumnTransformer:
        if scale_numeric:
            numeric_transformer: Any = Pipeline([("scaler", StandardScaler())])
        else:
            numeric_transformer = "passthrough"

        categorical_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, settings.NUMERIC_COLS),
                ("categorical", categorical_encoder, settings.UNORDERED_CATEGORICAL_COLS),
                ("ordinal", "passthrough", settings.ORDINAL_COLS),
                ("binary", "passthrough", settings.BINARY_COLS),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )


__all__ = ["BaseFamily", "PreprocessedEstimator"]
