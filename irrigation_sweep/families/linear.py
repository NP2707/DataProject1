from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from irrigation_sweep.config import settings
from irrigation_sweep.families.base import BaseFamily


class LinearFamily(BaseFamily):
    family_name = "linear"

    def build(self, config: dict[str, Any]) -> Pipeline:
        params = self.merge_with_defaults(settings.LINEAR_DEFAULTS, config)
        estimator = LogisticRegression(
            C=params["C"],
            solver=params["solver"],
            penalty=params["penalty"],
            max_iter=params["max_iter"],
            class_weight=params["class_weight"],
            n_jobs=-1,
            random_state=settings.SEED,
        )
        return Pipeline(
            steps=[
                ("preprocess", self.build_one_hot_preprocessor(scale_numeric=True)),
                ("model", estimator),
            ]
        )

    def get_search_space(self, trial: Any) -> dict[str, Any]:
        return self.sample_space(trial, settings.LINEAR_SEARCH_SPACE)


__all__ = ["LinearFamily"]
