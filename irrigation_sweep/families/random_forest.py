from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from irrigation_sweep.config import settings
from irrigation_sweep.families.base import BaseFamily


class RandomForestFamily(BaseFamily):
    family_name = "random_forest"

    def build(self, config: dict[str, Any]) -> Pipeline:
        params = self.merge_with_defaults(settings.RANDOM_FOREST_DEFAULTS, config)
        estimator = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            n_jobs=-1,
            random_state=settings.SEED,
        )
        return Pipeline(
            steps=[
                ("preprocess", self.build_one_hot_preprocessor(scale_numeric=False)),
                ("model", estimator),
            ]
        )

    def get_search_space(self, trial: Any) -> dict[str, Any]:
        return self.sample_space(trial, settings.RANDOM_FOREST_SEARCH_SPACE)


__all__ = ["RandomForestFamily"]
