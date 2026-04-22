from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from irrigation_sweep.config import settings
from irrigation_sweep.families.base import BaseFamily


class HistGBMFamily(BaseFamily):
    family_name = "hist_gbm"

    def build(self, config: dict[str, Any]) -> Pipeline:
        params = self.merge_with_defaults(settings.HIST_GBM_DEFAULTS, config)
        estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            max_iter=params["max_iter"],
            class_weight="balanced",
            early_stopping=False,
            random_state=settings.SEED,
        )
        return Pipeline(
            steps=[
                ("preprocess", self.build_one_hot_preprocessor(scale_numeric=False)),
                ("model", estimator),
            ]
        )

    def get_search_space(self, trial: Any) -> dict[str, Any]:
        return self.sample_space(trial, settings.HIST_GBM_SEARCH_SPACE)

    def fit(
        self,
        model: Pipeline,
        X_train,
        y_train,
        X_eval=None,
        y_eval=None,
    ) -> Pipeline:
        del X_eval, y_eval
        with threadpool_limits(limits=settings.HIST_GBM_OPENMP_THREADS, user_api="openmp"):
            model.fit(X_train, y_train)
        return model


__all__ = ["HistGBMFamily"]
