from __future__ import annotations

from typing import Any

from irrigation_sweep.config import settings
from irrigation_sweep.families.base import BaseFamily, PreprocessedEstimator


class XGBoostFamily(BaseFamily):
    family_name = "xgboost"

    def __init__(self) -> None:
        super().__init__()
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - exercised when dependency is missing
            self._estimator_cls = None
            self.set_unavailable(f"xgboost import failed: {type(exc).__name__}: {exc}")
        else:
            self._estimator_cls = XGBClassifier

    def build(self, config: dict[str, Any]) -> PreprocessedEstimator:
        if self._estimator_cls is None:
            raise RuntimeError(self.unavailable_reason)

        params = self.merge_with_defaults(settings.XGBOOST_DEFAULTS, config)
        estimator = self._estimator_cls(
            objective="multi:softprob",
            num_class=len(settings.TARGET_LABELS),
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            tree_method="hist",
            n_jobs=-1,
            eval_metric="mlogloss",
            random_state=settings.SEED,
        )
        return PreprocessedEstimator(
            preprocessor=self.build_one_hot_preprocessor(scale_numeric=False),
            estimator=estimator,
            eval_set_builder=lambda X_eval, y_eval: {
                "eval_set": [(X_eval, y_eval)],
                "verbose": False,
            },
            encode_target=True,
        )

    def fit(self, model, X_train, y_train, X_eval=None, y_eval=None):
        model.fit(X_train, y_train, X_eval=X_eval, y_eval=y_eval)
        return model

    def get_search_space(self, trial: Any) -> dict[str, Any]:
        return self.sample_space(trial, settings.XGBOOST_SEARCH_SPACE)


__all__ = ["XGBoostFamily"]
