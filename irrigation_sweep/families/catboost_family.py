from __future__ import annotations

from typing import Any

from irrigation_sweep.config import settings
from irrigation_sweep.families.base import BaseFamily, PreprocessedEstimator


class CatBoostFamily(BaseFamily):
    family_name = "catboost"

    def __init__(self) -> None:
        super().__init__()
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:  # pragma: no cover - exercised when dependency is missing
            self._estimator_cls = None
            self.set_unavailable(f"catboost import failed: {type(exc).__name__}: {exc}")
        else:
            self._estimator_cls = CatBoostClassifier

    def build(self, config: dict[str, Any]) -> PreprocessedEstimator:
        if self._estimator_cls is None:
            raise RuntimeError(self.unavailable_reason)

        params = self.merge_with_defaults(settings.CATBOOST_DEFAULTS, config)
        estimator = self._estimator_cls(
            loss_function="MultiClass",
            iterations=params["iterations"],
            learning_rate=params["learning_rate"],
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_strength=params["random_strength"],
            bagging_temperature=params["bagging_temperature"],
            thread_count=-1,
            random_seed=settings.SEED,
            verbose=False,
        )
        return PreprocessedEstimator(
            preprocessor=self.build_one_hot_preprocessor(scale_numeric=False),
            estimator=estimator,
            eval_set_builder=lambda X_eval, y_eval: {
                "eval_set": (X_eval, y_eval),
                "use_best_model": False,
                "verbose": False,
            },
            encode_target=True,
        )

    def fit(self, model, X_train, y_train, X_eval=None, y_eval=None):
        model.fit(X_train, y_train, X_eval=X_eval, y_eval=y_eval)
        return model

    def get_search_space(self, trial: Any) -> dict[str, Any]:
        return self.sample_space(trial, settings.CATBOOST_SEARCH_SPACE)


__all__ = ["CatBoostFamily"]
