from __future__ import annotations

import json
from contextlib import contextmanager

import optuna
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from irrigation_sweep.config import settings
from irrigation_sweep.config.validate import ConfigValidationError, startup_validate
from irrigation_sweep.families.base import BaseFamily
from irrigation_sweep.families.catboost_family import CatBoostFamily
from irrigation_sweep.families.hist_gbm import HistGBMFamily
from irrigation_sweep.families.lightgbm_family import LightGBMFamily
from irrigation_sweep.families.linear import LinearFamily
from irrigation_sweep.families.random_forest import RandomForestFamily
from irrigation_sweep.families.xgboost_family import XGBoostFamily
from irrigation_sweep.preprocessing import frame
from irrigation_sweep.run import run_experiment
from irrigation_sweep.search.allocator import UCB1Allocator
from irrigation_sweep.search.engine import SweepEngine, shortlist_families
from irrigation_sweep.search.stopper import Stopper
from irrigation_sweep.validation import validator


pytestmark = pytest.mark.filterwarnings("ignore")


@contextmanager
def override_settings(**overrides):
    original = {key: getattr(settings, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(settings, key, value)
        yield
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


@pytest.fixture(scope="session")
def training_frame():
    return frame.load_training_frame()


@pytest.fixture(scope="session")
def stratified_1000(training_frame):
    X, y = training_frame
    X_small, _, y_small, _ = train_test_split(
        X,
        y,
        train_size=1000,
        stratify=y,
        random_state=settings.SEED,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_small,
        y_small,
        test_size=0.2,
        stratify=y_small,
        random_state=settings.SEED,
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture(scope="module")
def small_sweep_result():
    with override_settings(
        UCB_INIT_TRIALS=2,
        MAX_TOTAL_TRIALS=15,
        MAX_WALL_CLOCK_HOURS=0.25,
        MIN_TRIALS=5,
    ):
        yield run_experiment(sample_size=5000)


def test_config_validation_catches_mismatch():
    bad_mapping = {
        "Crop_Growth_Stage": {
            "Sowing": 0,
            "Vegetative": 1,
            "Flowering": 2,
            "Harvest": 3,
            "ImpossibleStage": 4,
        }
    }
    with override_settings(ORDINAL_MAPPINGS=bad_mapping):
        with pytest.raises(ConfigValidationError, match="configured values absent from data"):
            startup_validate()

    summary = startup_validate()
    assert "Crop_Growth_Stage" in summary.categorical_values


def test_frame_prep_contracts(training_frame):
    X, y = training_frame

    assert "id" not in X.columns
    assert sorted(X["Mulching_Used"].unique().tolist()) == [0, 1]
    assert pd.api.types.is_integer_dtype(X["Mulching_Used"])

    expected_stage_codes = [
        settings.ORDINAL_MAPPINGS["Crop_Growth_Stage"][stage]
        for stage in settings.ALLOWED_CATEGORIES["Crop_Growth_Stage"]
    ]
    assert expected_stage_codes == sorted(expected_stage_codes)
    assert sorted(X["Crop_Growth_Stage"].unique().tolist()) == expected_stage_codes
    assert pd.api.types.is_integer_dtype(X["Crop_Growth_Stage"])

    for column in settings.UNORDERED_CATEGORICAL_COLS:
        assert str(X[column].dtype) == "category"

    assert set(y.unique().tolist()) == {"Low", "Medium", "High"}

    X2, y2 = frame.load_training_frame()
    pd.testing.assert_frame_equal(X, X2)
    pd.testing.assert_series_equal(y, y2)


def test_family_interface(stratified_1000):
    X_train, X_val, y_train, y_val = stratified_1000
    families = [
        LinearFamily(),
        RandomForestFamily(),
        HistGBMFamily(),
        XGBoostFamily(),
        LightGBMFamily(),
        CatBoostFamily(),
    ]

    names = [family.name() for family in families]
    assert len(names) == len(set(names))
    assert all(name for name in names)

    for family in families:
        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        config = family.get_search_space(trial)
        assert config

        model = family.build(config)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

        if family.name() in {"xgboost", "lightgbm", "catboost"}:
            family.fit(model, X_train, y_train, X_val, y_val)
        else:
            family.fit(model, X_train, y_train)

        preds = family.predict(model, X_val)
        proba = family.predict_proba(model, X_val)
        metrics = family.evaluate(model, X_val, y_val)

        assert preds.ndim == 1
        assert len(preds) == len(X_val)
        assert proba.shape == (len(X_val), 3)
        assert set(metrics) == {"accuracy", "macro_f1", "balanced_accuracy", "per_class_recall"}
        assert str(X_train["Season"].dtype) == "category"


def test_stopper_respects_warmup():
    stopper = Stopper(min_trials=20, patience=15, delta=0.001)
    for _ in range(19):
        stopper.update(0.5)
    assert stopper.should_stop() is False

    for _ in range(15):
        stopper.update(0.5)
        assert stopper.should_stop() is False

    stopper.update(0.5)
    assert stopper.should_stop() is True


def test_allocator_burn_in():
    families = [f"family_{idx}" for idx in range(6)]
    allocator = UCB1Allocator(families, burn_in=4)
    selected = [allocator.select_family() for _ in range(24)]

    counts = {family: selected.count(family) for family in families}
    assert counts == {family: 4 for family in families}

    allocator.record_result("family_0", 0.9)
    allocator.record_result("family_1", 0.7)
    allocator.record_result("family_2", 0.6)
    allocator.record_result("family_3", 0.5)
    allocator.record_result("family_4", 0.4)
    allocator.record_result("family_5", 0.3)
    assert allocator.select_family() == "family_0"

    allocator.mark_quarantined("family_0")
    for _ in range(10):
        assert allocator.select_family() != "family_0"


def test_phase0_holdout_is_locked(training_frame):
    X, y = training_frame
    dev_a, holdout_a = validator.global_split()
    dev_b, holdout_b = validator.global_split()

    assert len(dev_a) + len(holdout_a) == len(X)
    full_props = validator.class_proportions(y)
    for split in (dev_a, holdout_a):
        split_props = validator.class_proportions(split.y)
        for label, full_value in full_props.items():
            assert abs(split_props[label] - full_value) <= 0.01

    assert dev_a.indices.tolist() == dev_b.indices.tolist()
    assert holdout_a.indices.tolist() == holdout_b.indices.tolist()


def test_phase1_subsample_stratification():
    dev_set, _ = validator.global_split()
    train_set, val_set = validator.phase1_split(dev_set, sample_size=150_000, seed=42)

    assert len(train_set) + len(val_set) == 150_000
    dev_props = validator.class_proportions(dev_set.y)
    val_props = validator.class_proportions(val_set.y)
    for label, dev_value in dev_props.items():
        assert abs(val_props[label] - dev_value) <= 0.01


def test_shortlist_criteria():
    scores = {
        "a": 0.82,
        "b": 0.815,
        "c": 0.81,
        "d": 0.77,
        "e": 0.75,
    }
    assert shortlist_families(scores, top_k=3, abs_gap=0.005) == ["a", "b", "c"]
    assert shortlist_families(scores, top_k=3, abs_gap=0.01) == ["a", "b", "c"]


def test_report_json_schema(small_sweep_result):
    json_path = small_sweep_result["json_path"]
    data = json.loads(json_path.read_text(encoding="utf-8"))

    expected_keys = {
        "experiment_id",
        "seed",
        "dataset_hash",
        "package_versions",
        "timestamps",
        "baseline",
        "phase1",
        "phase2",
        "holdout",
    }
    assert expected_keys.issubset(data.keys())

    required_family_keys = {
        "name",
        "best_config",
        "best_score",
        "top_5_configs",
        "score_distribution",
        "convergence_history",
        "train_val_gap",
        "overfit_flag",
        "trial_durations",
        "failed_trial_count",
        "quarantine_status",
    }
    for entry in data["phase1"]["ranked_families"]:
        assert required_family_keys.issubset(entry.keys())


def test_dummy_baseline_is_floor(small_sweep_result):
    report = small_sweep_result["report"]
    console_output = small_sweep_result["console_output"]

    assert "phase1_validation" in report["baseline"]
    assert "holdout" in report["baseline"]
    assert "dummy_baseline" in console_output
    assert report["baseline"]["allocated_trials"] == 0
    assert all(row["name"] != "dummy_baseline" for row in report["phase1"]["ranked_families"])


class AlwaysFailFamily(BaseFamily):
    family_name = "always_fail"

    def build(self, config):
        return object()

    def fit(self, model, X_train, y_train, X_eval=None, y_eval=None):
        raise RuntimeError("forced failure")

    def get_search_space(self, trial):
        return {"dummy": 1}


def test_quarantine_on_consecutive_failures():
    with override_settings(MAX_TOTAL_TRIALS=20, MAX_WALL_CLOCK_HOURS=0.1):
        engine = SweepEngine([AlwaysFailFamily()])
        result = engine.run(sample_size=500)

    row = result["phase1"]["ranked_families"][0]
    assert row["failed_trial_count"] == settings.MAX_CONSECUTIVE_FAILS
    assert row["quarantine_status"]["is_quarantined"] is True
    assert row["quarantine_status"]["quarantine_reason"] == "max_consecutive_failures"
    assert "RuntimeError: forced failure" == row["quarantine_status"]["last_exception"]

    allocator = UCB1Allocator(["always_fail"], burn_in=1)
    allocator.mark_quarantined("always_fail")
    with pytest.raises(RuntimeError):
        allocator.select_family()


def test_phase1_phase2_holdout_not_mixed_in_report(small_sweep_result):
    report = small_sweep_result["report"]
    console_output = small_sweep_result["console_output"]

    assert {"phase1", "phase2", "holdout"}.issubset(report.keys())
    assert all("metrics" not in row for row in report["phase1"]["ranked_families"])
    assert all("metrics" not in row for row in report["phase2"]["ranked_families"])
    assert all("metrics" in row for row in report["holdout"]["ranked_families"])

    assert "Phase 1 Ranking" in console_output
    assert "Phase 2 Ranking" in console_output
    assert "Holdout Ranking" in console_output


def test_safety_caps_terminate_search():
    with override_settings(MAX_TOTAL_TRIALS=10, MAX_WALL_CLOCK_HOURS=0.1):
        engine = SweepEngine([HistGBMFamily()])
        result = engine.run(sample_size=1000)
    assert result["total_trials_run"] == 10
    assert result["stop_reason"] == "max_total_trials"

    with override_settings(MAX_TOTAL_TRIALS=100, MAX_WALL_CLOCK_HOURS=0.001):
        engine = SweepEngine([HistGBMFamily()])
        result = engine.run(sample_size=5000)
    assert result["stop_reason"] == "wall_clock_limit"
    assert "Wall clock limit reached" in result["console_output"]
