from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

SEED = 42
TARGET_COL = "Irrigation_Need"
DROP_COLS = ["id"]
TARGET_LABELS = ["Low", "Medium", "High"]
BASELINE_STRATEGY = "most_frequent"
SCORING_METRIC = "macro_f1"

NUMERIC_COLS = [
    "Soil_pH",
    "Soil_Moisture",
    "Organic_Carbon",
    "Electrical_Conductivity",
    "Temperature_C",
    "Humidity",
    "Rainfall_mm",
    "Sunlight_Hours",
    "Wind_Speed_kmh",
    "Field_Area_hectare",
    "Previous_Irrigation_mm",
]

UNORDERED_CATEGORICAL_COLS = [
    "Soil_Type",
    "Crop_Type",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Region",
]

ORDINAL_COLS = ["Crop_Growth_Stage"]
BINARY_COLS = ["Mulching_Used"]

ORDINAL_MAPPINGS = {
    "Crop_Growth_Stage": {
        "Sowing": 0,
        "Vegetative": 1,
        "Flowering": 2,
        "Harvest": 3,
    }
}

BINARY_MAPPINGS = {
    "Mulching_Used": {
        "No": 0,
        "Yes": 1,
    }
}

ALLOWED_CATEGORIES = {
    "Soil_Type": ["Clay", "Loamy", "Sandy", "Silt"],
    "Crop_Type": ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Wheat"],
    "Crop_Growth_Stage": ["Sowing", "Vegetative", "Flowering", "Harvest"],
    "Season": ["Kharif", "Rabi", "Zaid"],
    "Irrigation_Type": ["Canal", "Drip", "Rainfed", "Sprinkler"],
    "Water_Source": ["Groundwater", "Rainwater", "Reservoir", "River"],
    "Mulching_Used": ["No", "Yes"],
    "Region": ["Central", "East", "North", "South", "West"],
    TARGET_COL: TARGET_LABELS,
}

FEATURE_COLS = (
    NUMERIC_COLS
    + UNORDERED_CATEGORICAL_COLS
    + ORDINAL_COLS
    + BINARY_COLS
)

GLOBAL_HOLDOUT_SIZE = 0.20
PHASE1_SAMPLE_SIZE = 75_000
PHASE1_VAL_SIZE = 0.20
PHASE2_VAL_SIZE = 0.20

SHORTLIST_TOP_K = 3
SHORTLIST_ABS_GAP = 0.002
OVERFIT_GAP_THRESHOLD = 0.03

MIN_TRIALS = 12
PATIENCE = 10
DELTA = 0.001
UCB_INIT_TRIALS = 2
PHASE2_UCB_INIT_TRIALS = 1
PHASE1_TRIAL_FRACTION = 0.50
MAX_TOTAL_TRIALS = 24
MAX_WALL_CLOCK_HOURS = 1.0
MAX_CONSECUTIVE_FAILS = 5
TOP_CONFIGS_TO_KEEP = 5

FAMILY_ORDER = [
    "linear",
    "random_forest",
    "hist_gbm",
    "xgboost",
    "lightgbm",
    "catboost",
]

LINEAR_DEFAULTS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1200,
    "class_weight": "balanced",
}

LINEAR_SEARCH_SPACE = {
    "C": {
        "type": "float",
        "low": 1e-2,
        "high": 30.0,
        "log": True,
    },
    "solver": {
        "type": "categorical",
        "choices": ["lbfgs", "newton-cg"],
    },
    "max_iter": {
        "type": "int",
        "low": 800,
        "high": 1600,
    },
}

RANDOM_FOREST_DEFAULTS = {
    "n_estimators": 120,
    "max_depth": 12,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
}

RANDOM_FOREST_SEARCH_SPACE = {
    "n_estimators": {
        "type": "int",
        "low": 80,
        "high": 180,
    },
    "max_depth": {
        "type": "categorical",
        "choices": [8, 12, 16],
    },
    "min_samples_split": {
        "type": "int",
        "low": 2,
        "high": 20,
    },
    "min_samples_leaf": {
        "type": "int",
        "low": 2,
        "high": 10,
    },
    "max_features": {
        "type": "categorical",
        "choices": ["sqrt", "log2", None],
    },
}

HIST_GBM_DEFAULTS = {
    "learning_rate": 0.08,
    "max_depth": 10,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 30,
    "l2_regularization": 0.0,
    "max_iter": 220,
}
HIST_GBM_OPENMP_THREADS = 1

HIST_GBM_SEARCH_SPACE = {
    "learning_rate": {
        "type": "float",
        "low": 0.02,
        "high": 0.20,
        "log": True,
    },
    "max_depth": {
        "type": "categorical",
        "choices": [None, 6, 10, 14],
    },
    "max_leaf_nodes": {
        "type": "int",
        "low": 15,
        "high": 47,
    },
    "min_samples_leaf": {
        "type": "int",
        "low": 10,
        "high": 80,
    },
    "l2_regularization": {
        "type": "float",
        "low": 1e-8,
        "high": 10.0,
        "log": True,
    },
    "max_iter": {
        "type": "int",
        "low": 120,
        "high": 260,
    },
}

XGBOOST_DEFAULTS = {
    "n_estimators": 180,
    "learning_rate": 0.08,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}

XGBOOST_SEARCH_SPACE = {
    "n_estimators": {
        "type": "int",
        "low": 100,
        "high": 220,
    },
    "learning_rate": {
        "type": "float",
        "low": 0.02,
        "high": 0.20,
        "log": True,
    },
    "max_depth": {
        "type": "int",
        "low": 3,
        "high": 10,
    },
    "min_child_weight": {
        "type": "float",
        "low": 1.0,
        "high": 8.0,
    },
    "subsample": {
        "type": "float",
        "low": 0.6,
        "high": 1.0,
    },
    "colsample_bytree": {
        "type": "float",
        "low": 0.6,
        "high": 1.0,
    },
    "reg_lambda": {
        "type": "float",
        "low": 1e-2,
        "high": 10.0,
        "log": True,
    },
}

LIGHTGBM_DEFAULTS = {
    "n_estimators": 180,
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 0.0,
}

LIGHTGBM_SEARCH_SPACE = {
    "n_estimators": {
        "type": "int",
        "low": 100,
        "high": 220,
    },
    "learning_rate": {
        "type": "float",
        "low": 0.02,
        "high": 0.20,
        "log": True,
    },
    "num_leaves": {
        "type": "int",
        "low": 15,
        "high": 63,
    },
    "max_depth": {
        "type": "categorical",
        "choices": [-1, 4, 6, 8, 10],
    },
    "min_child_samples": {
        "type": "int",
        "low": 10,
        "high": 80,
    },
    "subsample": {
        "type": "float",
        "low": 0.6,
        "high": 1.0,
    },
    "colsample_bytree": {
        "type": "float",
        "low": 0.6,
        "high": 1.0,
    },
}

CATBOOST_DEFAULTS = {
    "iterations": 180,
    "learning_rate": 0.08,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 0.0,
}

CATBOOST_SEARCH_SPACE = {
    "iterations": {
        "type": "int",
        "low": 100,
        "high": 220,
    },
    "learning_rate": {
        "type": "float",
        "low": 0.02,
        "high": 0.20,
        "log": True,
    },
    "depth": {
        "type": "int",
        "low": 4,
        "high": 10,
    },
    "l2_leaf_reg": {
        "type": "float",
        "low": 1.0,
        "high": 10.0,
    },
    "random_strength": {
        "type": "float",
        "low": 1e-2,
        "high": 10.0,
        "log": True,
    },
    "bagging_temperature": {
        "type": "float",
        "low": 0.0,
        "high": 5.0,
    },
}
