# irrigation_sweep

`irrigation_sweep` is a structured benchmark runner for supervised multiclass classification on the Kaggle Playground Series S6E4 irrigation dataset. It is intentionally not AutoML: it compares a fixed set of model families, performs bounded adaptive search, and produces evidence for a human to review.

## Dataset contract

- Source files live in `data/`.
- `id` is always dropped. It is a synthetic identifier with no predictive value.
- Target column is `Irrigation_Need` with labels `Low`, `Medium`, `High`.
- `Mulching_Used` is deterministically mapped to `0/1` in `preprocessing/frame.py`.
- `Crop_Growth_Stage` is deterministically mapped to monotonically increasing ordinal integers in `preprocessing/frame.py`.
- `Season` is unordered and must remain a pandas `category`, not an ordinal code.
- `frame.py` never fits transforms. Family modules own encoding inside each model object.

## Startup rules

- `config/settings.py` is the single source of truth for paths, column groups, mappings, thresholds, and family defaults/search ranges.
- `config/validate.py` must run before the sweep starts. It checks file presence, train/test schema, categorical vocabularies, and mapping integrity against the dataset.
- If config and data disagree, fail fast with a descriptive error before any search begins.

## Validation stages

1. Global split
   Dev and locked holdout are created by a deterministic stratified split.
2. Phase 1
   All available families compete on a stratified dev sample with UCB1 allocation plus per-family early stopping.
3. Phase 2
   Shortlisted families continue on the full dev partition.
4. Holdout
   Finalists are retrained on dev and scored once on the locked holdout. Holdout is never used for tuning or stopping.

## Family contract

Every family must preserve the `BaseFamily` interface in `families/base.py`.

- `name()` returns a unique non-empty string.
- `build(config)` returns an estimator object with `.fit()`, `.predict()`, and `.predict_proba()`.
- `fit(model, X_train, y_train, X_eval=None, y_eval=None)` accepts optional eval data and silently ignores it if unused.
- `evaluate()` returns a metric dictionary, never a scalar.
- Family-specific encoding happens inside the built estimator or pipeline, not upstream.
- If an external package import fails, the family must disable itself cleanly and report the reason.

## Search rules

- One trial runs at a time in V1.
- `DummyClassifier` is a fixed baseline, not a search family.
- `Stopper` is per-family and per-phase. Never reuse a stopper across families.
- `UCB1Allocator` must never return unavailable, converged, or quarantined families.
- Five consecutive trial failures quarantine a family and remove it from allocation.
- `MAX_TOTAL_TRIALS` and `MAX_WALL_CLOCK_HOURS` are hard safety caps.

## Reporting contract

- JSON top-level keys must include:
  `experiment_id`, `seed`, `dataset_hash`, `package_versions`, `timestamps`, `baseline`, `phase1`, `phase2`, `holdout`
- Baseline is reported separately from ranked family tables.
- Phase 1, Phase 2, and holdout metrics stay in separate sections. Do not merge them into one score column.
- `results/report_<id>.json` and `results/report_<id>.md` must both be written before exit.

## Adapting to a new dataset

1. Update `config/settings.py`.
2. Run startup validation and resolve any schema or vocabulary failures.
3. Review each family search space and defaults in `settings.py`.
4. Re-run the test suite, especially config validation, frame contracts, and split invariants.
