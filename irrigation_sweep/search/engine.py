from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from irrigation_sweep.config import settings
from irrigation_sweep.families.base import BaseFamily
from irrigation_sweep.search.allocator import UCB1Allocator
from irrigation_sweep.search.stopper import Stopper
from irrigation_sweep.validation.validator import DatasetPartition, global_split, phase1_split, phase2_split

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class FamilyPhaseState:
    family: BaseFamily
    study: optuna.study.Study
    stopper: Stopper
    best_score: float | None = None
    best_config: dict[str, Any] | None = None
    best_train_score: float | None = None
    score_distribution: list[float] = field(default_factory=list)
    convergence_history: list[dict[str, Any]] = field(default_factory=list)
    top_configs: list[dict[str, Any]] = field(default_factory=list)
    trial_durations: list[float] = field(default_factory=list)
    failed_trial_count: int = 0
    consecutive_failures: int = 0
    quarantined: bool = False
    quarantine_reason: str | None = None
    last_exception: str | None = None
    train_val_gap: float | None = None
    overfit_flag: bool = False
    selected_count: int = 0


def shortlist_families(
    phase1_scores: dict[str, float],
    top_k: int,
    abs_gap: float,
) -> list[str]:
    ranked = sorted(phase1_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return []
    best_score = ranked[0][1]
    shortlist = {name for name, _ in ranked[:top_k]}
    shortlist.update(
        name
        for name, score in ranked
        if best_score - score <= abs_gap
    )
    return [name for name, _ in ranked if name in shortlist]


class SweepEngine:
    def __init__(
        self,
        families: list[BaseFamily],
        seed: int = settings.SEED,
        logger=None,
    ) -> None:
        self.families = families
        self.seed = seed
        self.logger = logger
        self.console_messages: list[str] = []
        self.total_trials_run = 0
        self.stop_reason: str | None = None
        self._start_time = 0.0
        self._deadline = 0.0

    def log(self, message: str) -> None:
        self.console_messages.append(message)
        if self.logger is not None:
            self.logger(message)

    def _metrics_from_predictions(self, y_true, y_pred) -> dict[str, Any]:
        recalls = recall_score(
            y_true,
            y_pred,
            labels=settings.TARGET_LABELS,
            average=None,
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "per_class_recall": {
                label: float(value)
                for label, value in zip(settings.TARGET_LABELS, recalls, strict=True)
            },
        }

    def _evaluate_dummy(
        self,
        train_set: DatasetPartition,
        eval_set: DatasetPartition,
    ) -> dict[str, Any]:
        model = DummyClassifier(strategy=settings.BASELINE_STRATEGY, random_state=self.seed)
        model.fit(train_set.X, train_set.y)
        preds = model.predict(eval_set.X)
        return self._metrics_from_predictions(eval_set.y, preds)

    def _phase_budget(self, available_family_count: int) -> int:
        burn_in_total = available_family_count * settings.UCB_INIT_TRIALS
        fraction_budget = math.floor(settings.MAX_TOTAL_TRIALS * settings.PHASE1_TRIAL_FRACTION)
        return min(settings.MAX_TOTAL_TRIALS, max(burn_in_total, fraction_budget))

    def _remaining_trial_budget(self) -> int:
        return max(settings.MAX_TOTAL_TRIALS - self.total_trials_run, 0)

    def _wall_clock_exceeded(self) -> bool:
        return time.monotonic() >= self._deadline

    def _initialize_states(
        self,
        families: list[BaseFamily],
        phase_tag: str,
        phase_seed_offset: int,
    ) -> dict[str, FamilyPhaseState]:
        states: dict[str, FamilyPhaseState] = {}
        for offset, family in enumerate(families):
            sampler = TPESampler(seed=self.seed + phase_seed_offset + offset)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            states[family.name()] = FamilyPhaseState(
                family=family,
                study=study,
                stopper=Stopper(
                    min_trials=settings.MIN_TRIALS,
                    patience=settings.PATIENCE,
                    delta=settings.DELTA,
                ),
            )
            if not family.is_available():
                self.log(
                    f"{phase_tag}: disabling {family.name()} because {family.unavailable_reason}"
                )
        return states

    def _record_top_config(
        self,
        state: FamilyPhaseState,
        config: dict[str, Any],
        score: float,
    ) -> None:
        state.top_configs.append({"config": config, "score": score})
        state.top_configs.sort(key=lambda item: item["score"], reverse=True)
        del state.top_configs[settings.TOP_CONFIGS_TO_KEEP :]

    def _serialize_state(self, state: FamilyPhaseState) -> dict[str, Any]:
        return {
            "name": state.family.name(),
            "best_config": state.best_config,
            "best_score": state.best_score,
            "top_5_configs": state.top_configs,
            "score_distribution": state.score_distribution,
            "convergence_history": state.convergence_history,
            "train_val_gap": state.train_val_gap,
            "overfit_flag": state.overfit_flag,
            "trial_durations": state.trial_durations,
            "failed_trial_count": state.failed_trial_count,
            "quarantine_status": {
                "is_quarantined": state.quarantined,
                "quarantine_reason": state.quarantine_reason,
                "last_exception": state.last_exception,
                "unavailable_reason": state.family.unavailable_reason,
            },
            "selected_count": state.selected_count,
        }

    def _rank_states(self, states: dict[str, FamilyPhaseState]) -> list[dict[str, Any]]:
        serialized = [self._serialize_state(state) for state in states.values()]
        return sorted(
            serialized,
            key=lambda item: (
                item["best_score"] is None,
                -item["best_score"] if item["best_score"] is not None else float("inf"),
                item["name"],
            ),
        )

    def _run_phase(
        self,
        phase_tag: str,
        families: list[BaseFamily],
        train_set: DatasetPartition,
        val_set: DatasetPartition,
        burn_in: int,
        phase_budget: int,
        phase_seed_offset: int,
    ) -> tuple[list[dict[str, Any]], dict[str, FamilyPhaseState]]:
        states = self._initialize_states(families, phase_tag, phase_seed_offset)
        allocator = UCB1Allocator(families, burn_in=burn_in)

        for family in families:
            if not family.is_available():
                allocator.mark_unavailable(family.name())

        phase_trials = 0
        while phase_trials < phase_budget and self.total_trials_run < settings.MAX_TOTAL_TRIALS:
            if self._wall_clock_exceeded():
                self.stop_reason = "wall_clock_limit"
                self.log("Wall clock limit reached; stopping sweep early.")
                break

            try:
                family_name = allocator.select_family()
            except RuntimeError:
                break

            state = states[family_name]
            state.selected_count += 1
            trial = state.study.ask()
            config: dict[str, Any] | None = None
            started = time.monotonic()
            try:
                config = state.family.get_search_space(trial)
                model = state.family.build(config)
                fitted = state.family.fit(model, train_set.X, train_set.y, val_set.X, val_set.y)
                train_metrics = state.family.evaluate(fitted, train_set.X, train_set.y)
                val_metrics = state.family.evaluate(fitted, val_set.X, val_set.y)
                score = float(val_metrics[settings.SCORING_METRIC])
                train_score = float(train_metrics[settings.SCORING_METRIC])

                state.study.tell(trial, score)
                allocator.record_result(family_name, score)
                state.stopper.update(score)
                state.score_distribution.append(score)
                state.consecutive_failures = 0
                self._record_top_config(state, config, score)

                if state.best_score is None or score > state.best_score:
                    state.best_score = score
                    state.best_config = config
                    state.best_train_score = train_score
                    state.train_val_gap = train_score - score
                    state.overfit_flag = state.train_val_gap > settings.OVERFIT_GAP_THRESHOLD

                state.convergence_history.append(
                    {
                        "trial": len(state.score_distribution),
                        "best_score": state.best_score,
                    }
                )

                if state.stopper.should_stop():
                    allocator.mark_converged(family_name)
                    self.log(f"{phase_tag}: {family_name} converged and was removed from allocation.")
            except Exception as exc:
                state.failed_trial_count += 1
                state.consecutive_failures += 1
                state.last_exception = f"{type(exc).__name__}: {exc}"
                state.study.tell(trial, state=TrialState.FAIL)
                self.log(f"{phase_tag}: trial failed for {family_name}: {state.last_exception}")
                if state.consecutive_failures >= settings.MAX_CONSECUTIVE_FAILS:
                    state.quarantined = True
                    state.quarantine_reason = "max_consecutive_failures"
                    allocator.mark_quarantined(family_name)
                    self.log(
                        f"{phase_tag}: quarantined {family_name} after "
                        f"{settings.MAX_CONSECUTIVE_FAILS} consecutive failures."
                    )
            finally:
                duration = time.monotonic() - started
                state.trial_durations.append(duration)
                self.total_trials_run += 1
                phase_trials += 1

        return self._rank_states(states), states

    def _holdout_evaluation(
        self,
        finalists: list[BaseFamily],
        phase2_states: dict[str, FamilyPhaseState],
        dev_set: DatasetPartition,
        holdout_set: DatasetPartition,
    ) -> list[dict[str, Any]]:
        holdout_rows: list[dict[str, Any]] = []
        for family in finalists:
            state = phase2_states[family.name()]
            if state.best_config is None:
                continue
            if self._wall_clock_exceeded():
                self.stop_reason = "wall_clock_limit"
                self.log("Wall clock limit reached before holdout evaluation completed.")
                break

            model = family.build(state.best_config)
            fitted = family.fit(model, dev_set.X, dev_set.y)
            metrics = family.evaluate(fitted, holdout_set.X, holdout_set.y)
            holdout_rows.append(
                {
                    "name": family.name(),
                    "best_config": state.best_config,
                    "metrics": metrics,
                }
            )
        return sorted(
            holdout_rows,
            key=lambda item: item["metrics"][settings.SCORING_METRIC],
            reverse=True,
        )

    def run(self, sample_size: int | None = None) -> dict[str, Any]:
        self._start_time = time.monotonic()
        self._deadline = self._start_time + (settings.MAX_WALL_CLOCK_HOURS * 3600.0)
        self.total_trials_run = 0
        self.stop_reason = None
        self.console_messages = []

        dev_set, holdout_set = global_split(seed=self.seed, sample_size=sample_size)
        phase1_train, phase1_val = phase1_split(
            dev_set,
            sample_size=min(settings.PHASE1_SAMPLE_SIZE, len(dev_set)),
            seed=self.seed,
        )
        phase2_train, phase2_val = phase2_split(dev_set, seed=self.seed)

        baseline = {
            "name": "dummy_baseline",
            "strategy": settings.BASELINE_STRATEGY,
            "allocated_trials": 0,
            "phase1_validation": self._evaluate_dummy(phase1_train, phase1_val),
            "holdout": self._evaluate_dummy(dev_set, holdout_set),
        }
        self.log(
            "Baseline dummy_baseline "
            f"phase1_macro_f1={baseline['phase1_validation'][settings.SCORING_METRIC]:.4f} "
            f"holdout_macro_f1={baseline['holdout'][settings.SCORING_METRIC]:.4f}"
        )

        available_families = [family for family in self.families if family.is_available()]
        phase1_budget = self._phase_budget(len(available_families))
        phase1_ranked, phase1_states = self._run_phase(
            phase_tag="phase1",
            families=self.families,
            train_set=phase1_train,
            val_set=phase1_val,
            burn_in=settings.UCB_INIT_TRIALS,
            phase_budget=phase1_budget,
            phase_seed_offset=100,
        )

        phase1_scores = {
            entry["name"]: entry["best_score"]
            for entry in phase1_ranked
            if entry["best_score"] is not None and not entry["quarantine_status"]["is_quarantined"]
        }
        shortlisted_names = shortlist_families(
            phase1_scores,
            top_k=settings.SHORTLIST_TOP_K,
            abs_gap=settings.SHORTLIST_ABS_GAP,
        )
        finalists = [family for family in self.families if family.name() in shortlisted_names]

        remaining_budget = self._remaining_trial_budget()
        phase2_ranked: list[dict[str, Any]] = []
        phase2_states: dict[str, FamilyPhaseState] = {}
        if finalists and remaining_budget > 0 and not self._wall_clock_exceeded():
            phase2_ranked, phase2_states = self._run_phase(
                phase_tag="phase2",
                families=finalists,
                train_set=phase2_train,
                val_set=phase2_val,
                burn_in=settings.PHASE2_UCB_INIT_TRIALS,
                phase_budget=remaining_budget,
                phase_seed_offset=10_000,
            )
        elif finalists:
            self.log("No remaining trial budget for phase2 finalists.")

        holdout_ranked = []
        if phase2_states:
            holdout_ranked = self._holdout_evaluation(finalists, phase2_states, dev_set, holdout_set)

        if self.stop_reason is None and self.total_trials_run >= settings.MAX_TOTAL_TRIALS:
            self.stop_reason = "max_total_trials"
            self.log("Maximum total trial cap reached; stopping sweep.")

        return {
            "baseline": baseline,
            "phase1": {
                "sample_size": len(phase1_train) + len(phase1_val),
                "ranked_families": phase1_ranked,
                "shortlist": shortlisted_names,
            },
            "phase2": {
                "ranked_families": phase2_ranked,
            },
            "holdout": {
                "ranked_families": holdout_ranked,
            },
            "stop_reason": self.stop_reason,
            "total_trials_run": self.total_trials_run,
            "console_output": "\n".join(self.console_messages),
        }


__all__ = ["SweepEngine", "shortlist_families"]
