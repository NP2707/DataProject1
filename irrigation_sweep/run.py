from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from irrigation_sweep.config import settings
from irrigation_sweep.config.validate import startup_validate
from irrigation_sweep.families.catboost_family import CatBoostFamily
from irrigation_sweep.families.hist_gbm import HistGBMFamily
from irrigation_sweep.families.lightgbm_family import LightGBMFamily
from irrigation_sweep.families.linear import LinearFamily
from irrigation_sweep.families.random_forest import RandomForestFamily
from irrigation_sweep.families.xgboost_family import XGBoostFamily
from irrigation_sweep.reporting.report import (
    assemble_report,
    render_console_report,
    render_markdown_report,
    utc_now_iso,
    write_report_files,
)
from irrigation_sweep.search.engine import SweepEngine


def build_families():
    return [
        LinearFamily(),
        RandomForestFamily(),
        HistGBMFamily(),
        XGBoostFamily(),
        LightGBMFamily(),
        CatBoostFamily(),
    ]


def run_experiment(sample_size: int | None = None, seed: int = settings.SEED) -> dict:
    startup_validate()
    started_at = utc_now_iso()
    engine = SweepEngine(build_families(), seed=seed)
    engine_result = engine.run(sample_size=sample_size)
    finished_at = utc_now_iso()
    experiment_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = assemble_report(
        engine_result=engine_result,
        experiment_id=experiment_id,
        seed=seed,
        started_at=started_at,
        finished_at=finished_at,
    )
    console_output = render_console_report(report)
    markdown = render_markdown_report(report)
    json_path, md_path = write_report_files(report, markdown)
    print(console_output)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    return {
        "report": report,
        "json_path": json_path,
        "markdown_path": md_path,
        "console_output": console_output,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the irrigation_sweep benchmark.")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional full-dataset sample size.")
    parser.add_argument("--seed", type=int, default=settings.SEED, help="Random seed override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(sample_size=args.sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
