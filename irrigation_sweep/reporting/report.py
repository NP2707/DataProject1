from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from irrigation_sweep.config import settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def dataset_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    package_names = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit_learn": "scikit-learn",
        "optuna": "optuna",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
    }
    for key, package_name in package_names.items():
        try:
            versions[key] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "missing"
    return versions


def assemble_report(
    engine_result: dict[str, Any],
    experiment_id: str,
    seed: int,
    started_at: str,
    finished_at: str,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "dataset_hash": dataset_hash(settings.TRAIN_PATH),
        "package_versions": collect_package_versions(),
        "timestamps": {
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
        },
        "baseline": engine_result["baseline"],
        "phase1": engine_result["phase1"],
        "phase2": engine_result["phase2"],
        "holdout": engine_result["holdout"],
        "stop_reason": engine_result["stop_reason"],
        "total_trials_run": engine_result["total_trials_run"],
    }


def _phase_score(row: dict[str, Any], holdout: bool = False) -> str:
    if holdout:
        return f"{row['metrics'][settings.SCORING_METRIC]:.4f}"
    score = row["best_score"]
    return "n/a" if score is None else f"{score:.4f}"


def _render_table(title: str, rows: list[dict[str, Any]], holdout: bool = False) -> list[str]:
    lines = [title]
    if not rows:
        lines.append("(no rows)")
        return lines

    header = f"{'name':<18} {'score':>10} {'fails':>7} {'quarantined':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        if holdout:
            lines.append(
                f"{row['name']:<18} {_phase_score(row, holdout=True):>10} {'-':>7} {'-':>12}"
            )
        else:
            quarantine = row["quarantine_status"]["is_quarantined"]
            lines.append(
                f"{row['name']:<18} {_phase_score(row):>10} "
                f"{row['failed_trial_count']:>7} {str(quarantine):>12}"
            )
    return lines


def render_console_report(report: dict[str, Any]) -> str:
    lines = [
        "Baseline",
        (
            "dummy_baseline "
            f"phase1_macro_f1={report['baseline']['phase1_validation'][settings.SCORING_METRIC]:.4f} "
            f"holdout_macro_f1={report['baseline']['holdout'][settings.SCORING_METRIC]:.4f}"
        ),
        "",
    ]
    lines.extend(_render_table("Phase 1 Ranking", report["phase1"]["ranked_families"]))
    lines.append("")
    lines.extend(_render_table("Phase 2 Ranking", report["phase2"]["ranked_families"]))
    lines.append("")
    lines.extend(_render_table("Holdout Ranking", report["holdout"]["ranked_families"], holdout=True))
    return "\n".join(lines)


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# irrigation_sweep report `{report['experiment_id']}`",
        "",
        f"- Seed: `{report['seed']}`",
        f"- Dataset hash: `{report['dataset_hash']}`",
        f"- Stop reason: `{report['stop_reason']}`",
        f"- Total trials run: `{report['total_trials_run']}`",
        "",
        "## Baseline",
        (
            f"- Dummy baseline macro F1 on Phase 1 validation: "
            f"`{report['baseline']['phase1_validation'][settings.SCORING_METRIC]:.4f}`"
        ),
        (
            f"- Dummy baseline macro F1 on holdout: "
            f"`{report['baseline']['holdout'][settings.SCORING_METRIC]:.4f}`"
        ),
        "",
        "## Phase 1",
        f"- Shortlist: `{report['phase1']['shortlist']}`",
        "",
        "| name | best_score | failed_trial_count | quarantined |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in report["phase1"]["ranked_families"]:
        score = _phase_score(row)
        lines.append(
            f"| {row['name']} | {score} | {row['failed_trial_count']} | "
            f"{row['quarantine_status']['is_quarantined']} |"
        )

    lines.extend(
        [
            "",
            "## Phase 2",
            "",
            "| name | best_score | failed_trial_count | quarantined |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in report["phase2"]["ranked_families"]:
        score = _phase_score(row)
        lines.append(
            f"| {row['name']} | {score} | {row['failed_trial_count']} | "
            f"{row['quarantine_status']['is_quarantined']} |"
        )

    lines.extend(
        [
            "",
            "## Holdout",
            "",
            "| name | macro_f1 |",
            "| --- | ---: |",
        ]
    )
    for row in report["holdout"]["ranked_families"]:
        lines.append(f"| {row['name']} | {_phase_score(row, holdout=True)} |")

    return "\n".join(lines) + "\n"


def write_report_files(report: dict[str, Any], markdown: str) -> tuple[Path, Path]:
    settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = settings.RESULTS_DIR / f"report_{report['experiment_id']}.json"
    md_path = settings.RESULTS_DIR / f"report_{report['experiment_id']}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path


__all__ = [
    "assemble_report",
    "collect_package_versions",
    "dataset_hash",
    "render_console_report",
    "render_markdown_report",
    "utc_now_iso",
    "write_report_files",
]
