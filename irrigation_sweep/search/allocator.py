from __future__ import annotations

import math
from typing import Any


class UCB1Allocator:
    def __init__(self, families: list[Any], burn_in: int) -> None:
        self.family_order = [family.name() if hasattr(family, "name") else str(family) for family in families]
        self.burn_in = burn_in
        self.selection_counts = {name: 0 for name in self.family_order}
        self.reward_sums = {name: 0.0 for name in self.family_order}
        self.quarantined: set[str] = set()
        self.converged: set[str] = set()
        self.unavailable: set[str] = set()

    def mark_quarantined(self, family_name: str) -> None:
        self.quarantined.add(family_name)

    def mark_converged(self, family_name: str) -> None:
        self.converged.add(family_name)

    def mark_unavailable(self, family_name: str) -> None:
        self.unavailable.add(family_name)

    def record_result(self, family_name: str, score: float | None) -> None:
        if score is None:
            return
        self.reward_sums[family_name] += score

    def _eligible_family_names(self) -> list[str]:
        blocked = self.quarantined | self.converged | self.unavailable
        return [name for name in self.family_order if name not in blocked]

    def select_family(self) -> str:
        eligible = self._eligible_family_names()
        if not eligible:
            raise RuntimeError("No eligible families remain for allocation.")

        for name in eligible:
            if self.selection_counts[name] < self.burn_in:
                self.selection_counts[name] += 1
                return name

        total_pulls = sum(self.selection_counts[name] for name in eligible)
        best_name = eligible[0]
        best_value = float("-inf")
        for name in eligible:
            count = self.selection_counts[name]
            mean_reward = self.reward_sums[name] / count if count else 0.0
            ucb_score = mean_reward + math.sqrt(2.0 * math.log(total_pulls) / count)
            if ucb_score > best_value:
                best_name = name
                best_value = ucb_score

        self.selection_counts[best_name] += 1
        return best_name


__all__ = ["UCB1Allocator"]
