from __future__ import annotations


class Stopper:
    def __init__(self, min_trials: int, patience: int, delta: float) -> None:
        self.min_trials = min_trials
        self.patience = patience
        self.delta = delta
        self.trial_count = 0
        self.best_score: float | None = None
        self._warmup_complete = False
        self._post_warmup_non_improving = 0

    def update(self, score: float) -> None:
        self.trial_count += 1
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            if self._warmup_complete:
                self._post_warmup_non_improving = 0
        elif self._warmup_complete:
            self._post_warmup_non_improving += 1

        if not self._warmup_complete and self.trial_count >= self.min_trials:
            self._warmup_complete = True
            self._post_warmup_non_improving = 0

    def should_stop(self) -> bool:
        if not self._warmup_complete:
            return False
        return self._post_warmup_non_improving >= self.patience


__all__ = ["Stopper"]
