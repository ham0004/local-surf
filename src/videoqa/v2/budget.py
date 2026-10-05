"""Answerer-call budget with TWO limits (fresh calls and seconds), persisted to disk.

Why: an earlier round raised its GPU-time cap but not its call cap and spent
6,433 fresh calls against a declared 3,000. This ledger enforces both limits
and stops at whichever is reached first.

Usage pattern (one question = one block, so tables are never left half done):

    budget = CallBudget(Path("runs/x/call_budget.json"), max_calls=500, max_seconds=3600)
    teacher = CachedTeacher(answerer, teacher_id, cache, identity=..., budget=budget)
    for question in questions:
        missing = count_uncached_requests(question)      # caller computes this from the cache
        if not budget.reserve(missing):
            break                                        # stop cleanly before starting the block
        ... teacher.quality(...) calls ...               # each real call is charged automatically

``reserve`` only checks that the whole block fits; ``charge`` (called by the
teacher after each real call) records what was actually spent. A call that
would exceed either limit raises BudgetExceeded instead of running.
"""

from __future__ import annotations

import json
from pathlib import Path


class BudgetExceeded(RuntimeError):
    """A real answerer call was requested beyond the declared limits."""


class CallBudget:
    def __init__(self, path: str | Path, max_calls: int, max_seconds: float) -> None:
        if max_calls < 0 or max_seconds <= 0:
            raise ValueError("max_calls must be >= 0 and max_seconds > 0")
        self.path = Path(path)
        self.max_calls, self.max_seconds = int(max_calls), float(max_seconds)
        state = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if state and (state.get("max_calls"), state.get("max_seconds")) != (self.max_calls, self.max_seconds):
            raise ValueError(f"{self.path} was created with different limits; a new limit needs a new ledger "
                             "(and a new authorisation), not a silent change")
        self.calls = int(state.get("calls", 0))
        self.seconds = float(state.get("seconds", 0.0))

    @property
    def calls_left(self) -> int:
        return self.max_calls - self.calls

    @property
    def seconds_left(self) -> float:
        return self.max_seconds - self.seconds

    def reserve(self, n_calls: int) -> bool:
        """True if a block of ``n_calls`` fresh calls fits in BOTH limits (nothing is spent)."""
        return n_calls <= self.calls_left and (n_calls == 0 or self.seconds_left > 0)

    def check_one(self) -> None:
        """Raise before a single real call that would break a limit."""
        if self.calls_left < 1 or self.seconds_left <= 0:
            raise BudgetExceeded(f"budget spent: {self.calls}/{self.max_calls} calls, "
                                 f"{self.seconds:.0f}/{self.max_seconds:.0f} s")

    def charge(self, seconds: float) -> None:
        """Record one completed real call and persist the ledger."""
        self.calls += 1
        self.seconds += float(seconds)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"max_calls": self.max_calls, "max_seconds": self.max_seconds,
                                         "calls": self.calls, "seconds": self.seconds}, indent=1), encoding="utf-8")
