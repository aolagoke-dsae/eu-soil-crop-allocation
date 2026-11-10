"""Record how many points enter and leave every filter (the sample flow).

The protocol requires that no point is dropped silently: every filter calls
:meth:`SampleFlow.apply`, which keeps the rows that pass, and records the
counts and the reason. The resulting table is the sample-flow table of the
write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class FlowStep:
    step: str
    reason: str
    n_in: int
    n_out: int

    @property
    def n_removed(self) -> int:
        return self.n_in - self.n_out


@dataclass
class SampleFlow:
    """An ordered log of filtering steps."""

    steps: list[FlowStep] = field(default_factory=list)

    def apply(self, df: pd.DataFrame, keep: pd.Series, step: str, reason: str) -> pd.DataFrame:
        """Keep the rows where ``keep`` is True and record the step.

        ``keep`` must be a boolean Series aligned with ``df``; missing values
        are treated as "do not keep", so a filter can never pass a row by
        accident.
        """
        if not keep.index.equals(df.index):
            raise ValueError(f"Filter '{step}': mask index does not match the data index")
        mask = keep.fillna(False).astype(bool)
        out = df.loc[mask]
        self.steps.append(FlowStep(step=step, reason=reason, n_in=len(df), n_out=len(out)))
        return out

    def note(self, df: pd.DataFrame, step: str, reason: str) -> None:
        """Record a step that changes values but keeps every row."""
        self.steps.append(FlowStep(step=step, reason=reason, n_in=len(df), n_out=len(df)))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "step": s.step,
                    "reason": s.reason,
                    "n_in": s.n_in,
                    "n_removed": s.n_removed,
                    "n_out": s.n_out,
                }
                for s in self.steps
            ]
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(path, index=False)
