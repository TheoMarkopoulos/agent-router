"""Track cumulative spend per pipeline run and enforce budget caps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class BudgetExceeded(Exception):
    """Raised when a stage or run exceeds its budget cap."""

    def __init__(self, message: str, spent: float, cap: float) -> None:
        super().__init__(message)
        self.spent = spent
        self.cap = cap


class StageBudget(BaseModel):
    """Budget configuration for a single pipeline stage."""

    max_budget_usd: float


class BudgetManager:
    """Enforce per-run and per-stage budget caps loaded from routes.yaml."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "routes.yaml"
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        self._stage_caps: dict[str, float] = {
            k: v["max_budget_usd"] for k, v in raw["routes"].items()
        }
        self._run_cap: float = sum(self._stage_caps.values())
        self._run_spent: float = 0.0
        self._stage_spent: dict[str, float] = {}

    @property
    def run_cap(self) -> float:
        return self._run_cap

    @property
    def run_spent(self) -> float:
        return self._run_spent

    @property
    def stage_spent(self) -> dict[str, float]:
        return dict(self._stage_spent)

    def remaining_run_budget(self) -> float:
        """Return the remaining dollar budget for this pipeline run."""
        return max(0.0, self._run_cap - self._run_spent)

    def remaining_stage_budget(self, stage: str) -> float:
        """Return the remaining dollar budget for a specific stage."""
        cap = self._stage_caps.get(stage, 0.0)
        spent = self._stage_spent.get(stage, 0.0)
        return max(0.0, cap - spent)

    def stage_cap(self, stage: str) -> float:
        """Return the budget cap for a specific stage."""
        return self._stage_caps.get(stage, 0.0)

    def record_spend(self, stage: str, amount: float) -> None:
        """Record spend and raise BudgetExceeded if any cap is breached."""
        new_stage = self._stage_spent.get(stage, 0.0) + amount
        new_run = self._run_spent + amount

        if new_run > self._run_cap:
            raise BudgetExceeded(
                f"Run budget exceeded: ${new_run:.6f} > ${self._run_cap:.6f}",
                spent=new_run,
                cap=self._run_cap,
            )

        stage_cap = self._stage_caps.get(stage)
        if stage_cap is not None and new_stage > stage_cap:
            raise BudgetExceeded(
                f"Stage '{stage}' budget exceeded: ${new_stage:.6f} > ${stage_cap:.6f}",
                spent=new_stage,
                cap=stage_cap,
            )

        self._stage_spent[stage] = new_stage
        self._run_spent = new_run

    def reset(self) -> None:
        """Reset spend tracking for a new run."""
        self._run_spent = 0.0
        self._stage_spent.clear()
