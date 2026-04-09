"""Arm Elimination bandit for cost-aware model selection per pipeline stage.

Each arm is a (stage, model_id) pair. The algorithm maintains running estimates of
cost and quality per arm, applies UCB1 exploration bonuses, eliminates
Pareto-dominated arms, and selects the best arm under a budget constraint.
The first WARM_START_ROUNDS runs use round-robin to build initial estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


WARM_START_ROUNDS = 50


@dataclass
class ArmStats:
    """Running statistics for a single arm."""

    total_cost: float = 0.0
    total_quality: float = 0.0
    pulls: int = 0

    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.pulls if self.pulls else 0.0

    @property
    def avg_quality(self) -> float:
        return self.total_quality / self.pulls if self.pulls else 0.0


@dataclass
class ArmKey:
    """Unique identifier for an arm: (stage, model_id)."""

    stage: str
    model_id: str

    def __hash__(self) -> int:
        return hash((self.stage, self.model_id))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArmKey):
            return NotImplemented
        return self.stage == other.stage and self.model_id == other.model_id


@dataclass
class ArmElimination:
    """Arm Elimination bandit with UCB1 exploration and Pareto-dominance pruning."""

    candidate_models: list[str] = field(default_factory=list)
    stats: dict[ArmKey, ArmStats] = field(default_factory=dict)
    eliminated: set[ArmKey] = field(default_factory=set)
    total_rounds: int = 0

    @classmethod
    def from_config(cls, config_path: Path | str | None = None) -> ArmElimination:
        """Load candidate models from models.yaml."""
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        model_ids = [m["id"] for m in raw["models"]]
        return cls(candidate_models=model_ids)

    def _ensure_arms(self, stage: str) -> None:
        """Initialise stats for any unseen (stage, model) pairs."""
        for model_id in self.candidate_models:
            key = ArmKey(stage, model_id)
            if key not in self.stats:
                self.stats[key] = ArmStats()

    def _active_arms(self, stage: str) -> list[ArmKey]:
        """Return non-eliminated arms for a stage."""
        self._ensure_arms(stage)
        return [
            ArmKey(stage, m)
            for m in self.candidate_models
            if ArmKey(stage, m) not in self.eliminated
        ]

    def _ucb_score(self, key: ArmKey) -> float:
        """UCB1 score: avg_quality + exploration bonus."""
        s = self.stats[key]
        if s.pulls == 0:
            return float("inf")
        exploration = math.sqrt(2 * math.log(max(self.total_rounds, 1)) / s.pulls)
        return s.avg_quality + exploration

    # --- Core API ---

    def select(self, stage: str, budget: float) -> str:
        """Pick the best model for *stage* given remaining *budget*.

        During warm-start (< WARM_START_ROUNDS total pulls for the stage),
        uses round-robin. After that, picks the active arm with the highest
        UCB score whose average cost fits the budget.
        """
        active = self._active_arms(stage)
        if not active:
            # All eliminated — reset and return first candidate
            self.eliminated -= {ArmKey(stage, m) for m in self.candidate_models}
            active = self._active_arms(stage)

        # Warm-start: round-robin across all candidates for this stage
        stage_pulls = sum(self.stats[k].pulls for k in active)
        if stage_pulls < WARM_START_ROUNDS:
            # Pick the arm with fewest pulls
            return min(active, key=lambda k: self.stats[k].pulls).model_id

        # Budget-filtered selection
        affordable = [k for k in active if self.stats[k].avg_cost <= budget]
        if not affordable:
            # Nothing fits budget — pick cheapest active arm
            return min(active, key=lambda k: self.stats[k].avg_cost).model_id

        # UCB selection among affordable arms
        return max(affordable, key=lambda k: self._ucb_score(k)).model_id

    def update(self, stage: str, model_id: str, cost: float, quality: float) -> None:
        """Record observed cost and quality for an arm pull."""
        key = ArmKey(stage, model_id)
        self._ensure_arms(stage)
        s = self.stats[key]
        s.total_cost += cost
        s.total_quality += quality
        s.pulls += 1
        self.total_rounds += 1

    def eliminate(self, stage: str) -> list[ArmKey]:
        """Remove Pareto-dominated arms for *stage*.

        Arm A dominates Arm B if A has both lower avg_cost and higher avg_quality.
        Only considers arms with at least 5 pulls to avoid premature elimination.
        """
        active = self._active_arms(stage)
        newly_eliminated: list[ArmKey] = []

        for arm_b in active:
            sb = self.stats[arm_b]
            if sb.pulls < 5:
                continue
            for arm_a in active:
                if arm_a == arm_b:
                    continue
                sa = self.stats[arm_a]
                if sa.pulls < 5:
                    continue
                if sa.avg_cost < sb.avg_cost and sa.avg_quality > sb.avg_quality:
                    self.eliminated.add(arm_b)
                    newly_eliminated.append(arm_b)
                    break

        # Never eliminate all arms for a stage
        remaining = [a for a in active if a not in self.eliminated]
        if not remaining:
            # Undo last round of eliminations
            for arm in newly_eliminated:
                self.eliminated.discard(arm)
            newly_eliminated.clear()

        return newly_eliminated

    def get_stats(self, stage: str) -> dict[str, dict[str, float | int]]:
        """Return readable stats for all arms in a stage."""
        self._ensure_arms(stage)
        result: dict[str, dict[str, float]] = {}
        for model_id in self.candidate_models:
            key = ArmKey(stage, model_id)
            s = self.stats[key]
            result[model_id] = {
                "avg_cost": s.avg_cost,
                "avg_quality": s.avg_quality,
                "pulls": float(s.pulls),
                "eliminated": float(key in self.eliminated),
            }
        return result
