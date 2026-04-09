"""Tests for arm elimination bandit and budget manager."""

from __future__ import annotations

import random

import pytest

from src.bandit.arm_elimination import ArmElimination, ArmKey, WARM_START_ROUNDS
from src.bandit.budget_manager import BudgetExceeded, BudgetManager


# ---------------------------------------------------------------------------
# Arm Elimination
# ---------------------------------------------------------------------------


class TestArmElimination:
    def _make_bandit(self, models: list[str] | None = None) -> ArmElimination:
        return ArmElimination(candidate_models=models or ["cheap", "mid", "expensive"])

    def test_warm_start_round_robin(self):
        """During warm-start, all models should be pulled roughly equally."""
        bandit = self._make_bandit()
        counts: dict[str, int] = {"cheap": 0, "mid": 0, "expensive": 0}
        for _ in range(30):
            model = bandit.select("stage_a", budget=1.0)
            counts[model] += 1
            bandit.update("stage_a", model, cost=0.01, quality=3.0)
        # Each model should have been pulled at least 8 times in 30 rounds
        assert all(c >= 8 for c in counts.values())

    def test_select_returns_candidate(self):
        bandit = self._make_bandit()
        model = bandit.select("stage_a", budget=1.0)
        assert model in bandit.candidate_models

    def test_update_records_stats(self):
        bandit = self._make_bandit()
        bandit.update("s", "cheap", cost=0.01, quality=4.0)
        bandit.update("s", "cheap", cost=0.03, quality=2.0)
        stats = bandit.get_stats("s")
        assert stats["cheap"]["pulls"] == 2.0
        assert abs(stats["cheap"]["avg_cost"] - 0.02) < 1e-9
        assert abs(stats["cheap"]["avg_quality"] - 3.0) < 1e-9

    def test_elimination_removes_dominated(self):
        """If arm A is strictly cheaper AND higher quality than B, B is eliminated."""
        bandit = self._make_bandit(["good", "bad"])
        # Give both arms enough pulls
        for _ in range(10):
            bandit.update("s", "good", cost=0.01, quality=4.5)
            bandit.update("s", "bad", cost=0.05, quality=2.0)
        eliminated = bandit.eliminate("s")
        assert ArmKey("s", "bad") in {e for e in eliminated}

    def test_elimination_preserves_at_least_one(self):
        """Elimination must never remove all arms for a stage."""
        bandit = self._make_bandit(["a", "b"])
        for _ in range(10):
            bandit.update("s", "a", cost=0.01, quality=5.0)
            bandit.update("s", "b", cost=0.02, quality=4.0)
        bandit.eliminate("s")
        active = bandit._active_arms("s")
        assert len(active) >= 1

    def test_budget_constraint_selects_affordable(self):
        """After warm-start, bandit should prefer affordable arms."""
        bandit = self._make_bandit(["cheap", "expensive"])
        # Burn through warm-start
        for _ in range(WARM_START_ROUNDS):
            bandit.update("s", "cheap", cost=0.01, quality=3.0)
            bandit.update("s", "expensive", cost=0.50, quality=4.5)
        bandit.total_rounds = WARM_START_ROUNDS * 2
        # With tight budget, should pick cheap
        model = bandit.select("s", budget=0.02)
        assert model == "cheap"

    def test_convergence_on_synthetic_data(self):
        """Over 200 rounds the bandit should converge toward the Pareto-optimal arm.

        Setup: 3 models.
          - 'best':      cost ~ 0.02, quality ~ 4.0 (Pareto-optimal)
          - 'mediocre':   cost ~ 0.03, quality ~ 3.0 (dominated by best)
          - 'expensive':  cost ~ 0.10, quality ~ 4.2 (high quality but costly)
        With a budget of 0.05, 'best' should be selected most often post-warmup.
        """
        rng = random.Random(42)
        bandit = self._make_bandit(["best", "mediocre", "expensive"])
        selection_counts: dict[str, int] = {"best": 0, "mediocre": 0, "expensive": 0}

        cost_map = {"best": 0.02, "mediocre": 0.03, "expensive": 0.10}
        quality_map = {"best": 4.0, "mediocre": 3.0, "expensive": 4.2}

        for i in range(200):
            model = bandit.select("s", budget=0.05)
            cost = cost_map[model] + rng.gauss(0, 0.002)
            quality = quality_map[model] + rng.gauss(0, 0.3)
            bandit.update("s", model, cost=max(0.001, cost), quality=max(1.0, quality))
            if i >= WARM_START_ROUNDS:
                selection_counts[model] += 1
            # Periodically eliminate
            if i % 20 == 19:
                bandit.eliminate("s")

        # 'best' should be the most-selected model post-warmup
        assert selection_counts["best"] >= selection_counts["mediocre"]
        assert selection_counts["best"] >= selection_counts["expensive"]


# ---------------------------------------------------------------------------
# Budget Manager
# ---------------------------------------------------------------------------


class TestBudgetManager:
    @pytest.fixture()
    def bm(self, tmp_path):
        config = tmp_path / "routes.yaml"
        config.write_text(
            """routes:
  stage_a:
    description: "test stage a"
    default_model: m1
    max_budget_usd: 0.10
  stage_b:
    description: "test stage b"
    default_model: m2
    max_budget_usd: 0.05
"""
        )
        return BudgetManager(config_path=config)

    def test_caps_loaded(self, bm: BudgetManager):
        assert bm.stage_cap("stage_a") == pytest.approx(0.10)
        assert bm.stage_cap("stage_b") == pytest.approx(0.05)
        assert bm.run_cap == pytest.approx(0.15)

    def test_record_spend(self, bm: BudgetManager):
        bm.record_spend("stage_a", 0.03)
        assert bm.run_spent == pytest.approx(0.03)
        assert bm.remaining_stage_budget("stage_a") == pytest.approx(0.07)
        assert bm.remaining_run_budget() == pytest.approx(0.12)

    def test_stage_budget_exceeded(self, bm: BudgetManager):
        bm.record_spend("stage_a", 0.08)
        with pytest.raises(BudgetExceeded, match="stage_a"):
            bm.record_spend("stage_a", 0.05)

    def test_run_budget_exceeded(self, bm: BudgetManager):
        # Both within stage caps but combined exceeds run cap (0.15)
        bm.record_spend("stage_a", 0.10)
        bm.record_spend("stage_b", 0.04)
        # 0.10 + 0.04 + 0.02 = 0.16 > 0.15 run cap; stage_b at 0.06 > 0.05 too,
        # but run cap is checked first
        with pytest.raises(BudgetExceeded, match="Run budget"):
            bm.record_spend("stage_b", 0.02)

    def test_reset(self, bm: BudgetManager):
        bm.record_spend("stage_a", 0.05)
        bm.reset()
        assert bm.run_spent == 0.0
        assert bm.remaining_run_budget() == pytest.approx(0.15)

    def test_budget_exceeded_attributes(self, bm: BudgetManager):
        with pytest.raises(BudgetExceeded) as exc_info:
            bm.record_spend("stage_b", 0.06)
        assert exc_info.value.spent == pytest.approx(0.06)
        assert exc_info.value.cap == pytest.approx(0.05)
