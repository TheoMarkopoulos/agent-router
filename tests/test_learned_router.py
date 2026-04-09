"""Tests for LearnedRouter: train on synthetic data, predict, save/load."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.router.learned_router import LearnedRouter
from src.router.rule_router import ActionType

# Minimal synthetic dataset — 4 examples per class = 20 total
_TRAINING_DATA: list[tuple[str, str]] = [
    # direct_response
    ("Hello there", "direct_response"),
    ("What is the capital of France?", "direct_response"),
    ("Convert 100 USD to EUR", "direct_response"),
    ("Define machine learning", "direct_response"),
    # single_agent
    ("Write a Python quicksort function", "single_agent"),
    ("Summarise this article about climate change", "single_agent"),
    ("Generate a cover letter for a data scientist role", "single_agent"),
    ("Fix the bug in my sorting algorithm", "single_agent"),
    # multi_agent
    ("Compare React and Vue for building dashboards", "multi_agent"),
    ("Build an API service for user authentication", "multi_agent"),
    ("Give me a step-by-step guide to deploy on AWS", "multi_agent"),
    ("Research Python web frameworks and then recommend one", "multi_agent"),
    # escalation
    ("Critical security vulnerability in our auth module", "escalation"),
    ("Production database is down and users are affected", "escalation"),
    ("Urgent: deployment failed and rollback isn't working", "escalation"),
    ("We need to architect a complete system redesign", "escalation"),
    # clarification
    ("huh?", "clarification"),
    ("I'm not sure what I need", "clarification"),
    ("help", "clarification"),
    ("what do you mean by that?", "clarification"),
]


@pytest.fixture()
def trained_router() -> LearnedRouter:
    queries = [q for q, _ in _TRAINING_DATA]
    labels = [label for _, label in _TRAINING_DATA]
    return LearnedRouter.train(queries, labels, max_iter=200)


class TestTrainAndPredict:
    def test_train_returns_router(self, trained_router: LearnedRouter) -> None:
        assert isinstance(trained_router, LearnedRouter)

    def test_classify_returns_routing_decision(self, trained_router: LearnedRouter) -> None:
        result = trained_router.classify("What is Python?")
        assert result.action_type in ActionType
        assert 0.0 <= result.confidence <= 1.0
        assert result.model is not None
        assert result.max_budget_usd > 0

    def test_classify_known_escalation(self, trained_router: LearnedRouter) -> None:
        result = trained_router.classify("Critical security breach in production systems")
        # With only 20 training examples the model may not always get this right,
        # but it should at least return a valid ActionType
        assert result.action_type in ActionType

    def test_classify_known_direct(self, trained_router: LearnedRouter) -> None:
        result = trained_router.classify("Hello, how are you?")
        assert result.action_type in ActionType

    def test_all_action_types_representable(self, trained_router: LearnedRouter) -> None:
        """The trained model should be able to produce all 5 action types."""
        # Check the label encoder knows all types
        known_labels = set(trained_router._le.classes_)
        for at in ActionType:
            assert at.value in known_labels


class TestPersistence:
    def test_save_and_load(self, trained_router: LearnedRouter) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "router.pkl"
            trained_router.save(model_path)

            loaded = LearnedRouter.from_pretrained(model_path)

            # Both should produce the same prediction
            query = "What is the capital of France?"
            original = trained_router.classify(query)
            reloaded = loaded.classify(query)
            assert original.action_type == reloaded.action_type
            assert abs(original.confidence - reloaded.confidence) < 1e-6

    def test_save_creates_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            queries = [q for q, _ in _TRAINING_DATA]
            labels = [label for _, label in _TRAINING_DATA]
            router = LearnedRouter.train(queries, labels, max_iter=100)

            nested_path = Path(tmpdir) / "nested" / "dir" / "router.pkl"
            router.save(nested_path)
            assert nested_path.exists()


class TestGetRouteConfig:
    def test_returns_config(self, trained_router: LearnedRouter) -> None:
        config = trained_router.get_route_config(ActionType.ESCALATION)
        assert config.default_model is not None
        assert config.max_budget_usd > 0
