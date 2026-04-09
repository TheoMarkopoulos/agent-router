"""Integration tests: send queries through pipeline, verify routing + logging."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.router.pipeline import Pipeline
from src.router.rule_router import ActionType, RuleRouter
from src.utils.cost import CostCalculator
from src.utils.llm_client import LLMClient, LLMResponse
from src.utils.logger import DuckDBLogger


def _make_pipeline(mock_llm: MagicMock) -> Pipeline:
    """Build a pipeline with a mocked LLM client."""
    client = LLMClient()
    client.complete = mock_llm
    return Pipeline(
        router=RuleRouter(),
        llm_client=client,
        cost_calculator=CostCalculator(),
        logger=DuckDBLogger(),
    )


def _fake_response(content: str = "Test response", model: str = "gpt-4o-mini") -> LLMResponse:
    return LLMResponse(
        content=content,
        model=model,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
    )


class TestPipelineRouting:
    def test_direct_response_query(self) -> None:
        mock = MagicMock(return_value=_fake_response())
        pipeline = _make_pipeline(mock)

        result = pipeline.run("Hello there")
        assert result["routing_decision"].action_type == ActionType.DIRECT_RESPONSE
        assert mock.called

    def test_escalation_query(self) -> None:
        mock = MagicMock(return_value=_fake_response(model="gpt-4o"))
        pipeline = _make_pipeline(mock)

        result = pipeline.run("Critical security vulnerability found in production")
        assert result["routing_decision"].action_type == ActionType.ESCALATION

    def test_clarification_skips_llm(self) -> None:
        mock = MagicMock(return_value=_fake_response())
        pipeline = _make_pipeline(mock)

        result = pipeline.run("huh?")
        assert result["routing_decision"].action_type == ActionType.CLARIFICATION
        # Clarification produces a canned response without calling the LLM
        mock.assert_not_called()
        assert result["llm_response"] is not None
        assert "more details" in result["llm_response"].content.lower()


class TestPipelineLogging:
    def test_logs_are_written(self) -> None:
        mock = MagicMock(return_value=_fake_response())
        pipeline = _make_pipeline(mock)

        pipeline.run("What is Python?")
        pipeline.run("Build an API service for payments")
        pipeline.run("help")

        assert pipeline.logger.count() == 3

    def test_log_fields(self) -> None:
        mock = MagicMock(return_value=_fake_response())
        pipeline = _make_pipeline(mock)

        pipeline.run("What is Python?")
        logs = pipeline.logger.query("SELECT * FROM request_logs")
        assert len(logs) == 1

        log = logs[0]
        assert log["query"] == "What is Python?"
        assert log["route_chosen"] == "direct_response"
        assert log["model_used"] is not None


class TestPipelineCost:
    def test_cost_computed(self) -> None:
        mock = MagicMock(return_value=_fake_response())
        pipeline = _make_pipeline(mock)

        result = pipeline.run("Hello")
        # 10 prompt tokens * 0.00015/1k + 20 completion tokens * 0.0006/1k
        assert result["cost_usd"] > 0

    def test_clarification_zero_cost(self) -> None:
        mock = MagicMock(return_value=_fake_response())
        pipeline = _make_pipeline(mock)

        result = pipeline.run("huh?")
        assert result["cost_usd"] == 0.0
