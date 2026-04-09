"""Unit tests for rule router classification across all 5 action types."""

from src.router.rule_router import ActionType, RuleRouter


def _router() -> RuleRouter:
    return RuleRouter()


class TestDirectResponse:
    def test_greeting(self) -> None:
        result = _router().classify("Hello, how are you?")
        assert result.action_type == ActionType.DIRECT_RESPONSE

    def test_factual_question(self) -> None:
        result = _router().classify("What is the capital of France?")
        assert result.action_type == ActionType.DIRECT_RESPONSE

    def test_simple_conversion(self) -> None:
        result = _router().classify("Convert 100 USD to EUR")
        assert result.action_type == ActionType.DIRECT_RESPONSE

    def test_list_request(self) -> None:
        result = _router().classify("List the top 5 popular programming languages")
        assert result.action_type == ActionType.DIRECT_RESPONSE


class TestSingleAgent:
    def test_standard_task(self) -> None:
        result = _router().classify("Write a Python function to sort a list using quicksort")
        assert result.action_type == ActionType.SINGLE_AGENT

    def test_code_review(self) -> None:
        result = _router().classify("Review this code snippet for bugs")
        assert result.action_type == ActionType.SINGLE_AGENT

    def test_default_fallback(self) -> None:
        # Queries that don't match any strong pattern fall to single_agent
        result = _router().classify("Generate a haiku about distributed systems")
        assert result.action_type == ActionType.SINGLE_AGENT


class TestMultiAgent:
    def test_comparison(self) -> None:
        result = _router().classify("Compare and contrast React and Vue for large projects")
        assert result.action_type == ActionType.MULTI_AGENT

    def test_step_by_step(self) -> None:
        result = _router().classify("Give me a step-by-step guide to setting up CI/CD")
        assert result.action_type == ActionType.MULTI_AGENT

    def test_build_system(self) -> None:
        result = _router().classify("Build an API service for user authentication")
        assert result.action_type == ActionType.MULTI_AGENT


class TestEscalation:
    def test_security(self) -> None:
        result = _router().classify("There's a security vulnerability in our auth module")
        assert result.action_type == ActionType.ESCALATION

    def test_critical_incident(self) -> None:
        result = _router().classify("Critical incident: database is down")
        assert result.action_type == ActionType.ESCALATION

    def test_production_failure(self) -> None:
        result = _router().classify("Production deploy failed with errors")
        assert result.action_type == ActionType.ESCALATION


class TestClarification:
    def test_vague_short(self) -> None:
        result = _router().classify("help")
        assert result.action_type == ActionType.CLARIFICATION

    def test_confused_user(self) -> None:
        result = _router().classify("I'm not sure what to do, I'm confused")
        assert result.action_type == ActionType.CLARIFICATION

    def test_very_short(self) -> None:
        result = _router().classify("huh?")
        assert result.action_type == ActionType.CLARIFICATION


class TestRoutingMetadata:
    def test_has_model(self) -> None:
        result = _router().classify("Hello")
        assert result.model is not None
        assert len(result.model) > 0

    def test_has_budget(self) -> None:
        result = _router().classify("Hello")
        assert result.max_budget_usd > 0

    def test_confidence_range(self) -> None:
        result = _router().classify("Compare and contrast Python and Java")
        assert 0.0 <= result.confidence <= 1.0
