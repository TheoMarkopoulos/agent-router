"""Rule-based classifier: maps queries → 5 action types using keyword/regex patterns."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ActionType(StrEnum):
    DIRECT_RESPONSE = "direct_response"
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    ESCALATION = "escalation"
    CLARIFICATION = "clarification"


class RouteConfig(BaseModel):
    """Configuration for a single route."""

    description: str
    default_model: str
    fallback_model: str | None = None
    max_budget_usd: float


class RoutingDecision(BaseModel):
    """Result of classifying a query."""

    action_type: ActionType
    model: str
    max_budget_usd: float
    confidence: float


# --- Pattern definitions ---

_ESCALATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(security|vulnerability|exploit|breach|audit)\b", re.I),
    re.compile(r"\b(production|deploy|migration|rollback)\b.*\b(issue|fail|broke|error)", re.I),
    re.compile(r"\b(critical|urgent|emergency|incident)\b", re.I),
    re.compile(r"\b(architect|redesign|refactor.*entire|rewrite)\b", re.I),
]

_MULTI_AGENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(compare|contrast|analyze.*and)\b", re.I),
    re.compile(r"\b(step[- ]by[- ]step|multi[- ]step|pipeline|workflow)\b", re.I),
    re.compile(r"\b(build|create|implement|develop)\b.*\b(system|application|service|api)\b", re.I),
    re.compile(r"\b(research|investigate|explore)\b.*\b(and|then)\b", re.I),
]

_CLARIFICATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(what|huh|how|why|help)\??\s*$", re.I),
    re.compile(r"\b(not sure|don't know|unclear|confused|what do you mean)\b", re.I),
    re.compile(r"^(it|this|that)\s+(doesn't|isn't|won't)\b", re.I),
]

_DIRECT_RESPONSE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(hi|hello|hey|thanks|thank you|bye|goodbye)\b", re.I),
    re.compile(r"\b(what is|define|explain|who is|when was|where is)\b", re.I),
    re.compile(r"\b(translate|convert|calculate|count)\b", re.I),
    re.compile(r"\b(list|name|enumerate)\b.*\b(top|best|common|popular)\b", re.I),
    re.compile(r"^(yes|no|ok|sure|got it)\b", re.I),
]


def _match_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    """Return True if any pattern matches."""
    return any(p.search(text) for p in patterns)


class RuleRouter:
    """Classify queries into action types using keyword/regex rules."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "routes.yaml"
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        self._routes: dict[ActionType, RouteConfig] = {
            ActionType(k): RouteConfig(**v) for k, v in raw["routes"].items()
        }

    def classify(self, query: str) -> RoutingDecision:
        """Classify a query into an action type with confidence.

        Priority order: escalation > multi_agent > direct_response > clarification > single_agent.
        First match wins at high confidence; ties go to higher-priority category.
        """
        query = query.strip()

        # Priority-ordered checks — first match wins
        checks: list[tuple[ActionType, list[re.Pattern[str]]]] = [
            (ActionType.ESCALATION, _ESCALATION_PATTERNS),
            (ActionType.MULTI_AGENT, _MULTI_AGENT_PATTERNS),
            (ActionType.DIRECT_RESPONSE, _DIRECT_RESPONSE_PATTERNS),
            (ActionType.CLARIFICATION, _CLARIFICATION_PATTERNS),
        ]

        for action_type, patterns in checks:
            if _match_any(query, patterns):
                route = self._routes[action_type]
                return RoutingDecision(
                    action_type=action_type,
                    model=route.default_model,
                    max_budget_usd=route.max_budget_usd,
                    confidence=0.8,
                )

        # Default: single_agent
        route = self._routes[ActionType.SINGLE_AGENT]
        return RoutingDecision(
            action_type=ActionType.SINGLE_AGENT,
            model=route.default_model,
            max_budget_usd=route.max_budget_usd,
            confidence=0.5,
        )

    def get_route_config(self, action_type: ActionType) -> RouteConfig:
        """Return the route configuration for an action type."""
        return self._routes[action_type]
