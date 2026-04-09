"""Cost calculator: maps (provider, model) → per-token pricing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ModelPricing(BaseModel):
    """Pricing info for a single model."""

    id: str
    provider: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    quality_tier: str


class UsageMetadata(BaseModel):
    """Token usage returned by an LLM call."""

    prompt_tokens: int
    completion_tokens: int


class CostCalculator:
    """Computes dollar cost from token usage and model identity."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        self._models: dict[str, ModelPricing] = {
            m["id"]: ModelPricing(**m) for m in raw["models"]
        }

    def get_pricing(self, model_id: str) -> ModelPricing:
        """Return pricing info for a model, raising KeyError if unknown."""
        if model_id not in self._models:
            raise KeyError(f"Unknown model: {model_id}")
        return self._models[model_id]

    def compute_cost(self, model_id: str, usage: UsageMetadata) -> float:
        """Return total cost in USD for the given usage."""
        pricing = self.get_pricing(model_id)
        input_cost = (usage.prompt_tokens / 1000) * pricing.cost_per_1k_input
        output_cost = (usage.completion_tokens / 1000) * pricing.cost_per_1k_output
        return input_cost + output_cost

    def list_models(self) -> list[ModelPricing]:
        """Return all registered model pricing entries."""
        return list(self._models.values())
