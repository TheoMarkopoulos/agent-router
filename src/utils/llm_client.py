"""LiteLLM wrapper with provider configs loaded from models.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

import litellm

load_dotenv()


class ModelConfig(BaseModel):
    """Configuration for a single LLM model."""

    id: str
    provider: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    quality_tier: str


class LLMResponse(BaseModel):
    """Standardised response from an LLM call."""

    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    """Thin wrapper around LiteLLM for multi-provider completion."""

    def __init__(self, config_path: Path | str | None = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        self._models: dict[str, ModelConfig] = {
            m["id"]: ModelConfig(**m) for m in raw["models"]
        }

    def get_model_config(self, model_id: str) -> ModelConfig:
        """Return config for a model, raising KeyError if unknown."""
        if model_id not in self._models:
            raise KeyError(f"Unknown model: {model_id}")
        return self._models[model_id]

    def _litellm_model_name(self, model_id: str) -> str:
        """Map our model id to litellm's expected format."""
        cfg = self.get_model_config(model_id)
        provider = cfg.provider
        if provider == "groq":
            return f"groq/{model_id}"
        if provider == "anthropic":
            return f"anthropic/{model_id}"
        # openai models pass through directly
        return model_id

    def complete(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a completion request via LiteLLM."""
        litellm_model = self._litellm_model_name(model_id)
        response = litellm.completion(model=litellm_model, messages=messages, **kwargs)
        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content,
            model=model_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    def list_models(self) -> list[ModelConfig]:
        """Return all registered model configurations."""
        return list(self._models.values())
