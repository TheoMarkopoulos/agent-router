"""Learned routing classifier using sentence-transformer embeddings + sklearn MLP.

Drop-in replacement for RuleRouter. Same classify() interface, same RoutingDecision output.

# TODO: Upgrade path — replace embedding+MLP with fine-tuned Qwen3-3B for higher accuracy.
#   1. scripts/train_router.py → LoRA fine-tune Qwen3-3B on routing_dataset.jsonl
#   2. scripts/export_onnx.py → Export to ONNX for fast CPU inference
#   3. This file → swap MLP predict for ONNX session.run()
#   See CLAUDE.md Phase 2 for details.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.router.rule_router import ActionType, RouteConfig, RoutingDecision

import yaml


class LearnedRouterConfig(BaseModel):
    """Serialised metadata for a trained LearnedRouter."""

    embedding_model: str
    action_types: list[str]
    accuracy: float | None = None


class LearnedRouter:
    """Classify queries using sentence embeddings + MLP. Same interface as RuleRouter."""

    def __init__(
        self,
        clf: MLPClassifier,
        label_encoder: LabelEncoder,
        embedder: SentenceTransformer,
        config: LearnedRouterConfig,
        routes_path: Path | str | None = None,
    ) -> None:
        self._clf = clf
        self._le = label_encoder
        self._embedder = embedder
        self._config = config
        # Load route configs for model/budget assignment
        if routes_path is None:
            routes_path = Path(__file__).resolve().parents[2] / "configs" / "routes.yaml"
        with open(routes_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        self._routes: dict[ActionType, RouteConfig] = {
            ActionType(k): RouteConfig(**v) for k, v in raw["routes"].items()
        }

    def classify(self, query: str) -> RoutingDecision:
        """Classify a query into an action type with confidence."""
        embedding = self._embedder.encode([query])
        proba = self._clf.predict_proba(embedding)[0]
        best_idx = int(np.argmax(proba))
        confidence = float(proba[best_idx])
        action_type = ActionType(self._le.inverse_transform([best_idx])[0])

        route = self._routes[action_type]
        return RoutingDecision(
            action_type=action_type,
            model=route.default_model,
            max_budget_usd=route.max_budget_usd,
            confidence=confidence,
        )

    def get_route_config(self, action_type: ActionType) -> RouteConfig:
        """Return the route configuration for an action type."""
        return self._routes[action_type]

    # --- Training ---

    @classmethod
    def train(
        cls,
        queries: list[str],
        labels: list[str],
        embedding_model: str = "all-MiniLM-L6-v2",
        hidden_layers: tuple[int, ...] = (128, 64),
        max_iter: int = 500,
        routes_path: Path | str | None = None,
    ) -> LearnedRouter:
        """Train a new LearnedRouter from labelled data."""
        embedder = SentenceTransformer(embedding_model)
        embeddings = embedder.encode(queries, show_progress_bar=len(queries) > 100)

        le = LabelEncoder()
        y = le.fit_transform(labels)

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        clf.fit(embeddings, y)

        config = LearnedRouterConfig(
            embedding_model=embedding_model,
            action_types=list(le.classes_),
        )
        return cls(clf, le, embedder, config, routes_path=routes_path)

    # --- Persistence ---

    def save(self, path: Path | str) -> None:
        """Save trained model to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "clf": self._clf,
            "label_encoder": self._le,
            "config": self._config.model_dump(),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def from_pretrained(
        cls,
        path: Path | str,
        routes_path: Path | str | None = None,
    ) -> LearnedRouter:
        """Load a trained model from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301

        config = LearnedRouterConfig(**payload["config"])
        embedder = SentenceTransformer(config.embedding_model)
        return cls(
            clf=payload["clf"],
            label_encoder=payload["label_encoder"],
            embedder=embedder,
            config=config,
            routes_path=routes_path,
        )
