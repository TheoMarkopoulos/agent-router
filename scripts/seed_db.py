"""Seed DuckDB with 100 synthetic pipeline runs for dashboard demo."""

from __future__ import annotations

import random
import time
from pathlib import Path

from src.utils.logger import DuckDBLogger, RequestLog

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "agent_router.duckdb"

MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-sonnet-4-20250514",
    "claude-haiku-3-5",
    "llama-3.1-70b-versatile",
]

ROUTES = [
    "direct_response",
    "single_agent",
    "multi_agent",
    "escalation",
    "clarification",
]

# Route weights: single_agent most common, escalation rare
ROUTE_WEIGHTS = [0.25, 0.35, 0.20, 0.10, 0.10]

# Cost ranges per model (realistic per-request costs)
COST_RANGES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.0001, 0.001),
    "gpt-4o": (0.002, 0.02),
    "claude-sonnet-4-20250514": (0.003, 0.025),
    "claude-haiku-3-5": (0.0005, 0.003),
    "llama-3.1-70b-versatile": (0.0003, 0.002),
}

SAMPLE_QUERIES = [
    "What is machine learning?",
    "Compare React and Vue for a new project",
    "Build a REST API for user management",
    "Explain the Transformer architecture",
    "Hello!",
    "What are the top 5 programming languages?",
    "Help me debug this production error",
    "Security audit for our auth system",
    "Translate this paragraph to French",
    "Step-by-step guide to deploy on AWS",
    "Who invented Python?",
    "Calculate 15% of 230",
    "Analyze customer churn data and recommend actions",
    "I'm confused",
    "Research best practices for microservices and then summarize",
    "Critical: database migration failed in prod",
    "What is the capital of France?",
    "Create a CI/CD pipeline for our Node.js app",
    "Redesign the entire notification system",
    "Convert this CSV to JSON",
]

# Model-route affinity: escalation → expensive, direct_response → cheap
ROUTE_MODEL_WEIGHTS: dict[str, list[float]] = {
    "direct_response": [0.50, 0.05, 0.05, 0.35, 0.05],
    "single_agent": [0.15, 0.25, 0.25, 0.15, 0.20],
    "multi_agent": [0.05, 0.35, 0.35, 0.05, 0.20],
    "escalation": [0.02, 0.45, 0.45, 0.03, 0.05],
    "clarification": [0.60, 0.05, 0.05, 0.25, 0.05],
}


def seed(n: int = 100) -> None:
    """Insert n synthetic pipeline runs into the DuckDB database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = DuckDBLogger(db_path=DB_PATH)

    base_time = time.time() - (n * 60)  # spread over last N minutes

    for i in range(n):
        route = random.choices(ROUTES, weights=ROUTE_WEIGHTS, k=1)[0]
        model_weights = ROUTE_MODEL_WEIGHTS[route]
        model = random.choices(MODELS, weights=model_weights, k=1)[0]

        cost_lo, cost_hi = COST_RANGES[model]
        cost = random.uniform(cost_lo, cost_hi)

        latency = random.uniform(200, 5000) if route != "clarification" else random.uniform(5, 50)
        quality = random.uniform(2.0, 5.0) if route != "clarification" else None

        query = random.choice(SAMPLE_QUERIES)

        logger.log(RequestLog(
            timestamp=base_time + (i * 60) + random.uniform(0, 30),
            query=query,
            route_chosen=route,
            model_used=model,
            latency_ms=round(latency, 1),
            cost_usd=round(cost, 6),
            quality_score=round(quality, 2) if quality else None,
        ))

    print(f"Seeded {n} records into {DB_PATH}")
    logger.close()


if __name__ == "__main__":
    seed()
