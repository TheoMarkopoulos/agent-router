"""DuckDB logger: logs every request with routing and cost metadata."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, Field


class RequestLog(BaseModel):
    """A single logged request."""

    timestamp: float = Field(default_factory=time.time)
    query: str
    route_chosen: str
    model_used: str
    latency_ms: float
    cost_usd: float
    quality_score: float | None = None


class DuckDBLogger:
    """Append-only logger backed by DuckDB."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._conn = duckdb.connect(str(db_path))
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                timestamp DOUBLE,
                query VARCHAR,
                route_chosen VARCHAR,
                model_used VARCHAR,
                latency_ms DOUBLE,
                cost_usd DOUBLE,
                quality_score DOUBLE
            )
        """)

    def log(self, entry: RequestLog) -> None:
        """Append a request log entry to the database."""
        self._conn.execute(
            """
            INSERT INTO request_logs
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                entry.timestamp,
                entry.query,
                entry.route_chosen,
                entry.model_used,
                entry.latency_ms,
                entry.cost_usd,
                entry.quality_score,
            ],
        )

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a SQL query and return results as a list of dicts."""
        result = self._conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def count(self) -> int:
        """Return the total number of logged requests."""
        return self._conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
