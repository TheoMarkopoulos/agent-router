"""Reusable Plotly chart builders for the dashboard."""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go


def route_distribution_pie(data: list[dict[str, Any]]) -> go.Figure:
    """Pie chart of route_chosen counts."""
    routes = [r["route_chosen"] for r in data]
    counts = [r["cnt"] for r in data]
    fig = px.pie(names=routes, values=counts, hole=0.4)
    fig.update_layout(margin=dict(t=30, b=10, l=10, r=10), height=350)
    return fig


def quality_histogram(scores: list[float]) -> go.Figure:
    """Histogram of quality scores."""
    fig = px.histogram(x=scores, nbins=10, labels={"x": "Quality Score"})
    fig.update_layout(
        margin=dict(t=30, b=40, l=40, r=10),
        height=350,
        xaxis_title="Quality Score",
        yaxis_title="Count",
    )
    return fig


def cost_by_model_bar(data: list[dict[str, Any]]) -> go.Figure:
    """Stacked bar: total cost per model, split by route."""
    models = []
    routes = []
    costs = []
    for row in data:
        models.append(row["model_used"])
        routes.append(row["route_chosen"])
        costs.append(row["total_cost"])

    fig = px.bar(
        x=models,
        y=costs,
        color=routes,
        labels={"x": "Model", "y": "Total Cost ($)", "color": "Route"},
        barmode="stack",
    )
    fig.update_layout(margin=dict(t=30, b=40, l=40, r=10), height=400)
    return fig


def cost_over_time(data: list[dict[str, Any]]) -> go.Figure:
    """Line chart of cumulative cost over time."""
    timestamps = [r["ts"] for r in data]
    cum_cost = [r["cum_cost"] for r in data]
    fig = go.Figure(go.Scatter(x=timestamps, y=cum_cost, mode="lines+markers", name="Cumulative"))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Cumulative Cost ($)",
        margin=dict(t=30, b=40, l=40, r=10),
        height=400,
    )
    return fig


def savings_gauge(actual: float, naive: float) -> go.Figure:
    """Gauge showing savings vs naive baseline."""
    savings_pct = ((naive - actual) / naive * 100) if naive > 0 else 0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=savings_pct,
            number={"suffix": "%"},
            delta={"reference": 0, "increasing": {"color": "green"}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2ecc71"},
                "steps": [
                    {"range": [0, 30], "color": "#fadbd8"},
                    {"range": [30, 60], "color": "#fdebd0"},
                    {"range": [60, 100], "color": "#d5f5e3"},
                ],
            },
            title={"text": "Cost Savings vs Naive"},
        )
    )
    fig.update_layout(margin=dict(t=60, b=10, l=30, r=30), height=300)
    return fig
