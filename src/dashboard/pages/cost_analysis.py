"""Cost analysis page: per-model breakdown, cost over time, savings vs naive."""

from __future__ import annotations

import streamlit as st

from src.dashboard.charts import cost_by_model_bar, cost_over_time, savings_gauge
from src.utils.logger import DuckDBLogger


def _get_naive_cost(logger: DuckDBLogger) -> float:
    """Compute what cost would be if every request used the most expensive model."""
    rows = logger.query("""
        SELECT model_used, cost_usd,
               latency_ms  -- unused, just to keep the query simple
        FROM request_logs
    """)
    if not rows:
        return 0.0
    # Find the highest avg cost-per-request model
    per_model = logger.query("""
        SELECT model_used, AVG(cost_usd) AS avg_cost
        FROM request_logs
        GROUP BY model_used
        ORDER BY avg_cost DESC
        LIMIT 1
    """)
    if not per_model:
        return 0.0
    most_expensive_avg = per_model[0]["avg_cost"]
    total_runs = logger.count()
    return most_expensive_avg * total_runs


def render(logger: DuckDBLogger) -> None:
    """Render the cost analysis page with breakdowns and savings gauge."""
    st.header("Cost Analysis")

    total = logger.count()
    if total == 0:
        st.info("No data yet.")
        return

    # Top-line cost metrics
    cost_stats = logger.query("""
        SELECT
            ROUND(SUM(cost_usd), 4) AS total_cost,
            ROUND(MIN(cost_usd), 6) AS min_cost,
            ROUND(MAX(cost_usd), 6) AS max_cost
        FROM request_logs
    """)[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Spend", f"${cost_stats['total_cost']:.4f}")
    c2.metric("Min Request Cost", f"${cost_stats['min_cost']:.6f}")
    c3.metric("Max Request Cost", f"${cost_stats['max_cost']:.6f}")

    st.divider()

    # Stacked bar: cost per model by route
    st.subheader("Cost by Model & Route")
    model_route = logger.query("""
        SELECT model_used, route_chosen, ROUND(SUM(cost_usd), 6) AS total_cost
        FROM request_logs
        GROUP BY model_used, route_chosen
        ORDER BY total_cost DESC
    """)
    st.plotly_chart(cost_by_model_bar(model_route), use_container_width=True)

    # Cost over time
    st.subheader("Cumulative Cost Over Time")
    time_data = logger.query("""
        SELECT
            timestamp AS ts,
            SUM(cost_usd) OVER (ORDER BY timestamp) AS cum_cost
        FROM request_logs
        ORDER BY timestamp
    """)
    st.plotly_chart(cost_over_time(time_data), use_container_width=True)

    # Savings gauge
    st.subheader("Cost Efficiency")
    actual = cost_stats["total_cost"]
    naive = _get_naive_cost(logger)
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.plotly_chart(savings_gauge(actual, naive), use_container_width=True)
    with col_right:
        st.metric("Actual Total", f"${actual:.4f}")
        st.metric("Naive Baseline", f"${naive:.4f}")
        saved = naive - actual
        st.metric("Saved", f"${saved:.4f}", delta=f"{saved / naive * 100:.1f}%" if naive else "0%")
