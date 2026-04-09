"""Overview page: summary stats, route distribution, quality histogram."""

from __future__ import annotations

import streamlit as st

from src.dashboard.charts import quality_histogram, route_distribution_pie
from src.utils.logger import DuckDBLogger


def render(logger: DuckDBLogger) -> None:
    """Render the overview page with summary stats and charts."""
    st.header("Overview")

    total = logger.count()
    if total == 0:
        st.info("No data yet. Run some queries or use `python scripts/seed_db.py`.")
        return

    # Summary stats
    stats = logger.query("""
        SELECT
            COUNT(*) AS total_runs,
            ROUND(AVG(cost_usd), 6) AS avg_cost,
            ROUND(AVG(latency_ms), 1) AS avg_latency,
            ROUND(AVG(quality_score), 2) AS avg_quality,
            ROUND(SUM(cost_usd), 4) AS total_cost
        FROM request_logs
    """)[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Runs", f"{stats['total_runs']:,}")
    c2.metric("Avg Cost", f"${stats['avg_cost']:.6f}")
    c3.metric("Avg Latency", f"{stats['avg_latency']:.0f} ms")
    c4.metric("Avg Quality", f"{stats['avg_quality']:.2f}" if stats["avg_quality"] else "N/A")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Route Distribution")
        route_data = logger.query("""
            SELECT route_chosen, COUNT(*) AS cnt
            FROM request_logs
            GROUP BY route_chosen
            ORDER BY cnt DESC
        """)
        st.plotly_chart(route_distribution_pie(route_data), use_container_width=True)

    with col_right:
        st.subheader("Quality Score Distribution")
        scores_raw = logger.query("""
            SELECT quality_score FROM request_logs
            WHERE quality_score IS NOT NULL
        """)
        scores = [r["quality_score"] for r in scores_raw]
        if scores:
            st.plotly_chart(quality_histogram(scores), use_container_width=True)
        else:
            st.info("No quality scores recorded yet.")
