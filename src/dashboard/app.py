"""Main Streamlit dashboard with sidebar navigation."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.dashboard.pages import cost_analysis, live_demo, overview
from src.utils.logger import DuckDBLogger

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "agent_router.duckdb"


@st.cache_resource
def get_logger() -> DuckDBLogger:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return DuckDBLogger(db_path=DB_PATH)


def main() -> None:
    """Launch the Streamlit dashboard with sidebar navigation."""
    st.set_page_config(page_title="AgentRouter Dashboard", layout="wide")
    st.title("AgentRouter Dashboard")

    logger = get_logger()

    page = st.sidebar.radio("Navigation", ["Overview", "Cost Analysis", "Live Demo"])

    if page == "Overview":
        overview.render(logger)
    elif page == "Cost Analysis":
        cost_analysis.render(logger)
    else:
        live_demo.render(logger)


if __name__ == "__main__":
    main()
