"""Live demo page: run a query through the pipeline and display results."""

from __future__ import annotations

import streamlit as st

from src.utils.logger import DuckDBLogger


def render(logger: DuckDBLogger) -> None:
    """Render the live demo page for interactive query routing."""
    st.header("Live Demo")

    st.markdown("Type a query below to see how the router classifies it and which model handles it.")

    query = st.text_input("Query", placeholder="e.g. What are the top 3 AI startups?")

    if st.button("Run", type="primary") and query:
        with st.spinner("Routing and executing..."):
            try:
                from src.router.pipeline import Pipeline

                pipeline = Pipeline(logger=logger, enable_bandit=False)
                result = pipeline.run(query)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                return

        decision = result.get("routing_decision")
        response = result.get("llm_response")
        error = result.get("error")

        # Routing info
        st.subheader("Routing Decision")
        r1, r2, r3 = st.columns(3)
        r1.metric("Action Type", decision.action_type.value if decision else "N/A")
        r2.metric("Model", decision.model if decision else "N/A")
        r3.metric("Confidence", f"{decision.confidence:.0%}" if decision else "N/A")

        # Model override (bandit)
        override = result.get("model_override")
        if override and override != (decision.model if decision else None):
            st.info(f"Bandit override: using **{override}** instead of default **{decision.model}**")

        st.divider()

        # Cost & latency
        st.subheader("Performance")
        p1, p2 = st.columns(2)
        p1.metric("Cost", f"${result.get('cost_usd', 0):.6f}")
        p2.metric("Latency", f"{result.get('latency_ms', 0):.0f} ms")

        st.divider()

        # Response
        st.subheader("Response")
        if error:
            st.error(error)
        elif response:
            st.markdown(response.content)
            with st.expander("Token usage"):
                st.json({
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "total_tokens": response.total_tokens,
                })
        else:
            st.warning("No response generated.")
