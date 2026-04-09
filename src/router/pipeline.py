"""LangGraph StateGraph pipeline with RouterNode, agent executors, and response aggregator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable

from langgraph.graph import END, StateGraph

from src.bandit.arm_elimination import ArmElimination
from src.bandit.budget_manager import BudgetExceeded, BudgetManager
from src.router.rule_router import ActionType, RuleRouter, RoutingDecision
from src.utils.cost import CostCalculator, UsageMetadata
from src.utils.llm_client import LLMClient, LLMResponse
from src.utils.logger import DuckDBLogger, RequestLog


@runtime_checkable
class Router(Protocol):
    def classify(self, query: str) -> RoutingDecision: ...
    def get_route_config(self, action_type: ActionType) -> Any: ...


class PipelineState(TypedDict, total=False):
    """State flowing through the pipeline."""

    query: str
    routing_decision: RoutingDecision
    model_override: str | None
    llm_response: LLMResponse | None
    cost_usd: float
    latency_ms: float
    error: str | None


def _default_router(router_type: str = "rule") -> Router:
    """Instantiate a router by type name."""
    if router_type == "learned":
        from src.router.learned_router import LearnedRouter

        default_model_path = Path(__file__).resolve().parents[2] / "models" / "router.pkl"
        if not default_model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {default_model_path}. "
                "Run: python scripts/train_router.py"
            )
        return LearnedRouter.from_pretrained(default_model_path)
    return RuleRouter()


class Pipeline:
    """Main routing pipeline backed by LangGraph.

    After the RouterNode picks an action type, the bandit selector picks the
    model for each stage (falling back to the route's default_model when the
    bandit has insufficient data or is disabled).
    """

    def __init__(
        self,
        router: Router | None = None,
        llm_client: LLMClient | None = None,
        cost_calculator: CostCalculator | None = None,
        logger: DuckDBLogger | None = None,
        bandit: ArmElimination | None = None,
        budget_manager: BudgetManager | None = None,
        router_type: str = "rule",
        enable_bandit: bool = True,
    ) -> None:
        self._router = router or _default_router(router_type)
        self._llm = llm_client or LLMClient()
        self._cost = cost_calculator or CostCalculator()
        self._logger = logger or DuckDBLogger()
        self._bandit = bandit if enable_bandit else None
        self._budget = budget_manager
        self._graph = self._build_graph()

    # --- Node functions ---

    def _route_node(self, state: dict[str, Any]) -> dict[str, Any]:
        decision = self._router.classify(state["query"])

        # Reset budget tracking for each new run
        if self._budget:
            self._budget.reset()

        # Let the bandit override the model if available
        model_override: str | None = None
        if self._bandit is not None:
            stage = decision.action_type.value
            budget = (
                self._budget.remaining_stage_budget(stage)
                if self._budget
                else decision.max_budget_usd
            )
            model_override = self._bandit.select(stage, budget)
            # Verify the model is known to LLM client; fall back if not
            try:
                self._llm.get_model_config(model_override)
            except KeyError:
                model_override = None

        return {"routing_decision": decision, "model_override": model_override}

    def _agent_node(self, state: dict[str, Any]) -> dict[str, Any]:
        decision: RoutingDecision = state["routing_decision"]
        model = state.get("model_override") or decision.model
        query: str = state["query"]

        if decision.action_type == ActionType.CLARIFICATION:
            return {
                "llm_response": LLMResponse(
                    content="Could you please provide more details about what you need?",
                    model=model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                ),
                "cost_usd": 0.0,
                "latency_ms": 0.0,
            }

        messages = [{"role": "user", "content": query}]
        start = time.perf_counter()
        try:
            response = self._llm.complete(model, messages)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "llm_response": None,
                "error": str(e),
                "cost_usd": 0.0,
                "latency_ms": elapsed,
            }
        elapsed = (time.perf_counter() - start) * 1000

        usage = UsageMetadata(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        cost = self._cost.compute_cost(model, usage)

        # Record spend in budget manager
        stage = decision.action_type.value
        if self._budget:
            try:
                self._budget.record_spend(stage, cost)
            except BudgetExceeded:
                pass  # Log the cost anyway; budget is advisory here

        # Feed back to bandit (quality placeholder = 3.0 until LLM-as-judge is wired)
        if self._bandit is not None:
            self._bandit.update(stage, model, cost=cost, quality=3.0)

        return {
            "llm_response": response,
            "cost_usd": cost,
            "latency_ms": elapsed,
        }

    def _log_node(self, state: dict[str, Any]) -> dict[str, Any]:
        decision: RoutingDecision = state["routing_decision"]
        model = state.get("model_override") or decision.model
        self._logger.log(
            RequestLog(
                query=state["query"],
                route_chosen=decision.action_type.value,
                model_used=model,
                latency_ms=state.get("latency_ms", 0.0),
                cost_usd=state.get("cost_usd", 0.0),
            )
        )
        return {}

    # --- Graph construction ---

    def _build_graph(self) -> Any:
        graph = StateGraph(PipelineState)
        graph.add_node("router", self._route_node)
        graph.add_node("agent", self._agent_node)
        graph.add_node("logger", self._log_node)

        graph.set_entry_point("router")
        graph.add_edge("router", "agent")
        graph.add_edge("agent", "logger")
        graph.add_edge("logger", END)

        return graph.compile()

    # --- Public API ---

    def run(self, query: str) -> PipelineState:
        """Run a query through the full pipeline."""
        initial_state: PipelineState = {"query": query}
        result = self._graph.invoke(initial_state)
        return result

    @property
    def logger(self) -> DuckDBLogger:
        return self._logger

    @property
    def bandit(self) -> ArmElimination | None:
        return self._bandit

    @property
    def budget_manager(self) -> BudgetManager | None:
        return self._budget
