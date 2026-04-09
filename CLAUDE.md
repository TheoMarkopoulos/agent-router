# CLAUDE.md — AgentRouter Project

## CLAUDE CODE OUTPUT RULES
- No summaries after writing files. The file speaks for itself.
- No listing what you're "about to do." Just do it.
- No recap of what you built at the end. Just say "Done. Tests pass." or "Done. 2 failures:" + the errors.
- Batch reads: one grep beats reading 5 files sequentially.
- Never re-read a file you already read this session.


## TOKEN-EFFICIENT OPERATIONAL PROTOCOL

Every token costs money. Deliver maximum output quality while minimizing waste. Route tasks to the right model tier and effort level. No bloat. No filler. No over-engineering.

### Model Routing

| Tier | Use For | Examples in This Project |
|------|---------|------------------------|
| **Haiku** | Predictable, template-based | Config files, boilerplate endpoints, data formatting, git ops, running tests |
| **Sonnet** | Judgment + multi-file (DEFAULT) | Feature implementation, API integrations, debugging, dashboard components |
| **Opus** | High-stakes / Sonnet failed | Architecture decisions, bandit algorithm design, novel optimization logic |

### Effort Levels

- 1 file → Low | 2-5 files → Medium | 6+ files → High | Failed at High → Max
- Never start at Max. Earn it by failing at High.

### Output Rules

**Eliminate:** Restating problems, preambles, post-task summaries, explaining stdlib, comments on obvious code.
**Never cut:** Error handling, architecture rationale, non-obvious logic, complete implementations, real warnings.
Output tokens cost 5x input — cutting verbosity 30% saves more than cutting context 60%.

---

## PROJECT: AgentRouter

**What:** Cost-aware routing layer that decides which agent/model handles each sub-task in a multi-step pipeline, optimizing quality under a dollar budget.

**Papers:** AgentGate (arXiv:2604.06696) — two-stage structured routing. AgentOpt (arXiv:2604.06296) — Arm Elimination bandit for model selection.

**Stack:** Python 3.12 · LangGraph · Qwen3-3B (router) · Modal · LiteLLM · DuckDB · Streamlit · ONNX Runtime

---

## BUILD PLAN — Execute phases in order. Each phase has atomic tasks.

### PHASE 1: Foundation (Sonnet, Medium effort)

**Goal:** Working LangGraph pipeline with rule-based routing + LiteLLM multi-provider + DuckDB logging.

- [ ] `src/router/pipeline.py` — LangGraph StateGraph with RouterNode entry, agent executor nodes, and response aggregator
- [ ] `src/router/rule_router.py` — Rule-based classifier: maps queries → 5 action types (single_agent, multi_agent, direct_response, escalation, clarification) using keyword/regex patterns
- [ ] `src/utils/llm_client.py` — LiteLLM wrapper with provider configs (OpenAI, Anthropic, Groq). Load API keys from env.
- [ ] `src/utils/logger.py` — DuckDB logger: every request logs (timestamp, query, route_chosen, model_used, latency_ms, cost_usd, quality_score). Create table on init.
- [ ] `src/utils/cost.py` — Cost calculator: maps (provider, model) → per-token pricing. Compute cost from usage metadata.
- [ ] `configs/models.yaml` — Model registry: for each candidate model, list provider, model_id, cost_per_1k_input, cost_per_1k_output, quality_tier
- [ ] `configs/routes.yaml` — Route definitions: for each action type, list default_model, fallback_model, max_budget_usd
- [ ] `tests/test_rule_router.py` — Unit tests for rule router classification
- [ ] `tests/test_pipeline.py` — Integration test: send 3 queries through pipeline, verify routing + logging

### PHASE 2: Routing Classifier (Sonnet→Opus for training logic, Medium effort)

**Goal:** Fine-tuned Qwen3-3B that classifies queries into action types with 90%+ accuracy.

- [ ] `scripts/generate_training_data.py` — Use frontier model (claude-sonnet) to generate 5K query→action_type pairs with hard negatives. Output JSONL.
- [ ] `scripts/train_router.py` — LoRA fine-tune Qwen3-3B on the dataset. Use HuggingFace PEFT + bitsandbytes. Save adapter.
- [ ] `scripts/export_onnx.py` — Export fine-tuned model to ONNX for fast CPU inference.
- [ ] `src/router/learned_router.py` — Drop-in replacement for rule_router: loads ONNX model, classifies queries. Same interface.
- [ ] `scripts/eval_router.py` — Evaluate on held-out set, print classification report. Target: 90%+ accuracy.
- [ ] `data/routing_dataset.jsonl` — Generated training data (committed as sample, full generation via script)

### PHASE 3: Bandit Model Selector (Opus for algorithm, Medium effort)

**Goal:** Arm Elimination algorithm that finds Pareto-optimal model assignments per pipeline stage.

- [ ] `src/bandit/arm_elimination.py` — Core algorithm: maintain per-stage (model → cost, quality) estimates. Eliminate Pareto-dominated arms. Select best arm given budget constraint. UCB-style exploration bonus for under-sampled arms.
- [ ] `src/bandit/budget_manager.py` — Track cumulative spend per pipeline run. Enforce per-run and per-stage budget caps from configs.
- [ ] `src/router/pipeline.py` — UPDATE: integrate bandit selector. RouterNode picks action type, then bandit picks model for each stage.
- [ ] `tests/test_bandit.py` — Unit tests: verify elimination logic, budget enforcement, convergence on synthetic cost/quality data.

### PHASE 4: Dashboard (Sonnet, Medium effort)

**Goal:** Streamlit app showing cost breakdown, model utilization, quality metrics, and live demo.

- [ ] `src/dashboard/app.py` — Main Streamlit app with 3 tabs: Overview, Cost Analysis, Live Demo
- [ ] `src/dashboard/pages/overview.py` — Summary stats from DuckDB: total runs, avg cost, avg quality, route distribution pie chart
- [ ] `src/dashboard/pages/cost_analysis.py` — Per-stage cost breakdown bar chart, model utilization over time, cost savings vs naive (always-use-best-model) baseline
- [ ] `src/dashboard/pages/live_demo.py` — Text input → show routing decision, model assignment, response, cost, quality score in real time
- [ ] `src/dashboard/charts.py` — Reusable Plotly chart builders

### PHASE 5: Deploy & Polish (Sonnet, Low-Medium effort)

- [ ] `scripts/deploy_modal.py` — Modal deployment for routing classifier (GPU serverless)
- [ ] `pyproject.toml` — Project metadata, dependencies
- [ ] `README.md` — Architecture diagram (mermaid), quickstart, demo GIF placeholder, resume bullets
- [ ] `.env.example` — Required env vars
- [ ] `Makefile` — Common commands: `make install`, `make test`, `make run`, `make dashboard`, `make deploy`

---

## ARCHITECTURE NOTES

```
Query → RouterNode (action type classifier)
           │
           ├─ direct_response → single LLM call (cheapest model)
           ├─ single_agent → bandit picks best model for task
           ├─ multi_agent → LangGraph subgraph, bandit picks model per stage
           ├─ escalation → always use frontier model
           └─ clarification → ask user for more info
           │
           ▼
      Response + cost/quality logged to DuckDB
```

**Bandit warm-start:** First 50 runs use round-robin across all candidate models to build initial estimates. After that, Arm Elimination kicks in.

**Quality scoring:** For now, use LLM-as-judge (cheap model rates response 1-5). Later: user feedback signals.

---

## FILE CONVENTIONS

- Type hints on all functions
- Pydantic models for all configs and data structures
- `src/` is the package, `scripts/` are one-off runners
- Tests mirror src structure: `tests/test_<module>.py`
- No classes where a function suffices. No abstract base classes unless 3+ implementations exist.

## COMMANDS

```bash
# Setup
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Run pipeline
python -m src.router.pipeline "What are the top 3 AI startups?"

# Run dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest tests/ -v

# Generate training data
python scripts/generate_training_data.py --n 5000 --output data/routing_dataset.jsonl

# Train router
python scripts/train_router.py --data data/routing_dataset.jsonl --output models/router-lora

# Deploy
modal deploy scripts/deploy_modal.py
```
