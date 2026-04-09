# AgentRouter — Complete Setup Guide
## From Zero to GitHub Repo Using Claude Code

---

## PART 1: PREREQUISITES (One-Time Setup)

### 1A — Install Required Software

**Node.js (required for Claude Code)**
```bash
# macOS
brew install node

# Windows — download from https://nodejs.org (LTS version)

# Linux
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

**Python 3.12+**
```bash
# macOS
brew install python@3.12

# Windows — download from https://python.org

# Linux
sudo apt install python3.12 python3.12-venv
```

**uv (fast Python package manager)**
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Git**
```bash
# macOS
brew install git

# Windows — download from https://git-scm.com

# Linux
sudo apt install git
```

**GitHub CLI (optional but makes repo creation easier)**
```bash
# macOS
brew install gh

# Windows
winget install GitHub.cli

# Linux
sudo apt install gh
```

### 1B — Install Claude Code

```bash
npm install -g @anthropic-ai/claude-code
```

Verify it works:
```bash
claude --version
```

### 1C — Set Up API Keys

You need at least ONE of these for the project to call LLMs. All three is ideal.

```bash
# Add to your shell profile (~/.zshrc or ~/.bashrc)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

Then reload:
```bash
source ~/.zshrc   # or source ~/.bashrc
```

---

## PART 2: CREATE THE PROJECT

### Step 1 — Create the project folder

```bash
mkdir agent-router
cd agent-router
```

### Step 2 — Unzip the scaffold I gave you

Take the `agent-router.zip` file I provided and unzip it INTO your `agent-router` folder. You should end up with this structure:

```
agent-router/
├── CLAUDE.md              ← Claude Code reads this automatically
├── pyproject.toml          ← Dependencies
├── Makefile                ← Common commands
├── .env.example
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml          ← GitHub Actions CI
├── README.md               ← Professional GitHub README
├── LICENSE
├── configs/
│   ├── models.yaml         ← Model registry + pricing
│   └── routes.yaml         ← Route definitions
├── src/
│   ├── __init__.py
│   ├── router/
│   │   └── __init__.py
│   ├── bandit/
│   │   └── __init__.py
│   ├── dashboard/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── data/
├── scripts/
└── docs/
    └── architecture.png    ← (generated later)
```

**If you didn't unzip but want to create it manually:**
Just copy each file I provide below into the matching path.

### Step 3 — Create your .env file

```bash
cp .env.example .env
```

Edit `.env` and paste your real API keys.

### Step 4 — Initialize git

```bash
git init
git add .
git commit -m "Initial scaffold: project structure, configs, CLAUDE.md"
```

---

## PART 3: LET CLAUDE CODE BUILD IT

### Step 1 — Launch Claude Code

```bash
cd agent-router
claude
```

Claude Code will automatically read `CLAUDE.md` and understand the entire project plan.

### Step 2 — Feed it the prompts below, ONE PHASE AT A TIME

Wait for each phase to finish before starting the next. Review the output between phases.

---

**PROMPT 1 — Phase 1: Foundation**

```
Build Phase 1 from CLAUDE.md. Implement every checked task in order:

1. src/utils/cost.py — cost calculator mapping (provider, model) → per-token pricing
2. src/utils/logger.py — DuckDB logger, create table on init, log every request
3. src/utils/llm_client.py — LiteLLM wrapper loading keys from .env, configs from configs/models.yaml
4. src/router/rule_router.py — rule-based classifier mapping queries → 5 action types using keyword/regex
5. src/router/pipeline.py — LangGraph StateGraph with RouterNode entry, agent executors, response aggregator
6. tests/test_rule_router.py — unit tests for all 5 action types
7. tests/test_pipeline.py — integration test sending 3 queries through the pipeline

Use Pydantic models for all data structures. Type hints everywhere. Load configs from YAML files in configs/. Run pytest at the end to verify everything passes.
```

**After it finishes, review and commit:**
```bash
git add .
git commit -m "Phase 1: Foundation — LangGraph pipeline, rule router, DuckDB logging"
```

---

**PROMPT 2 — Phase 2: Routing Classifier**

```
Build Phase 2 from CLAUDE.md. The routing classifier:

1. scripts/generate_training_data.py — Use litellm to call claude-sonnet to generate 5K query→action_type training pairs. Include hard negatives (queries that look like one type but are another). Output as JSONL to data/routing_dataset.jsonl. Include a --sample mode that generates 50 examples for testing without burning API credits.

2. src/router/learned_router.py — Drop-in replacement for rule_router. For now, implement it as a lightweight classifier using sentence-transformers embeddings + a small sklearn MLP trained on the JSONL data. Same interface as rule_router (takes query string, returns action type). Include a train() classmethod and a from_pretrained() classmethod.

3. scripts/train_router.py — Train the learned router on the JSONL dataset. Save model artifact to models/router.pkl.

4. scripts/eval_router.py — Evaluate on 20% held-out split, print sklearn classification_report. 

5. Update src/router/pipeline.py to accept either router via config flag.

6. tests/test_learned_router.py — test train + predict cycle on 20 synthetic examples.

Skip the Qwen3 fine-tuning and ONNX export for now — we'll use the embedding+MLP approach as v1 since it works on CPU with no GPU needed. Add a TODO comment noting the Qwen3 upgrade path.
```

**After it finishes:**
```bash
git add .
git commit -m "Phase 2: Learned router — embedding classifier with train/eval pipeline"
```

---

**PROMPT 3 — Phase 3: Bandit Model Selector**

```
Build Phase 3 from CLAUDE.md. The bandit model selector:

1. src/bandit/arm_elimination.py — Arm Elimination algorithm. Each "arm" is a (pipeline_stage, model) pair. Maintain running estimates of cost and quality per arm. Use UCB1 exploration bonus for under-sampled arms. Eliminate arms that are Pareto-dominated (worse cost AND worse quality than another arm). Select best arm given a budget constraint. Include a warm_start mode that round-robins first 50 runs.

2. src/bandit/budget_manager.py — Track cumulative spend per pipeline run. Enforce per-run and per-stage budget caps from configs/routes.yaml. Raise BudgetExceeded if a stage would blow the budget.

3. Update src/router/pipeline.py — After RouterNode picks action type, bandit picks the model for each execution stage. Fall back to config default_model if bandit has insufficient data.

4. tests/test_bandit.py — Test elimination logic (verify dominated arms get removed), budget enforcement (verify BudgetExceeded raised), and convergence (run 200 simulated rounds with known cost/quality, verify it converges to optimal).

Run pytest at the end.
```

**After it finishes:**
```bash
git add .
git commit -m "Phase 3: Bandit model selector — Arm Elimination with budget management"
```

---

**PROMPT 4 — Phase 4: Dashboard**

```
Build Phase 4 from CLAUDE.md. Streamlit dashboard:

1. src/dashboard/app.py — Main app with sidebar nav, 3 pages
2. src/dashboard/pages/overview.py — Query DuckDB for: total runs, avg cost, avg latency, route distribution (pie chart), quality distribution (histogram). Use plotly.
3. src/dashboard/pages/cost_analysis.py — Per-model cost breakdown (stacked bar), cost over time (line chart), cost savings vs always-use-best-model baseline (big number + comparison chart). 
4. src/dashboard/pages/live_demo.py — Text input box. On submit: run query through pipeline, show routing decision, model chosen, response, cost, quality score. Show a step-by-step trace of the routing decision.
5. src/dashboard/charts.py — Reusable plotly chart builder functions.

Seed DuckDB with 100 synthetic logged runs so the dashboard has data to show on first load. Put the seed script in scripts/seed_db.py.
```

**After it finishes:**
```bash
git add .
git commit -m "Phase 4: Streamlit dashboard with cost analysis and live demo"
```

---

**PROMPT 5 — Phase 5: Polish & Deploy**

```
Build Phase 5 from CLAUDE.md. Polish everything:

1. scripts/deploy_modal.py — Modal stub that deploys the routing classifier as a serverless endpoint. Include a health check route.

2. Update README.md — Make it professional:
   - Add a mermaid architecture diagram showing the full query flow
   - Add a "Quick Start" section with exact commands
   - Add a "How It Works" section explaining the 3 core components (router, bandit, dashboard)
   - Add a "Results" section with placeholder metrics
   - Add the resume impact bullets from the project doc
   - Add a "Built With" section listing the tech stack with links

3. Run ruff check and fix any linting issues across the entire codebase.

4. Run pytest and make sure everything passes.

5. Make sure every module has a docstring. Make sure every public function has a one-line docstring.
```

**After it finishes:**
```bash
git add .
git commit -m "Phase 5: Polish — README, linting, deploy script, docstrings"
```

---

## PART 4: PUSH TO GITHUB

### Option A — Using GitHub CLI (easiest)

```bash
gh repo create agent-router --public --description "Cost-aware routing layer for multi-agent LLM pipelines. Routes each sub-task to the cheapest model that can handle it." --push --source .
```

### Option B — Manual

1. Go to https://github.com/new
2. Name: `agent-router`
3. Description: `Cost-aware routing layer for multi-agent LLM pipelines`
4. Public, no README (we already have one), no .gitignore (we already have one)
5. Create, then:

```bash
git remote add origin https://github.com/YOUR_USERNAME/agent-router.git
git branch -M main
git push -u origin main
```

### Add GitHub Topics (makes it discoverable)

Go to your repo page → click the gear icon next to "About" → add topics:
```
llm, multi-agent, routing, cost-optimization, langgraph, bandit-algorithm, litellm, python
```

---

## PART 5: TROUBLESHOOTING

**Claude Code says "I don't have access to..."**
→ Make sure your API keys are exported in your shell AND in the `.env` file.

**Claude Code tries to over-engineer or goes off-plan**
→ Say: "Stop. Re-read CLAUDE.md. Follow the build plan exactly. Do not add features not listed."

**Tests fail after a phase**
→ Say: "Tests are failing. Run pytest -v, read the errors, and fix them. Do not move to the next phase until all tests pass."

**Claude Code runs out of context mid-phase**
→ Start a new Claude Code session. Say: "Read CLAUDE.md. Phase N is partially complete. Run the tests to see what's working, then finish the remaining tasks."

**Want to save tokens**
→ Your CLAUDE.md already has the token protocol baked in. Claude Code will route to the right effort level automatically.
