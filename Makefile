.PHONY: install test run dashboard deploy lint

install:
	uv venv && source .venv/bin/activate && uv pip install -e ".[dev,dashboard]"

test:
	pytest tests/ -v

run:
	python -m src.router.pipeline "What are the top 3 AI startups?"

dashboard:
	streamlit run src/dashboard/app.py

lint:
	ruff check src/ tests/

deploy:
	modal deploy scripts/deploy_modal.py
