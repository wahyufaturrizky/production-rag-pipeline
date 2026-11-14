.PHONY: install test lint eval ask run up down

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

eval:        ## golden QA set + release gates (exit 1 on regression)
	rag eval

ask:         ## make ask Q="How much does a BI-FAST transfer cost?"
	rag ask "$(Q)" --debug

run:         ## API on :8000 with in-memory store (no database needed)
	uvicorn rag.api:app --reload --port 8000

up:          ## Postgres + pgvector and the API via docker compose
	docker compose up --build

down:
	docker compose down -v
