.PHONY: up down build test test-unit test-eval clean

# Start the stack
up:
	docker compose up -d postgres memory-engine

# Stop everything
down:
	docker compose down -v

# Build images
build:
	docker compose build

# Run all tests (requires "make up" first)
test:
	cd tests && pip install -q -r requirements.txt 2>/dev/null; pytest -v --tb=short

# Run unit/integration tests only (fast)
test-unit:
	cd tests && pip install -q -r requirements.txt 2>/dev/null; pytest -v --tb=short --ignore=test_agent_evals.py

# Run quality evals only
test-eval:
	cd tests && pip install -q -r requirements.txt 2>/dev/null; pytest -v --tb=short test_agent_evals.py

# Start engines in background for development
dev:
	docker compose up -d postgres memory-engine

# Full CI flow: build, start, test, stop
ci: build up test down

# Clean everything
clean:
	docker compose down -v
	docker image rm -f pg-agent-memory-demo-memory-engine pg-agent-memory-demo-demo-agents 2>/dev/null; true
