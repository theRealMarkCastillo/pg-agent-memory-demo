COMPOSE = docker compose --env-file .env --env-file .env.auth
PYTHON ?= python
ENGINE_PYTHON ?= $(PYTHON)
.PHONY: up down build seed demo test test-unit test-eval ci clean
up:
	$(COMPOSE) up -d postgres memory-engine
down:
	$(COMPOSE) down
build:
	$(COMPOSE) build
seed:
	$(COMPOSE) --profile setup run --rm demo-seed
demo:
	$(COMPOSE) run --rm demo-agents
test:
	$(PYTHON) tests/run_isolated.py --engine-python $(ENGINE_PYTHON)
test-unit: test
# Live model quality evaluation is explicit; this target runs deterministic logic checks.
test-eval: test
ci: test
# Explicitly destructive reset; normal down/ci preserve persistent volumes.
clean:
	$(COMPOSE) down -v
