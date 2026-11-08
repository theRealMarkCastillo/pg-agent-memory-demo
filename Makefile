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
# Seeding is idempotent and resets the demo companion user, so every run starts clean.
demo: seed
	$(COMPOSE) run --rm demo-agents
test:
	$(PYTHON) tests/run_isolated.py --engine-python $(ENGINE_PYTHON)
test-unit: test
extensions-demo:
	$(PYTHON) tests/run_isolated.py --engine-python $(ENGINE_PYTHON) --extensions-demo
extensions-eval:
	$(PYTHON) -m demo_agents.extensions.evaluate --mode scripted --output /tmp/pg-extension-evaluation.json
# Live model quality evaluation is explicit; this target runs deterministic logic checks.
test-eval: test
ci: test
# Explicitly destructive reset; normal down/ci preserve persistent volumes.
clean:
	$(COMPOSE) down -v
