.PHONY: install install-dev dataset tune train validate test eval plot infer serve docker-up docker-down help

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

dataset:
	python scripts/build_dataset.py

tune:
	python scripts/tune.py --horizon 120 --trials 60

train:
	python scripts/train.py

validate:
	python scripts/validate.py

test:
	python scripts/test_suite.py

eval:
	python scripts/evaluate.py

plot:
	python scripts/plot.py

infer:
	python scripts/infer.py --mode live

serve:
	python scripts/serve.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

help:
	@grep -E '^[a-zA-Z_-]+:' $(MAKEFILE_LIST) | awk -F: '{printf "  %-14s\n", $$1}'
