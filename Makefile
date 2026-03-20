.PHONY: install train validate infer serve docker-up docker-down help

install:
	pip install .

train:
	python scripts/train.py

validate:
	python scripts/validate.py

infer:
	python scripts/infer.py --mode live

serve:
	python scripts/serve.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

help:
	@grep -E '^[a-zA-Z_-]+:.*' $(MAKEFILE_LIST) | awk -F: '{printf "  %-12s\n", $$1}'
