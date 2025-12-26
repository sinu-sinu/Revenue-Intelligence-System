.PHONY: help setup start stop clean test lint format

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: ## Initial setup (Docker, database, demo data)
	@chmod +x scripts/setup.sh
	@./scripts/setup.sh

start: ## Start all services
	cd docker && docker-compose up -d
	@echo " Services started. Access UI at http://localhost:8501"

stop: ## Stop all services
	cd docker && docker-compose down
	@echo " Services stopped"

restart: stop start ## Restart all services

logs: ## View logs
	cd docker && docker-compose logs -f

db-shell: ## Open database shell
	docker exec -it revenue_intel_db psql -U app -d revenue_intel

db-reset: ## Reset database and reseed
	docker exec -it revenue_intel_db psql -U app -d revenue_intel -c "TRUNCATE accounts, sales_reps, products, deals, deal_stage_history CASCADE;"
	python database/seeds/seed_demo_data.py

clean: ## Remove containers and volumes
	cd docker && docker-compose down -v
	@echo "✅ Cleaned up"

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=. --cov-report=html --cov-report=term

lint: ## Run linters
	flake8 .
	mypy .

format: ## Format code
	black .
	isort .

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install dev dependencies
	pip install -r requirements.txt
	pre-commit install

train: ## Run model training
	cd docker && docker-compose --profile training up trainer

notebook: ## Start Jupyter notebook
	jupyter notebook notebooks/

