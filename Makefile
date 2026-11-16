UV				?= uv
UV_PROJECT_ENV	:= .venv
CACHE_DIR		:= .cache
UV_CACHE_DIR	:= $(CACHE_DIR)/uv
HF_CACHE_DIR	:= $(CACHE_DIR)/huggingface
UV_ENV			:= UV_CACHE_DIR=$(UV_CACHE_DIR) UV_PROJECT_ENVIRONMENT=$(UV_PROJECT_ENV) HF_HOME=$(HF_CACHE_DIR)

ARGS			?=

.PHONY: install run debug lint clean clean_data fclean

install:
	$(UV_ENV) $(UV) sync
	$(UV_ENV) $(UV) run hf download Qwen/Qwen3-0.6B

run:
	$(UV_ENV) $(UV) run python -B -m src $(ARGS)

debug:
	$(UV_ENV) $(UV) run python -B -m pdb -m src $(ARGS)

lint: clean
	$(UV_ENV) $(UV) run flake8 src
	$(UV_ENV) $(UV) run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

clean_data: clean
	rm -rf data/processed/
	rm -rf data/output/

fclean: clean
	rm -rf $(UV_PROJECT_ENV)
	rm -rf $(CACHE_DIR)
