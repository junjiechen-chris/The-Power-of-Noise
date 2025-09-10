# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core Python code (e.g., `generate_answers_llm.py`, `utils.py`, `index.py`, `retriever.py`).
- `conf/`: Hydra configs (`generation_config.yaml`, `corpus/*.yaml`) driving paths, retrievers, and LLM settings.
- `example_scripts/`: End‑to‑end shell scripts for common runs.
- `data/`: Large artifacts referenced by configs (not tracked). See README for sources.
- Docs: `README.md` (overview), `HYDRA_USAGE.md` (config examples).

## Build, Test, and Development Commands
- Environment (recommended): `./init_uv.sh` then `uv sync`.
- Run generation (Hydra overrides):
  - `uv run python src/generate_answers_llm.py generation.num_documents_in_context=2 generation.gold_position=0`
- Alternative setup (conda/pip):
  - `conda create -n power_of_noise python=3.11 -y && conda activate power_of_noise`
  - `pip install -r requirements.txt`
- Examples: `bash example_scripts/run_generation.sh`, `bash example_scripts/run_read_gen_res.sh`.

## Coding Style & Naming Conventions
- Python ≥ 3.11. Use 4‑space indentation and type hints where practical.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Prefer small, pure helpers in `src/`; keep notebooks exploratory.
- Formatting/linting: no enforced tool; prefer Black and Ruff locally (`black src && ruff check .`).

## Testing Guidelines
- Framework: pytest. Place tests under `tests/`, mirroring `src/` (e.g., `tests/test_utils.py`).
- Naming: files `test_*.py`, tests `test_*`.
- Keep tests fast: use tiny fixtures and set `llm.debug=true` via Hydra overrides.
- Run: `uv run pytest -q` (or `pytest -q`).

## Commit & Pull Request Guidelines
- Commits: imperative, present‑tense, concise (e.g., "add s3 sync script", "update contriever data config").
- PRs must include: summary, motivation, key commands/config overrides, before/after impact, linked issues, and doc updates when behavior changes.

## Security & Configuration Tips
- Do not commit credentials or large datasets. Use `.env` for tokens (e.g., HF/OpenAI/AWS); `dotenv` is loaded in runtime.
- Prefer config over code: adjust paths and behavior via Hydra (`conf/` + CLI overrides) rather than hardcoding.

## Architecture Overview
- RAG pipeline: retrieval (FAISS/Contriever/BM25) → context assembly → LLM generation (vLLM wrapper) → evaluation.
- Outputs are organized by Hydra in `gen_res/...` based on model/corpus and run time.

