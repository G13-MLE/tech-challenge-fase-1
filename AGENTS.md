# AGENTS.md - TechChallenge1ChurnTelco

ML pipeline for telecom churn prediction. Python 3.12+, PyTorch MLP, scikit-learn baselines, MLflow tracking, FastAPI inference.

## Philosophy: Simplicity First

- **Prefer simple solutions** - avoid over-engineering
- **Question every addition** - does this really need to be added?
- **Function over form** - working code beats perfect architecture
- **Less is more** - fewer lines, fewer files, fewer dependencies

### Code Sharing Between Experiments

Three ML pipelines (dummy, MLP, logistic) share common infrastructure but keep experiment-specific code explicit:

**Shared (extracted to modules):**
- Data loading: `src.data.load.load_telco_data()`
- Data split: `src.data.load.split_train_test_stratified()`
- MLflow setup: `src.training.mlflow_tracking.setup_mlflow()`, `build_mlflow_inputs()`
- Metrics: `src.training.metrics.compute_binary_classification_metrics()`

**Duplicated OK (kept per-pipeline for clarity):**
- Model-specific param logging (each model logs different values)
- Preprocessing (Dummy uses strings, MLP uses numeric tensors)
- Results post-processing (CSV comparison vs torch model saving)
- CLI argument parsing (only where needed)

**Rule:** Extract helpers only when genuinely identical AND extraction does not increase abstraction complexity. Prefer explicit duplication over clever abstractions.

## Commands

Uses `uv` as package manager. Always prefix Python commands with `uv run`.

```bash
# Setup
make setup                    # uv sync + pre-commit install + DVC setup
uv sync                       # Install deps (add --no-dev for CI/Docker)

# Testing (80% coverage minimum enforced)
make test                     # Full test suite with coverage
uv run pytest tests/ -v       # Run all tests
uv run pytest -m fast -v      # Quick tests only
uv run pytest -m slow -v      # Integration tests only

# Quality
make lint                     # Check with ruff
make format                   # Format with ruff
uv run mypy src/              # Type check (strict mode)

# MLflow (requires .env)
make docker-up                # Start MLflow + PostgreSQL + MinIO
make docker-down              # Stop containers
```

## ML Experiments

Three baseline experiments tracked in MLflow:

| Experiment | Command | Status |
|------------|---------|--------|
| Dummy Baseline | `make train-dummy` or `uv run python -m src.pipelines.run_dummy_baseline` | Ready |
| MLP | `make train-mlp` or `uv run python -m src.pipelines.run_mlp` | Ready |
| Logistic Regression | `make train-logistic` (planned) | Future |

All experiments share:
- Data loading: `src.data.prepare_telco_dataset.load_telco_data()`
- Split: `src.data.splitting.split_train_test_stratified()`
- MLflow setup: `src.training.mlflow_tracking.setup_mlflow()`
- Metrics: `src.training.metrics.compute_binary_classification_metrics()`

## Code Style (Strict)

- **Line length:** 79 characters (not 88)
- **Python:** 3.12+ with `from __future__ import annotations` always
- **Type hints:** Required on all functions (`disallow_untyped_defs = true`)
- **Import order:** stdlib → third-party → local (enforced by ruff I rule)
- **Use `TYPE_CHECKING`** for imports only needed for type hints

### Patterns

```python
# Dataclasses for config
@dataclass(frozen=True)
class Config:
    value: str

# Protocols for interfaces
class Trainer(Protocol):
    def train(self, data: Data) -> Model: ...

# Error handling
except SpecificException as e:  # Be specific
except Exception as e:  # noqa: BLE001  # Rare, mark with noqa
```

## Project Structure

### Gerenciamento de Ambiente
- Use arquivo `.env` para configuração local (copie de `.env.example`)
- Nunca commite arquivos `.env`
- Use `python-dotenv` para carregar variáveis de ambiente
- Docker Compose usa arquivo `.env` automaticamente

### Dependências
- Deps de produção: listadas em `[project] dependencies`
- Deps de dev: listadas em `[dependency-groups] dev`
- Use `uv add <package>` para adicionar dependências de produção
- Use `uv add --dev <package>` para adicionar dependências de dev
- Arquivo de lock `uv.lock` deve ser commitado

### Fluxo de Trabalho Git
- **NUNCA crie commits ou faça `git push` sozinho.** Sempre aguarde o usuário pedir ou deixe para que o usuário faça o commit e o push.
- Hooks do pre-commit rodam automaticamente no commit
- CI corrige automaticamente PRs com pre-commit
- Use mensagens de commit convencionais

## Arquitetura do Projeto

### Estrutura de Diretórios
```
src/
├── api/           # FastAPI (empty - .gitkeep)
├── data/          # Data loading (empty - .gitkeep)
├── eda/           # EDA helpers (empty - .gitkeep)
├── features/      # Feature pipelines (empty - .gitkeep)
├── inference/     # Prediction service (empty - .gitkeep)
├── pipelines/     # End-to-end scripts (empty - .gitkeep)
├── schemas/       # Pydantic models (empty - .gitkeep)
└── training/      # Model training (empty - .gitkeep)

data/              # DVC-tracked datasets
├── raw/           # Original data
└── processed/     # Transformed data

models/            # Saved model artifacts
tests/             # pytest tests (test_*.py)
notebooks/         # Exploration only - NO artifacts
docs/              # Documentation
```

## Critical Rules

1. **NO artifacts from notebooks** - notebooks are exploration only; final artifacts must come from `src/pipelines/` scripts
2. **All code needs type hints** - mypy runs in strict mode
3. **Coverage minimum 80%** - enforced in CI via pyproject.toml
4. **Test markers:** `@pytest.mark.fast` for quick tests, `@pytest.mark.slow` for integration
5. **NO emojis anywhere** - use ASCII text equivalents (see Text Style section below)
6. **NO unrequested documentation** - do not create .md files, CHANGELOGs, or docs unless explicitly asked
7. **NO deprecated code** - if code is not used, remove it; do not keep "for future use" or "backward compatibility"

## Text and Documentation Style

**NO emojis in any file** - this includes source code, documentation, comments, commit messages, and shell scripts. Use ASCII text equivalents:

| Instead of | Use |
|------------|-----|
| Checkmark | `[OK]`, `Success:`, `Done:` |
| Cross | `[ERROR]`, `Error:` |
| Warning | `[WARN]`, `Warning:` |
| Whale | `Docker:` |
| Rocket | `Starting...`, `Launching...` |
| Lightbulb | `Tip:`, `Note:` |
| Chart | `Results:`, `Metrics:` |
| Target | `Goal:`, `Next:` |
| Wrench | `Config:`, `Setup:` |
| Folder | `Directory:`, `Folder:` |
| Clipboard | `List:`, `Summary:` |

**NO emoji in:**
- Source code (Python files)
- Documentation (README, AGENTS.md, etc)
- Comments (inline or block)
- Commit messages
- Shell scripts (Makefile, .sh files)
- Configuration files

## Environment Setup

```bash
# Required before running MLflow or pipeline scripts
cp .env.example .env          # Then edit with your values
```

Key env vars:
- `MLFLOW_TRACKING_URI=http://localhost:5000`
- `MLFLOW_S3_ENDPOINT_URL=http://localhost:9000`
- `MLFLOW_DUMMY_EXPERIMENT_NAME=tech-challenge-dummy-baseline`
- `DVC_ONEDRIVE_REMOTE_URL=` (set during `make setup`)

## MLflow Integration

```python
from dotenv import load_dotenv
import mlflow

load_dotenv()

with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

## Dependencies

- Add prod: `uv add <package>`
- Add dev: `uv add --dev <package>`
- Lock file: `uv.lock` (must be committed)

Key deps: fastapi, pandas, scikit-learn, torch, mlflow, pytest, ruff, mypy

## Important
- Always run make test and make lint after any code update
