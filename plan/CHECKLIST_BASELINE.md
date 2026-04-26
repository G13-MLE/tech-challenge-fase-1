"""Checklist local da Issue #20 - DummyClassifier baseline.
"""

# Issue #20 — Checklist local (pré-PR)

## Objetivo
Executar e validar localmente o baseline `DummyClassifier` com estratégias:

- `most_frequent`
- `stratified`
- `uniform`

## Pré-requisitos

1. Ambiente sincronizado:

```bash
uv sync
```

2. Dados disponíveis localmente (DVC):

```bash
dvc status
```

3. Stack MLflow local ativa (Docker):

```bash
docker compose -f docker/docker-compose.yml up -d
```

## Execução do pipeline

```bash
uv run python -m src.pipelines.run_dummy_baseline
```

Saídas esperadas:

- 4 runs no MLflow (`dummy_most_frequent`, `dummy_stratified`,
  `dummy_uniform`, `dummy_comparison_summary`)
- arquivo `models/dummy_baseline_comparison.csv`

## Validação de qualidade

```bash
uv run ruff check src/pipelines/run_dummy_baseline.py src/data/validation.py src/data/splitting.py src/data/versioning.py src/training/metrics.py src/training/mlflow_tracking.py tests/pipelines/test_run_dummy_baseline.py
uv run pytest tests/pipelines/test_run_dummy_baseline.py tests/data/test_prepare_telco_dataset.py -v
```

## Observações

- `random_seed=42` está aplicado no split e no `DummyClassifier`.
- O pipeline em `src/pipelines/` atua como orquestrador, reutilizando
  funções modulares em `src/data/` e `src/training/`.
- Tags de contexto no MLflow incluem:
  - `issue=20`
  - `baseline_family=dummy`
  - `model_baseline=dummy_classifier`
  - `random_seed=42`
- Nenhum commit/push foi realizado durante estas fases.
