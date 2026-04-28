Moveu src/config/logging.py para src/api/logging.py e atualizou todos os imports em src/api, src/pipelines e tests para evitar falhas de importação no container Docker da API.

Problemas adicionais resolvidos:
- Marcadores de conflito de merge (`<<<<<<< HEAD`) em `src/pipelines/run_mlp.py`
- Erros de tipo do mypy em `src/data/preprocessing.py`, `src/training/dummy_trainer.py`, `src/pipelines/run_mlp.py` e `src/pipelines/run_mlp_tuning.py`
- Dependências faltando no hook do mypy do pre-commit (`matplotlib` e `optuna`)
