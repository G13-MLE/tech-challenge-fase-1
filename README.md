# tech-challenge-fase-1

Estrutura inicial de pastas do projeto.

## Estrutura

```text
tech-challenge-fase-1/
├── src/
│   ├── data/
│   ├── features/
│   ├── eda/
│   ├── training/
│   ├── inference/
│   ├── schemas/
│   ├── api/
│   └── pipelines/
├── data/
├── models/
├── tests/
├── notebooks/
└── docs/
```

## Regra de execução (pipelines)

- Não gerar artefatos finais (modelos treinados, bases finais, tracking de experimento) a partir de notebooks.
- Notebooks são para exploração.
- Artefatos finais devem ser gerados por scripts parametrizáveis em `src/pipelines/`, executados via terminal.

## Sincronização do Ambiente

Para sincronizar o ambiente incluindo pacotes de desenolvimento utilizando o uv, use:

```bash
uv sync
```
Para sincronizar sem dependências de desenolvimento, útil para CI e contâiners docker, use:
```bash
uv sync --no-dev
```
## Comandos

```bash
# Setup
make setup         # Instala dependências

# Desenvolvimento
make mlflow        # UI do MLflow
make train         # Treinar modelo
make test          # Rodar testes
make lint          # Verificar código
make format        # Formatar código
make help          # Todos os comandos
```

## Variáveis de Ambiente

- `MLFLOW_PORT`: Porta padrão mlflow (5000)
- `MLFLOW_HOST`: Host padrão mlfow (127.0.0.1)
- `MLFLOW_TRACKING_URI`: URI de tracking (padrão: `file:./mlruns`)
- `MLFLOW_EXPERIMENT_NAME`: Nome do experimento
