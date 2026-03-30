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

## Regra de execucao (pipelines)

- Nao gerar artefatos finais (modelos treinados, bases finais, tracking de experimento) a partir de notebooks.
- Notebooks sao para exploracao.
- Artefatos finais devem ser gerados por scripts parametrizaveis em `src/pipelines/`, executados via terminal.

---

## IMPORTANTE: Ambiente Virtual

**Sempre utilize um ambiente virtual (venv) para isolar as dependencias do projeto.**

O projeto suporta duas formas de gerenciar o ambiente virtual:

### Opcao 1: Com uv (Recomendado)

O [uv](https://docs.astral.sh/uv/) e um gerenciador de pacotes extremamente rapido (escrito em Rust) que gerencia automaticamente o ambiente virtual e dependencias.

**Instalar o uv:**

Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Ou com pip:
```bash
pip install uv
```

**Setup completo:**
```bash
make setup
```

Este comando ira:
1. Criar o ambiente virtual (`.venv`)
2. Instalar todas as dependencias do `pyproject.toml`
3. Instalar os pre-commit hooks

### Opcao 2: Com venv tradicional

Se preferir usar o fluxo tradicional:

```bash
# Criar ambiente virtual
python3 -m venv .venv

# Ativar ambiente virtual
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Instalar dependencias
uv pip install -e .

# Instalar dependencias de desenvolvimento
uv pip install -e ".[dev]"
```

> **Nota:** Usar `uv pip install` e muito mais rapido que `pip install`, mas requer que o ambiente virtual esteja ativado.

---

## Pre-commit Hooks

Este projeto utiliza [pre-commit](https://pre-commit.com/) para verificar automaticamente o codigo antes de cada commit. Os hooks incluem:

- **Ruff**: Linter e formatter para Python
- **MyPy**: Verificacao de tipos estaticos
- **Verificacoes gerais**: trailing whitespace, YAML valido, arquivos grandes, etc.

**Instalar hooks (incluido no `make setup`):**
```bash
# Com uv (automatico)
make setup

# Ou manualmente
uv run pre-commit install
```

**Executar verificacao manualmente:**
```bash
uv run pre-commit run --all-files
```

**Atualizar hooks para versoes mais recentes:**
```bash
uv run pre-commit autoupdate
```

---

## Executar Comandos

### Com uv (Recomendado)

Use `uv run` para executar comandos no ambiente virtual sem precisar ativa-lo:

```bash
# Executar script Python
uv run python src/train_example.py

# Executar MLflow UI
uv run mlflow ui --port 5000

# Executar testes
uv run pytest tests/ -v --cov=src

# Executar linting
uv run ruff check .
uv run ruff format .

# Verificar tipos
uv run mypy src/
```

### Com venv ativado

```bash
# Ativar ambiente virtual primeiro
source .venv/bin/activate  # Linux/Mac

# Depois executar comandos normalmente
python src/train_example.py
pytest tests/ -v --cov=src
mlflow ui --port 5000
```

---

## Gerenciar Dependencias

Adicione ou remova dependencias diretamente com `uv`:

```bash
# Adicionar dependencia de producao
uv add pandas

# Adicionar dependencia de desenvolvimento
uv add --dev pytest

# Remover dependencia
uv remove pandas

# Sincronizar ambiente
uv sync

# Listar pacotes instalados
uv pip list
```

Ou edite o `pyproject.toml` manualmente e execute `uv sync`.

---

## Comandos Uteis

```bash
# Setup inicial
make setup           # Criar ambiente, instalar dependencias e pre-commit hooks

# Ambiente
make sync            # Sincronizar ambiente com uv
make sync-no-dev      # Sincronizar sem dependencias dev
make uv-list          # Listar pacotes instalados

# MLflow
make mlflow          # Iniciar MLflow UI
make mlflow-clean    # Limpar runs (CUIDADO!)

# Desenvolvimento
make train           # Treinar modelo de exemplo
make test            # Rodar testes
make lint            # Verificar codigo com ruff
make format          # Formatar codigo com ruff
make typecheck       # Verificar tipos com mypy

# Utilitarios
make clean           # Limpar arquivos temporarios
make help            # Mostrar todos os comandos
```

---

## MLflow Setup

### Iniciar MLflow UI

```bash
make mlflow
# ou diretamente
uv run mlflow ui --port 5000
```

Acesse a interface em: **http://localhost:5000**

### Executar exemplo de treino

Para validar que o MLflow esta funcionando corretamente:

```bash
make train
# ou diretamente
uv run python src/train_example.py
```

### Estrutura de Pastas do MLflow

```
mlruns/
├── 0/                    # Experimento Default
├── 1/                    # Experimento "validacao-mlflow"
└── ...
```

**Nota**: A pasta `mlruns/` e criada automaticamente e esta no `.gitignore`.

### Variaveis de Ambiente

- `MLFLOW_TRACKING_URI`: URI de tracking (padrao: `file:./mlruns`)
- `MLFLOW_EXPERIMENT_NAME`: Nome do experimento (padrao: `tech-challenge-fase-1`)
- `MLFLOW_PORT`: Porta da UI (padrao: 5000)

Exemplo:
```bash
export MLFLOW_TRACKING_URI="file:./mlruns"
export MLFLOW_EXPERIMENT_NAME="meu-experimento"
make mlflow
```

### Configuracao por Ambiente

O projeto suporta tres ambientes:

```python
from src.config.mlflow_config import setup_mlflow, Environment, MLflowConfig

# Desenvolvimento (padrao) - arquivo local
setup_mlflow(environment=Environment.DEVELOPMENT)

# Staging - SQLite database
setup_mlflow(environment=Environment.STAGING)

# Producao - servidor remoto (requer MLFLOW_TRACKING_URI)
setup_mlflow(environment=Environment.PRODUCTION)
```

### Uso no Codigo

```python
from src.config.mlflow_config import setup_mlflow, MLflowConfig, Environment
import mlflow

# Configurar MLflow (desenvolvimento)
setup_mlflow(experiment_name="meu-experimento")

# Ou com configuracao avancada
config = MLflowConfig.for_env(Environment.DEVELOPMENT)
config.setup()

# Iniciar run e logar metricas
with mlflow.start_run(run_name="baseline"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(modelo, "model")
```

## Arquitetura

```
src/config/
├── __init__.py
└── mlflow_config.py        # Configuracao centralizada

src/
└── train_example.py         # Exemplo de treinamento

tests/
└── test_*.py               # Testes automatizados

scripts/
└── start_mlflow.sh         # Script de inicializacao
```

### Fluxo de Trabalho

1. **Setup**: `make setup` configura ambiente e pre-commit hooks
2. **Train**: Script de treino loga parametros e metricas
3. **Track**: MLflow armazena runs em `mlruns/`
4. **Visualize**: MLflow UI mostra resultados
