# Makefile para o Tech Challenge Fase 1
# Configuracao de MLflow tracking
# Gerenciamento de ambiente com uv

.PHONY: help setup venv venv-activate sync sync-no-dev mlflow mlflow-clean train test lint format typecheck clean uv-list pip-install pip-install-dev

# Variaveis
MLFLOW_PORT = 5000
MLFLOW_HOST = 127.0.0.1
VENV_DIR = .venv

# Verificar se uv esta instalado
UV_CHECK := $(shell which uv 2>/dev/null)
ifdef UV_CHECK
	UV := uv
else
	UV := echo "[ERRO] uv nao encontrado. Instale com: curl -LsSf https://astral.sh/uv/install.sh | sh" && false
endif

# Help padrao
help:
	@echo "Tech Challenge Fase 1 - Comandos Disponiveis"
	@echo ""
	@echo "IMPORTANTE: Sempre use o ambiente virtual (venv)!"
	@echo "  Opcao 1: 'make setup' para setup automatico com uv"
	@echo "  Opcao 2: 'make venv' + ativar manualmente"
	@echo ""
	@echo "Setup inicial:"
	@echo "  make setup          - Setup completo (venv + deps + pre-commit)"
	@echo "  make venv           - Criar ambiente virtual (.venv)"
	@echo ""
	@echo "Ambiente (com uv - recomendado):"
	@echo "  make sync           - Sincronizar ambiente com uv"
	@echo "  make sync-no-dev    - Sincronizar sem dependencias de dev"
	@echo "  make uv-list        - Listar pacotes instalados"
	@echo ""
	@echo "Ambiente (com venv ativado - alternativa):"
	@echo "  make pip-install    - Instalar dependencias (requer venv ativado)"
	@echo "  make pip-install-dev- Instalar com dependencias de dev (requer venv ativado)"
	@echo ""
	@echo "MLflow:"
	@echo "  make mlflow         - Iniciar MLflow UI"
	@echo "  make mlflow-clean   - Limpar runs (CUIDADO!)"
	@echo ""
	@echo "Desenvolvimento:"
	@echo "  make train          - Executar exemplo de treino"
	@echo "  make test           - Rodar testes"
	@echo "  make lint           - Verificar codigo com ruff"
	@echo "  make format         - Formatar codigo com ruff"
	@echo "  make typecheck      - Verificar tipos com mypy"
	@echo ""
	@echo "Utilitarios:"
	@echo "  make clean          - Limpar arquivos temporarios"
	@echo "  make help           - Mostrar esta mensagem"

# ------------------------------------------------------------------
# SETUP INICIAL
# ------------------------------------------------------------------

# Setup completo: criar venv, sincronizar e instalar pre-commit hooks
setup:
	@echo "=========================================="
	@echo "Configurando ambiente virtual..."
	@echo "=========================================="
	$(UV) sync
	@echo ""
	@echo "Instalando pre-commit hooks..."
	$(UV) run pre-commit install
	@echo ""
	@echo "=========================================="
	@echo "Setup concluido com sucesso!"
	@echo "=========================================="
	@echo ""
	@echo "O ambiente virtual (.venv) foi criado."
	@echo "Pre-commit hooks instalados."
	@echo ""
	@echo "Proximos passos:"
	@echo "  - Use 'uv run <comando>' para executar sem ativar o venv"
	@echo "  - Ou ative o venv: source .venv/bin/activate"
	@echo ""

# Criar ambiente virtual manualmente
venv:
	@echo "Criando ambiente virtual em $(VENV_DIR)/..."
	@echo ""
	$(UV) venv
	@echo ""
	@echo "Ambiente virtual criado!"
	@echo ""
	@echo "Para ativar, execute:"
	@echo "  source $(VENV_DIR)/bin/activate   (Linux/Mac)"
	@echo "  $(VENV_DIR)\\Scripts\\activate    (Windows)"
	@echo ""
	@echo "Depois instale as dependencias:"
	@echo "  make pip-install"
	@echo ""

# ------------------------------------------------------------------
# COMANDOS DE AMBIENTE (uv - recomendado)
# ------------------------------------------------------------------

# Sincronizar ambiente (padrao)
sync:
	@echo "Sincronizando ambiente com uv..."
	$(UV) sync
	@echo "Ambiente sincronizado!"

# Sincronizar sem dependencias de desenvolvimento
sync-no-dev:
	@echo "Sincronizando ambiente (sem dev deps)..."
	$(UV) sync --no-dev
	@echo "Ambiente sincronizado!"

# Listar pacotes instalados
uv-list:
	@echo "Pacotes instalados:"
	$(UV) pip list

# ------------------------------------------------------------------
# COMANDOS DE AMBIENTE (venv ativado - alternativa)
# ------------------------------------------------------------------

# Instalar dependencias com pip (requer venv ativado)
pip-install:
ifndef VIRTUAL_ENV
	@echo "ERRO: Ambiente virtual nao esta ativado!"
	@echo ""
	@echo "Ative o ambiente virtual primeiro:"
	@echo "  source .venv/bin/activate   (Linux/Mac)"
	@echo "  .venv\\Scripts\\activate    (Windows)"
	@echo ""
	@echo "Ou use 'make sync' para sincronizar com uv automaticamente."
	@exit 1
endif
	@echo "Instalando dependencias com uv pip..."
	$(UV) pip install -e .

# Instalar dependencias com dev (requer venv ativado)
pip-install-dev:
ifndef VIRTUAL_ENV
	@echo "ERRO: Ambiente virtual nao esta ativado!"
	@echo ""
	@echo "Ative o ambiente virtual primeiro:"
	@echo "  source .venv/bin/activate   (Linux/Mac)"
	@echo "  .venv\\Scripts\\activate    (Windows)"
	@echo ""
	@echo "Ou use 'make sync' para sincronizar com uv automaticamente."
	@exit 1
endif
	@echo "Instalando dependencias (incluindo dev) com uv pip..."
	$(UV) pip install -e ".[dev]"

# ------------------------------------------------------------------
# COMANDOS MLFLOW
# ------------------------------------------------------------------

# Iniciar MLflow UI
mlflow:
	@echo "Iniciando MLflow UI na porta $(MLFLOW_PORT)..."
	@mkdir -p mlruns
	$(UV) run mlflow ui --port $(MLFLOW_PORT) --host $(MLFLOW_HOST) --backend-store-uri file:./mlruns

# Limpar runs (CUIDADO: remove todo historico!)
mlflow-clean:
	@echo "ATENCAO: Isso removera todos os runs locais!"
	@read -p "Tem certeza? (yes/no): " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		rm -rf mlruns/; \
		mkdir -p mlruns; \
		rm -f mlflow.db; \
		echo "Dados do MLflow limpos!"; \
	else \
		echo "Operacao cancelada."; \
	fi

# ------------------------------------------------------------------
# COMANDOS DE DESENVOLVIMENTO
# ------------------------------------------------------------------

# Treinar modelo de exemplo
train:
	@echo "Executando treinamento de exemplo..."
	$(UV) run python src/train_example.py

# Testes
test:
	@echo "Executando testes..."
	$(UV) run pytest tests/ -v --cov=src --cov-report=term-missing

# Verificar codigo
lint:
	@echo "Verificando codigo com ruff..."
	$(UV) run ruff check .
	@echo "Verificacao concluida"

# Formatar codigo
format:
	@echo "Formatando codigo com ruff..."
	$(UV) run ruff format .
	@echo "Formatacao concluida"

# Verificar tipos
typecheck:
	@echo "Verificando tipos com mypy..."
	$(UV) run mypy src/ --explicit-package-bases --ignore-missing-imports
	@echo "Verificacao de tipos concluida"

# ------------------------------------------------------------------
# UTILITARIOS
# ------------------------------------------------------------------

# Limpar arquivos temporarios
clean:
	@echo "Limpando arquivos temporarios..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	@echo "Limpeza concluida"
