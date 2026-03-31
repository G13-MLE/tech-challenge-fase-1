# Makefile para o TechChallenge Fase 1
# Comandos essenciais para desenvolvimento

.PHONY: setup mlflow test lint format help

# Variaveis
MLFLOW_PORT = 5000
MLFLOW_HOST = 127.0.0.1

# Help padrao
help:
	@echo "Tech Challenge Fase 1 - Comandos Disponiveis"
	@echo ""
	@echo "Setup:"
	@echo "  make setup    - Configurar ambiente (uv sync + pre-commit)"
	@echo ""
	@echo "MLflow:"
	@echo "  make mlflow   - Iniciar MLflow UI"
	@echo ""
	@echo "Desenvolvimento:"
	@echo "  make train    - Executar exemplo de treino"
	@echo "  make test     - Rodar testes"
	@echo "  make lint     - Verificar codigo com ruff"
	@echo "  make format   - Formatar codigo com ruff"
	@echo ""

# Setup inicial
setup:
	@echo "Configurando ambiente..."
	uv sync
	uv run pre-commit install
	@echo "Setup concluido!"

# Iniciar MLflow UI
mlflow:
	@echo "Iniciando MLflow UI na porta $(MLFLOW_PORT)..."
	@mkdir -p mlruns
	uv run mlflow ui --port $(MLFLOW_PORT) --host $(MLFLOW_HOST) --backend-store-uri file:./mlruns

# Treinar modelo de exemplo
train:
	@echo "Executando treinamento de exemplo..."
	uv run python src/train_example.py

# Testes
test:
	@echo "Executando testes..."
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Verificar codigo
lint:
	@echo "Verificando codigo com ruff..."
	uv run ruff check .

# Formatar codigo
format:
	@echo "Formatando codigo com ruff..."
	uv run ruff format .
