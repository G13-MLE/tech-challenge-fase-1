# Makefile para o TechChallenge Fase 1
# Comandos essenciais para desenvolvimento

.PHONY: setup test lint format help docker-up docker-down api-up api-down api-test train train-dummy train-mlp train-logistic analyze tune-mlp

# Verifica se o arquivo .env existe
CHECK_ENV := $(shell test -f .env && echo 1 || echo 0)
ifeq ($(CHECK_ENV),0)
  ENV_ERROR = @echo "[ERROR] ERRO: Arquivo .env não encontrado!" && echo "Tip: Copie .env.example para .env:" && echo "   cp .env.example .env" && echo "" && exit 1
endif

# Help padrão
help:
	@echo "Tech Challenge Fase 1 - Comandos Disponiveis"
	@echo ""
	@echo "Setup:"
	@echo "  make setup      - Configurar ambiente (uv sync + pre-commit)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up   - Iniciar MLflow em background (requer .env)"
	@echo "  make docker-down - Parar todos os containers MLflow"
	@echo "  make api-up      - Iniciar API FastAPI em background com hot-reload"
	@echo "  make api-down    - Parar container da API"
	@echo "  make api-test    - Testar endpoint de predição via cURL"
	@echo ""
	@echo "Desenvolvimento:"
	@echo "  make test       - Rodar testes"
	@echo "  make lint       - Verificar código com ruff"
	@echo "  make format     - Formatar código com ruff"
	@echo ""
	@echo "ML:"
	@echo "  make train          - Treinar todos os modelos (requer .env + MLflow)"
	@echo "  make train-dummy    - Treinar baseline DummyClassifier"
	@echo "  make train-mlp      - Treinar modelo MLP"
	@echo "  make train-logistic - Treinar modelo Logistic Regression"
	@echo ""

# Setup inicial
setup:
	@echo "Configurando ambiente..."
	@if [ ! -f .env ]; then \
		echo "Criando .env a partir de .env.example..."; \
		cp .env.example .env; \
	fi
	uv sync
	uv run pre-commit install
	@echo "Instalando DVC via uv tools..."
	uv tool install dvc
	@echo "Configurando DVC remote..."
	@URL=$$(grep -E '^DVC_ONEDRIVE_REMOTE_URL=' .env | cut -d '=' -f2); \
	if [ -t 0 ]; then \
		printf "Caminho atual do DVC remoto no .env: [$$URL]\nDigite um novo caminho ou pressione Enter para manter: "; \
		read user_input </dev/tty; \
	else \
		user_input=""; \
	fi; \
	if [ -n "$$user_input" ]; then \
		awk -v val="$$user_input" '{if ($$0 ~ /^DVC_ONEDRIVE_REMOTE_URL=/) print "DVC_ONEDRIVE_REMOTE_URL=" val; else print $$0}' .env > .env.tmp && mv .env.tmp .env; \
		URL="$$user_input"; \
	fi; \
	if [ -n "$$URL" ]; then \
		dvc remote add -d onedrive_remote "$$URL"; \
		echo "[OK] DVC remote configurado para: $$URL"; \
	else \
		echo "[WARN] Aviso: URL remota do DVC não definida."; \
	fi
	@echo "Setup concluído!"

# Testes
test:
	@echo "Executando testes..."
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Verificar código
lint:
	@echo "Verificando código com ruff..."
	uv run ruff check .

# Formatar código
format:
	@echo "Formatando código com ruff..."
	uv run ruff format .

# Iniciar Docker em background
docker-up:
	$(ENV_ERROR)
	@echo "Docker: Iniciando MLflow em background..."
	docker compose -f docker/docker-compose.yml --env-file .env up -d
	@echo "[OK] MLflow iniciado! Acesse http://localhost:$$(grep -E '^MLFLOW_PORT=' .env | cut -d '=' -f2) para usar."

# Parar Docker
docker-down:
	@echo "[STOP] Parando containers MLflow..."
	docker compose -f docker/docker-compose.yml --env-file .env down
	@echo "[OK] Containers parados!"

# Treinar todos os modelos
train:
	$(ENV_ERROR)
	@echo "Treinando todos os modelos..."
	make train-dummy
	make train-mlp
	make train-logistic
	@echo "Todos os treinamentos concluidos!"

# Treinar baseline DummyClassifier
train-dummy:
	$(ENV_ERROR)
	@echo "Treinando baseline DummyClassifier..."
	uv run python -m src.pipelines.run_dummy_baseline
	@echo "Baseline DummyClassifier concluído!"

# Treinar modelo MLP
train-mlp:
	$(ENV_ERROR)
	@echo "Treinando modelo MLP..."
	uv run python -m src.pipelines.run_mlp
	@echo "Treinamento MLP concluído!"

# Futuro: Treinar modelo Logistic Regression
train-logistic:
	$(ENV_ERROR)
	@echo "Treinando modelo Logistic Regression..."
	uv run python -m src.pipelines.run_logistic_regression
	@echo "Treinamento Logistic Regression concluido!"

# Analisar experimentos do MLflow
analyze:
	$(ENV_ERROR)
	@echo "Analisando experimentos no MLflow..."
	uv run python -m src.tools.analyze_experiments --output reports/mlflow_analysis.csv
	@echo "Analise concluida! CSV salvo em reports/mlflow_analysis.csv"

# Iniciar API no Docker
api-up:
	@echo "Starting API in background with hot-reload..."
	docker compose -f docker/docker-compose.api.yml up --build -d
	@echo "[OK] API started! Access Swagger at http://localhost:$${API_PORT:-8000}/docs"

# Parar API
api-down:
	@echo "[STOP] Stopping API container..."
	docker compose -f docker/docker-compose.api.yml down
	@echo "[OK] API stopped!"

# Testar API
api-test:
	@echo "Testing prediction endpoint (/predict)..."
	curl -X POST "http://localhost:$${API_PORT:-8000}/predict" \
	     -H "Content-Type: application/json" \
	     -d '{"customerID": "1234-ABCD", "tenure": 5, "MonthlyCharges": 50.0, "Contract": "Month-to-month"}'
	@echo "\nTeste concluído!"

# Tuning de hiperparametros do MLP com Optuna
tune-mlp:
	$(ENV_ERROR)
	@echo "Tuning de hiperparametros MLP com Optuna..."
	uv run python -m src.pipelines.run_mlp_tuning --n-trials 20
	@echo "Tuning concluido! Relatorio em reports/optuna_study.csv"
