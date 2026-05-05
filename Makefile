# Makefile para o TechChallenge Fase 1
# Comandos essenciais para desenvolvimento

.PHONY: setup test lint format help docker-up docker-down api-up api-down api-test api-load api-load-watch train train-dummy train-mlp train-logistic compare-models analyze tune-mlp recover-model validate-model

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
	@echo "  make api-up     - Iniciar API FastAPI + Prometheus + Grafana em background"
	@echo "  make api-down   - Parar containers da API, Prometheus e Grafana"
	@echo "  make api-test   - Testar endpoint de predição via cURL"
	@echo "  make api-load   - Teste de carga batch (default: 50 reqs)"
	@echo "  make api-load-watch - Carga continua ate Ctrl+C (default: 5 req/s)"
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
	@echo "  make compare-models  - Comparar MLP vs baselines e gerar relatorio"
	@echo "  make analyze        - Analisar experimentos e gerar relatorio"
	@echo "  make recover-model   - Recuperar modelo do MLflow (requer .env)"
	@echo "  make validate-model  - Validar modelo contra baseline (monitoramento)"
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
	make compare-models
	@echo "Todos os treinamentos e comparacao concluidos!"

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

# Treinar modelo Logistic Regression
train-logistic:
	$(ENV_ERROR)
	@echo "Treinando modelo Logistic Regression..."
	uv run python -m src.pipelines.run_logistic_regression
	@echo "Treinamento Logistic Regression concluido!"

# Comparar MLP vs modelos baseline
compare-models:
	@echo "Comparando MLP vs modelos baseline..."
	uv run python -m src.pipelines.run_compare_models
	@echo "Comparacao concluida! Relatorio em MLP_VERSUS_BASELINE.md"

# Analisar experimentos do MLflow
analyze:
	$(ENV_ERROR)
	@echo "Analisando experimentos no MLflow..."
	uv run python -m src.tools.analyze_experiments --output reports/mlflow_analysis.csv
	uv run python -m src.tools.analyze_report --input reports/mlflow_analysis.csv --output reports/experiment_comparison.md
	@echo "Analise concluida! CSV salvo em reports/mlflow_analysis.csv, relatorio em reports/experiment_comparison.md"

# Iniciar API + Prometheus + Grafana no Docker
api-up:
	@echo "Starting API + Prometheus + Grafana in background..."
	docker compose -f docker/docker-compose.api.yml up --build -d
	@echo "[OK] API:       http://localhost:$${API_PORT:-8000}/docs"
	@echo "[OK] Prometheus: http://localhost:9090"
	@echo "[OK] Targets:    http://localhost:9090/targets"
	@echo "[OK] Grafana:    http://localhost:3000  (admin/admin)"

# Parar API + Prometheus + Grafana
api-down:
	@echo "[STOP] Stopping API + Prometheus + Grafana..."
	docker compose -f docker/docker-compose.api.yml down
	@echo "[OK] Stopped!"

# Testar API
api-test:
	@echo "Testing prediction endpoint (/predict)..."
	curl -X POST "http://localhost:$${API_PORT:-8000}/predict" \
	     -H "Content-Type: application/json" \
	     -d '{"customerID":"7590-VHVEG","gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85}'
	@echo "\nTeste concluido!"

# Teste de carga batch (envia N requisicoes e gera relatorio)
api-load:
	@echo "Teste de carga batch..."
	uv run python -m src.pipelines.explore_metrics --requests $${REQUESTS:-100}
	@echo "Carga concluida!"

# Teste de carga continua (envia requisicoes ate Ctrl+C)
api-load-watch:
	@echo "Carga continua (Ctrl+C para parar)..."
	uv run python -m src.pipelines.explore_metrics --watch --rate $${RATE:-5}

# Tuning de hiperparametros do MLP com Optuna
tune-mlp:
	$(ENV_ERROR)
	@echo "Tuning de hiperparametros MLP com Optuna..."
	uv run python -m src.pipelines.run_mlp_tuning --n-trials 20
	@echo "Tuning concluido! Relatorio em reports/optuna_study.csv"

# Recuperar modelo do MLflow
recover-model:
	$(ENV_ERROR)
	@echo "Recuperando modelo do MLflow..."
	@echo -n "Tipo de modelo (mlp/logistic/dummy): " && read model_type; \
	uv run python -m src.inference.recover_model --model-type $$model_type --output models/recovered
	@echo "Modelo recuperado com sucesso!"

# Validar modelo contra baseline (monitoramento periodico)
validate-model:
	@echo "Validando modelo MLP contra baseline..."
	uv run python -m src.tools.validate_model
	@echo "Validacao concluida! Relatorio em reports/model_validation.json"
