# Tech Challenge — Fase 1

Pipeline end-to-end de ML para previsão de churn em telecomunicações — MLP com PyTorch, baselines Scikit-Learn, rastreamento com MLflow e API de inferência com FastAPI. Tech Challenge Fase 1 · PÓS TECH FIAP.

## Contexto do problema

Em telecom, churn representa o cancelamento de clientes. Antecipar esse comportamento ajuda o negócio a:

- reduzir perda de receita;
- priorizar ações de retenção;
- melhorar a experiência do cliente com decisões orientadas por dados.

## Estrutura do repositório

```text
tech-challenge-fase-1/
├── src/
│   ├── data/       # Importação e parse de dados
│   ├── features/   # Limpeza, feature engineering e seleção
│   ├── eda/        # Funções auxiliares para exploração
│   ├── training/   # Split, treino e tuning
│   ├── inference/  # Predição desacoplada da camada web
│   ├── schemas/    # Schemas de validação (Pandera/Pydantic)
│   ├── api/        # Camada FastAPI
│   └── pipelines/  # Scripts de execução/orquestração
├── data/           # Dados do projeto
├── models/         # Artefatos de modelo
├── tests/          # Testes automatizados
├── notebooks/      # Exploração e análises
└── docs/           # Documentação complementar
```

## Dataset base — Telco Customer Churn (IBM)

- Arquivo usado no projeto:
  `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Integridade (SHA256):
  `88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a`
- Dicionário de dados:
  `docs/telco_customer_churn_data_dictionary.md`

Fonte e referência pública:

- IBM Community:
  https://community.ibm.com/community/user/businessanalytics/blogs/monil-shah/2019/07/31/how-to-calculate-customer-churn-rate-in-spss-modeler
- Kaggle (espelho amplamente utilizado):
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn

> Observação de licença/uso: o dataset é amplamente usado para estudo e
> demonstração. Antes de uso comercial/produção, valide os termos da fonte
> escolhida e as políticas internas.

Validação local do dataset:

```bash
python src/data/prepare_telco_dataset.py
```

## Regra de execução (pipelines)

- Não gerar artefatos finais (modelos treinados, bases finais, tracking de experimento) a partir de notebooks.
- Notebooks são para exploração.
- Artefatos finais devem ser gerados por scripts parametrizáveis em `src/pipelines/`, executados via terminal.

## Instalação (setup de ambiente)

Pré-requisitos:

- Python `>=3.12,<3.14`
- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado
- (Opcional) Docker + Docker Compose para subir o MLflow local

```bash
# clonar repositório
git clone https://github.com/G13-MLE/tech-challenge-fase-1.git
cd tech-challenge-fase-1

# sincronizar dependências (runtime + dev)
uv sync

# instalar hooks de qualidade
uv run pre-commit install
```

## Sincronização do Ambiente

Para sincronizar o ambiente incluindo pacotes de desenvolvimento utilizando o uv, use:

```bash
uv sync
```

Para sincronizar sem dependências de desenvolvimento, útil para CI e containers Docker, use:

```bash
uv sync --no-dev
```

Para aplicar mudanças no `pyproject.toml` com lockfile atualizado, use:

```bash
uv lock
uv sync
```

## Stack e versões definidas

Referência consolidada com base no `pyproject.toml` atual da branch.

- **Python:** `>=3.12,<3.14`
- **Dependências base:**
  - `dotenv>=0.9.9`
  - `fastapi>=0.135.2`
  - `pandas>=2.3.3`
  - `scikit-learn>=1.8.0`
  - `protobuf<5.0.0`
- **Dependências de desenvolvimento:**
  - `mlflow>=3.10.1`
  - `ruff>=0.15.8`
  - `mypy>=1.20.0`
  - `pytest>=9.0.2`
  - `pytest-cov>=7.1.0`
  - `pre-commit>=4.5.1`
  - `torch>=2.11.0`
  - `httpx>=0.27.0`
  - `rich>=13.0.0`
  - `boto3>=1.28.0`

### Qualidade e testes (padrões iniciais)

- lint e formatação com **Ruff**;
- tipagem estática com **MyPy**;
- testes com **Pytest**;
- cobertura mínima alvo em `src`: **80%**.

## Configuração de Variáveis de Ambiente

Para configuração local, copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Preencha no `.env` as variáveis essenciais para começar (baseadas no `.env.example`):

```bash
MLFLOW_PORT=5000
MLFLOW_WORKERS=2
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_secure_password_2024
POSTGRES_DB=mlflow_db
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin_secret_key_2024
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin_secret_key_2024
MLFLOW_DUMMY_EXPERIMENT_NAME=tech-challenge-dummy-baseline
```

> Security: nunca versionar o arquivo `.env` com credenciais reais.

## Comandos básicos

Comandos disponíveis no momento (validação local):

```bash
# verificar branch atual
git branch --show-current

# ver mudanças locais
git status
```

## Comandos Disponíveis (Makefile)

O projeto inclui um Makefile com comandos essenciais para desenvolvimento:

### Setup

```bash
make setup        # Configurar ambiente (uv sync + pre-commit)
```

### Docker (MLflow)

```bash
make docker-up    # Iniciar MLflow em background (requer .env)
make docker-down  # Parar todos os containers MLflow
```

### Treinamento do Modelo MLP

Para treinar o modelo MLP (Multi-Layer Perceptron) com PyTorch:

**Pré-requisitos:**
- Arquivo `.env` configurado (veja seção "Configuração de Variáveis de Ambiente")
- MLflow rodando localmente (`make docker-up`)
- Dataset Telco no caminho `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Executar treino:**

```bash
# Opção 1: Usando o Makefile (recomendado)
make train

# Opção 2: Executando diretamente com uv
uv run python -m src.pipelines.train_mlp

# Opção 3: Com argumentos customizados
uv run python -m src.pipelines.train_mlp \
    --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --experiment-name churn-mlp-v1
```

**O que o treino faz:**
1. Carrega e pré-processa os dados (one-hot encoding, padronização)
2. Divide em treino/teste (80/20) com estratificação
3. Treina MLP com arquitetura configurável (padrão: 128→64→32 neurônios)
4. Aplica early stopping e learning rate scheduling
5. Registra métricas e modelo no MLflow
6. Salva o melhor modelo em `models/churn_mlp_best.pt`

**Métricas geradas:**
- Acurácia, Precisão, Recall, F1-Score, AUC-ROC
- Visualização no MLflow UI (http://localhost:5000)

### API (FastAPI)

Para testar e desenvolver a API localmente com hot-reload, utilize os seguintes comandos:

```bash
make api-up       # Sobe o container da API (acessível em http://localhost:8000/docs)
make api-test     # Envia um payload de teste para o endpoint /predict via cURL
make api-down     # Para o container da API
```

### Desenvolvimento

```bash
make test         # Rodar testes com cobertura
make lint         # Verificar código com ruff
make format       # Formatar código com ruff
```

### Ajuda

```bash
make help         # Mostrar todos os comandos disponíveis
```

### Acessar

- MLflow UI: http://localhost:5000

## Roadmap / Próximos passos

Resumo conciso do cronograma de execução (issue de controle):

- **Etapa 0 (26/03 → 06/04):** setup de repositório e base técnica;
- **Etapa 1 (07/04 → 20/04):** entendimento dos dados (EDA) e baselines;
- **Etapa 2 (21/04 → 30/04):** modelagem com MLP (PyTorch);
- **Etapa 3 (24/04 → 04/05):** engenharia (API, testes, integração);
- **Etapa 4 (01/05 → 05/05):** documentação final e entrega.

Marcos críticos:

- **04/05/2026:** gravação do vídeo STAR;
- **05/05/2026:** entrega final.

## Boas práticas e módulos previstos (baseado na Issue #3)

> [OK] **Status:** esta seção descreve direcionadores e componentes **previstos** para evolução do projeto.

Boas práticas de engenharia previstas:

- TDD desde o início dos módulos críticos;
- padronização de lint/format (ruff);
- tipagem estática gradual (mypy em CI + pyright no IDE);
- organização por pipeline reproduzível (ambiente declarativo e containerização);
- documentação contínua de decisões técnicas e do fluxo de execução.

Módulos e componentes previstos:

- **Qualidade e automação:** pre-commit, testes automatizados e cobertura;
- **ML pipeline:** `sklearn.Pipeline`, rastreamento com MLflow e serialização de artefatos;
- **API e contratos:** FastAPI + Pydantic v2 para inferência e validação de entradas;
- **Operação:** Docker multi-stage, variáveis de ambiente e CI com GitHub Actions;
- **Suporte ao ciclo de dados:** uso opcional de DVC e organização de notebooks (com possibilidade de jupytext).

## Autores

- Eduardo Pereira (@eduardonunesp)
- Bruno Fructuoso (@BrunoFructuoso)
- Fernando Failla Foschiani (@FernandoFailla)
- Rafael (@gabipasse)
- Ygor Martinelli (@ygormartinelli)

## Links úteis

- Roadmap e controle de execução: https://github.com/G13-MLE/tech-challenge-fase-1/issues/7
- Boas práticas de desenvolvimento (Issue #3): https://github.com/G13-MLE/tech-challenge-fase-1/issues/3
- Kickoff Planning (Miro): https://miro.com/app/board/uXjVGt4ginw=/
