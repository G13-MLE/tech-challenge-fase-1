# Roteiro do Video STAR — Tech Challenge Fase 1

> Duracao total: 5 minutos
> Formato: STAR (Situation, Task, Action, Result)

---

## Slide 1 — Abertura (0:00 - 0:10)

**Titulo:** Tech Challenge Fase 1 — Previsao de Churn em Telecom

**Conteudo visual:**
- Nome do projeto
- Grupo 13 - MLE
- Integrantes: Eduardo, Fernando, Rafael, Bruno, Ygor
- POS TECH FIAP

**Narracao:**
> "Ola. Somos o Grupo 13 do Tech Challenge Fase 1 da POS TECH FIAP.
> Neste video, vamos apresentar nosso pipeline end-to-end de Machine
> Learning para previsao de churn em telecomunicacoes."

---

## S — SITUATION (0:10 - 0:40)

### Slide 2 — O Problema de Churn (0:10 - 0:25)

**Titulo:** Churn em Telecomunicacoes: Um Problema de R$ 187 Mil

**Conteudo visual:**
- Taxa de churn no setor de telecom: ~27% (dataset Telco)
- Impacto financeiro: cada cliente perdido = LTV de R$ 500
- Dados: dataset Telco Customer Churn da IBM, ~7043 clientes, 20 atributos
- Grafico: distribuicao churn vs nao-churn (barras)

**Narracao:**
> "Em telecomunicacoes, churn e o cancelamento de clientes. No dataset
> Telco da IBM, cerca de 27% dos clientes cancelam. Cada cliente perdido
> representa R$ 500 de LTV. Sem um modelo preditivo, o custo de nao
> detectar esses cancelamentos e de R$ 187 mil em falsos negativos."

### Slide 3 — Contexto do Dataset (0:25 - 0:40)

**Titulo:** Telco Customer Churn — Caracteristicas Principais

**Conteudo visual:**
- ~7032 registros apos limpeza
- 20 atributos originais (contrato, tenure, servicos, pagamento)
- Classe desbalanceada: 73% nao-churn / 27% churn
- Features relevantes identificadas no EDA:
  - Contract (Month-to-month = maior churn)
  - Tenure (clientes novos = maior risco)
  - PaymentMethod (Electronic check = maior churn)
  - MonthlyCharges (valores altos = maior propensao)

**Narracao:**
> "O dataset da IBM possui cerca de 7 mil registros com 20 atributos
> como tipo de contrato, tempo como cliente e forma de pagamento.
> A classe e desbalanceada: apenas 27% sao churners. Nosso EDA revelou
> que contratos mensais, clientes novos e pagamento por electronic check
> estao fortemente associados ao churn."

---

## T — TASK (0:40 - 1:25)

### Slide 4 — O Que Precisava Ser Feito (0:40 - 0:55)

**Titulo:** Objetivo: Pipeline End-to-End de ML

**Conteudo visual:**
- Pipeline completo: dados -> modelo -> API -> monitoramento
- Entregas obrigatorias:
  - [x] EDA e baseline models
  - [x] MLP em PyTorch
  - [x] API FastAPI (/predict, /health)
  - [x] Testes automatizados (>= 80% cobertura)
  - [x] MLflow experiment tracking
  - [x] Model Card completo
  - [x] Video STAR

**Narracao:**
> "O desafio exigia um pipeline completo de Machine Learning: desde a
> analise exploratoria ate uma API funcional com monitoramento. As
> entregas incluiam modelos baseline e MLP com PyTorch, rastreamento
> de experimentos no MLflow, API REST com FastAPI, testes automatizados
> com cobertura minima de 80 por cento, e um Model Card documentando
> limitacoes e vieses."

### Slide 5 — Abordagem e Metodologia (0:55 - 1:10)

**Titulo:** Metodologia: Baseline-First + Pair Programming

**Conteudo visual:**
- Abordagem: baseline-first (Dummy -> Logistic -> MLP)
- 5 integrantes com skill matrix complementar
- Pair programming entre experts e aprendizes
- 4 sprints: Setup, EDA, Modelagem, API/Deploy
- Ferramentas: Python 3.12+, PyTorch, scikit-learn, MLflow, FastAPI

**Narracao:**
> "Adotamos a abordagem baseline-first: primeiro estabelecemos
> referencias com DummyClassifier e Logistic Regression, e depois
> buscamos ganhos com o MLP. O time de 5 pessoas usou pair
> programming para nivelar skills, com 4 sprints progressivas do
> setup ate o deploy."

### Slide 6 — Decisoes Tecnicas Iniciais (1:10 - 1:25)

**Titulo:** Decisoes Arquiteturais

**Conteudo visual:**
- Stack: PyTorch MLP + scikit-learn baselines + MLflow + FastAPI
- Qualidade: ruff (lint), mypy (tipagem estrita), pytest (testes)
- Reprodutibilidade: Docker, DVC, Makefile, uv (package manager)
- Custo de negocio como metrica guia: FN = R$ 500, FP = R$ 50
- Metrica primaria: PR-AUC (classes desbalanceadas)

**Narracao:**
> "Definimos o custo de negocio como metrica guia. Em telecom, nao
> detectar um churner custa 10 vezes mais que uma retencao
> desnecessaria: R$ 500 contra R$ 50. Por isso, priorizamos PR-AUC
> sobre ROC-AUC, ja que o dataset e desbalanceado. Para reprodutibilidade,
> usamos Docker, DVC e Makefile."

---

## A — ACTION (1:25 - 3:25)

### Slide 7 — Arquitetura do Pipeline (1:25 - 1:45)

**Titulo:** Pipeline End-to-End

**Conteudo visual:**
- Diagrama de fluxo:
  ```
  data/raw/ -> prepare_telco_dataset -> split_train_test_stratified
      |                                    |
      v                                    v
  src/features/       ->    src/training/    ->    models/
  (preprocessamento)      (treino + tuning)       (artefatos)
      |                                            |
      v                                            v
  src/pipelines/          MLflow tracking       src/inference/
  (scripts CLI)            (experimentos)        (predicao)
      |                                            |
      v                                            v
  tests/                docker/                  src/api/
  (pytest >=80%)        (MLflow+Postgres+MinIO)  (FastAPI)
  ```

**Narracao:**
> "O pipeline segue um fluxo linear e reprodutivel: os dados brutos
> passam pela etapa de preparacao e split estratificado, seguida por
> preprocessamento com one-hot encoding e StandardScaler fitado apenas
> no treino para evitar data leakage. Os scripts em src/pipelines
> orquestram o treino, registrando tudo no MLflow. A inferencia e
> desacoplada da API, e toda a infraestrutura roda em Docker."

### Slide 8 — Modelos: Baseline (1:45 - 2:00)

**Titulo:** Modelos Baseline: Referencia de Performance

**Conteudo visual:**
- DummyClassifier (3 estrategias: most_frequent, stratified, uniform)
  - ROC-AUC = 0.50 (chute aleatorio)
  - Custo total: R$ 110 mil a R$ 187 mil
- Logistic Regression (5-fold CV estratificado)
  - ROC-AUC = 0.8357
  - F1-Score = 0.6091
  - Custo total: R$ 85.350
  - Brier Score = 0.1402 (boa calibracao)

**Narracao:**
> "Os baselines estabeleceram o piso e o teto de expectativa. O
> DummyClassifier com ROC-AUC de 0.5 confirma que o problema exige
> aprendizado. A Logistic Regression ja alcancou ROC-AUC de 0.84 e
> custo de R$ 85 mil, servindo como referencia forte para o MLP
> superar."

### Slide 9 — Modelo MLP: Arquitetura (2:00 - 2:15)

**Titulo:** MLP em PyTorch — Arquitetura e Treinamento

**Conteudo visual:**
- Arquitetura MLP:
  ```
  Input (~45 features) -> Linear(45,128) -> BatchNorm -> ReLU -> Dropout(0.3)
                        -> Linear(128,64) -> BatchNorm -> ReLU -> Dropout(0.3)
                        -> Linear(64,32)  -> BatchNorm -> ReLU -> Dropout(0.3)
                        -> Linear(32,1)   -> Sigmoid
  ```
- Otimizador: Adam (lr=0.001, weight_decay=1e-5)
- Loss: BCEWithLogitsLoss
- Scheduler: ReduceLROnPlateau (patience=3)
- Early stopping: patience=5, min_delta=0.001
- Batch size: 64, max epochs: 100

**Narracao:**
> "O MLP tem 3 camadas ocultas com 128, 64 e 32 neuronios, cada uma
> com BatchNorm, ReLU e Dropout de 30% para regularizacao. Usamos
> BCEWithLogitsLoss, otimizador Adam com learning rate scheduling e
> early stopping com paciencia de 5 epocas. O treino e rastreado
> integralmente no MLflow."

### Slide 10 — API e Monitoramento (2:15 - 2:35)

**Titulo:** API FastAPI + Monitoramento em Producao

**Conteudo visual:**
- Endpoints:
  - POST /predict — recebe dados do cliente, retorna probabilidade + predicao
  - GET /health — status da API
  - GET /metrics — metricas no formato Prometheus
- Monitoramento:
  - Data drift: PSI (Population Stability Index) por janela circular
  - Latencia: middleware com SLO de 500ms
  - Request ID: rastreamento ponta-a-ponta
  - Logging estruturado: JSON com request_id, probabilidade, latencia
- Schemas: Pydantic v2 com validacao estrita (extra=forbid)
- Docker: docker-compose.api.yml

**Narracao:**
> "A API FastAPI expoe tres endpoints: predict para inferencia, health
> para verificacao e metrics para Prometheus. Implementamos
> monitoramento de data drift com PSI, que compara a distribuicao das
> requisicoes contra a baseline de treino usando um buffer circular de
> 200 amostras. O logging estruturado em JSON inclui request ID para
> rastreamento, latencia e probabilidade predita."

### Slide 11 — Demonstracao Rapida (2:35 - 3:00)

**Titulo:** Demonstracao: Treino + API

**Conteudo visual:**
- [TELA AO VIVO - gravar demonstracao real]
- Terminal 1: `make docker-up` (sobe MLflow + Postgres + MinIO)
- Terminal 2: `make train` (treina MLP e registra no MLflow)
- Navegador: MLflow UI (http://localhost:5000) mostrando experimentos
- Terminal 3: `make api-up` (sobe API)
- Terminal 4: `make api-test` (curl para /predict)
- Browser: Swagger UI (http://localhost:8000/docs)

**Narracao:**
> "Vamos a demonstracao pratica. Com make docker-up subimos o MLflow.
> Com make train executamos o pipeline MLP que registra metricas e
> artefatos no MLflow. Aqui vemos os 3 experimentos comparados. Entao
> subimos a API com make api-up e testamos o endpoint de predicao.
> O Swagger documenta os contratos automaticamente."

### Slide 12 — Qualidade e Testes (3:00 - 3:15)

**Titulo:** Qualidade: Testes, Tipagem, Lint

**Conteudo visual:**
- pytest: cobertura >= 80% em src/
- Tipagem estatica: mypy strict mode
- Lint e formato: ruff (line length 79)
- Pre-commit hooks: rodam automaticamente no commit
- CI: GitHub Actions com lint + test + coverage
- Testes por categoria:
  - Unitarios (funcoes criticas)
  - Schema (validacao Pydantic)
  - Integracao (API + modelo)
  - Smoke (sobe e responde)

**Narracao:**
> "Garantimos qualidade com pytest, cobertura minima de 80 por cento,
> tipagem estatica com mypy em modo estrito e lint com ruff. Pre-commit
> hooks rodam automaticamente a cada commit, e o CI valida tudo no
> GitHub Actions."

### Slide 13 — Infraestrutura e DevOps (3:15 - 3:25)

**Titulo:** Infraestrutura: Docker + DVC + CI

**Conteudo visual:**
- Docker Compose: MLflow + PostgreSQL + MinIO + API
- DVC: versionamento do dataset com SHA256 de integridade
- Makefile: comandos padronizados (setup, train, test, api-up, lint)
- uv: gerenciador de dependencias (lockfile commitado)
- GitHub Actions: CI automatico em push/PR

**Narracao:**
> "A infraestrutura usa Docker Compose para MLflow com PostgreSQL e
> MinIO. O dataset e versionado com DVC e checksum SHA256. Comandos
> padronizados no Makefile garantem reprodutibilidade, e o CI valida
> cada push automaticamente."

---

## R — RESULT (3:25 - 4:25)

### Slide 14 — Comparacao de Modelos (3:25 - 3:45)

**Titulo:** Resultados: MLP vs Baselines

**Conteudo visual:**
- Tabela comparativa:

| Metrica    | Dummy (strat) | Logistic | MLP   |
|------------|:------------:|:--------:|:-----:|
| ROC-AUC    | 0.5074       | 0.8357   | 0.8314 |
| F1-Score   | 0.2773       | 0.6091   | 0.6067 |
| Recall     | 0.2781       | 0.5749   | 0.6043 |
| Precision  | 0.2766       | 0.6476   | 0.6092 |
| PR-AUC     | 0.2688       | 0.6215   | 0.6147 |
| Brier Score| 0.3852       | 0.1402   | 0.1430 |

- Grafico: radar de metricas (reports/comparison_metrics_radar.png)

**Narracao:**
> "Nos resultados, Logistic Regression e MLP tiveram performance muito
> proxima. Logistic ganhou em ROC-AUC com 0.8357 contra 0.8314 do MLP.
> Ja o MLP teve recall ligeiramente superior: 0.60 contra 0.57,
> detectando mais churners. O Brier Score de ambos abaixo de 0.15
> indica boa calibracao das probabilidades."

### Slide 15 — Analise de Custo de Negocio (3:45 - 4:00)

**Titulo:** O Que Importa: Custo de Negocio

**Conteudo visual:**
- Custo total por modelo (FN x R$500 + FP x R$50):

| Modelo          | FN  | FP  | Custo Total |
|-----------------|:---:|:---:|:-----------:|
| Dummy (most_freq)| 374 | 0  | R$ 187.000 |
| Logistic        | 159 | 117 | R$ 85.350  |
| MLP             | 148 | 145 | R$ 81.250  |

- MLP: menor custo total (R$ 81.250) — 56% de reducao vs Dummy
- Grafico: comparison_cost.png

**Narracao:**
> "A metrica que importa para o negocio e o custo. O MLP reduziu o
> custo total em 56 por cento comparado ao Dummy: de R$ 187 mil para
> R$ 81 mil. Detectou 226 churners contra 0 do Dummy mais frequente,
> com 148 falsos negativos contra 374. Em telecom, cada churner
> detectado e receita salva."

### Slide 16 — Threshold Otimo e Bandas de Risco (4:00 - 4:15)

**Titulo:** Otimizacao de Threshold + Bandas de Risco

**Conteudo visual:**
- Threshold otimo (minimiza custo): 0.05
  - Custo otimo MLP: R$ 36.900 (FN=7, FP=668)
  - Recall em 98% — quase todos churners detectados
- Bandas de risco MLP:
  - Low (< 0.30): populacao mayor, taxa de churn baixa
  - Medium (0.30-0.60): populacao intermediaria
  - High (> 0.60): populacao menor, alta concentracao de churners
- Grafico: comparison_threshold_tradeoff.png

**Narracao:**
> "Com threshold otimo de 0.05, o custo cai para R$ 36.900 com recall
> de 98 por cento. Isso significa que, ao priorizar a deteccao de
> churners, o modelo captura quase todos, aceitando mais falsos
> positivos. As bandas de risco permitem ao negocio priorizar campanhas
> nos clientes de risco alto e medio."

### Slide 17 — Diagnostico de Overfitting (4:15 - 4:25)

**Titulo:** Overfitting Controlado

**Conteudo visual:**
- Gap treino/validacao: +0.015 (minimo)
- Gap validacao/teste: -0.009 (minimo)
- Diagnostico: overfitting minimo
- Early stopping + Dropout(0.3) + BatchNorm efetivos
- Grafico: loss_curve.png

**Narracao:**
> "O diagnostico de overfitting mostra gaps minimos entre treino e
> validacao. O early stopping, dropout e batch normalization foram
> eficazes na regularizacao. O modelo generaliza bem."

---

## Encerramento (4:25 - 5:00)

### Slide 18 — Aprendizados (4:25 - 4:40)

**Titulo:** Aprendizados

**Conteudo visual:**
- Baseline-first e essencial: Dummy -> Logistic -> MLP
- Custo de negocio e mais relevante que acuracia
- Classes desbalanceadas exigem PR-AUC, nao apenas ROC-AUC
- Modelos lineares podem ser suficientes (Logistic ~= MLP em AUC)
- API com monitoramento e tao importante quanto o modelo
- Pair programming acelera nivelamento do time

**Narracao:**
> "Os principais aprendizados: baseline-first e essencial para medir
> ganho real; custo de negocio importa mais que acuracia; classes
> desbalanceadas exigem PR-AUC; e a Logistic Regression mostrou que
> modelos lineares podem ser suficientes. A API com monitoramento de
> drift e tao importante quanto o modelo em si."

### Slide 19 — Proximos Passos (4:40 - 4:50)

**Titulo:** Proximos Passos

**Conteudo visual:**
- Gradient Boosting (XGBoost/LightGBM) para superar Logistic+MLP
- Deploy em cloud (AWS/Azure/GCP)
- A/B testing do threshold otimo em producao
- Coleta de metricas de negocio real (ROI de campanhas)
- Re-treino periodico com dados atualizados
- Features adicionais: historico de reclamacoes, NPS, uso real

**Narracao:**
> "Proximos passos incluem testar gradient boosting, fazer deploy em
> cloud, implementar A/B testing do threshold otimo e coletar metricas
> de negocio real para validar as predicoes."

### Slide 20 — Encerramento (4:50 - 5:00)

**Titulo:** Obrigado

**Conteudo visual:**
- Grupo 13 - MLE
- Tech Challenge Fase 1 — POS TECH FIAP
- Repositorio: github.com/G13-MLE/tech-challenge-fase-1
- Integrantes: Eduardo, Fernando, Rafael, Bruno, Ygor

**Narracao:**
> "Obrigado. Este foi o projeto do Grupo 13 para o Tech Challenge
> Fase 1. O repositorio esta disponivel no GitHub. Todos os 5
> integrantes participaram ativamente com pair programming e
> contribuicoes nas areas de ML, API, infraestrutura e documentacao."

---

## Resumo de Tempos

| Secao   | Tempo     | Slides |
|---------|-----------|--------|
| Abertura | 0:00-0:10 | 1 |
| S - Situation | 0:10-0:40 | 2-3 |
| T - Task | 0:40-1:25 | 4-6 |
| A - Action | 1:25-3:25 | 7-13 |
| R - Result | 3:25-4:25 | 14-17 |
| Encerramento | 4:25-5:00 | 18-20 |
| **Total** | **5:00** | **20 slides** |

## Notas de Producao

- Slide 11 (Demonstracao): gravar tela ao vivo com execucao real dos comandos
- Slides com graficos: usar imagens de reports/ (roc_curve, cost, radar, etc.)
- Transicoes: manter simples, sem efeitos elaborados
- Audio: gravar narracao clara e em ritmo constante
- Background: usar fundo escuro com texto claro para contraste