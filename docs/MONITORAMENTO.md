# Plano de Monitoramento -- Churn Prediction API

Este documento define o plano de monitoramento do servico de predicao de
churn. O objetivo e detectar degradacao de performance, mudancas nos dados de
entrada (data drift), degradacao do modelo (model drift) e problemas
operacionais antes que impactem os usuarios finais.

Links relacionados:
- [Arquitetura de Deploy](ARQUITETURA_DE_DEPLOY.md)

## 1. Visao Geral

O monitoramento abrange tres dimensoes principais:
- **Infraestrutura**: saude da API, latencia, disponibilidade.
- **Dados**: distribuicao das features de entrada, data drift.
- **Modelo**: qualidade das predicoes, model drift, degradacao de metricas.

## 2. Metricas Tecnicas

### 2.1 Metricas de Infraestrutura

#### Disponibilidade
- **Metrica**: taxa de uptime do endpoint `/health` (HTTP 200).
- **Target**: >= 99.5% no periodo de 24h.
- **Frequencia**: verificacao a cada 30 segundos.

#### Latencia
- **Metricas**: p50, p95, p99 do tempo de resposta do `POST /predict`.
- **Targets**:
  - p50: < 100ms
  - p95: < 300ms
  - p99: < 500ms
- **Frequencia**: coleta a cada requisicao.

#### Erros HTTP
- **Metricas**: taxa de respostas 4xx (erro do cliente) e 5xx
  (erro do servidor).
- **Targets**:
  - 4xx: < 1% (indica problemas de validacao de entrada)
  - 5xx: < 0.1% (indica falhas internas)
- **Frequencia**: coleta a cada requisicao.

#### Recursos do Container
- **Metricas**: uso de CPU e memoria do container da API.
- **Targets**:
  - CPU: < 70% em media, < 90% em pico
  - Memoria: < 80% em media
- **Frequencia**: coleta a cada 15 segundos.

### 2.2 Data Drift (Desvio dos Dados)

O data drift ocorre quando a distribuicao dos dados de entrada muda em
relacao ao conjunto de treinamento. Isso pode levar a predicoes menos
confiaveis.

#### Features Monitoradas
- **Numericas**: `tenure`, `MonthlyCharges`, `TotalCharges`.
- **Categoricas**: `Contract`, `InternetService`, `PaymentMethod`.

#### Metricas de Drift
- **Population Stability Index (PSI)**:
  - Calculado entre a distribuicao atual e a de treinamento.
  - PSI < 0.1: sem drift.
  - PSI entre 0.1 e 0.25: drift moderado.
  - PSI > 0.25: drift severo.
- **Kolmogorov-Smirnov (KS)**:
  - Para features numericas.
  - p-value < 0.05: distribuicao significativamente diferente.
- **Chi-squared**:
  - Para features categoricas.
  - p-value < 0.05: distribuicao significativamente diferente.

#### Distribuicao de Entradas
- **Metrica**: media, mediana e desvio padrao das features numericas por
  janela de tempo (diaria).
- **Objetivo**: detectar mudancas abruptas nos valores de entrada.

### 2.3 Model Drift (Desvio do Modelo)

#### Taxa de Predicoes Positivas
- **Metrica**: proporcao de predicoes `churn_prediction=True` nas ultimas
  24h em relacao ao baseline de treinamento.
- **Target**: variacao <= 15% em relacao ao baseline.
- **Interpretacao**: mudancas bruscas podem indicar shift no comportamento
  dos clientes.

#### Distribuicao de Probabilidades
- **Metrica**: media e desvio padrao de `churn_probability` nas ultimas
  24h.
- **Alvo**: detectar concentracao em torno de 0.5 (incerteza maxima) ou
  mudancas bruscas na distribuicao.

#### Performance em Validacao Periodica
- **Metricas**: AUC-ROC, F1-score, precisao, recall.
- **Target**: AUC-ROC >= 0.78, F1-score >= 0.65.
- **Frequencia**: avaliacao semanal em conjunto de validacao.

### 2.4 Metricas do MLflow (Tracking)

As metricas registradas durante o treinamento servem como baseline:
- `accuracy`: acuracia no conjunto de teste.
- `precision`: precisao para a classe positiva.
- `recall`: recall para a classe positiva.
- `f1_score`: F1 medio.
- `roc_auc`: AUC-ROC.
- `pr_auc`: AUC-PR (Precision-Recall).

Estas metricas sao comparadas periodicamente (semanalmente) com novos dados
para detectar degradacao.

## 3. Thresholds e Alertas

### 3.1 Thresholds de Infraestrutura

| Metrica | Warning | Critical |
|---------|---------|----------|
| Uptime (< 24h) | < 99.5% | < 99% |
| Latencia p95 | > 300ms | > 500ms |
| Latencia p99 | > 500ms | > 2000ms |
| Taxa de erros 5xx | > 0.1% | > 1% |
| Taxa de erros 4xx | > 1% | > 5% |
| CPU container | > 70% | > 90% |
| Memoria container | > 80% | > 95% |

### 3.2 Thresholds de Data e Model Drift

| Metrica | Warning | Critical |
|---------|---------|----------|
| PSI (features) | > 0.10 | > 0.25 |
| p-value KS-test | < 0.10 | < 0.05 |
| Variacao taxa positiva | > 10% do baseline | > 15% do baseline |
| AUC-ROC | < 0.78 | < 0.72 |
| F1-score | < 0.65 | < 0.55 |

### 3.3 Canais de Notificacao

- **Slack / Discord**:
  - Canal: `#ml-alerts` ou `#tech-challenge`.
  - Uso: Warning e Critical.
  - Integracao: webhook.
- **Email**:
  - Lista: time de ML (Tech Leads + Rafael).
  - Uso: Critical e resumo diario de Warning.
- **MLflow Tags**:
  - Runs com anomalias sao marcados com a tag `status=ANOMALY`.
  - Permite rastreabilidade e auditoria.

## 4. Ferramentas Recomendadas

### 4.1 Em Operacao (Stack Atual)

- **MLflow UI** (`http://localhost:5000`): comparacao de runs, metricas de
  treinamento, versionamento de modelos.
- **Docker Logs**: logs de container para diagnostico de problemas.
- **FastAPI `/health`**: health check basico (ja implementado).

### 4.2 Em Producao (Adicionar)

- **Prometheus + Grafana**:
  - Coleta de metricas de infraestrutura (CPU, memoria, latencia).
  - Dashboards visuais e alertas configuraveis.
- **Evidently AI** (ou WhyLabs / Fiddler):
  - Deteccao automatica de data drift e model drift.
  - Relatorios gerados via script ou API.
- **PagerDuty / OpsGenie** (opcional):
  - Escalonamento para Critical.

## 5. Frequencia de Monitoramento

| Metrica | Frequencia | Ferramenta |
|---------|------------|------------|
| `/health` | A cada 30s | Probe (Prometheus / script custom) |
| Latencia p95/p99 | A cada requisicao | Prometheus Metrics |
| Erros HTTP | A cada requisicao | Prometheus Metrics |
| CPU/Memoria | A cada 15s | Docker Stats / Prometheus |
| Data Drift (PSI) | Semanal | Evidently AI / Script custom |
| Model Drift (AUC) | Semanal | Script de validacao + MLflow |
| Taxa de predicoes positivas | Diaria | Agregacao via logs |

## 6. Procedimentos de Retreinamento

### 6.1 Trigger

- **Agendado**: retreino mensal com dados dos ultimos 3 meses.
- **Por drift**:
  - PSI > 0.25 em qualquer feature = retreino imediato.
  - AUC-ROC caiu abaixo de 0.72 = retreino imediato.
- **Por negocio**: mudanca significativa no produto (novo plano, alteracao
  de precos).

### 6.2 Processo

1. **Coleta**: reunir dados rotulados recentes.
2. **Validacao**: aplicar `src/data/validation.py` para garantir
   qualidade.
3. **Treinamento**: executar `make train-mlp`.
4. **Comparacao**: comparar metricas do novo modelo com o baseline no
   MLflow UI.
5. **Promocao**: se novo modelo superior, promover via DVC
   (`dvc add` e `dvc push`).
6. **Deploy**: implantar novo modelo e restartar a API.
7. **Monitoramento**: acompanhar metricas nas proximas 24h.

### 6.3 Validacao do Modelo Retreinado

- AUC-ROC deve ser >= ao modelo atual.
- F1-score deve ser >= ao modelo atual.
- Nao deve haver overfitting (comparar metricas de treino vs teste).
- PSI entre treino e teste deve ser < 0.10 (sanity check).

## 7. Dashboards

### 7.1 MLflow (Ja Disponivel)

- Comparacao de experimentos (runs).
- Metricas de treinamento (loss, AUC-ROC, etc.).
- Artefatos versionados (modelos, scalers).

### 7.2 Grafana (Recomendado)

Painel principal com:
- Latencia e throughput da API (tempo real).
- Taxa de erros (tempo real).
- Uso de recursos do container (tempo real).
- Drift semanal (tabela de PSI).
- Comparacao de modelos (AUC-ROC ao longo do tempo).

## 8. Referencias

- **Arquitetura**: ver [ARQUITETURA_DE_DEPLOY.md](ARQUITETURA_DE_DEPLOY.md).
- **MLflow UI**: `http://localhost:5000` (requer `make docker-up`).
- **Dataset e validacao**: `src/data/validation.py`.
