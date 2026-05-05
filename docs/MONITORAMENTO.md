# Plano de Monitoramento -- Churn Prediction API

Este documento define o plano de monitoramento do servico de predição de
churn. O objetivo e detectar degradação de performance, mudanças nos dados de
entrada (data drift), degradação do modelo (model drift) e problemas
operacionais antes que impactêm os usuários finais.

Links relacionados:
- [Arquitetura de Deploy](ARQUITETURA_DE_DEPLOY.md)

## 1. Visão Geral

O monitoramento abrange tres dimensões principais:
- **Infraestrutura**: saude da API, latência, disponibilidade.
- **Dados**: distribuição das features de entrada, data drift.
- **Modelo**: qualidade das predições, model drift, degradação de métricas.

### Implementação Atual

O sistema ja possui monitoramento integrado em runtime:

| Componente | Modulo | Função |
|------------|--------|--------|
| `/metrics` (GET) | `src/api/main.py` | Exposição de métricas Prometheus |
| `/health` (GET) | `src/api/main.py` | Health check da API |
| LatencyMiddleware | `src/api/middleware/latency.py` | Latência + SLO + métricas HTTP |
| DriftMiddleware | `src/api/middleware/drift.py` | Detecção drift per-request + PSI |
| RequestIDMiddleware | `src/api/middleware/request_id.py` | Tracing distribu ido |
| Prometheus | `docker/docker-compose.api.yml` | Scraping a cada 15s |
| Grafana | `docker/docker-compose.api.yml` | Dashboards pré-provisionados |
| Load Tester | `src/api/monitoring/load_tester.py` | Teste de carga + relatorio PromQL |

## 2. Métricas Técnicas

### 2.1 Métricas de Infraestrutura

#### Disponibilidade
- **Métrica**: taxa de uptime do endpoint `/health` (HTTP 200).
- **Target**: >= 99.5% no periodo de 24h.
- **Frequência**: verificação a cada 30 segundos.
- **Implementação**: Prometheus scraping do `/health` ou_probe externo.

#### Latência
- **Métricas**: p50, p95, p99 do tempo de resposta do `POST /predict`.
- **Targets**:
  - p50: < 100ms
  - p95: < 300ms
  - p99: < 500ms
- **Frequência**: coleta a cada requisição.
- **Implementação**: `LatencyMiddleware` registra
  `http_request_duration_seconds` (Histogram Prometheus) com buckets de
  5ms a 10s. SLO de 500ms configurável via `PREDICTION_SLO_MS`.
- **Dashboard Grafana**: painel "Latência /predict (ms)" no dashboard
  "API Churn - Métricas Operacionais".

#### Erros HTTP
- **Métricas**: taxa de respostas 4xx (erro do cliente) e 5xx
  (erro do servidor).
- **Targets**:
  - 4xx: < 1% (indica problemas de validação de entrada)
  - 5xx: < 0.1% (indica falhas internas)
- **Frequência**: coleta a cada requisição.
- **Implementação**: `LatencyMiddleware` registra
  `http_requests_total` (Counter Prometheus) com labels `method`,
  `status_code`, `path`.
- **Dashboard Grafana**: "Taxa de Erros 5xx" e "Distribuição de Status HTTP"
  no dashboard de métricas operacionais.

#### Recursos do Container
- **Métricas**: uso de CPU e memória do container da API.
- **Targets**:
  - CPU: < 70% em média, < 90% em pico
  - Memória: < 80% em média
- **Frequência**: coleta a cada 15 segundos.
- **Implementação**: Docker Stats ou Node Exporter (a adicionar em
  produção).

### 2.2 Data Drift (Desvio dos Dados)

O data drift ocorre quando a distribuição dos dados de entrada muda em
relação ao conjunto de treinamento. Isso pode levar a predições menos
confiáveis. O sistema implementa duas camadas de detecção:

#### 2.2.1 Detecção Per-Request (Runtime)

Implementada em `src/api/drift.py` e `src/api/middleware/drift.py`.

- **Features monitoradas**: todas as 19 features do `PredictRequest`
  (3 numéricas + 16 categoricas).
- **Lógica para features numéricas** (`tenure`, `MonthlyCharges`,
  `TotalCharges`):
  - Verifica se o valor esta dentro do range [min, max] da baseline de
    treinamento (`reference_stats.json`).
  - Score 1.0 se fora do range, 0.0 se dentro.
- **Lógica para features categoricas** (`Contract`, `InternetService`,
  etc.):
  - Verifica se a categoria existe na lista de categorias da baseline.
  - Score 1.0 se categoria inédita, 0.0 se conhecida.
- **Métricas Prometheus**: `drift_detections_total` (Counter por feature e
  `drift_detected` label).
- **Logging**: WARNING quando drift detectado, DEBUG quando estável.
- **Baseline**: `src/api/reference_stats.json` (gerado a partir dos dados
  de treino).

#### 2.2.2 Monitoramento por Janela (PSI)

Implementada em `src/api/drift_monitor.py` e `src/api/middleware/drift.py`.

- **Metodo**: Population Stability Index (PSI) calculado a cada 50
  requisições, comparando a distribuição da janela contra a baseline.
- **Features monitoradas**: `tenure`, `MonthlyCharges`, `TotalCharges`.
- **Tamanho da janela**: 200 amostras (buffer circular).
- **Baseline**: bins de proporcao do dataset de treino
  (`reference_stats.json`).
- **Interpretacao do PSI**:
  - PSI < 0.1: estável ("stable").
  - PSI entre 0.1 e 0.25: drift moderado ("moderate").
  - PSI > 0.25: drift severo ("significant").
- **Métricas Prometheus**: `drift_psi_score` (Gauge por feature).
- **Logging**: WARNING quando PSI moderate ou significant.
- **Dashboard Grafana**: painel "PSI por Feature (Janela 200 reqs)" no
  dashboard "API Churn - Data Drift", com thresholds visuais em 0.1
  (amarelo) e 0.25 (vermelho).

#### 2.2.3 Métricas de Drift Teoricas (Produção)

Para ambientes de produção, recomenda-se complementar com:
- **Kolmogorov-Smirnov (KS)**: para features numéricas.
  p-value < 0.05: distribuição significativamente diferente.
- **Chi-squared**: para features categoricas.
  p-value < 0.05: distribuição significativamente diferente.

#### 2.2.4 Distribuição de Entradas
- **Métrica**: média, médiana e desvio padrão das features numéricas por
  janela de tempo (diária).
- **Objetivo**: detectar mudanças abruptas nos valores de entrada.

### 2.3 Model Drift (Desvio do Modelo)

#### Taxa de Predições Positivas
- **Métrica**: proporcao de predições `churn_prediction=True` nas últimas
  24h em relação ao baseline de treinamento.
- **Target**: variacao <= 15% em relação ao baseline.
- **Interpretacao**: mudanças bruscas podem indicar shift no comportamento
  dos clientes.

#### Distribuição de Probabilidades
- **Métrica**: média e desvio padrão de `churn_probability` nas últimas
  24h.
- **Implementação**: `prediction_probability` (Histogram Prometheus) com
  buckets de 0.05, observado pelo `/predict` endpoint.
- **Dashboard Grafana**: "Distribuição de Probabilidades de Churn" em
  ambos dashboards (métricas operacionais e drift).
- **Alvo**: detectar concentração em torno de 0.5 (incerteza maxima) ou
  mudanças bruscas na distribuição.

#### Performance em Validação Periodica
- **Métricas**: AUC-ROC, F1-score, precisão, recall.
- **Target**: AUC-ROC >= 0.78, F1-score >= 0.55.
- **Frequência**: avaliacao semanal em conjunto de validação.
- **Implementação**: script `src/tools/validate_model.py` executado
  via scheduler (cron ou Airflow). Saída em `reports/model_validation.json`.
- **Uso**:
  ```bash
  uv run python -m src.tools.validate_model
  uv run python -m src.tools.validate_model --threshold-roc 0.80
  ```

### 2.4 Métricas do MLflow (Tracking)

As métricas registradas durante o treinamento servem como baseline:
- `accuracy`: acurácia no conjunto de teste.
- `precision`: precisão para a classe positiva.
- `recall`: recall para a classe positiva.
- `f1_score`: F1 médio.
- `roc_auc`: AUC-ROC.
- `pr_auc`: AUC-PR (Precision-Recall).

Estas métricas são comparadas periodicamente (semanalmente) com novos dados
para detectar degradação.

### 2.5 Resumo de Métricas Prometheus Implementadas

| Nome | Tipo | Labels | Descrição |
|------|------|--------|-----------|
| `http_requests_total` | Counter | method, status_code, path | Total de requisições HTTP |
| `http_request_duration_seconds` | Histogram | method, path | Latência das requisições (s) |
| `prediction_probability` | Histogram | (sem labels) | Distribuição de probabilidades de churn |
| `drift_detections_total` | Counter | feature, drift_detected | Detecções de drift por feature |
| `drift_psi_score` | Gauge | feature | PSI score por janela vs baseline |

## 3. Thresholds e Alertas

### 3.1 Thresholds de Infraestrutura

| Métrica | Warning | Critical |
|---------|---------|----------|
| Uptime (< 24h) | < 99.5% | < 99% |
| Latência p95 | > 300ms | > 500ms |
| Latência p99 | > 500ms | > 2000ms |
| Taxa de erros 5xx | > 0.1% | > 1% |
| Taxa de erros 4xx | > 1% | > 5% |
| CPU container | > 70% | > 90% |
| Memória container | > 80% | > 95% |

### 3.2 Thresholds de Data e Model Drift

| Métrica | Warning | Critical |
|---------|---------|----------|
| PSI (features) | > 0.10 | > 0.25 |
| Detecções per-request | > 5% das requisições | > 15% das requisições |
| Variacao taxa positiva | > 10% do baseline | > 15% do baseline |
| AUC-ROC | < 0.78 | < 0.72 |
| F1-score | < 0.65 | < 0.55 |

### 3.3 Regras de Alerta Prometheus (Implementado)

As regras de alerta estao definidas em `docker/prometheus_alerts.yml`
e montadas no container Prometheus via `docker/docker-compose.api.yml`.

```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate5xx
        expr: >
          (sum(rate(http_requests_total{status_code=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m]))) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Taxa de erros 5xx acima de 1%"

      - alert: HighLatencyP95
        expr: >
          histogram_quantile(0.95,
          sum(rate(http_request_duration_seconds_bucket{path="/predict"}[5m]))
          by (le)) * 1000 > 500
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Latência p95 /predict acima de 500ms"

      - alert: HighLatencyP99
        expr: >
          histogram_quantile(0.99,
          sum(rate(http_request_duration_seconds_bucket{path="/predict"}[5m]))
          by (le)) * 1000 > 2000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Latência p99 /predict acima de 2000ms"

      - alert: DriftDetected
        expr: >
          (sum(rate(drift_detections_total{drift_detected="true"}[5m]))
          / sum(rate(drift_detections_total[5m]))) > 0.05
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detectado em mais de 5% das requisições"

      - alert: SignificantPSI
        expr: drift_psi_score > 0.25
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PSI significativo detectado na feature {{ $labels.feature }}"
```

### 3.4 Canais de Notificação

- **Slack / Discord**:
  - Canal: `#ml-alerts` ou `#tech-challenge`.
  - Uso: Warning e Critical.
  - Integracao: webhook.
- **Email**:
  - Lista: time de ML (Tech Leads + Rafael).
  - Uso: Critical e resumo diario de Warning.
- **MLflow Tags**:
  - Runs com anomalias são marcados com a tag `status=ANOMALY`.
  - Permite rastreabilidade e auditoria.
- **Grafana Alerts** (em produção):
  - Configurar alert rules no Grafana apontando para os canais acima.
  - Dashboard "API Churn - Data Drift" ja possui thresholds visuais.

## 4. Ferramentas

### 4.1 Em Operação (Implementado)

| Ferramenta | Endpoint | Função |
|------------|----------|--------|
| FastAPI `/health` | `:8000/health` | Health check da API |
| FastAPI `/metrics` | `:8000/metrics` | Exposição de métricas Prometheus |
| Prometheus | `:9090` | Scraping de métricas a cada 15s |
| Grafana | `:3000` | Dashboards pré-provisionados |
| MLflow UI | `:5000` | Tracking de experimentos e modelos |
| Docker Logs | `docker logs` | Logs estruturados JSON dos containers |

**Dashboards Grafana provisionados:**

1. **API Churn - Métricas Operacionais** (`api-metrics.json`):
   - Total de requisições (stat).
   - Taxa de erros 5xx (stat).
   - Latência p99 /predict (stat).
   - Total de predições (stat).
   - Taxa de requisições por endpoint (timeseries).
   - Latência /predict p50/p95/p99 (timeseries).
   - Distribuição de probabilidades de churn (histogram).
   - Distribuição de status HTTP (timeseries).
   - Auto-refresh: 5s.

2. **API Churn - Data Drift** (`api-drift.json`):
   - Detecções de drift (stat).
   - Taxa de data drift % (timeseries).
   - Detecções de drift por feature (timeseries).
   - Distribuição de probabilidades de churn (histogram).
   - Status de drift OK/DRIFT (stat).
   - PSI por feature com thresholds 0.1 e 0.25 (timeseries).
   - Auto-refresh: 5s.

**Middlewares de monitoramento:**

| Middleware | Modulo | Métricas |
|-----------|--------|----------|
| LatencyMiddleware | `src/api/middleware/latency.py` | `http_requests_total`, `http_request_duration_seconds`, SLO breach log |
| DriftMiddleware | `src/api/middleware/drift.py` | `drift_detections_total`, `drift_psi_score`, drift WARNING log |
| RequestIDMiddleware | `src/api/middleware/request_id.py` | `X-Request-ID` header + ContextVar para tracing |

### 4.2 Em Produção (A Adicionar)

- **Grafana Alerting**: configurar regras de alerta apontando para
  Slack/Discord webhooks.
- **Evidently AI** (ou WhyLabs / Fiddler):
  - Detecção automatica de data drift e model drift.
  - Relatorios gerados via script ou API.
- **PagerDuty / OpsGenie** (opcional):
  - Escalonamento para Critical.
- **Node Exporter**: métricas de CPU/memória do host no Prometheus.

## 5. Frequência de Monitoramento

| Métrica | Frequência | Ferramenta |
|---------|------------|------------|
| `/health` | A cada 15s | Prometheus scrape + Grafana |
| `/metrics` | A cada 15s | Prometheus scrape |
| Latência p95/p99 | A cada requisição | LatencyMiddleware + Prometheus |
| Erros HTTP | A cada requisição | LatencyMiddleware + Prometheus |
| CPU/Memória | A cada 15s | Docker Stats / Node Exporter (produção) |
| Data Drift per-request | A cada requisição /predict | DriftMiddleware + Prometheus |
| Data Drift PSI | A cada 50 requisições | DriftMiddleware (janela 200) + Grafana |
| Model Drift (AUC) | Semanal | Script de validação + MLflow |
| Taxa de predições positivas | Diária | Agregacao via `prediction_probability` histogram |

## 6. Procedimentos de Retreinamento

### 6.1 Trigger

- **Agendado**: retreino mensal com dados dos últimos 3 meses.
- **Por drift**:
  - PSI > 0.25 em qualquer feature = retreino imédiato.
  - Detecção per-request em mais de 15% das requisições = investigar e
    possívelmente retreinar.
  - AUC-ROC caiu abaixo de 0.72 = retreino imédiato.
- **Por negócio**: mudança significativa no produto (novo plano, alteracao
  de precos).

### 6.2 Processo

1. **Coleta**: reunir dados rotulados recentes.
2. **Validação**: aplicar `src/data/cleaning.py`
   (`clean_telco_data`) para garantir qualidade.
3. **Treinamento**: executar `make train-mlp`.
4. **Comparação**: comparar métricas do novo modelo com o baseline no
   MLflow UI.
5. **Promoção**: se novo modelo superior, promover via DVC
   (`dvc add` e `dvc push`).
6. **Deploy**: implantar novo modelo e restartar a API.
7. **Regeneração de baseline**: executar
  `uv run python -m src.tools.generate_reference_stats` para atualizar
  o `reference_stats.json`.
8. **Monitoramento**: acompanhar métricas nas próximas 24h.

### 6.3 Validação do Modelo Retreinado

- AUC-ROC deve ser >= ao modelo atual.
- F1-score deve ser >= ao modelo atual.
- Não deve haver overfitting (comparar métricas de treino vs teste).
- PSI entre treino e teste deve ser < 0.10 (sanity check).

## 7. Dashboards

### 7.1 Grafana (Ja Provisionado)

Dois dashboards estao pre-configurados e carregados automaticamente ao
subir o `docker-compose.api.yml`:

**Dashboard 1: API Churn - Métricas Operacionais**
- UID: `api-churn-metrics`
- Painel superior: total de requisições, taxa 5xx, latência p99, total de
  predições.
- Painel médio: taxa de requisições por endpoint, latência /predict
  (p50/p95/p99).
- Painel inferior: distribuição de probabilidades de churn (histogram),
  distribuição de status HTTP.
- Refresh: 5s, janela padrão: últimos 15 minutos.

**Dashboard 2: API Churn - Data Drift**
- UID: `api-churn-drift`
- Painel superior: detecções de drift, taxa de data drift %.
- Painel médio: detecções por feature, histograma de probabilidades,
  status OK/DRIFT.
- Painel inferior: PSI por feature com thresholds visuais (0.1 amarelo,
  0.25 vermelho).
- Refresh: 5s, janela padrão: últimos 15 minutos.

### 7.2 MLflow (Ja Disponivel)

- Comparação de experimentos (runs).
- Métricas de treinamento (loss, AUC-ROC, etc.).
- Artefatos versionados (modelos, scalers).

### 7.3 Logs Estruturados

Todos os logs da API são emitidos em formato JSON estruturado
(`src/api/logging.py`), incluindo:

- `timestamp`: horario do evento.
- `level`: nivel de severidade.
- `logger`: nome do modulo.
- `request_id`: ID único da requisição (via `X-Request-ID`).
- `message`: mensagem do evento.
- Campos extras: `latency_ms`, `slo_breached`, `drift_score`, etc.

Comando para consultar logs:

```bash
docker logs churn-api-dev --tail 100 -f
```

Exemplo de log estruturado:

```json
{
  "timestamp": "2026-05-04T10:30:00",
  "level": "INFO",
  "logger": "src.api.middleware.latency",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Requisição concluida",
  "method": "POST",
  "path": "/predict",
  "status_code": "200",
  "latency_ms": 45.23,
  "slo_ms": 500.0,
  "slo_breached": false
}
```

## 8. Playbook de Incidentes

### 8.1 Classificação de Severidade

| Severidade | Critério | SLA de Resposta |
|------------|----------|-----------------|
| P1 - Crítico | API indisponível ou taxa 5xx > 5% | 15 minutos |
| P2 - Alto | Latência p99 > 2s ou drift severo (PSI > 0.25) | 30 minutos |
| P3 - Médio | Latência p95 > 500ms ou drift moderado (PSI > 0.1) | 2 horas |
| P4 - Baixo | Warning de métricas menores | Próximo dia útil |

### 8.2 Incidente: API Indisponível (P1)

**Sintomas**: endpoint `/health` retorna erro ou timeout.

**Diagnóstico**:

1. Verificar se o container esta rodando:
   ```bash
   docker ps -a | grep churn-api-dev
   ```
2. Verificar logs do container:
   ```bash
   docker logs churn-api-dev --tail 50
   ```
3. Verificar se o modelo e artefatos existem:
   ```bash
   ls -la models/churn_mlp_best.pt models/scaler.pkl models/feature_names.json
   ```
4. Verificar se a porta 8000 esta em uso:
   ```bash
   lsof -i :8000
   ```

**Ações corretivas**:

| Causa | Acao |
|-------|------|
| Container parado | `docker restart churn-api-dev` ou `make api-up` |
| Artefatos ausentes | Restaurar via DVC: `dvc pull models/churn_mlp_best.pt.dvc models/scaler.pkl.dvc` |
| Erro de memória | Aumentar recursos do container ou reduzir `MLFLOW_WORKERS` |
| Porta em conflito | Parar o processo conflitante ou alterar `API_PORT` no `.env` |
| Modelo corrompido | Restaurar backup via DVC e restartar |

**Rollback**:

1. Se a versão atual do container falhou, usar imagem anterior:
   ```bash
   docker images | grep churn-api
   docker run -d --name churn-api-rollback <imagem-anterior>
   ```
2. Se o modelo esta corrompido, restaurar versão anterior via DVC:
   ```bash
   git checkout HEAD~1 -- models/churn_mlp_best.pt.dvc
   dvc pull models/churn_mlp_best.pt.dvc
   make api-up
   ```
3. Verificar health check: `curl http://localhost:8000/health`
4. Verificar predição: `make api-test`

### 8.3 Incidente: Alta Latência (P2/P3)

**Sintomas**: latência p95 > 500ms (P3) ou p99 > 2s (P2) nos dashboards
Grafana ou logs WARNING de SLO breach.

**Diagnóstico**:

1. Verificar métricas de latência no Grafana (dashboard "Métricas
   Operacionais").
2. Verificar uso de recursos do container:
   ```bash
   docker stats churn-api-dev
   ```
3. Verificar volume de requisições no Grafana (throughput).
4. Verificar logs de SLO breach:
   ```bash
   docker logs churn-api-dev 2>&1 | grep "SLO"
   ```

**Ações corretivas**:

| Causa | Acao |
|-------|------|
| CPU alta > 90% | Escalar horizontalmente (adicionar replicas) |
| Memória alta > 95% | Reiniciar container, investigar memory leak |
| Spike de tráfego | Ativar auto-scaling ou rate limiting |
| Modelo demorando para carregar | Modelo ja e lazy-loaded uma vez; se lento, verificar I/O do volume |
| Garbage collection | Ajustar `MLFLOW_WORKERS` ou recursos do container |

### 8.4 Incidente: Data Drift Detectado (P2/P3)

**Sintomas**: PSI > 0.25 em alguma feature (P2) ou > 0.1 (P3),
ou mais de 5% das requisições com drift per-request.

**Diagnóstico**:

1. Verificar dashboard "API Churn - Data Drift" no Grafana.
2. Identificar quais features estao com drift:
   - Painel "Detecções de Drift por Feature".
   - Painel "PSI por Feature".
3. Verificar distribuição das probabilidades de churn no histograma.
4. Verificar logs de drift:
   ```bash
   docker logs churn-api-dev 2>&1 | grep "Data drift"
   ```

**Ações corretivas**:

| Causa | Acao |
|-------|------|
| Drift moderado (PSI 0.1-0.25) | Monitorar, agendar retreino na próxima janela |
| Drift severo (PSI > 0.25) | Retreino imédiato (ver Secao 6.2) |
| Categoria inédita | Verificar se e erro de dados ou mudança legitima de negócio |
| Range fora do baseline | Verificar se e erro de entrada ou shift real nos dados |
| Valores invalidos no payload | Verificar integracao do cliente (4xx errors) |

**Escalonamento**:

1. Se drift e causado por mudança de negócio (ex: novo plano),
   comúnicar time de produto e agendar retreino.
2. Se drift e causado por erro de dados, corrigir na fonte e
   investigar impacto nas predições ja realizadas.
3. Se retreino urgente, seguir processo da Secao 6.2 e comúnicar
  o time via Slack.

### 8.5 Incidente: Model Drift / Degradação (P2)

**Sintomas**: AUC-ROC < 0.78 (P3) ou < 0.72 (P2) na validação semanal,
ou distribuição de probabilidades anormal no Grafana.

**Diagnóstico**:

1. Executar validação periodica no conjunto de held-out.
2. Comparar métricas atuais vs baseline no MLflow UI.
3. Verificar distribuição de `prediction_probability` no Grafana
  (concentração em 0.5 indica incerteza).
4. Verificar data drift (model drift frequentemente e causado por
  data drift).

**Ações corretivas**:

1. Retreinar modelo seguindo Secao 6.2.
2. Se retreino não melhora métricas, investigar:
   - Mudança na distribuição do target (churn rate).
   - Features perdendo poder preditivo.
   - Necessidade de novas features.
3. Comúnicar time de produto sobre possível perda de performance.

### 8.6 Incidente: Stack de Monitoramento Indisponível (P3)

**Sintomas**: Grafana ou Prometheus não responde.

**Diagnóstico**:

1. Verificar containers:
   ```bash
   docker ps -a | grep -E "churn-(prometheus|grafana)"
   ```
2. Verificar network:
   ```bash
   docker network ls | grep api-network
   ```

**Ações corretivas**:

1. Reiniciar stack de monitoramento:
   ```bash
   make api-down && make api-up
   ```
2. Se apenas Grafana falhou:
   ```bash
   docker restart churn-grafana
   ```
3. Se apenas Prometheus falhou:
   ```bash
   docker restart churn-prometheus
   ```
4. A API contínua funcionando sem monitoramento (NAO e bloqueante).

### 8.7 Escalonamento

| Nivel | Responsavel | Contato |
|-------|-------------|---------|
| L1 | Desenvolvedor de plantao | Slack: #ml-alerts |
| L2 | Tech Lead (Eduardo) | Slack DM + email |
| L3 | Time de infraestrutura | Email infra + PagerDuty |
| Negócio | Product Owner | Email produto |

**Regra de escalonamento**:
- P1: L1 responde em 15min, escala para L2 em 30min se não resolvido.
- P2: L1 responde em 30min, escala para L2 em 2h se não resolvido.
- P3: L1 responde em 2h, escala para L2 em 8h se não resolvido.
- P4: tratado no próximo dia útil.

### 8.8 Pós-Incidente

Apos resolucao de qualquer incidente P1 ou P2:

1. **Post-mortem**: documento descrevendo causa raiz, timeline, impacto
   e ações preventivas.
2. **Atualizar monitoramento**: adicionar alertas ou dashboards que
   poderiam ter detectado o problema mais cedo.
3. **Atualizar este playbook**: incluir lições aprendidas.
4. **Comúnicar**: resumo para stakeholders via Slack/email.
5. **MLflow**: registrar run com tag `incident=yes` e link para
   post-mortem.

## 9. Teste de Carga

A ferramenta `src/pipelines/explore_metrics` permite validar a
capacidade da API e gerar métricas para o Prometheus, usando internamente
o módulo `src/api/monitoring/load_tester.py`.

### 9.1 Uso

```bash
# Teste discreto: 100 requisições
uv run python -m src.pipelines.explore_metrics --requests 100

# Teste contínuo: ~5 requisições por segundo (Ctrl+C para parar)
uv run python -m src.pipelines.explore_metrics --watch --rate 5

# Via Makefile
make api-load           # batch (default: 100 reqs)
make api-load-watch     # contínuo (default: 5 req/s, Ctrl+C para parar)
```

### 9.2 Requisitos

- API rodando em `http://localhost:8000` (`make api-up`).
- Prometheus rodando em `http://localhost:9090` para relatorio
  (opcional).

### 9.3 Métricas Coletadas

Apos o teste, o script consulta o Prometheus e exibe:
- Requisições totais por endpoint e status code.
- Latência média dos últimos 5 minutos.
- Total de predições no histograma.
- Distribuição de probabilidades de churn (buckets).

## 10. Referências

- **Arquitetura**: ver [ARQUITETURA_DE_DEPLOY.md](ARQUITETURA_DE_DEPLOY.md).
- **MLflow UI**: `http://localhost:5000` (requer `make docker-up`).
- **Grafana**: `http://localhost:3000` (requer `make api-up`, login:
  `admin/admin`).
- **Prometheus**: `http://localhost:9090` (requer `make api-up`).
- **Dataset e cleaning**: `src/data/cleaning.py` (`clean_telco_data`).
- **Drift detection**: `src/api/drift.py`, `src/api/drift_monitor.py`.
- **Métricas Prometheus**: `src/api/metrics.py`.
- **Load tester**: `src/api/monitoring/load_tester.py`.