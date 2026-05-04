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

### Implementacao Atual

O sistema ja possui monitoramento integrado em runtime:

| Componente | Modulo | Funcao |
|------------|--------|--------|
| `/metrics` (GET) | `src/api/main.py` | Exposicao de metricas Prometheus |
| `/health` (GET) | `src/api/main.py` | Health check da API |
| LatencyMiddleware | `src/api/middleware/latency.py` | Latencia + SLO + metricas HTTP |
| DriftMiddleware | `src/api/middleware/drift.py` | Deteccao drift per-request + PSI |
| RequestIDMiddleware | `src/api/middleware/request_id.py` | Tracing distribu ido |
| Prometheus | `docker/docker-compose.api.yml` | Scraping a cada 15s |
| Grafana | `docker/docker-compose.api.yml` | Dashboards pre-provisionados |
| Load Tester | `src/api/monitoring/load_tester.py` | Teste de carga + relatorio PromQL |

## 2. Metricas Tecnicas

### 2.1 Metricas de Infraestrutura

#### Disponibilidade
- **Metrica**: taxa de uptime do endpoint `/health` (HTTP 200).
- **Target**: >= 99.5% no periodo de 24h.
- **Frequencia**: verificacao a cada 30 segundos.
- **Implementacao**: Prometheus scraping do `/health` ou_probe externo.

#### Latencia
- **Metricas**: p50, p95, p99 do tempo de resposta do `POST /predict`.
- **Targets**:
  - p50: < 100ms
  - p95: < 300ms
  - p99: < 500ms
- **Frequencia**: coleta a cada requisicao.
- **Implementacao**: `LatencyMiddleware` registra
  `http_request_duration_seconds` (Histogram Prometheus) com buckets de
  5ms a 10s. SLO de 500ms configuravel via `PREDICTION_SLO_MS`.
- **Dashboard Grafana**: painel "Latencia /predict (ms)" no dashboard
  "API Churn - Metricas Operacionais".

#### Erros HTTP
- **Metricas**: taxa de respostas 4xx (erro do cliente) e 5xx
  (erro do servidor).
- **Targets**:
  - 4xx: < 1% (indica problemas de validacao de entrada)
  - 5xx: < 0.1% (indica falhas internas)
- **Frequencia**: coleta a cada requisicao.
- **Implementacao**: `LatencyMiddleware` registra
  `http_requests_total` (Counter Prometheus) com labels `method`,
  `status_code`, `path`.
- **Dashboard Grafana**: "Taxa de Erros 5xx" e "Distribuicao de Status HTTP"
  no dashboard de metricas operacionais.

#### Recursos do Container
- **Metricas**: uso de CPU e memoria do container da API.
- **Targets**:
  - CPU: < 70% em media, < 90% em pico
  - Memoria: < 80% em media
- **Frequencia**: coleta a cada 15 segundos.
- **Implementacao**: Docker Stats ou Node Exporter (a adicionar em
  producao).

### 2.2 Data Drift (Desvio dos Dados)

O data drift ocorre quando a distribuicao dos dados de entrada muda em
relacao ao conjunto de treinamento. Isso pode levar a predicoes menos
confiaveis. O sistema implementa duas camadas de deteccao:

#### 2.2.1 Deteccao Per-Request (Runtime)

Implementada em `src/api/drift.py` e `src/api/middleware/drift.py`.

- **Features monitoradas**: todas as 19 features do `PredictRequest`
  (3 numericas + 16 categoricas).
- **Logica para features numericas** (`tenure`, `MonthlyCharges`,
  `TotalCharges`):
  - Verifica se o valor esta dentro do range [min, max] da baseline de
    treinamento (`reference_stats.json`).
  - Score 1.0 se fora do range, 0.0 se dentro.
- **Logica para features categoricas** (`Contract`, `InternetService`,
  etc.):
  - Verifica se a categoria existe na lista de categorias da baseline.
  - Score 1.0 se categoria inedita, 0.0 se conhecida.
- **Metricas Prometheus**: `drift_detections_total` (Counter por feature e
  `drift_detected` label).
- **Logging**: WARNING quando drift detectado, DEBUG quando estavel.
- **Baseline**: `src/api/reference_stats.json` (gerado a partir dos dados
  de treino).

#### 2.2.2 Monitoramento por Janela (PSI)

Implementada em `src/api/drift_monitor.py` e `src/api/middleware/drift.py`.

- **Metodo**: Population Stability Index (PSI) calculado a cada 50
  requisicoes, comparando a distribuicao da janela contra a baseline.
- **Features monitoradas**: `tenure`, `MonthlyCharges`, `TotalCharges`.
- **Tamanho da janela**: 200 amostras (buffer circular).
- **Baseline**: bins de proporcao do dataset de treino
  (`reference_stats.json`).
- **Interpretacao do PSI**:
  - PSI < 0.1: estavel ("stable").
  - PSI entre 0.1 e 0.25: drift moderado ("moderate").
  - PSI > 0.25: drift severo ("significant").
- **Metricas Prometheus**: `drift_psi_score` (Gauge por feature).
- **Logging**: WARNING quando PSI moderate ou significant.
- **Dashboard Grafana**: painel "PSI por Feature (Janela 200 reqs)" no
  dashboard "API Churn - Data Drift", com thresholds visuais em 0.1
  (amarelo) e 0.25 (vermelho).

#### 2.2.3 Metricas de Drift Teoricas (Producao)

Para ambientes de producao, recomenda-se complementar com:
- **Kolmogorov-Smirnov (KS)**: para features numericas.
  p-value < 0.05: distribuicao significativamente diferente.
- **Chi-squared**: para features categoricas.
  p-value < 0.05: distribuicao significativamente diferente.

#### 2.2.4 Distribuicao de Entradas
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
- **Implementacao**: `prediction_probability` (Histogram Prometheus) com
  buckets de 0.05, observado pelo `/predict` endpoint.
- **Dashboard Grafana**: "Distribuicao de Probabilidades de Churn" em
  ambos dashboards (metricas operacionais e drift).
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

### 2.5 Resumo de Metricas Prometheus Implementadas

| Nome | Tipo | Labels | Descricao |
|------|------|--------|-----------|
| `http_requests_total` | Counter | method, status_code, path | Total de requisicoes HTTP |
| `http_request_duration_seconds` | Histogram | method, path | Latencia das requisicoes (s) |
| `prediction_probability` | Histogram | (sem labels) | Distribuicao de probabilidades de churn |
| `drift_detections_total` | Counter | feature, drift_detected | Deteccoes de drift por feature |
| `drift_psi_score` | Gauge | feature | PSI score por janela vs baseline |

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
| Deteccoes per-request | > 5% das requisicoes | > 15% das requisicoes |
| Variacao taxa positiva | > 10% do baseline | > 15% do baseline |
| AUC-ROC | < 0.78 | < 0.72 |
| F1-score | < 0.65 | < 0.55 |

### 3.3 Regras de Alerta Prometheus (Recomendadas)

```yaml
# docker/prometheus_alerts.yml (a criar para producao)
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
          summary: "Latencia p95 /predict acima de 500ms"

      - alert: HighLatencyP99
        expr: >
          histogram_quantile(0.99,
          sum(rate(http_request_duration_seconds_bucket{path="/predict"}[5m]))
          by (le)) * 1000 > 2000
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Latencia p99 /predict acima de 2000ms"

      - alert: DriftDetected
        expr: >
          (sum(rate(drift_detections_total{drift_detected="true"}[5m]))
          / sum(rate(drift_detections_total[5m]))) > 0.05
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detectado em mais de 5% das requisicoes"

      - alert: SignificantPSI
        expr: drift_psi_score > 0.25
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PSI significativo detectado na feature {{ $labels.feature }}"
```

### 3.4 Canais de Notificacao

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
- **Grafana Alerts** (em producao):
  - Configurar alert rules no Grafana apontando para os canais acima.
  - Dashboard "API Churn - Data Drift" ja possui thresholds visuais.

## 4. Ferramentas

### 4.1 Em Operacao (Implementado)

| Ferramenta | Endpoint | Funcao |
|------------|----------|--------|
| FastAPI `/health` | `:8000/health` | Health check da API |
| FastAPI `/metrics` | `:8000/metrics` | Exposicao de metricas Prometheus |
| Prometheus | `:9090` | Scraping de metricas a cada 15s |
| Grafana | `:3000` | Dashboards pre-provisionados |
| MLflow UI | `:5000` | Tracking de experimentos e modelos |
| Docker Logs | `docker logs` | Logs estruturados JSON dos containers |

**Dashboards Grafana provisionados:**

1. **API Churn - Metricas Operacionais** (`api-metrics.json`):
   - Total de requisicoes (stat).
   - Taxa de erros 5xx (stat).
   - Latencia p99 /predict (stat).
   - Total de predicoes (stat).
   - Taxa de requisicoes por endpoint (timeseries).
   - Latencia /predict p50/p95/p99 (timeseries).
   - Distribuicao de probabilidades de churn (histogram).
   - Distribuicao de status HTTP (timeseries).
   - Auto-refresh: 5s.

2. **API Churn - Data Drift** (`api-drift.json`):
   - Deteccoes de drift (stat).
   - Taxa de data drift % (timeseries).
   - Deteccoes de drift por feature (timeseries).
   - Distribuicao de probabilidades de churn (histogram).
   - Status de drift OK/DRIFT (stat).
   - PSI por feature com thresholds 0.1 e 0.25 (timeseries).
   - Auto-refresh: 5s.

**Middlewares de monitoramento:**

| Middleware | Modulo | Metricas |
|-----------|--------|----------|
| LatencyMiddleware | `src/api/middleware/latency.py` | `http_requests_total`, `http_request_duration_seconds`, SLO breach log |
| DriftMiddleware | `src/api/middleware/drift.py` | `drift_detections_total`, `drift_psi_score`, drift WARNING log |
| RequestIDMiddleware | `src/api/middleware/request_id.py` | `X-Request-ID` header + ContextVar para tracing |

### 4.2 Em Producao (A Adicionar)

- **Grafana Alerting**: configurar regras de alerta apontando para
  Slack/Discord webhooks.
- **Evidently AI** (ou WhyLabs / Fiddler):
  - Deteccao automatica de data drift e model drift.
  - Relatorios gerados via script ou API.
- **PagerDuty / OpsGenie** (opcional):
  - Escalonamento para Critical.
- **Node Exporter**: metricas de CPU/memoria do host no Prometheus.

## 5. Frequencia de Monitoramento

| Metrica | Frequencia | Ferramenta |
|---------|------------|------------|
| `/health` | A cada 15s | Prometheus scrape + Grafana |
| `/metrics` | A cada 15s | Prometheus scrape |
| Latencia p95/p99 | A cada requisicao | LatencyMiddleware + Prometheus |
| Erros HTTP | A cada requisicao | LatencyMiddleware + Prometheus |
| CPU/Memoria | A cada 15s | Docker Stats / Node Exporter (producao) |
| Data Drift per-request | A cada requisicao /predict | DriftMiddleware + Prometheus |
| Data Drift PSI | A cada 50 requisicoes | DriftMiddleware (janela 200) + Grafana |
| Model Drift (AUC) | Semanal | Script de validacao + MLflow |
| Taxa de predicoes positivas | Diaria | Agregacao via `prediction_probability` histogram |

## 6. Procedimentos de Retreinamento

### 6.1 Trigger

- **Agendado**: retreino mensal com dados dos ultimos 3 meses.
- **Por drift**:
  - PSI > 0.25 em qualquer feature = retreino imediato.
  - Deteccao per-request em mais de 15% das requisicoes = investigar e
    possivelmente retreinar.
  - AUC-ROC caiu abaixo de 0.72 = retreino imediato.
- **Por negocio**: mudanca significativa no produto (novo plano, alteracao
  de precos).

### 6.2 Processo

1. **Coleta**: reunir dados rotulados recentes.
2. **Validacao**: aplicar `src/data/cleaning.py`
   (`clean_telco_data`) para garantir qualidade.
3. **Treinamento**: executar `make train-mlp`.
4. **Comparacao**: comparar metricas do novo modelo com o baseline no
   MLflow UI.
5. **Promocao**: se novo modelo superior, promover via DVC
   (`dvc add` e `dvc push`).
6. **Deploy**: implantar novo modelo e restartar a API.
7. **Regeneracao de baseline**: executar
  `uv run python -m src.tools.generate_reference_stats` para atualizar
  o `reference_stats.json`.
8. **Monitoramento**: acompanhar metricas nas proximas 24h.

### 6.3 Validacao do Modelo Retreinado

- AUC-ROC deve ser >= ao modelo atual.
- F1-score deve ser >= ao modelo atual.
- Nao deve haver overfitting (comparar metricas de treino vs teste).
- PSI entre treino e teste deve ser < 0.10 (sanity check).

## 7. Dashboards

### 7.1 Grafana (Ja Provisionado)

Dois dashboards estao pre-configurados e carregados automaticamente ao
subir o `docker-compose.api.yml`:

**Dashboard 1: API Churn - Metricas Operacionais**
- UID: `api-churn-metrics`
- Painel superior: total de requisicoes, taxa 5xx, latencia p99, total de
  predicoes.
- Painel medio: taxa de requisicoes por endpoint, latencia /predict
  (p50/p95/p99).
- Painel inferior: distribuicao de probabilidades de churn (histogram),
  distribuicao de status HTTP.
- Refresh: 5s, janela padrao: ultimos 15 minutos.

**Dashboard 2: API Churn - Data Drift**
- UID: `api-churn-drift`
- Painel superior: deteccoes de drift, taxa de data drift %.
- Painel medio: deteccoes por feature, histograma de probabilidades,
  status OK/DRIFT.
- Painel inferior: PSI por feature com thresholds visuais (0.1 amarelo,
  0.25 vermelho).
- Refresh: 5s, janela padrao: ultimos 15 minutos.

### 7.2 MLflow (Ja Disponivel)

- Comparacao de experimentos (runs).
- Metricas de treinamento (loss, AUC-ROC, etc.).
- Artefatos versionados (modelos, scalers).

### 7.3 Logs Estruturados

Todos os logs da API sao emitidos em formato JSON estruturado
(`src/api/logging.py`), incluindo:

- `timestamp`: horario do evento.
- `level`: nivel de severidade.
- `logger`: nome do modulo.
- `request_id`: ID unico da requisicao (via `X-Request-ID`).
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
  "message": "Requisicao concluida",
  "method": "POST",
  "path": "/predict",
  "status_code": "200",
  "latency_ms": 45.23,
  "slo_ms": 500.0,
  "slo_breached": false
}
```

## 8. Playbook de Incidentes

### 8.1 Classificacao de Severidade

| Severidade | Criterio | SLA de Resposta |
|------------|----------|-----------------|
| P1 - Critico | API indisponivel ou taxa 5xx > 5% | 15 minutos |
| P2 - Alto | Latencia p99 > 2s ou drift severo (PSI > 0.25) | 30 minutos |
| P3 - Medio | Latencia p95 > 500ms ou drift moderado (PSI > 0.1) | 2 horas |
| P4 - Baixo | Warning de metricas menores | Proximo dia util |

### 8.2 Incidente: API Indisponivel (P1)

**Sintomas**: endpoint `/health` retorna erro ou timeout.

**Diagnostico**:

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

**Acoes corretivas**:

| Causa | Acao |
|-------|------|
| Container parado | `docker restart churn-api-dev` ou `make api-up` |
| Artefatos ausentes | Restaurar via DVC: `dvc pull models/churn_mlp_best.pt.dvc models/scaler.pkl.dvc` |
| Erro de memoria | Aumentar recursos do container ou reduzir `MLFLOW_WORKERS` |
| Porta em conflito | Parar o processo conflitante ou alterar `API_PORT` no `.env` |
| Modelo corrompido | Restaurar backup via DVC e restartar |

**Rollback**:

1. Se a versao atual do container falhou, usar imagem anterior:
   ```bash
   docker images | grep churn-api
   docker run -d --name churn-api-rollback <imagem-anterior>
   ```
2. Se o modelo esta corrompido, restaurar versao anterior via DVC:
   ```bash
   git checkout HEAD~1 -- models/churn_mlp_best.pt.dvc
   dvc pull models/churn_mlp_best.pt.dvc
   make api-up
   ```
3. Verificar health check: `curl http://localhost:8000/health`
4. Verificar predicao: `make api-test`

### 8.3 Incidente: Alta Latencia (P2/P3)

**Sintomas**: latencia p95 > 500ms (P3) ou p99 > 2s (P2) nos dashboards
Grafana ou logs WARNING de SLO breach.

**Diagnostico**:

1. Verificar metricas de latencia no Grafana (dashboard "Metricas
   Operacionais").
2. Verificar uso de recursos do container:
   ```bash
   docker stats churn-api-dev
   ```
3. Verificar volume de requisicoes no Grafana (throughput).
4. Verificar logs de SLO breach:
   ```bash
   docker logs churn-api-dev 2>&1 | grep "SLO"
   ```

**Acoes corretivas**:

| Causa | Acao |
|-------|------|
| CPU alta > 90% | Escalar horizontalmente (adicionar replicas) |
| Memoria alta > 95% | Reiniciar container, investigar memory leak |
| Spike de trafego | Ativar auto-scaling ou rate limiting |
| Modelo demorando para carregar | Modelo ja e lazy-loaded uma vez; se lento, verificar I/O do volume |
| Garbage collection | Ajustar `MLFLOW_WORKERS` ou recursos do container |

### 8.4 Incidente: Data Drift Detectado (P2/P3)

**Sintomas**: PSI > 0.25 em alguma feature (P2) ou > 0.1 (P3),
ou mais de 5% das requisicoes com drift per-request.

**Diagnostico**:

1. Verificar dashboard "API Churn - Data Drift" no Grafana.
2. Identificar quais features estao com drift:
   - Painel "Deteccoes de Drift por Feature".
   - Painel "PSI por Feature".
3. Verificar distribuicao das probabilidades de churn no histograma.
4. Verificar logs de drift:
   ```bash
   docker logs churn-api-dev 2>&1 | grep "Data drift"
   ```

**Acoes corretivas**:

| Causa | Acao |
|-------|------|
| Drift moderado (PSI 0.1-0.25) | Monitorar, agendar retreino na proxima janela |
| Drift severo (PSI > 0.25) | Retreino imediato (ver Secao 6.2) |
| Categoria inedita | Verificar se e erro de dados ou mudanca legitima de negocio |
| Range fora do baseline | Verificar se e erro de entrada ou shift real nos dados |
| Valores invalidos no payload | Verificar integracao do cliente (4xx errors) |

**Escalonamento**:

1. Se drift e causado por mudanca de negocio (ex: novo plano),
   comunicar time de produto e agendar retreino.
2. Se drift e causado por erro de dados, corrigir na fonte e
   investigar impacto nas predicoes ja realizadas.
3. Se retreino urgente, seguir processo da Secao 6.2 e comunicar
  o time via Slack.

### 8.5 Incidente: Model Drift / Degradacao (P2)

**Sintomas**: AUC-ROC < 0.78 (P3) ou < 0.72 (P2) na validacao semanal,
ou distribuicao de probabilidades anormal no Grafana.

**Diagnostico**:

1. Executar validacao periodica no conjunto de held-out.
2. Comparar metricas atuais vs baseline no MLflow UI.
3. Verificar distribuicao de `prediction_probability` no Grafana
  (concentracao em 0.5 indica incerteza).
4. Verificar data drift (model drift frequentemente e causado por
  data drift).

**Acoes corretivas**:

1. Retreinar modelo seguindo Secao 6.2.
2. Se retreino nao melhora metricas, investigar:
   - Mudanca na distribuicao do target (churn rate).
   - Features perdendo poder preditivo.
   - Necessidade de novas features.
3. Comunicar time de produto sobre possivel perda de performance.

### 8.6 Incidente: Stack de Monitoramento Indisponivel (P3)

**Sintomas**: Grafana ou Prometheus nao responde.

**Diagnostico**:

1. Verificar containers:
   ```bash
   docker ps -a | grep -E "churn-(prometheus|grafana)"
   ```
2. Verificar network:
   ```bash
   docker network ls | grep api-network
   ```

**Acoes corretivas**:

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
4. A API continua funcionando sem monitoramento (NAO e bloqueante).

### 8.7 Escalonamento

| Nivel | Responsavel | Contato |
|-------|-------------|---------|
| L1 | Desenvolvedor de plantao | Slack: #ml-alerts |
| L2 | Tech Lead (Eduardo) | Slack DM + email |
| L3 | Time de infraestrutura | Email infra + PagerDuty |
| Negocio | Product Owner | Email produto |

**Regra de escalonamento**:
- P1: L1 responde em 15min, escala para L2 em 30min se n resolvido.
- P2: L1 responde em 30min, escala para L2 em 2h se n resolvido.
- P3: L1 responde em 2h, escala para L2 em 8h se n resolvido.
- P4: tratado no proximo dia util.

### 8.8 Pos-Incidente

Apos resolucao de qualquer incidente P1 ou P2:

1. **Post-mortem**: documento descrevendo causa raiz, timeline, impacto
   e acoes preventivas.
2. **Atualizar monitoramento**: adicionar alertas ou dashboards que
   poderiam ter detectado o problema mais cedo.
3. **Atualizar this playbook**: incluir licoes aprendidas.
4. **Comunicar**: resumo para stakeholders via Slack/email.
5. **MLflow**: registrar run com tag `incident=yes` e link para
   post-mortem.

## 9. Teste de Carga

A ferramenta `src/api/monitoring/load_tester.py` permite validar a
capacidade da API e gerar metricas para o Prometheus.

### 9.1 Uso

```bash
# Teste discreto: 100 requisicoes
uv run python -m src.api.monitoring.load_tester --requests 100

# Teste continuo: ~5 requisicoes por segundo (Ctrl+C para parar)
uv run python -m src.api.monitoring.load_tester --continuous --rate 5
```

### 9.2 Requisitos

- API rodando em `http://localhost:8000` (`make api-up`).
- Prometheus rodando em `http://localhost:9090` para relatorio
  (opcional).

### 9.3 Metricas Coletadas

Apos o teste, o script consulta o Prometheus e exibe:
- Requisicoes totais por endpoint e status code.
- Latencia media dos ultimos 5 minutos.
- Total de predicoes no histograma.
- Distribuicao de probabilidades de churn (buckets).

## 10. Referencias

- **Arquitetura**: ver [ARQUITETURA_DE_DEPLOY.md](ARQUITETURA_DE_DEPLOY.md).
- **MLflow UI**: `http://localhost:5000` (requer `make docker-up`).
- **Grafana**: `http://localhost:3000` (requer `make api-up`, login:
  `admin/admin`).
- **Prometheus**: `http://localhost:9090` (requer `make api-up`).
- **Dataset e cleaning**: `src/data/cleaning.py` (`clean_telco_data`).
- **Drift detection**: `src/api/drift.py`, `src/api/drift_monitor.py`.
- **Metricas Prometheus**: `src/api/metrics.py`.
- **Load tester**: `src/api/monitoring/load_tester.py`.