# Model Card - Churn Prediction Telco (Template)

## Sobre os marcadores [MLFLOW:...]

Os valores marcados com `[MLFLOW:nome_da_metrica]` sao placeholders. O model
card autoritativo com metricas populadas e gerado automaticamente durante o
treino e registrado como artefato `model_card.json` em cada run do MLflow.

Para visualizar:
```bash
make docker-up          # Inicia MLflow
make train-dummy        # Ou make train-mlp / make train-logistic
# Acessar http://localhost:5000, abrir o run, clicar em Artifacts > model_card.json
```

## 1. Model Details

| Campo               | Valor                                        |
|---------------------|----------------------------------------------|
| Modelo principal     | MLP PyTorch (128, 64, 32)                   |
| Baselines           | DummyClassifier, LogisticRegression          |
| Framework           | PyTorch 2.x + scikit-learn                   |
| Versao              | v1.0 (baseline comparison)                   |
| Data do treino      | [MLFLOW:run_start_time]                      |
| Seed                | 42                                           |
| Autores             | G13-MLE                                      |
| Repo                | G13-MLE/tech-challenge-fase-1                |
| Licenca do dataset  | IBM Sample Data (uso educacional)            |

### Arquitetura MLP

- Input: ~45 features (one-hot encoded)
- Hidden 1: Linear(45, 128) -> BatchNorm -> ReLU -> Dropout(0.3)
- Hidden 2: Linear(128, 64) -> BatchNorm -> ReLU -> Dropout(0.3)
- Hidden 3: Linear(64, 32) -> BatchNorm -> ReLU -> Dropout(0.3)
- Output: Linear(32, 1) -> Sigmoid
- Loss: BCEWithLogitsLoss
- Otimizador: Adam (lr=0.001, weight_decay=1e-5)
- Scheduler: ReduceLROnPlateau (patience=3)
- Early stopping: patience=5, min_delta=0.001
- Batch size: 64, max epochs: 100, val_split: 0.2

### Baselines

- **DummyClassifier**: 3 estrategias (most_frequent,
  stratified, uniform). Sem aprendizado, serve como
  referencia inferior de performance.
- **LogisticRegression**: sklearn, max_iter=1000,
  5-fold CV estratificado com StandardScaler por fold.

## 2. Intended Use

### Uso primario

Identificar clientes com alto risco de churn em operadora
de telecomunicacoes para direcionar acoes de retencao
proativamente (campanhas de desconto, upgrade, suporte
dedicado).

### Usuarios previstos

- Equipes de Customer Success / Retencao
- Analistas de dados da operadora
- Sistemas automatizados de priorizacao de campanhas

### Uso NAO pretendido

- Decisao automatica de cancelamento de contrato sem
  revisao humana
- Aplicacao direta em outros setores (banco, seguro,
  SaaS) sem revalidacao
- Unica base para decisoes financeiras (precos,
  cobranca)
- Discriminacao de clientes baseada em risco de churn

## 3. Factors

### Fatores de variabilidade

**Desbalanceamento de classes:** ~27% churn / ~73%
nao-churn. O modelo pode tender a prever a classe
majoritaria. A metrica PR-AUC e mais informativa que
ROC-AUC neste cenario.

**Tipo de contrato:**
- Month-to-month: taxa de churn significativamente
  maior
- One year / Two year: taxa de churn menor

**Tenure (tempo como cliente):**
- Clientes novos (tenure < 12 meses): maior risco
- Clientes antigos (tenure > 60 meses): menor risco
  mas sub-representados no churn

**Servicos:**
- Clientes sem servicos adicionais (OnlineSecurity,
  TechSupport, etc.): maior churn
- Fibra optica: taxa de churn maior que DSL

**Pagamento:**
- Electronic check: maior taxa de churn
- Automatico (credit card, bank transfer): menor churn

**Gasto:**
- MonthlyCharges elevado: maior propensao a churn
- TotalCharges baixo + tenure alto: possivel anomalia

### Atributos nao disponiveis (limitacao)

- Dados demograficos (idade, genero, renda)
- Historico de reclamacoes / suporte
- Uso real do servico (minutos, dados consumidos)
- Concorrencia local / precos de mercado

## 4. Metrics

### Metricas primarias

| Metrica        | Descricao                                     |
|----------------|------------------------------------------------|
| ROC-AUC        | Capacidade discriminativa geral               |
| PR-AUC         | Performance em classes desbalanceadas         |
| F1-Score       | Balanco harmonico precision/recall             |
| Recall         | % de churners detectados                      |
| Precision      | % de predicoes churn corretas                 |

### Metricas complementares

| Metrica          | Descricao                                   |
|------------------|----------------------------------------------|
| Accuracy         | Acuracia geral                              |
| Brier Score      | Calibracao das probabilidades               |
| ECE              | Expected Calibration Error                  |
| Custo total      | FN x 500 + FP x 50 (R$)                    |
| Precision@k      | Precision nos top-k clientes                |
| Risk Band        | Baixo (<0.30), Medio (0.30-0.60),          |
|                  | Alto (>0.60)                                |

### Metricas por modelo (conjunto de teste)

| Metrica         | Dummy (stratified) | Logistic |   MLP    |
|-----------------|--------------------|----------|----------|
| accuracy        | [MLFLOW:dummy_acc] | [MLFLOW:logistic_test_accuracy] | [MLFLOW:mlp_test_accuracy] |
| precision       | [MLFLOW:dummy_prec] | [MLFLOW:logistic_test_precision] | [MLFLOW:mlp_test_precision] |
| recall           | [MLFLOW:dummy_rec] | [MLFLOW:logistic_test_recall] | [MLFLOW:mlp_test_recall] |
| f1_score         | [MLFLOW:dummy_f1] | [MLFLOW:logistic_test_f1_score] | [MLFLOW:mlp_test_f1_score] |
| roc_auc          | [MLFLOW:dummy_auc] | [MLFLOW:logistic_test_roc_auc] | [MLFLOW:mlp_test_roc_auc] |
| pr_auc           | [MLFLOW:dummy_prauc] | [MLFLOW:logistic_test_pr_auc] | [MLFLOW:mlp_test_pr_auc] |
| brier_score      | [MLFLOW:dummy_brier] | [MLFLOW:logistic_test_brier_score] | [MLFLOW:mlp_test_brier_score] |

### Matriz de confusao (conjunto de teste)

| Modelo     |   TN  |   FP  |   FN  |   TP  |
|------------|-------|-------|-------|-------|
| Dummy      | [MLFLOW:dummy_tn] | [MLFLOW:dummy_fp] | [MLFLOW:dummy_fn] | [MLFLOW:dummy_tp] |
| Logistic   | [MLFLOW:logistic_tn] | [MLFLOW:logistic_fp] | [MLFLOW:logistic_fn] | [MLFLOW:logistic_tp] |
| MLP        | [MLFLOW:mlp_tn] | [MLFLOW:mlp_fp] | [MLFLOW:mlp_fn] | [MLFLOW:mlp_tp] |

## 5. Evaluation Data

### Dataset

- Nome: Telco Customer Churn (IBM Sample Data)
- Fonte: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Versionamento: DVC (`[MLFLOW:dataset_version]`)
- Registros apos limpeza: ~7032
- Colunas originais: 20

### Split treino/teste

- Metodo: estratificado por Churn (proporcao 0.8/0.2)
- Seed: 42
- Test set: ~1407 amostras (~27% churn)
- Train set: ~5625 amostras (~27% churn)

### Limpeza aplicada

1. TotalCharges: convertido para numerico
2. 11 linhas com TotalCharges NaN removidas (~0.16%)
3. customerID: removido (sem valor preditivo)
4. Validacao de dominio: tenure 0-120, charges >= 0

### Preprocessamento

1. Target: Churn (Yes=1, No=0)
2. Features categoricas: one-hot encoding
   (drop_first=True para evitar multicolinearidade)
3. StandardScaler: fit APENAS no treino,
   transform em treino e teste (evita data leakage)
4. Resultado: ~45 features numericas

## 6. Training Data

### Dataset de treino

- Fonte: mesmo dataset Telco Customer Churn
- Amostras de treino: ~5625
- Churn rate: ~27%
- Features apos preprocessamento: ~45

### Validacao interna (MLP)

- 20% do treino separado para validacao (~1125 amostras)
- Early stopping基于 validation loss
- Modelo salvo no checkpoint de menor val_loss

### Cross-validation (Logistic)

- 5-fold estratificado
- StandardScaler aplicado dentro de cada fold
  (fit no treino do fold, transform na validacao)
- Metricas: cv_accuracy, cv_precision, cv_recall,
  cv_f1, cv_roc_auc (mean +/- std)

### Metricas de treino (MLP - melhor epoca)

| Metrica         | Treino  | Validacao |
|-----------------|---------|-----------|
| loss            | [MLFLOW:mlp_train_loss] | [MLFLOW:mlp_val_loss] |
| f1_score        | [MLFLOW:mlp_val_f1_best] | [MLFLOW:mlp_val_f1_best] |

### Preprocessamento de treino

- Identico ao evaluation data (mesma pipeline)
- StandardScaler fitado apenas no treino
- Scaler salvo em `models/scaler.pkl` para inferencia

## 7. Quantitative Analyses

### Analise de custo

Custo de erro assimetrico: FN (nao detectar churn)
custa mais que FP (retencao desnecessaria).

| Parametro        | Valor  | Justificativa                         |
|------------------|--------|---------------------------------------|
| Custo FN         | R$ 500 | LTV medio de cliente perdido          |
| Custo FP         | R$ 50  | Custo de campanha/contato             |

**Nota:** Valores placeholder. Ajustar com dados reais
de negocio (ver issue #19).

### Custo total por modelo (conjunto de teste)

| Modelo    | Custo FN (R$) | Custo FP (R$) | Custo Total (R$) |
|-----------|---------------|---------------|-------------------|
| Dummy     | [MLFLOW:dummy_cost_fn] | [MLFLOW:dummy_cost_fp] | [MLFLOW:dummy_total_cost] |
| Logistic  | [MLFLOW:logistic_cost_fn] | [MLFLOW:logistic_cost_fp] | [MLFLOW:logistic_total_cost] |
| MLP       | [MLFLOW:mlp_test_cost_fn] | [MLFLOW:mlp_test_cost_fp] | [MLFLOW:mlp_test_total_cost] |

### Threshold otimo (minimiza custo)

| Modelo    | Threshold Otimo | Custo Minimo (R$) |
|-----------|-----------------|-------------------|
| MLP       | [MLFLOW:mlp_optimal_threshold_cost] | [MLFLOW:mlp_optimal_threshold_total_cost] |

### Bandas de risco (MLP)

| Banda   | Probabilidade   | % da populacao | Taxa de churn | % churners capturados |
|---------|-----------------|----------------|---------------|----------------------|
| Low     | < 0.30          | [MLFLOW:mlp_pct_low] | [MLFLOW:mlp_churn_rate_low] | - |
| Medium  | 0.30 - 0.60     | [MLFLOW:mlp_pct_medium] | [MLFLOW:mlp_churn_rate_medium] | - |
| High    | > 0.60          | [MLFLOW:mlp_pct_high] | [MLFLOW:mlp_churn_rate_high] | [MLFLOW:mlp_capture_high] |
| Med+High| >= 0.30         | -              | -             | [MLFLOW:mlp_capture_medium_high] |

### Calibracao (MLP)

| Metrica               | Valor                          |
|------------------------|--------------------------------|
| Brier Score            | [MLFLOW:mlp_test_brier_score]  |
| ECE (10 bins)          | [MLFLOW:mlp_test_ece]          |

### Precision@k / Recall@k (MLP)

| k       | Precision        | Recall           |
|---------|------------------|------------------|
| 100     | [MLFLOW:mlp_p@100] | [MLFLOW:mlp_r@100] |
| 250     | [MLFLOW:mlp_p@250] | [MLFLOW:mlp_r@250] |
| 500     | [MLFLOW:mlp_p@500] | [MLFLOW:mlp_r@500] |
| 5%      | [MLFLOW:mlp_p@5pct] | [MLFLOW:mlp_r@5pct] |
| 10%     | [MLFLOW:mlp_p@10pct] | [MLFLOW:mlp_r@10pct] |
| 20%     | [MLFLOW:mlp_p@20pct] | [MLFLOW:mlp_r@20pct] |

### Logistic Regression - Feature Importance (top 10)

[MLFLOW:logistic_feature_importance_top10]

**Nota:** Disponivel em
`models/logistic_feature_importance.csv` apos treino.

## 8. Ethical Considerations

### Vieses identificados

**Desbalanceamento de classes:** A taxa de churn de ~27%
pode levar o modelo a favorecer a classe majoritaria
(nao-churn), resultando em recall baixo para churners.
Clientes que efetivamente cancelam podem nao ser
detectados, especialmentese o threshold for alto.

**Custo assimetrico:** A definicao de FN = R$500 e
FP = R$50 favorece recall sobre precision. Isso pode
gerar ofertas de retencao desnecessarias para clientes
leais, impactando a experiencia e o orcamento do
programa de retencao.

**Ausencia de atributos protegidos:** O dataset nao
contem dados demograficos (genero, raca, idade,
renda). Isso impede a analise de vieses em subgrupos
protegidos, mas tambem impede que o modelo
discrimine diretamente por esses atributos.

**Correlacoes proxy:** Features como PaymentMethod
(Electronic check) e Contract (Month-to-month) podem
correlacionar com nivel socioeconomico. Decisoes de
retencao baseadas nessas features podem impactar
desproporcionalmente grupos de menor renda.

**Generalizacao limitada:** O dataset reflete uma
unica operadora num periodo especifico. Padroes de
churn variam entre operadoras, regioes e periodos.

### Mitigacoes

- Monitorar precision e recall por segmento de
  contrato e pagamento
- Ajustar threshold com base em impacto por subgrupo
- Auditar periodicamente decisoes de retencao
- Nao usar o modelo como unica base para decisoes
  afetando clientes

## 9. Caveats and Recommendations

### Limitacoes

- Dataset de uma unica operadora; performance em
  outras operadoras requer revalidacao completa
- Custo FN/FP (R$500/R$50) sao placeholders;
  ajustar com dados financeiros reais da operadora
- Threshold padrao 0.5; pode nao ser otimo para
  o custo de negocio
- Modelos sao baselines; performance pode melhorar
  com gradient boosting, ensemble ou feature
  engineering
- Ausencia de dados temporais impede modelar
  sazonalidade ou trends de churn
- Preprocessamento (one-hot encoding) assume que
  categorias do treino cobrem todas as categorias
  possiveis em producao

### Recomendacoes

- Usar threshold otimo do custo (ver secao 7) em
  lugar do padrao 0.5
- Monitorar data drift em producao (MonthlyCharges,
  tenure, distribuicao de contratos)
- Re-treinar periodicamente com dados atualizados
- Coletar metricas de negocio (retencao efetiva,
  ROI de campanhas) para validar predicoes
- Implementar A/B testing antes de deploy completo
- Considerar features adicionais: historico de
  reclamacao, uso de servico, NPS

### Sobre os marcadores [MLFLOW:...]

Valores marcados com `[MLFLOW:nome_da_metrica]`
devem ser preenchidos apos executar os pipelines de
treino e consultar o MLflow:

```bash
# Executar pipelines
make train-dummy
make train-mlp
make train-logistic

# O model_card.json estara nos artefatos de cada run
# http://localhost:5000
```

Experimentos MLflow:
- `tech-challenge-dummy-baseline`
- `tech-challenge-mlp`
- `tech-challenge-logistic-regression`

## 10. Cenarios de Falha

### Falhas de dados

- **Novas categorias em producao:** Features
  categoricas nao vistas no treino causam erro no
  one-hot encoding. Mitigacao: tratar categorias
  desconhecidas como desconhecidas ou "Other".
- **TotalCharges vazio:** Input com valor ausente
  ou nao numerico causa falha no preprocessing.
  Mitigacao: imputar valor ou rejeitar com alerta.
- **Data drift:** Mudanca na distribuicao de
  MonthlyCharges ou tipo de contrato ao longo do
  tempo degrada performance. Mitigacao: monitorar
  distribuicao de entrada e re-treinar.

### Falhas de modelo

- **Clientes de alto tenure sub-representados:**
  Poucos exemplos de churn em tenure > 60 meses.
  O modelo pode nao detectar churn nesses casos.
- **Novos servicos ou planos:** Features de servicos
  (StreamingTV, OnlineSecurity, etc.) podem mudar
  com novidades da operadora.
- **Calibracao degradada:** Se as probabilidades
  perdem calibracao, campanhas de retencao serao
  priorizadas incorretamente. Mitigacao: monitorar
  Brier Score e ECE periodicamente.

### Falhas de negocio

- **Threshold inadequado:** Usar 0.5 em vez do
  threshold otimo de custo pode resultar em custo
  total elevado (FN caros ou FP excessivos).
- **Orcamento limitado de retencao:** Mesmo com
  ranking correto, nao ha orcamento para reter
  todos os clientes de risco alto. Mitigacao: usar
  Precision@k para otimizar alocacao.
- **Efeito de rede:** Churn de um cliente pode
  influenciar churn de outros (efeito contagio).
  O modelo nao captura essa dependencia.
- **Mudancas macroeconomicas:** Recessao ou
  entrada de concorrente alteram padroes de churn
  fora da distribuicao de treino.

### Falhas de infraestrutura

- **API indisponivel:** O servico FastAPI pode ficar
  fora do ar. Mitigacao: health check, retry,
  fallback para regra de negocio.
- **Latencia SLO:** Requisicoes acima de 500ms
  (limiar SLO) impactam experiencia do operador.
  Mitigacao: monitorar latencia via middleware.
- **Modelo desatualizado:** Modelo sem re-treino
  por periodo estendido perde relevancia.
  Mitigacao: agendamento de re-treino periodico.

---

## Referencias

- Mitchell, M. et al. (2019). Model Cards for Model
  Reporting. ACM FAccT.
- IBM Telco Customer Churn Dataset:
  https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- MLflow Tracking:
  http://localhost:5000 (requer `make docker-up`)
