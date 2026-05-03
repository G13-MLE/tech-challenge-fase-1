# Comparacao: MLP vs Modelos Baseline para Predicao de Churn

Este documento apresenta a comparacao completa entre o modelo MLP e os modelos baseline (DummyClassifier e Logistic Regression) para predicao de churn.

## 1. Tabela Comparativa de Metricas

| Metrica | DummyClassifier_most_frequent | DummyClassifier_stratified | DummyClassifier_uniform | LogisticRegression | MLP |
|---------|-------|-------|-------|-------|-------|
| accuracy | 0.7342 | 0.6148 | 0.5217 | 0.8038 | 0.7918 |
| precision | 0.0000 | 0.2766 | 0.2879 | 0.6476 | 0.6092 |
| recall | 0.0000 | 0.2781 | 0.5428 | 0.5749 | 0.6043 |
| f1_score | 0.0000 | 0.2773 | 0.3763 | 0.6091 | 0.6067 |
| roc_auc | 0.5000 | 0.5074 | 0.5000 | 0.8357 | 0.8314 |
| pr_auc | 0.2658 | 0.2688 | 0.2658 | 0.6215 | 0.6147 |
| brier_score | 0.2658 | 0.3852 | 0.2500 | 0.1402 | 0.1430 |
| total_cost | 187000.0000 | 148600.0000 | 110600.0000 | 85350.0000 | 81250.0000 |
| normalized_cost | 132.9069 | 105.6148 | 78.6070 | 60.6610 | 57.7470 |
| cost_false_negatives | 187000.0000 | 135000.0000 | 85500.0000 | 79500.0000 | 74000.0000 |
| cost_false_positives | 0.0000 | 13600.0000 | 25100.0000 | 5850.0000 | 7250.0000 |
| true_negatives | 1033.0000 | 761.0000 | 531.0000 | 916.0000 | 888.0000 |
| false_positives | 0.0000 | 272.0000 | 502.0000 | 117.0000 | 145.0000 |
| false_negatives | 374.0000 | 270.0000 | 171.0000 | 159.0000 | 148.0000 |
| true_positives | 0.0000 | 104.0000 | 203.0000 | 215.0000 | 226.0000 |

## 2. Analise de Trade-off FN/FP (Custo de Negocio)

Em telecom, o custo de um **False Negative** (nao detectar um churner) e tipicamente muito maior que o custo de um **False Positive** (oferecer retencao para um cliente leal).

- **Custo FN (LTV perdido por churner nao detectado):** R$ 500
- **Custo FP (campanha de retencao desnecessaria):** R$ 50

- **DummyClassifier_most_frequent:** FN=374, FP=0, Custo total=R$ 187,000
- **DummyClassifier_stratified:** FN=270, FP=272, Custo total=R$ 148,600
- **DummyClassifier_uniform:** FN=171, FP=502, Custo total=R$ 110,600
- **LogisticRegression:** FN=159, FP=117, Custo total=R$ 85,350
- **MLP:** FN=148, FP=145, Custo total=R$ 81,250

## 3. Threshold Otimo por Modelo

O threshold otimo e definido como o valor que minimiza o custo total de negocio.

| model | optimal_threshold | optimal_total_cost | optimal_precision | optimal_recall | optimal_f1_score | false_negatives | false_positives |
|-------|-------|-------|-------|-------|-------|-------|-------|
| LogisticRegression | 0.0500 | 36800.0000 | 0.3610 | 0.9759 | 0.5271 | 9 | 646 |
| MLP | 0.0500 | 36900.0000 | 0.3546 | 0.9813 | 0.5209 | 7 | 668 |

## 4. Matrizes de Confusao

| Modelo | TN (No Churn correto) | FP (Falso alarme) | FN (Churner nao detectado) | TP (Churner detectado) |
|--------|----------------------|-------------------|----------------------------|----------------------|
| DummyClassifier_most_frequent | 1033 | 0 | 374 | 0 |
| DummyClassifier_stratified | 761 | 272 | 270 | 104 |
| DummyClassifier_uniform | 531 | 502 | 171 | 203 |
| LogisticRegression | 916 | 117 | 159 | 215 |
| MLP | 888 | 145 | 148 | 226 |

## 5. Calibracao das Probabilidades

- **LogisticRegression:** Brier Score = 0.1402, ECE = 0.0247
- **MLP:** Brier Score = 0.1430, ECE = 0.0364

## 6. Conclusoes

### Melhor Modelo por ROC-AUC: **LogisticRegression**
- ROC-AUC = 0.8357

### Melhor Modelo por F1-Score: **LogisticRegression**
- F1-Score = 0.6091

### Menor Custo de Negocio: **MLP**
- Custo total = R$ 81,250

### MLP vs Logistic Regression

A Logistic Regression superou o MLP em ROC-AUC (0.8357 vs 0.8314), indicando que o modelo linear e suficiente.

A Logistic Regression obteve F1-Score superior (0.6091 vs 0.6067).

### MLP vs Dummy Baseline

O MLP apresenta ROC-AUC de 0.8314 contra 0.5074 do DummyClassifier (stratified), confirmando ganho significativo sobre o baseline.

### Recomendacao

**Recomendacao: Analise case-by-case** - Os modelos tem trade-offs diferentes. Considere o custo de implantacao e a necessidade de explicabilidade.

## 7. Visualizacoes

As seguintes visualizacoes foram geradas em `reports/`:

- `comparison_roc_curve.png`: Curva ROC comparativa
- `comparison_pr_curve.png`: Curva Precision-Recall comparativa
- `comparison_confusion_matrices.png`: Matrizes de confusao
- `comparison_cost.png`: Custo total por modelo
- `comparison_threshold_tradeoff.png`: Trade-off precision/recall/custo por threshold
- `comparison_metrics_radar.png`: Radar de metricas
