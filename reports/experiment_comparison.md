# Relatorio de Analise: Dummy Baseline vs Logistic Regression vs MLP

## 1. Dummy Baseline

### Estrategia: most_frequent

| Metrica | Media | Std | Min | Max |
|---------|-------|-----|-----|-----|
| accuracy | 0.7342 | 0.0001 | 0.7342 | 0.7346 |
| f1_score | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| precision | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| recall | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| roc_auc | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| pr_auc | 0.2658 | 0.0001 | 0.2654 | 0.2658 |
| brier_score | 0.2658 | 0.0001 | 0.2654 | 0.2658 |

### Estrategia: stratified

| Metrica | Media | Std | Min | Max |
|---------|-------|-----|-----|-----|
| accuracy | 0.6159 | 0.0026 | 0.6148 | 0.6217 |
| f1_score | 0.2795 | 0.0048 | 0.2773 | 0.2903 |
| precision | 0.2787 | 0.0047 | 0.2766 | 0.2891 |
| recall | 0.2803 | 0.0050 | 0.2781 | 0.2914 |
| roc_auc | 0.5089 | 0.0033 | 0.5074 | 0.5163 |
| pr_auc | 0.2694 | 0.0013 | 0.2688 | 0.2723 |
| brier_score | 0.3848 | 0.0017 | 0.3783 | 0.3852 |

### Estrategia: uniform

| Metrica | Media | Std | Min | Max |
|---------|-------|-----|-----|-----|
| accuracy | 0.5159 | 0.0130 | 0.4869 | 0.5217 |
| f1_score | 0.3686 | 0.0173 | 0.3299 | 0.3763 |
| precision | 0.2820 | 0.0132 | 0.2525 | 0.2879 |
| recall | 0.5316 | 0.0249 | 0.4759 | 0.5428 |
| roc_auc | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| pr_auc | 0.2658 | 0.0001 | 0.2654 | 0.2658 |
| brier_score | 0.2500 | 0.0000 | 0.2500 | 0.2500 |

## 2. Logistic Regression

**Runs validos analisados:** 9

| Metrica | Media | Std |
|---------|-------|-----|
| cv_accuracy_mean | 0.8023 | 0.0000 |
| cv_accuracy_std | 0.0115 | 0.0000 |
| cv_f1_mean | 0.5956 | 0.0000 |
| cv_f1_std | 0.0236 | 0.0000 |
| cv_precision_mean | 0.6529 | 0.0000 |
| cv_precision_std | 0.0278 | 0.0000 |
| cv_recall_mean | 0.5478 | 0.0000 |
| cv_recall_std | 0.0244 | 0.0000 |
| cv_roc_auc_mean | 0.8461 | 0.0000 |
| cv_roc_auc_std | 0.0052 | 0.0000 |

| Metrica | Media | Std | Min | Mediana | Max |
|---------|-------|-----|-----|---------|-----|
| test_accuracy | 0.8038 | 0.0000 | 0.8038 | 0.8038 | 0.8038 |
| test_f1_score | 0.6091 | 0.0000 | 0.6091 | 0.6091 | 0.6091 |
| test_precision | 0.6476 | 0.0000 | 0.6476 | 0.6476 | 0.6476 |
| test_recall | 0.5749 | 0.0000 | 0.5749 | 0.5749 | 0.5749 |
| test_roc_auc | 0.8357 | 0.0000 | 0.8357 | 0.8357 | 0.8357 |
| test_pr_auc | 0.6215 | 0.0000 | 0.6215 | 0.6215 | 0.6215 |
| test_brier_score | 0.1402 | 0.0000 | 0.1402 | 0.1402 | 0.1402 |

### Melhor Run Logistic (por Test ROC-AUC)

- **Run:** judicious-lark-675 (`2d1d842f`)
- **Test ROC-AUC:** 0.8356727976766699
- **Test F1-Score:** 0.6090651558073654

## 3. MLP

**Runs validos analisados:** 96

| Metrica | Media | Std | Min | Mediana | Max |
|---------|-------|-----|-----|---------|-----|
| test_accuracy | 0.7715 | 0.0369 | 0.6169 | 0.7896 | 0.8031 |
| test_f1_score | 0.5802 | 0.0870 | 0.0000 | 0.5944 | 0.6300 |
| test_precision | 0.5729 | 0.1054 | 0.0000 | 0.6092 | 0.6682 |
| test_recall | 0.6128 | 0.1427 | 0.0000 | 0.5802 | 0.8850 |
| test_roc_auc | 0.8273 | 0.0345 | 0.5000 | 0.8314 | 0.8412 |
| test_pr_auc | 0.6137 | 0.0381 | 0.2658 | 0.6187 | 0.6339 |
| test_brier_score | 0.1537 | 0.0190 | 0.1383 | 0.1430 | 0.2228 |

## 4. Analise de Overfitting / Underfitting

| Indicador | Valor |
|-----------|-------|
| Train Loss (medio) | 0.4974 |
| Val Loss (medio) | 0.5125 |
| Gap Loss medio (val - train) | 0.0152 |
| Gap Loss std | 0.0375 |
| Val AUC (medio) | 0.8364 |
| Test AUC (medio) | 0.8273 |
| Gap AUC medio (val - test) | 0.0091 |
| Gap AUC std | 0.0113 |

**Diagnostico:**  Overfitting MINIMO: gaps treino/validacao e validacao/teste sao pequenos.

## 5. Melhor Run MLP (por Test ROC-AUC)

- **Run:** debonair-mink-266 (`e134d0b3`)
- **Test ROC-AUC:** 0.841220388023457
- **Test F1-Score:** 0.5857988165680473
