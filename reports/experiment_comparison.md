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
| accuracy | 0.6158 | 0.0024 | 0.6148 | 0.6217 |
| f1_score | 0.2792 | 0.0045 | 0.2773 | 0.2903 |
| precision | 0.2784 | 0.0044 | 0.2766 | 0.2891 |
| recall | 0.2800 | 0.0047 | 0.2781 | 0.2914 |
| roc_auc | 0.5086 | 0.0031 | 0.5074 | 0.5163 |
| pr_auc | 0.2693 | 0.0012 | 0.2688 | 0.2723 |
| brier_score | 0.3848 | 0.0016 | 0.3783 | 0.3852 |

### Estrategia: uniform

| Metrica | Media | Std | Min | Max |
|---------|-------|-----|-----|-----|
| accuracy | 0.5167 | 0.0122 | 0.4869 | 0.5217 |
| f1_score | 0.3697 | 0.0162 | 0.3299 | 0.3763 |
| precision | 0.2829 | 0.0124 | 0.2525 | 0.2879 |
| recall | 0.5332 | 0.0234 | 0.4759 | 0.5428 |
| roc_auc | 0.5000 | 0.0000 | 0.5000 | 0.5000 |
| pr_auc | 0.2658 | 0.0001 | 0.2654 | 0.2658 |
| brier_score | 0.2500 | 0.0000 | 0.2500 | 0.2500 |

## 2. Logistic Regression

**Runs validos analisados:** 13

| Metrica | Media | Std |
|---------|-------|-----|
| cv_accuracy_mean | 0.7877 | 0.0219 |
| cv_accuracy_std | 0.0099 | 0.0023 |
| cv_f1_mean | 0.6047 | 0.0157 |
| cv_f1_std | 0.0204 | 0.0056 |
| cv_precision_mean | 0.6138 | 0.0587 |
| cv_precision_std | 0.0217 | 0.0092 |
| cv_recall_mean | 0.6229 | 0.1127 |
| cv_recall_std | 0.0227 | 0.0026 |
| cv_roc_auc_mean | 0.8460 | 0.0001 |
| cv_roc_auc_std | 0.0052 | 0.0000 |

| Metrica | Media | Std | Min | Mediana | Max |
|---------|-------|-----|-----|---------|-----|
| test_accuracy | 0.7820 | 0.0328 | 0.7328 | 0.8038 | 0.8038 |
| test_f1_score | 0.6086 | 0.0007 | 0.6075 | 0.6091 | 0.6091 |
| test_precision | 0.6017 | 0.0689 | 0.4983 | 0.6476 | 0.6476 |
| test_recall | 0.6374 | 0.0938 | 0.5749 | 0.5749 | 0.7781 |
| test_roc_auc | 0.8349 | 0.0011 | 0.8332 | 0.8357 | 0.8357 |
| test_pr_auc | 0.6214 | 0.0001 | 0.6213 | 0.6215 | 0.6215 |
| test_brier_score | 0.1498 | 0.0145 | 0.1402 | 0.1402 | 0.1716 |

### Melhor Run Logistic (por Test ROC-AUC)

- **Run:** judicious-lark-675 (`2d1d842f`)
- **Test ROC-AUC:** 0.8356727976766699
- **Test F1-Score:** 0.6090651558073654

## 3. MLP

**Runs validos analisados:** 99

| Metrica | Media | Std | Min | Mediana | Max |
|---------|-------|-----|-----|---------|-----|
| test_accuracy | 0.7722 | 0.0365 | 0.6169 | 0.7906 | 0.8031 |
| test_f1_score | 0.5811 | 0.0858 | 0.0000 | 0.5950 | 0.6300 |
| test_precision | 0.5743 | 0.1040 | 0.0000 | 0.6092 | 0.6682 |
| test_recall | 0.6125 | 0.1405 | 0.0000 | 0.5909 | 0.8850 |
| test_roc_auc | 0.8274 | 0.0340 | 0.5000 | 0.8314 | 0.8412 |
| test_pr_auc | 0.6138 | 0.0375 | 0.2658 | 0.6183 | 0.6339 |
| test_brier_score | 0.1533 | 0.0188 | 0.1383 | 0.1430 | 0.2228 |

## 4. Analise de Overfitting / Underfitting

| Indicador | Valor |
|-----------|-------|
| Train Loss (medio) | 0.4948 |
| Val Loss (medio) | 0.5100 |
| Gap Loss medio (val - train) | 0.0152 |
| Gap Loss std | 0.0370 |
| Val AUC (medio) | 0.8365 |
| Test AUC (medio) | 0.8274 |
| Gap AUC medio (val - test) | 0.0090 |
| Gap AUC std | 0.0112 |

**Diagnostico:**  Overfitting MINIMO: gaps treino/validacao e validacao/teste sao pequenos.

## 5. Melhor Run MLP (por Test ROC-AUC)

- **Run:** debonair-mink-266 (`e134d0b3`)
- **Test ROC-AUC:** 0.841220388023457
- **Test F1-Score:** 0.5857988165680473
