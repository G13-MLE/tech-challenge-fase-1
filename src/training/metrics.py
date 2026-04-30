"""Métricas genéricas para classificação binária.

Este módulo fornece funções utilitárias para computar métricas
padrão de classificação binária, suportando pandas Series,
arrays numpy e tensores PyTorch.

Relevância das métricas para churn de telecom:
- AUC-ROC: Mede capacidade discriminativa geral do modelo,
    independente de threshold. Essencial para comparar modelos
    em datasets desbalanceados como churn (~27% positivos).
    Um modelo com ROC-AUC = 0.5 é equivalente a aleatório;
    acima de 0.7 indica discriminação útil.
- PR-AUC: Mais informativo que ROC quando classes estão
    desbalanceadas. Foca na capacidade de encontrar os
    positivos (clientes que de fato vão churn).
    PR-AUC alto significa que o modelo consegue identificar
    a maioria dos churners sem muitos falsos alarmes.
- F1-Score: Balanço harmônico entre precisão e recall.
    Ideal quando tanto FP quanto FN têm custos significativos,
    como em campanhas de retenção com orçamento limitado.
- Precision: % de clientes identificados como churn que de
    fato churnam. Baixa precision = desperdício de ações
    de retenção em clientes leais, erodindo ROI do programa.
- Recall: % de clientes que de fato churnam que foram
    identificados. Baixo recall = perda de receita por
    não detectar churners, especialmente clientes de alto
    valor (high-ARPU).
- Matriz de confusão: Base quantitativa para análise de custo.
    Permite quantificar FP (retenção desnecessária) vs
    FN (churn não detectado). Em telecom, FN geralmente
    custa mais que FP porque o LTV (Lifetime Value) de um
    cliente perdido supera o custo de uma oferta de retenção.
- Classification report: Visão consolidada por classe com
    precision, recall e f1 para positivos e negativos.
    Permite diagnosticar se o modelo está enviesado para
    uma classe (ex: alta acurácia apenas por prever sempre
    "não churn" no dataset desbalanceado).
- Calibration curve / Brier score: Verifica se probabilidades
    preditas refletem frequências observadas. Importante para
    priorizar ações de retenção por probabilidade. Um modelo
    calibrado com prob=0.8 deve ter ~80% de positivos.
- Custo de erro / trade-off: Em telecom, FN (perder cliente)
    tipicamente custa mais que FP (oferecer retenção).
    A análise de trade-off permite ajustar o threshold de
    decisão para minimizar o custo total considerando a
    matriz de custo do negócio. Isso transforma a métrica
    técnica em decisão de negócio concreta.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if TYPE_CHECKING:
    import torch


def _to_numpy_array(
    data: pd.Series | np.ndarray | torch.Tensor,
) -> np.ndarray:
    """Converte Series pandas ou array para numpy array.

    Args:
        data: Series pandas, array numpy ou tensor PyTorch

    Returns:
        Array numpy unidimensional

    Raises:
        TypeError: Se o tipo não for suportado
    """
    if isinstance(data, pd.Series):
        return np.asarray(data.values)
    if isinstance(data, np.ndarray):
        return data
    # Lazy import para suportar torch sem dependência obrigatória
    try:
        # nosec: B404 - lazy import necessário para evitar dependência
        import torch  # noqa: PLC0415

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
    except ImportError:
        pass
    raise TypeError(
        f"Tipo não suportado: {type(data)}. "
        "Esperado pd.Series, np.ndarray ou torch.Tensor"
    )


def compute_binary_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_proba_positive: pd.Series | np.ndarray | None = None,
    positive_label: str | None = None,
) -> dict[str, float]:
    """Calcula métricas escalares para classificação binária.

    Esta função suporta múltiplos formatos de entrada:
    - pandas Series (requer positive_label para converter strings)
    - numpy arrays (assume valores já numéricos 0/1)
    - tensores PyTorch (converte automaticamente)

    Retorna apenas métricas escalares (float) para compatibilidade
    com logging em MLflow e DataFrames. Para matriz de confusão,
    classification report e curvas, use as funções especializadas
    deste módulo.

    Args:
        y_true: Rótulos verdadeiros. Se Series, deve conter strings
            como "Yes"/"No" com positive_label especificado, ou valores
            numéricos 0/1.
        y_pred: Rótulos preditos. Mesmo formato que y_true ou int.
        y_proba_positive: Probabilidades para classe positiva (0-1).
            Opcional, mas requerido para métricas AUC.
        positive_label: Label da classe positiva quando y_true/y_pred
            são Series de strings. Opcional se dados já numéricos.

    Returns:
        Dicionário com métricas escalares:
            - accuracy: Acurácia geral
            - precision: Precisão (VPP)
            - recall: Recall (sensibilidade)
            - f1_score: F1 score
            - roc_auc: Área sob curva ROC (se y_proba_positive)
            - pr_auc: Área sob curva PR (se y_proba_positive)

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Com pandas Series e labels string
        >>> y_true = pd.Series(["Yes", "No", "Yes"])
        >>> y_pred = pd.Series(["Yes", "No", "No"])
        >>> y_prob = pd.Series([0.9, 0.1, 0.3])
        >>> metrics = compute_binary_classification_metrics(
        ...     y_true, y_pred, y_prob, positive_label="Yes"
        ... )
        >>> # Com numpy arrays numéricos
        >>> y_true = np.array([1, 0, 1])
        >>> y_pred = np.array([1, 0, 0])
        >>> metrics = compute_binary_classification_metrics(
        ...     y_true, y_pred
        ... )
    """
    # Converte para numpy arrays
    y_true_arr = _to_numpy_array(y_true)
    y_pred_arr = _to_numpy_array(y_pred)

    # Converte labels string para binário se necessário
    if positive_label is not None:
        y_true_bin = (y_true_arr == positive_label).astype(int)
        y_pred_bin = (y_pred_arr == positive_label).astype(int)
    else:
        # Assume já numérico (0/1)
        y_true_bin = y_true_arr.astype(int)
        y_pred_bin = y_pred_arr.astype(int)

    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "f1_score": f1_score(y_true_bin, y_pred_bin, zero_division=0),
    }

    # Adiciona métricas baseadas em probabilidades se disponíveis
    if y_proba_positive is not None:
        y_proba_arr = _to_numpy_array(y_proba_positive)
        try:
            metrics["roc_auc"] = roc_auc_score(y_true_bin, y_proba_arr)
            metrics["pr_auc"] = average_precision_score(
                y_true_bin, y_proba_arr
            )
            metrics["brier_score"] = brier_score_loss(y_true_bin, y_proba_arr)
        except ValueError:
            # AUC indefinido quando ha apenas uma classe
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0
            metrics["brier_score"] = 0.25

    return metrics


def compute_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    positive_label: str | None = None,
) -> dict[str, int]:
    """Computa matriz de confusão como dicionário de inteiros.

    Retorna TN, FP, FN, TP explicitamente nomeados para facilitar
    a análise de custo e trade-off no contexto de churn de telecom.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        positive_label: Label da classe positiva para strings.

    Returns:
        Dicionário com:
            - true_negatives (TN): Clientes leais corretamente
                identificados como não-churn
            - false_positives (FP): Clientes leais incorretamente
                classificados como churn (retenção desnecessária)
            - false_negatives (FN): Clientes churners não
                detectados (perda de receita)
            - true_positives (TP): Clientes churners corretamente
                identificados (oportunidade de retenção)
    """
    y_true_arr = _to_numpy_array(y_true)
    y_pred_arr = _to_numpy_array(y_pred)

    if positive_label is not None:
        y_true_bin = (y_true_arr == positive_label).astype(int)
        y_pred_bin = (y_pred_arr == positive_label).astype(int)
    else:
        y_true_bin = y_true_arr.astype(int)
        y_pred_bin = y_pred_arr.astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def compute_classification_report(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    positive_label: str | None = None,
) -> dict[str, dict[str, float]]:
    """Computa classification report estruturado.

    Retorna o report do sklearn como dict aninhado, permitindo
    acesso programático a precision, recall e f1 por classe
    (negativa e positiva) e agregados (macro, weighted).

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        positive_label: Label da classe positiva para strings.

    Returns:
        Dicionário aninhado com chaves por classe
        (ex: '0', '1', 'accuracy', 'macro avg', 'weighted avg')
        e sub-chaves 'precision', 'recall', 'f1-score', 'support'.
    """
    y_true_arr = _to_numpy_array(y_true)
    y_pred_arr = _to_numpy_array(y_pred)

    if positive_label is not None:
        y_true_bin = (y_true_arr == positive_label).astype(int)
        y_pred_bin = (y_pred_arr == positive_label).astype(int)
        target_names = [f"not_{positive_label}", positive_label]
    else:
        y_true_bin = y_true_arr.astype(int)
        y_pred_bin = y_pred_arr.astype(int)
        target_names = ["class_0", "class_1"]

    report = classification_report(
        y_true_bin,
        y_pred_bin,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return report  # type: ignore[no-any-return]


def compute_calibration_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba_positive: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Computa métricas de calibração das probabilidades preditas.

    A calibração mede o quanto as probabilidades preditas pelo
    modelo correspondem às frequências observadas. Em telecom,
    isso é crucial para priorizar campanhas: um cliente com
    prob=0.8 deve de fato ter ~80% de chance de churn.

    Args:
        y_true: Rótulos verdadeiros (0/1 ou booleano).
        y_proba_positive: Probabilidades preditas para classe
            positiva, no intervalo [0, 1].
        n_bins: Número de bins para cálculo do ECE
            (Expected Calibration Error). Default 10.

    Returns:
        Dicionário com:
            - brier_score: Brier score (média do erro quadrático
                das probabilidades). 0 = perfeito, 0.25 = pior.
            - expected_calibration_error: ECE (diferença média
                entre confiança predita e acurácia observada).
    """
    y_true_arr = _to_numpy_array(y_true).astype(int)
    y_proba_arr = _to_numpy_array(y_proba_positive)

    brier = brier_score_loss(y_true_arr, y_proba_arr)

    # Expected Calibration Error (ECE) com bins igualmente espaçados
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_proba_arr >= bin_boundaries[i]) & (
            y_proba_arr < bin_boundaries[i + 1]
        )
        if i == n_bins - 1:  # Último bin fecha no 1.0
            in_bin = (y_proba_arr >= bin_boundaries[i]) & (
                y_proba_arr <= bin_boundaries[i + 1]
            )
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true_arr[in_bin])
            bin_conf = np.mean(y_proba_arr[in_bin])
            ece += bin_size * np.abs(bin_acc - bin_conf)
    ece /= len(y_proba_arr)

    return {
        "brier_score": float(brier),
        "expected_calibration_error": float(ece),
    }


def compute_cost_analysis(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
    positive_label: str | None = None,
) -> dict[str, float]:
    """Análise de custo de erros para classificação binária.

    Quantifica o custo total considerando que False Negatives
    (clientes churn não detectados) e False Positives (clientes
    leais com retenção desnecessária) têm custos diferentes.

    Em telecom, FN geralmente custa mais que FP porque:
    - Custo de aquisição de novo cliente (CAC) > custo de
      retenção (discount, upgrade, suporte dedicado)
    - Churn de cliente de alto ARPU gera perda recorrente
    - Efeito de rede: churn contagia outros clientes

    # TODO: Ajustar custos reais (cost_fn, cost_fp) em outro PR.
    Valores placeholder. Ver issue #19 para contexto de
    negócio e levantamento de valores financeiros.

    Args:
        y_true: Rótulos verdadeiros.
        y_pred: Rótulos preditos.
        cost_fn: Custo de cada False Negative (cliente churn
            não detectado). Default 1.0 (placeholder).
        cost_fp: Custo de cada False Positive (retenção
            aplicada em cliente leal). Default 1.0 (placeholder).
        positive_label: Label da classe positiva para strings.

    Returns:
        Dicionário com:
            - total_cost: Custo total da matriz de confusão
            - normalized_cost: Custo médio por amostra
            - cost_false_negatives: Custo total de FN
            - cost_false_positives: Custo total de FP
            - savings_vs_no_action: Número de TP * cost_fn,
                representando churn evitado se ação fosse perfeita
            - wasted_retention: Número de FP * cost_fp,
                representando retenção desnecessária
    """
    y_true_arr = _to_numpy_array(y_true)
    y_pred_arr = _to_numpy_array(y_pred)

    if positive_label is not None:
        y_true_bin = (y_true_arr == positive_label).astype(int)
        y_pred_bin = (y_pred_arr == positive_label).astype(int)
    else:
        y_true_bin = y_true_arr.astype(int)
        y_pred_bin = y_pred_arr.astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    _tn, fp, fn, tp = cm.ravel()
    total_cost = fn * cost_fn + fp * cost_fp
    n_samples = len(y_true_bin)
    normalized_cost = total_cost / n_samples if n_samples > 0 else 0.0

    return {
        "total_cost": float(total_cost),
        "normalized_cost": float(normalized_cost),
        "cost_false_negatives": float(fn * cost_fn),
        "cost_false_positives": float(fp * cost_fp),
        "savings_vs_no_action": float(tp * cost_fn),
        "wasted_retention": float(fp * cost_fp),
    }


def analyze_threshold_tradeoff(
    y_true: pd.Series | np.ndarray,
    y_proba_positive: pd.Series | np.ndarray,
    thresholds: np.ndarray | None = None,
    cost_fn: float = 500.0,
    cost_fp: float = 50.0,
) -> pd.DataFrame:
    """Análise de trade-off precision/recall ao longo de thresholds.

    Permite visualizar como precision, recall, f1 e custo estimado
    variam conforme o threshold de decisão muda. Essencial para
    definir o ponto de operação ótimo considerando custos de
    negócio.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_proba_positive: Probabilidades preditas para classe
            positiva.
        thresholds: Array de thresholds para avaliar. Se None,
            usa np.arange(0.05, 1.0, 0.05).
        cost_fn: Custo de cada False Negative (cliente churn
            não detectado). Default 500 (LTV estimado).
        cost_fp: Custo de cada False Positive (retenção
            aplicada em cliente leal). Default 50 (custo de
            campanha/contato).

    Returns:
        DataFrame com colunas:
            - threshold: Limiar de decisão
            - precision, recall, f1_score, accuracy: Métricas
            - total_cost: Custo estimado com custos de negocio
            - false_positives, false_negatives: Contagens
    """
    y_true_arr = _to_numpy_array(y_true).astype(int)
    y_proba_arr = _to_numpy_array(y_proba_positive)

    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)

    records: list[dict[str, float | int]] = []
    for thresh in thresholds:
        y_pred_thresh = (y_proba_arr >= thresh).astype(int)

        prec = precision_score(y_true_arr, y_pred_thresh, zero_division=0)
        rec = recall_score(y_true_arr, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_true_arr, y_pred_thresh, zero_division=0)
        acc = accuracy_score(y_true_arr, y_pred_thresh)

        cm = confusion_matrix(y_true_arr, y_pred_thresh, labels=[0, 1])
        _tn, fp, fn, _tp = cm.ravel()

        total_cost = fn * cost_fn + fp * cost_fp

        records.append(
            {
                "threshold": round(float(thresh), 2),
                "precision": float(prec),
                "recall": float(rec),
                "f1_score": float(f1),
                "accuracy": float(acc),
                "total_cost": float(total_cost),
                "cost_false_negatives": int(fn * cost_fn),
                "cost_false_positives": int(fp * cost_fp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
        )

    return pd.DataFrame(records)


def compute_precision_at_k(
    y_true: pd.Series | np.ndarray,
    y_proba_positive: pd.Series | np.ndarray,
    k_values: tuple[int, ...] = (
        100,
        500,
        1000,
        int(0.05 * 1500),
        int(0.1 * 1500),
        int(0.2 * 1500),
    ),
) -> dict[str, float]:
    """Precision e Recall nos top-k clientes por probabilidade.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_proba_positive: Probabilidades preditas para classe positiva.
        k_values: Tupla de k absolutos ou percentuais.

    Returns:
        Dicionario com precision@k e recall@k para cada k.
    """
    y_true_arr = _to_numpy_array(y_true).astype(int)
    y_proba_arr = _to_numpy_array(y_proba_positive)

    # Ordena por probabilidade decrescente
    order = np.argsort(-y_proba_arr)
    y_true_sorted = y_true_arr[order]

    metrics: dict[str, float] = {}
    for kv in k_values:
        n_at_k = min(kv, len(y_true_arr))
        top_k_true = y_true_sorted[:n_at_k]
        precision_k = np.mean(top_k_true) if n_at_k > 0 else 0.0
        recall_k = (
            np.sum(top_k_true) / np.sum(y_true_arr)
            if np.sum(y_true_arr) > 0
            else 0.0
        )
        metrics[f"precision_at_{kv}"] = float(precision_k)
        metrics[f"recall_at_{kv}"] = float(recall_k)

    return metrics


def predict_risk_band(
    y_proba_positive: pd.Series | np.ndarray,
    thresholds: tuple[float, float] = (0.30, 0.60),
) -> np.ndarray:
    """Classifica clientes em bandas de risco baseado na probabilidade.

    Args:
        y_proba_positive: Probabilidades preditas (0-1).
        thresholds: Tupla com (limiar_medium, limiar_high).
            Default: (0.30, 0.60).

    Returns:
        Array com codigos: 0=Low, 1=Medium, 2=High.
    """
    y_proba_arr = _to_numpy_array(y_proba_positive)
    low_thr, high_thr = thresholds

    bands = np.zeros_like(y_proba_arr, dtype=int)
    bands[y_proba_arr >= low_thr] = 1  # Medium
    bands[y_proba_arr >= high_thr] = 2  # High
    return bands


def compute_risk_band_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba_positive: pd.Series | np.ndarray,
    thresholds: tuple[float, float] = (0.30, 0.60),
) -> dict[str, float]:
    """Métricas por banda de risco.

    Retorna taxa de churn e percentual da populacao em cada banda.

    Args:
        y_true: Rótulos verdadeiros (0/1).
        y_proba_positive: Probabilidades preditas.
        thresholds: Limites para Low/Medium/High.

    Returns:
        Dicionario com:
            - pct_low/medium/high: Percentual da amostra em cada banda
            - churn_rate_low/medium/high: Taxa de churn observada
            - capture_high: Percentual de churners capturados na banda High
            - capture_medium_high: Percentual na banda Medium+High
    """
    y_true_arr = _to_numpy_array(y_true).astype(int)
    bands = predict_risk_band(y_proba_positive, thresholds)

    total = len(y_true_arr)
    total_churners = np.sum(y_true_arr)

    results: dict[str, float] = {}
    for band_name, band_code in [("low", 0), ("medium", 1), ("high", 2)]:
        mask = bands == band_code
        count = np.sum(mask)
        churn_count = np.sum(y_true_arr[mask])
        results[f"pct_{band_name}"] = (
            float(count / total) if total > 0 else 0.0
        )
        results[f"churn_rate_{band_name}"] = (
            float(churn_count / count) if count > 0 else 0.0
        )

    high_mask = bands == 2  # noqa: PLR2004
    medium_high_mask = bands >= 1
    results["capture_high"] = (
        float(np.sum(y_true_arr[high_mask]) / total_churners)
        if total_churners > 0
        else 0.0
    )
    results["capture_medium_high"] = (
        float(np.sum(y_true_arr[medium_high_mask]) / total_churners)
        if total_churners > 0
        else 0.0
    )

    return results
