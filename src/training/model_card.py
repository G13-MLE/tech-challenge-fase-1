"""Model Card builder para artefatos MLflow.

Segue o framework Model Cards for Model Reporting (Mitchell et al., 2019).
Gera dict JSON-serializavel para ser logado via mlflow.log_dict().
"""

from __future__ import annotations

from typing import Any

_MODEL_INFO: dict[str, dict[str, str]] = {
    "dummy": {
        "model_name": "DummyClassifier Baseline",
        "framework": "scikit-learn",
        "architecture": (
            "DummyClassifier - 3 estrategias "
            "(most_frequent, stratified, uniform)"
        ),
    },
    "mlp": {
        "model_name": "MLP PyTorch - Churn Prediction",
        "framework": "PyTorch 2.x",
        "architecture": "MLP([45]->128->64->32->1) + BatchNorm + Dropout(0.3)",
    },
    "logistic": {
        "model_name": "Logistic Regression Baseline",
        "framework": "scikit-learn",
        "architecture": "LogisticRegression(max_iter=1000) + 5-fold CV",
    },
}

_METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-Score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "brier_score": "Brier Score",
    "ece": "Expected Calibration Error (ECE)",
}

_INTENDED_USE = {
    "primary": (
        "Identificar clientes com alto risco de churn em operadora de "
        "telecomunicacoes para direcionar acoes de retencao proativas."
    ),
    "users": [
        "Equipes de Customer Success / Retencao",
        "Analistas de dados da operadora",
        "Sistemas automatizados de priorizacao de campanhas",
    ],
    "out_of_scope": [
        "Decisao automatica de cancelamento sem revisao humana",
        "Aplicacao direta em outros setores sem revalidacao",
        "Unica base para decisoes financeiras",
        "Discriminacao de clientes baseada em risco de churn",
    ],
}

_FACTORS = {
    "class_imbalance": {
        "description": "~27% churn / ~73% nao-churn",
        "impact": (
            "Modelo pode tender a prever a classe majoritaria; "
            "PR-AUC recomendada"
        ),
    },
    "contract_type": {
        "month_to_month": "Taxa de churn significativamente maior",
        "one_year_two_year": "Taxa de churn menor",
    },
    "tenure": {
        "low": "Clientes novos (tenure < 12 meses): maior risco",
        "high": (
            "Clientes antigos (tenure > 60 meses): "
            "menor risco mas sub-representados"
        ),
    },
    "services": {
        "description": (
            "Clientes sem servicos adicionais (OnlineSecurity, TechSupport) "
            "apresentam maior churn; Fibra optica > DSL em churn"
        )
    },
    "payment": {
        "description": (
            "Electronic check: maior churn; Pagamento automatico: menor churn"
        )
    },
    "charges": {
        "description": (
            "MonthlyCharges elevado: maior propensao a churn; "
            "TotalCharges baixo + tenure alto: possivel anomalia"
        )
    },
    "missing_attributes": [
        "Dados demograficos (idade, genero, renda)",
        "Historico de reclamacoes / suporte",
        "Uso real do servico (minutos, dados consumidos)",
        "Concorrencia local / precos de mercado",
    ],
}

_EVALUATION_DATA = {
    "dataset": "Telco Customer Churn (IBM Sample Data)",
    "source": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "split": "Estratificado por Churn (0.8/0.2), seed=42",
    "cleaning": [
        "TotalCharges convertido para numerico",
        "11 linhas com NaN removidas (~0.16%)",
        "customerID removido (sem valor preditivo)",
    ],
    "preprocessing": (
        "One-hot encoding + StandardScaler (fit apenas no treino)"
    ),
}

_TRAINING_DATA = {
    "same_as_eval": "Mesmo dataset Telco Customer Churn",
    "train_samples": 5625,
    "churn_rate": 0.27,
    "features_after_preprocessing": 45,
}

_ETHICAL_CONSIDERATIONS = {
    "biases": [
        {
            "name": "Desbalanceamento de classes",
            "description": (
                "~27% churn pode favorecer classe majoritaria, "
                "resultando em recall baixo para churners"
            ),
            "mitigation": (
                "Usar PR-AUC como metrica principal e ajustar threshold"
            ),
        },
        {
            "name": "Custo assimetrico",
            "description": (
                "FN=R$500 vs FP=R$50 favorece recall sobre precision, "
                "potencialmente gerando ofertas desnecessarias "
                "para clientes leais"
            ),
            "mitigation": (
                "Balancear threshold com base em orcamento de retencao"
            ),
        },
        {
            "name": "Ausencia de atributos protegidos",
            "description": (
                "Dataset nao contem genero, raca, idade, renda - "
                "impede analise de fairness mas tambem impede "
                "discriminacao direta"
            ),
            "mitigation": None,
        },
        {
            "name": "Correlacoes proxy",
            "description": (
                "PaymentMethod e Contract podem correlacionar com "
                "nivel socioeconomico, impactando grupos de menor renda"
            ),
            "mitigation": (
                "Monitorar precision/recall por segmento de contrato "
                "e pagamento"
            ),
        },
        {
            "name": "Generalizacao limitada",
            "description": (
                "Dataset de uma unica operadora; padroes de churn "
                "variam entre regioes e periodos"
            ),
            "mitigation": "Revalidar em novas operadoras antes de usar",
        },
    ],
    "general_mitigations": [
        "Monitorar metricas por segmento de contrato e pagamento",
        "Ajustar threshold com base em impacto por subgrupo",
        "Auditar periodicamente decisoes de retencao",
        "Nao usar modelo como unica base para decisoes afetando clientes",
    ],
}

_CAVEATS = {
    "limitations": [
        ("Dataset de uma unica operadora; revalidacao necessaria em outras"),
        (
            "Custos FN/FP (R$500/R$50) sao placeholders; "
            "ajustar com dados financeiros reais"
        ),
        (
            "Modelos sao baselines; performance pode melhorar "
            "com ensemble ou feature engineering"
        ),
        "Ausencia de dados temporais impede modelar sazonalidade",
        "Categorias novas em producao podem quebrar one-hot encoding",
    ],
    "recommendations": [
        "Usar threshold otimo de custo em vez do padrao 0.5",
        "Monitorar data drift em producao",
        "Re-treinar periodicamente com dados atualizados",
        (
            "Coletar metricas de negocio "
            "(retencao efetiva, ROI) para validar predicoes"
        ),
        "Implementar A/B testing antes de deploy completo",
    ],
}

_FAILURE_SCENARIOS = {
    "data_failures": [
        {
            "scenario": "Novas categorias em producao",
            "impact": (
                "One-hot encoding quebra com categorias nao vistas no treino"
            ),
            "mitigation": "Tratar como 'Other' ou rejeitar com alerta",
        },
        {
            "scenario": "TotalCharges vazio",
            "impact": "Valor ausente causa falha no preprocessing",
            "mitigation": "Imputar valor ou rejeitar com alerta",
        },
        {
            "scenario": "Data drift",
            "impact": (
                "Mudanca na distribuicao de features degrada performance"
            ),
            "mitigation": "Monitorar PSI e re-treinar quando necessario",
        },
    ],
    "model_failures": [
        {
            "scenario": "Clientes de alto tenure sub-representados",
            "impact": "Poucos exemplos de churn em tenure > 60 meses",
            "mitigation": "Ponderar amostras ou coletar mais dados",
        },
        {
            "scenario": "Novos servicos ou planos",
            "impact": (
                "Features de servico podem mudar com novidades da operadora"
            ),
            "mitigation": "Retreinar apos mudancas no portfolio",
        },
        {
            "scenario": "Calibracao degradada",
            "impact": "Probabilidades perdem calibracao ao longo do tempo",
            "mitigation": "Monitorar Brier Score e ECE periodicamente",
        },
    ],
    "business_failures": [
        {
            "scenario": "Threshold inadequado",
            "impact": "Usar 0.5 em vez do otimo gera custo total elevado",
            "mitigation": "Calcular e usar threshold otimo de custo",
        },
        {
            "scenario": "Orcamento limitado de retencao",
            "impact": "Impossivel reter todos os clientes de risco alto",
            "mitigation": (
                "Usar Precision@k para otimizar alocacao de recursos"
            ),
        },
    ],
    "infra_failures": [
        {
            "scenario": "API indisponivel",
            "impact": "Servico FastAPI fora do ar",
            "mitigation": (
                "Health check, retry, fallback para regra de negocio"
            ),
        },
        {
            "scenario": "Latencia SLO excedida",
            "impact": "Requisicoes acima de 500ms impactam experiencia",
            "mitigation": "Monitorar latencia via middleware",
        },
    ],
}

_COST_PARAMS = {"cost_fn": 500.0, "cost_fp": 50.0, "unit": "R$"}

_RISK_BANDS = [
    {"band": "Low", "range": "< 0.30"},
    {"band": "Medium", "range": "0.30 - 0.60"},
    {"band": "High", "range": "> 0.60"},
]


def build_model_card(  # noqa: PLR0912
    model_type: str, **values: str | float
) -> dict[str, Any]:
    """Constroi um Model Card dict para artefato MLflow.

    Args:
        model_type: Um de 'dummy', 'mlp', 'logistic'.
        **values: Valores do run para popular o card
            (ex: accuracy=0.85, roc_auc=0.92, dataset_version="abc123").

    Returns:
        Dict com 10 secoes do Model Card, pronto para mlflow.log_dict().
    """
    info = _MODEL_INFO.get(model_type, _MODEL_INFO["mlp"])
    v: dict[str, Any] = dict(values)

    # --- Secao 1: Model Details ---
    model_details = {
        "model_name": info["model_name"],
        "model_type": model_type,
        "framework": info["framework"],
        "architecture": info["architecture"],
        "version": v.get("model_version", "v1.0"),
        "seed": v.get("random_seed", 42),
        "authors": "G13-MLE",
        "dataset_version": v.get("dataset_version", "unknown"),
    }

    # --- Secao 4: Metrics ---
    metric_entries: list[dict[str, Any]] = []
    for key, label in _METRIC_LABELS.items():
        if key in v:
            metric_entries.append(
                {"metric": label, "key": key, "value": v[key]}
            )

    # Confusion matrix
    cm = {}
    for k in ("tn", "fp", "fn", "tp"):
        if k in v:
            cm[k] = v[k]

    metrics_section: dict[str, Any] = {
        "primary_metrics": metric_entries,
    }
    if cm:
        metrics_section["confusion_matrix"] = cm

    # --- Secao 7: Quantitative Analyses ---
    cost = {
        "cost_fn": v.get("cost_fn", _COST_PARAMS["cost_fn"]),
        "cost_fp": v.get("cost_fp", _COST_PARAMS["cost_fp"]),
        "unit": _COST_PARAMS["unit"],
    }
    if "total_cost" in v:
        cost["total_cost"] = v["total_cost"]

    qa: dict[str, Any] = {"cost_analysis": cost}

    if "optimal_threshold" in v:
        qa["optimal_threshold"] = v["optimal_threshold"]
        if "optimal_total_cost" in v:
            qa["optimal_total_cost"] = v["optimal_total_cost"]

    # Risk bands
    risk_band_metrics = {}
    for prefix in ("pct", "churn_rate", "capture"):
        for band in ("low", "medium", "high"):
            key = f"{prefix}_{band}"
            if key in v:
                risk_band_metrics[key] = v[key]
            elif f"test_{key}" in v:
                risk_band_metrics[key] = v[f"test_{key}"]
    if risk_band_metrics:
        qa["risk_bands"] = {
            "definitions": _RISK_BANDS,
            "metrics": risk_band_metrics,
        }

    pk_metrics = {}
    for key, val in v.items():
        if key.startswith(("precision_at_", "recall_at_")):
            pk_metrics[key] = val
    if pk_metrics:
        qa["precision_recall_at_k"] = pk_metrics

    if "ece" in v or "brier_score" in v:
        qa["calibration"] = {k: v[k] for k in ("brier_score", "ece") if k in v}

    return {
        "model_details": model_details,
        "intended_use": _INTENDED_USE,
        "factors": _FACTORS,
        "metrics": metrics_section,
        "evaluation_data": _EVALUATION_DATA,
        "training_data": _TRAINING_DATA,
        "quantitative_analyses": qa,
        "ethical_considerations": _ETHICAL_CONSIDERATIONS,
        "caveats_and_recommendations": _CAVEATS,
        "failure_scenarios": _FAILURE_SCENARIOS,
        "_framework": (
            "Model Cards for Model Reporting (Mitchell et al., ACM FAccT 2019)"
        ),
    }
