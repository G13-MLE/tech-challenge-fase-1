"""Pipeline sklearn reprodutivel para previsao de churn.

Este modulo fornece builders de pipelines que combinam
pre-processamento, tratamento de desbalanceamento e modelo
em um unico objeto sklearn serializavel.
"""

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import RANDOM_SEED


def build_logistic_pipeline(
    max_iter: int = 1000,
    random_seed: int = RANDOM_SEED,
    use_smote: bool = True,
    smote_k_neighbors: int = 5,
) -> ImbPipeline:
    """Constroi pipeline sklearn para Logistic Regression.

    O pipeline inclui:
        - Imputacao de missing values (mediana para numericas,
          constante 'missing' para categoricas)
        - Scaling (StandardScaler) para features numericas
        - One-hot encoding para features categoricas
        - SMOTE para tratamento de desbalanceamento (opcional)
        - Logistic Regression como classificador

    Args:
        max_iter: Iteracoes maximas do LogisticRegression.
        random_seed: Seed para reprodutibilidade.
        use_smote: Se True, aplica SMOTE apos pre-processamento.
        smote_k_neighbors: Numero de vizinhos para SMOTE.

    Returns:
        ImbPipeline pronto para fit/predict com DataFrame.
    """
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = make_column_transformer(
        (
            numeric_transformer,
            make_column_selector(dtype_include=np.number),
        ),
        (
            categorical_transformer,
            make_column_selector(dtype_include="object"),
        ),
        remainder="drop",
    )

    steps: list[tuple[str, object]] = [("preprocessor", preprocessor)]

    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(
                    random_state=random_seed,
                    k_neighbors=smote_k_neighbors,
                ),
            )
        )

    steps.append(
        (
            "classifier",
            LogisticRegression(
                max_iter=max_iter,
                random_state=random_seed,
            ),
        )
    )

    return ImbPipeline(steps)
