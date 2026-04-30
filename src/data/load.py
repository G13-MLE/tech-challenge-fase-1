"""Carregamento do dataset Telco Customer Churn."""

from __future__ import annotations

import pandas as pd

from src.data.cleaning import clean_telco_data


def load_telco_data(
    filepath: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    *,
    apply_cleaning: bool = True,
) -> pd.DataFrame:
    """Carrega o dataset Telco Customer Churn de arquivo CSV.

    Por padrao aplica a limpeza EDA (clean_telco_data) para garantir
    que todos os pipelines usem dados consistentes com a
    fonte da verdade da analise exploratoria.

    Args:
        filepath: Caminho para o arquivo CSV.
            Default: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
        apply_cleaning: Se True (padrao), aplica clean_telco_data()
            automaticamente apos o carregamento.

    Returns:
        DataFrame pandas com o dataset carregado (e limpo se
            apply_cleaning=True)

    Raises:
        FileNotFoundError: Se o arquivo nao existir. Inclui
            mensagem sugerindo rodar 'dvc pull'.

    Example:
        >>> df = load_telco_data()
        >>> print(df.shape)
        (7032, 20)
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Arquivo '{filepath}' nao encontrado. "
            "Voce lembrou de rodar 'dvc pull' para baixar os dados localmente?"
        ) from e

    if apply_cleaning:
        df = clean_telco_data(df)

    return df


if __name__ == "__main__":  # pragma: no cover
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _logger = logging.getLogger(__name__)
    _df = load_telco_data()
    _logger.info("Dados carregados:\n%s", _df.head().to_string())
