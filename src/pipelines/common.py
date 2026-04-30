"""Funções utilitárias comuns para pipelines de ML."""

from __future__ import annotations

import os

import numpy as np
import torch

from src.data.versioning import get_dataset_version_from_dvc


def load_dotenv_silent() -> None:
    """Carrega variáveis de ambiente do arquivo .env silenciosamente."""
    try:
        # nosec: B404 - lazy import necessário para fallback gracioso
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()  # nosec: B108
    except ImportError:
        pass


def set_global_seed(seed: int) -> None:
    """Define seed global para reprodutibilidade.

    Configura seed em PyTorch, NumPy e CUDA para garantir
    resultados reproduzíveis em pipelines de ML.

    Args:
        seed: Valor da semente para geração de números aleatórios.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_experiment_name(
    cli_arg: str | None,
    env_var_name: str,
    default_name: str,
) -> str:
    """Obtém nome do experimento com prioridade: CLI > env var > default."""
    if cli_arg:
        return cli_arg

    env_value = os.getenv(env_var_name)
    if env_value:
        return env_value

    return default_name


def safe_get_dataset_version() -> str:
    """Obtém versão do dataset via DVC com fallback gracioso.

    Returns:
        String com versão do dataset ou "unknown".
    """
    try:
        return get_dataset_version_from_dvc()
    except (FileNotFoundError, ValueError):
        return "unknown"
