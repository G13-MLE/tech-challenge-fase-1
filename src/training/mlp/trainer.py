"""Loop de treino principal para modelo MLP com integração MLflow.

Este módulo implementa o MLPTrainer, que orquestra o treino do modelo MLP.
Inclui:
- Loop de treino com validação
- Detecção automática de dispositivo (GPU/CPU)
- Suporte a múltiplos otimizadores e schedulers
- Early stopping para prevenir overfitting
- Integração MLflow para tracking de experimentos
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.config.models import TrainingConfig
from src.constants import THRESHOLD
from src.training.metrics import compute_binary_classification_metrics
from src.training.mlp.checkpoint import save_best_model
from src.training.mlp.early_stopping import EarlyStopping
from src.training.mlp.model import MLPForTraining

logger = logging.getLogger(__name__)


class MLPTrainer:
    """Orquestrador de treino para modelo MLP.

    Esta classe gerencia todo o ciclo de vida do treino: configuração do
    dispositivo, preparação do otimizador, loop de treino/validação,
    ajuste de learning rate, early stopping e salvamento do melhor modelo.

    Attributes (Atributos):
        model: Instância MLPForTraining a ser treinada
        config: TrainingConfig com hiperparâmetros
        device: Dispositivo de computação ('cuda', 'mps' ou 'cpu')
        optimizer: Otimizador (Adam ou SGD)
        scheduler: Agendador opcional de learning rate
        early_stopping: Callback de parada antecipada
        history: Registro de métricas por época

    Exemplo:
        >>> config = TrainingConfig(lr=0.001, batch_size=64)
        >>> trainer = MLPTrainer(model, config)
        >>> history = trainer.fit(X_train, y_train)
        >>> trainer.log_to_mlflow()
    """

    def __init__(
        self,
        model: MLPForTraining,
        config: TrainingConfig,
        device: str = "auto",
    ) -> None:
        """Inicializa o MLPTrainer com modelo e configuração.

        Configura automaticamente o dispositivo (GPU/CPU), otimizador,
        scheduler e callbacks de treino.

        Args:
            model: Modelo MLPForTraining a ser treinado
            config: Configuração com hiperparâmetros de treino
            device: Dispositivo para computação ('auto', 'cuda', 'mps', 'cpu').
                'auto' detecta automaticamente GPU disponível
        """
        self.model = model
        self.config = config

        # Detecta dispositivo automaticamente se não especificado
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"  # NVIDIA GPU
            elif torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon (M1/M2/M3)
            else:
                self.device = "cpu"  # Fallback para CPU
        else:
            self.device = device

        # Move modelo para o dispositivo selecionado
        self.model.to(self.device)

        # Configura o otimizador
        if config.optimizer.lower() == "adam":
            self.optimizer: Adam | SGD = torch.optim.Adam(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        elif config.optimizer.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            raise ValueError(f"Otimizador não suportado: {config.optimizer}")

        # Configura o agendador de learning rate
        scheduler: ReduceLROnPlateau | StepLR | None
        if config.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=config.scheduler_patience,
            )
        elif config.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        else:
            scheduler = None
        self.scheduler: ReduceLROnPlateau | StepLR | None = scheduler

        # Inicializa early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode="min",
        )

        # Histórico de métricas
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_auc": [],
        }

    def fit(  # noqa: PLR0914
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        model_save_path: str = "models/best_model.pt",
    ) -> dict[str, list[float]]:
        """Executa o loop de treino completo.

        Treina o modelo por múltiplas épocas, validando a cada época,
        aplicando early stopping e salvando o melhor modelo.

        Args:
            X_train: Features de treino de shape (n_samples, n_features)
            y_train: Rótulos de treino de shape (n_samples,)
            X_val: Features de validação opcionais. Se None, usa val_split
            y_val: Rótulos de validação opcionais. Se None, usa val_split
            model_save_path: Caminho para salvar o melhor modelo

        Returns:
            Histórico de treino com métricas por época:
            {'train_loss': [...], 'val_loss': [...], 'val_f1': [...], ...}
        """
        # Cria split de validação se não fornecido
        if X_val is None or y_val is None:
            val_size = int(len(X_train) * self.config.val_split)
            indices = np.random.permutation(len(X_train))
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]

            X_val_split = X_train[val_indices]
            y_val_split = y_train[val_indices]
            X_train_split = X_train[train_indices]
            y_train_split = y_train[train_indices]
        else:
            X_train_split = X_train
            y_train_split = y_train
            X_val_split = X_val
            y_val_split = y_val

        # Cria datasets PyTorch
        train_dataset = TensorDataset(
            torch.tensor(X_train_split, dtype=torch.float32),
            torch.tensor(y_train_split, dtype=torch.float32),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_split, dtype=torch.float32),
            torch.tensor(y_val_split, dtype=torch.float32),
        )

        # Cria data loaders para batching
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,  # Embaralha a cada época
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        best_val_loss = float("inf")

        # Loop principal de treino
        for epoch in range(self.config.max_epochs):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader)

            # Registra métricas no histórico
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_metrics["f1_score"])
            if "roc_auc" in val_metrics:
                self.history["val_auc"].append(val_metrics["roc_auc"])

            logger.info(
                f"Epoch {epoch + 1}/{self.config.max_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val F1: {val_metrics['f1_score']:.4f}"
            )

            # Atualiza learning rate se houver scheduler
            if self.scheduler is not None:
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Salva melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_model(
                    self.model.model,
                    model_save_path,
                    metadata={"epoch": epoch, "val_loss": val_loss},
                )

            # Verifica early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping acionado na época {epoch + 1}")
                break

        return self.history

    def _train_epoch(self, loader: DataLoader[Any]) -> float:
        """Executa uma época de treino.

        Args:
            loader: DataLoader com dados de treino

        Returns:
            Perda média da época
        """
        self.model.train()  # Modo treino (ativa dropout, batch norm)
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in loader:
            # Move batches para o dispositivo
            batch_x_dev = batch_x.to(self.device)
            batch_y_dev = batch_y.to(self.device)

            # Zera gradientes acumulados
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_x_dev, targets=batch_y_dev)
            loss = outputs["loss"]

            # Backward pass (computa gradientes)
            loss.backward()

            # Atualiza pesos
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate(
        self, loader: DataLoader[Any]
    ) -> tuple[
        float,
        dict[str, float],
    ]:
        """Executa validação em um epoch.

        Args:
            loader: DataLoader com dados de validação

        Returns:
            Tupla (perda média, dicionário de métricas)
        """
        self.model.eval()  # Modo avaliação (desativa dropout)
        total_loss = 0.0
        num_batches = 0
        all_preds: list[np.ndarray[Any, Any]] = []
        all_probs: list[np.ndarray[Any, Any]] = []
        all_targets: list[np.ndarray[Any, Any]] = []

        # Desativa cálculo de gradientes (mais rápido, menos memória)
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x_dev = batch_x.to(self.device)
                batch_y_dev = batch_y.to(self.device)

                outputs = self.model(batch_x_dev, targets=batch_y_dev)
                loss = outputs["loss"]

                total_loss += loss.item()
                num_batches += 1

                # Converte probabilidades para predições binárias
                probs = outputs["probs"].cpu().numpy()
                preds = (probs > THRESHOLD).astype(int)

                all_preds.append(preds)
                all_probs.append(probs)
                all_targets.append(batch_y_dev.cpu().numpy())

        avg_loss = total_loss / num_batches

        # Calcula métricas no conjunto completo de validação
        # positive_label=None indica dados já numéricos (0/1)
        metrics = compute_binary_classification_metrics(
            y_true=np.concatenate(all_targets),
            y_pred=np.concatenate(all_preds),
            y_proba_positive=np.concatenate(all_probs),
            positive_label=None,
        )

        return avg_loss, metrics

    def log_to_mlflow(self) -> None:
        """Registra parâmetros e métricas no MLflow.

        Deve ser chamado após fit() para registrar o experimento completo.
        """
        mlflow.log_params(
            {
                "optimizer": self.config.optimizer,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
                "max_epochs": self.config.max_epochs,
                "early_stopping_patience": (
                    self.config.early_stopping_patience
                ),
            }
        )

        # Registra métricas de cada época
        for key, values in self.history.items():
            for idx, value in enumerate(values):
                mlflow.log_metric(key, value, step=idx)
