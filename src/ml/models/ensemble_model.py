"""
Ensemble model combining XGBoost, LSTM, and CNN predictions.
"""
import logging
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class VotingMethod(Enum):
    """Ensemble voting methods."""
    HARD = "hard"           # Majority voting on predictions
    SOFT = "soft"           # Average of probabilities
    WEIGHTED = "weighted"   # Weighted average of probabilities


@dataclass
class ModelPrediction:
    """Prediction from a single model."""
    model_name: str
    prob_up: float
    prob_down: float
    prediction: int
    weight: float


class EnsembleModel:
    """
    Ensemble model combining multiple model predictions.

    Supports:
    - Hard voting (majority vote)
    - Soft voting (probability averaging)
    - Weighted voting (weighted probability average)
    """

    def __init__(
        self,
        voting_method: VotingMethod = VotingMethod.SOFT,
        model_weights: Optional[Dict[str, float]] = None,
        sequence_length: int = 20
    ):
        """
        Initialize ensemble model.

        Args:
            voting_method: How to combine predictions
            model_weights: Dict of model_name -> weight (for weighted voting)
            sequence_length: Sequence length for LSTM/CNN models
        """
        self.voting_method = voting_method
        self.sequence_length = sequence_length

        # Default weights (can be tuned based on individual model performance)
        self.model_weights = model_weights or {
            "xgboost": 1.0,
            "lstm": 0.8,
            "cnn": 0.8,
            "transformer": 0.9
        }

        # Model instances
        self.xgboost_model = None
        self.lstm_model = None
        self.cnn_model = None
        self.transformer_model = None

        self.is_loaded = False
        self.active_models: List[str] = []

        # Feature names for XGBoost
        self.feature_names: Optional[List[str]] = None

    def load_models(
        self,
        xgboost_name: Optional[str] = "trading_model",
        lstm_name: Optional[str] = "lstm_trading_model",
        cnn_name: Optional[str] = "cnn_trading_model",
        transformer_name: Optional[str] = "transformer_trading_model"
    ) -> List[str]:
        """
        Load available models.

        Args:
            xgboost_name: Name of saved XGBoost model
            lstm_name: Name of saved LSTM model
            cnn_name: Name of saved CNN model
            transformer_name: Name of saved Transformer model

        Returns:
            List of successfully loaded model names
        """
        self.active_models = []

        # Load XGBoost
        if xgboost_name and self.model_weights.get("xgboost", 0) > 0:
            try:
                from src.ml.models.xgboost_model import XGBoostModel
                self.xgboost_model = XGBoostModel()
                self.xgboost_model.load(xgboost_name)
                self.active_models.append("xgboost")
                self.feature_names = self.xgboost_model.feature_names
                logger.info(f"Loaded XGBoost model: {xgboost_name}")
            except FileNotFoundError:
                logger.warning(f"XGBoost model not found: {xgboost_name}")
            except Exception as e:
                logger.error(f"Error loading XGBoost: {e}")

        # Load LSTM
        if lstm_name and self.model_weights.get("lstm", 0) > 0:
            try:
                from src.ml.models.lstm_model import LSTMModel
                self.lstm_model = LSTMModel(sequence_length=self.sequence_length)
                self.lstm_model.load(lstm_name)
                self.active_models.append("lstm")
                logger.info(f"Loaded LSTM model: {lstm_name}")
            except FileNotFoundError:
                logger.warning(f"LSTM model not found: {lstm_name}")
            except Exception as e:
                logger.error(f"Error loading LSTM: {e}")

        # Load CNN
        if cnn_name and self.model_weights.get("cnn", 0) > 0:
            try:
                from src.ml.models.cnn_model import CNNModel
                self.cnn_model = CNNModel(sequence_length=self.sequence_length)
                self.cnn_model.load(cnn_name)
                self.active_models.append("cnn")
                logger.info(f"Loaded CNN model: {cnn_name}")
            except FileNotFoundError:
                logger.warning(f"CNN model not found: {cnn_name}")
            except Exception as e:
                logger.error(f"Error loading CNN: {e}")

        # Load Transformer
        if transformer_name and self.model_weights.get("transformer", 0) > 0:
            try:
                from src.ml.models.transformer_model import TransformerModel
                self.transformer_model = TransformerModel(sequence_length=self.sequence_length)
                self.transformer_model.load(transformer_name)
                self.active_models.append("transformer")
                logger.info(f"Loaded Transformer model: {transformer_name}")
            except FileNotFoundError:
                logger.warning(f"Transformer model not found: {transformer_name}")
            except Exception as e:
                logger.error(f"Error loading Transformer: {e}")

        if not self.active_models:
            raise RuntimeError("No models could be loaded for ensemble")

        self.is_loaded = True
        logger.info(f"Ensemble loaded with models: {self.active_models}")

        return self.active_models

    def predict(
        self,
        X_flat: pd.DataFrame,
        X_seq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X_flat: Flat features for XGBoost (n_samples, n_features)
            X_seq: Sequential features for LSTM/CNN (n_samples, seq_len, n_features)

        Returns:
            Array of predicted labels (0 or 1)
        """
        proba = self.predict_proba(X_flat, X_seq)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_proba(
        self,
        X_flat: pd.DataFrame,
        X_seq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get ensemble prediction probabilities.

        Args:
            X_flat: Flat features for XGBoost
            X_seq: Sequential features for LSTM/CNN

        Returns:
            Array of shape (n_samples, 2) with [prob_class_0, prob_class_1]
        """
        self._ensure_loaded()

        all_probas = []
        weights = []

        # XGBoost predictions
        if "xgboost" in self.active_models and self.xgboost_model:
            try:
                proba = self.xgboost_model.predict_proba(X_flat)
                all_probas.append(proba)
                weights.append(self.model_weights["xgboost"])
            except Exception as e:
                logger.error(f"XGBoost prediction error: {e}")

        # LSTM predictions (need sequential data)
        if "lstm" in self.active_models and self.lstm_model and X_seq is not None:
            try:
                proba = self.lstm_model.predict_proba(X_seq)
                all_probas.append(proba)
                weights.append(self.model_weights["lstm"])
            except Exception as e:
                logger.error(f"LSTM prediction error: {e}")

        # CNN predictions (need sequential data)
        if "cnn" in self.active_models and self.cnn_model and X_seq is not None:
            try:
                proba = self.cnn_model.predict_proba(X_seq)
                all_probas.append(proba)
                weights.append(self.model_weights["cnn"])
            except Exception as e:
                logger.error(f"CNN prediction error: {e}")

        # Transformer predictions (need sequential data)
        if "transformer" in self.active_models and self.transformer_model and X_seq is not None:
            try:
                proba = self.transformer_model.predict_proba(X_seq)
                all_probas.append(proba)
                weights.append(self.model_weights["transformer"])
            except Exception as e:
                logger.error(f"Transformer prediction error: {e}")

        if not all_probas:
            logger.error("All model predictions failed — returning neutral prediction [0.5, 0.5]")
            return np.array([0.5, 0.5])

        # Combine based on voting method
        if self.voting_method == VotingMethod.HARD:
            return self._hard_vote(all_probas)
        elif self.voting_method == VotingMethod.SOFT:
            return self._soft_vote(all_probas)
        else:  # WEIGHTED
            return self._weighted_vote(all_probas, weights)

    def predict_single(
        self,
        X_flat: pd.DataFrame,
        feature_history: Optional[pd.DataFrame] = None
    ) -> Tuple[int, float, Dict[str, float]]:
        """
        Predict for a single sample with detailed breakdown.

        Args:
            X_flat: Single row of flat features
            feature_history: DataFrame with sequence_length rows for LSTM/CNN

        Returns:
            (prediction, confidence, model_probabilities)
        """
        self._ensure_loaded()

        model_probs: Dict[str, float] = {}
        weights_used: Dict[str, float] = {}

        # XGBoost
        if "xgboost" in self.active_models and self.xgboost_model:
            try:
                proba = self.xgboost_model.predict_proba(X_flat)[0]
                model_probs["xgboost"] = proba[1]  # prob_up
                weights_used["xgboost"] = self.model_weights["xgboost"]
            except Exception as e:
                logger.error(f"XGBoost single prediction error: {e}")

        # Create sequence for LSTM/CNN if history provided
        if feature_history is not None and len(feature_history) >= self.sequence_length:
            from src.ml.sequence_utils import create_single_sequence

            try:
                X_seq = create_single_sequence(feature_history, self.sequence_length)

                if "lstm" in self.active_models and self.lstm_model:
                    try:
                        proba = self.lstm_model.predict_proba(X_seq)[0]
                        model_probs["lstm"] = proba[1]
                        weights_used["lstm"] = self.model_weights["lstm"]
                    except Exception as e:
                        logger.error(f"LSTM single prediction error: {e}")

                if "cnn" in self.active_models and self.cnn_model:
                    try:
                        proba = self.cnn_model.predict_proba(X_seq)[0]
                        model_probs["cnn"] = proba[1]
                        weights_used["cnn"] = self.model_weights["cnn"]
                    except Exception as e:
                        logger.error(f"CNN single prediction error: {e}")

                if "transformer" in self.active_models and self.transformer_model:
                    try:
                        proba = self.transformer_model.predict_proba(X_seq)[0]
                        model_probs["transformer"] = proba[1]
                        weights_used["transformer"] = self.model_weights["transformer"]
                    except Exception as e:
                        logger.error(f"Transformer single prediction error: {e}")

            except Exception as e:
                logger.error(f"Sequence creation error: {e}")

        if not model_probs:
            return 0, 0.5, {}

        # Combine predictions based on voting method
        if self.voting_method == VotingMethod.HARD:
            # Majority vote
            votes = [1 if p > 0.5 else 0 for p in model_probs.values()]
            prediction = 1 if sum(votes) > len(votes) / 2 else 0
            final_prob = sum(model_probs.values()) / len(model_probs)
        elif self.voting_method == VotingMethod.SOFT:
            # Simple average — matches _soft_vote() in predict_proba()
            final_prob = sum(model_probs.values()) / len(model_probs)
            prediction = 1 if final_prob > 0.5 else 0
        else:
            # Weighted voting
            total_weight = sum(weights_used.values())
            if total_weight == 0:
                return 0, 0.5, model_probs

            final_prob = sum(
                prob * weights_used.get(model, 1.0)
                for model, prob in model_probs.items()
            ) / total_weight

            prediction = 1 if final_prob > 0.5 else 0

        confidence = max(final_prob, 1 - final_prob)

        return prediction, confidence, model_probs

    def _hard_vote(self, all_probas: List[np.ndarray]) -> np.ndarray:
        """Majority voting on predictions."""
        predictions = [(p[:, 1] > 0.5).astype(int) for p in all_probas]
        stacked = np.stack(predictions, axis=1)
        vote_ratio = np.mean(stacked, axis=1)

        # Return vote ratio as probability (e.g., 3/4 voted up → 0.75)
        result = np.zeros((len(vote_ratio), 2))
        result[:, 1] = vote_ratio
        result[:, 0] = 1 - vote_ratio
        return result

    def _soft_vote(self, all_probas: List[np.ndarray]) -> np.ndarray:
        """Average of probabilities."""
        stacked = np.stack(all_probas, axis=0)
        return np.mean(stacked, axis=0)

    def _weighted_vote(
        self,
        all_probas: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """Weighted average of probabilities."""
        weights_arr = np.array(weights)
        weights_arr = weights_arr / np.sum(weights_arr)  # Normalize
        stacked = np.stack(all_probas, axis=0)
        weighted = np.tensordot(weights_arr, stacked, axes=([0], [0]))
        return weighted

    def save_config(self, name: str = "ensemble_config") -> Path:
        """Save ensemble configuration."""
        config = {
            "voting_method": self.voting_method.value,
            "model_weights": self.model_weights,
            "sequence_length": self.sequence_length,
            "active_models": self.active_models
        }

        MODELS_DIR.mkdir(exist_ok=True)
        config_path = MODELS_DIR / f"{name}.json"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved ensemble config to {config_path}")
        return config_path

    def load_config(self, name: str = "ensemble_config") -> None:
        """Load ensemble configuration."""
        config_path = MODELS_DIR / f"{name}.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Ensemble config not found: {config_path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.voting_method = VotingMethod(config["voting_method"])
        self.model_weights = config["model_weights"]
        self.sequence_length = config["sequence_length"]

        logger.info(f"Loaded ensemble config from {config_path}")

    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            "voting_method": self.voting_method.value,
            "model_weights": self.model_weights,
            "active_models": self.active_models,
            "sequence_length": self.sequence_length,
            "is_loaded": self.is_loaded
        }

    def _ensure_loaded(self) -> None:
        """Ensure models are loaded."""
        if not self.is_loaded:
            raise RuntimeError("Ensemble not loaded. Call load_models() first.")
