"""
MLP model for stock price direction prediction using scikit-learn.
Faster alternative to TensorFlow-based models on CPU.
"""
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class MLPSequenceModel:
    """
    MLP classifier for predicting stock price direction from sequences.
    Uses scikit-learn for efficient CPU training.
    """

    def __init__(
        self,
        sequence_length: int = 20,
        n_features: int = 75,
        hidden_layers: Tuple[int, ...] = (64, 32),
        learning_rate: float = 0.001,
        max_iter: int = 200
    ):
        """
        Initialize MLP model.

        Args:
            sequence_length: Number of time steps in each sequence.
            n_features: Number of features per time step.
            hidden_layers: Tuple of hidden layer sizes.
            learning_rate: Initial learning rate.
            max_iter: Maximum number of iterations.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.model: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False

        logger.info(
            f"Initialized MLPSequenceModel: seq_len={sequence_length}, "
            f"features={n_features}, hidden={hidden_layers}"
        )

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """Flatten 3D sequences to 2D for MLP input."""
        if len(X.shape) == 3:
            # (samples, seq_len, features) -> (samples, seq_len * features)
            return X.reshape(X.shape[0], -1)
        return X

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the MLP model.

        Args:
            X: Training sequences of shape (n_samples, sequence_length, n_features).
            y: Training labels.
            eval_set: Optional validation set (X_val, y_val) - used for logging only.

        Returns:
            Dictionary with training metrics.
        """
        logger.info(f"Training MLP with {len(X)} sequences")

        # Update dimensions
        if len(X.shape) == 3:
            self.sequence_length = X.shape[1]
            self.n_features = X.shape[2]

        # Flatten sequences
        X_flat = self._flatten_sequences(X)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_flat)

        # Create and train model
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layers,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            verbose=True
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        metrics = self._calculate_metrics(y, y_pred)

        logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.4f}")

        # Log validation performance if eval_set provided
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_flat = self._flatten_sequences(X_val)
            X_val_scaled = self.scaler.transform(X_val_flat)
            y_val_pred = self.model.predict(X_val_scaled)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)
            logger.info(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Feature sequences.

        Returns:
            Array of predicted labels (0 or 1).
        """
        self._ensure_trained()

        X_flat = self._flatten_sequences(X)
        X_scaled = self.scaler.transform(X_flat)

        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Feature sequences.

        Returns:
            Array of shape (n_samples, 2) with [prob_class_0, prob_class_1].
        """
        self._ensure_trained()

        X_flat = self._flatten_sequences(X)
        X_scaled = self.scaler.transform(X_flat)

        return self.model.predict_proba(X_scaled)

    def save(self, name: str = "mlp_model") -> Path:
        """
        Save model to disk.

        Args:
            name: Model name.

        Returns:
            Path to saved model file.
        """
        self._ensure_trained()

        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{name}.pkl"

        data = {
            "model": self.model,
            "scaler": self.scaler,
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter
        }

        with open(model_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def load(self, name: str = "mlp_model") -> None:
        """
        Load model from disk.

        Args:
            name: Model name.
        """
        model_path = MODELS_DIR / f"{name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.sequence_length = data["sequence_length"]
        self.n_features = data["n_features"]
        self.hidden_layers = data["hidden_layers"]
        self.learning_rate = data["learning_rate"]
        self.max_iter = data["max_iter"]

        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        input_size = self.sequence_length * self.n_features
        layers = [input_size] + list(self.hidden_layers) + [1]

        total_params = 0
        summary = f"MLPSequenceModel\n{'='*50}\n"
        summary += f"Input: {self.sequence_length} timesteps x {self.n_features} features = {input_size}\n"

        for i in range(len(layers) - 1):
            params = (layers[i] + 1) * layers[i+1]  # weights + bias
            total_params += params
            summary += f"Dense {layers[i]} -> {layers[i+1]}: {params:,} params\n"

        summary += f"{'='*50}\n"
        summary += f"Total params: {total_params:,}\n"

        return summary

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    def _ensure_trained(self) -> None:
        """Ensure model is trained before prediction."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
