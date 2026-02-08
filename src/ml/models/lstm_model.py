"""
LSTM model for stock price direction prediction.
"""
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
import numpy as np

# Import TensorFlow BEFORE pandas to avoid model.fit() deadlock
# (pandas 2.3+ / TF 2.20+ import order bug on macOS)
from src.ml.device_config import configure_tensorflow_device, log_device_info
_DEVICE = configure_tensorflow_device()
import tensorflow as tf

import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config.settings import MODELS_DIR
from src.ml.sequence_utils import normalize_features, apply_normalization

logger = logging.getLogger(__name__)


class LSTMModel:
    """
    LSTM classifier for predicting stock price direction.
    """

    def __init__(
        self,
        sequence_length: int = 20,
        n_features: int = 75,
        lstm_units: Tuple[int, int] = (64, 32),
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.

        Args:
            sequence_length: Number of time steps in each sequence.
            n_features: Number of features per time step.
            lstm_units: Tuple of units for each LSTM layer.
            dropout_rate: Dropout rate for regularization.
            learning_rate: Learning rate for Adam optimizer.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model: Optional[Sequential] = None
        self.normalization_stats: Optional[Dict] = None
        self.feature_names: List[str] = []
        self.is_trained = False
        self.history = None

        logger.info(
            f"Initialized LSTMModel: seq_len={sequence_length}, "
            f"features={n_features}, units={lstm_units}"
        )

    def build_model(self) -> Sequential:
        """
        Build the LSTM architecture.

        Returns:
            Compiled Keras Sequential model.
        """
        model = Sequential([
            # First LSTM layer
            LSTM(
                self.lstm_units[0],
                return_sequences=True,
                input_shape=(self.sequence_length, self.n_features),
                kernel_regularizer=l2(1e-4)
            ),
            BatchNormalization(),
            Dropout(self.dropout_rate),

            # Second LSTM layer
            LSTM(self.lstm_units[1], return_sequences=False, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(self.dropout_rate),

            # Dense layers
            Dense(16, activation='relu'),
            Dropout(self.dropout_rate / 2),

            # Output layer
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"Built LSTM model with {model.count_params():,} parameters (device: {_DEVICE})")

        self.model = model
        return model

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ) -> Dict[str, float]:
        """
        Train the LSTM model.

        Args:
            X: Training sequences of shape (n_samples, sequence_length, n_features).
            y: Training labels.
            eval_set: Optional validation set (X_val, y_val).
            epochs: Maximum training epochs.
            batch_size: Batch size for training.
            early_stopping_patience: Patience for early stopping.

        Returns:
            Dictionary with training metrics.
        """
        logger.info(f"Training LSTM with {len(X)} sequences")

        # Update n_features based on actual data
        self.n_features = X.shape[2]
        self.sequence_length = X.shape[1]

        # Normalize data
        if eval_set is not None:
            X_val, y_val = eval_set
            X, X_val, self.normalization_stats = normalize_features(X, X_val)
            validation_data = (X_val, y_val)
        else:
            X, _, self.normalization_stats = normalize_features(X)
            validation_data = None

        # Build model
        self.model = self.build_model()

        # Callbacks — monitor val_accuracy for classification (val_loss can
        # decrease at low-accuracy epochs due to calibration overshoot)
        es_monitor = 'val_accuracy' if validation_data else 'loss'
        es_mode = 'max' if validation_data else 'min'
        callbacks = [
            EarlyStopping(
                monitor=es_monitor,
                mode=es_mode,
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor=es_monitor,
                mode=es_mode,
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train
        self.history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0  # Silent - verbose output causes broken pipe on MPS GPU
        )

        self.is_trained = True

        # Calculate training metrics
        y_pred = (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()
        metrics = self._calculate_metrics(y, y_pred)

        logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.4f}")

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

        # Normalize input
        if self.normalization_stats is not None:
            X = apply_normalization(X, self.normalization_stats)

        proba = self.model.predict(X, verbose=0)
        return (proba > 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Feature sequences.

        Returns:
            Array of shape (n_samples, 2) with [prob_class_0, prob_class_1].
        """
        self._ensure_trained()

        # Normalize input
        if self.normalization_stats is not None:
            X = apply_normalization(X, self.normalization_stats)

        prob_1 = self.model.predict(X, verbose=0).flatten()
        prob_0 = 1 - prob_1

        return np.column_stack([prob_0, prob_1])

    def save(self, name: str = "lstm_model") -> Path:
        """
        Save model to disk.

        Args:
            name: Model name.

        Returns:
            Path to saved model directory.
        """
        self._ensure_trained()

        MODELS_DIR.mkdir(exist_ok=True)
        model_dir = MODELS_DIR / name
        model_dir.mkdir(exist_ok=True)

        # Save Keras model with .keras extension (required by Keras 3)
        model_path = model_dir / "model.keras"
        self.model.save(model_path)

        # Save metadata and normalization stats
        metadata = {
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "lstm_units": list(self.lstm_units),
            "dropout_rate": self.dropout_rate,
            "learning_rate": self.learning_rate,
            "feature_names": self.feature_names,
            "normalization_stats": {
                "mean": self.normalization_stats["mean"].tolist(),
                "std": self.normalization_stats["std"].tolist()
            } if self.normalization_stats else None
        }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_dir}")
        return model_dir

    def load(self, name: str = "lstm_model") -> None:
        """
        Load model from disk.

        Args:
            name: Model name.
        """
        model_dir = MODELS_DIR / name

        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")

        # Load Keras model (check for .keras file first, then legacy format)
        model_path = model_dir / "model.keras"
        if model_path.exists():
            self.model = keras.models.load_model(model_path)
        else:
            # Legacy format - directory or .h5
            self.model = keras.models.load_model(model_dir)

        # Load metadata
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.sequence_length = metadata.get("sequence_length", self.sequence_length)
        self.n_features = metadata.get("n_features", self.n_features)
        self.lstm_units = tuple(metadata.get("lstm_units", self.lstm_units))
        self.dropout_rate = metadata.get("dropout_rate", self.dropout_rate)
        self.learning_rate = metadata.get("learning_rate", self.learning_rate)
        self.feature_names = metadata.get("feature_names", [])

        if metadata.get("normalization_stats"):
            self.normalization_stats = {
                "mean": np.array(metadata["normalization_stats"]["mean"]),
                "std": np.array(metadata["normalization_stats"]["std"])
            }

        self.is_trained = True
        logger.info(f"Model loaded from {model_dir}")

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        epochs: int = 30
    ) -> Dict[str, list]:
        """
        Perform time series cross-validation.

        Args:
            X: Feature sequences.
            y: Target labels.
            n_splits: Number of CV splits.
            epochs: Training epochs per fold.

        Returns:
            Dictionary with metrics for each fold.
        """
        logger.info(f"Running {n_splits}-fold time series cross-validation")

        results = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

        # Calculate fold sizes
        fold_size = len(X) // (n_splits + 1)

        for fold in range(n_splits):
            # Training set grows with each fold
            train_end = (fold + 2) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, len(X))

            if val_end <= val_start:
                break

            X_train = X[:train_end]
            y_train = y[:train_end]
            X_val = X[val_start:val_end]
            y_val = y[val_start:val_end]

            # Create fresh model for each fold
            fold_model = LSTMModel(
                sequence_length=self.sequence_length,
                n_features=self.n_features,
                lstm_units=self.lstm_units,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate
            )

            # Train on this fold
            fold_model.train(
                X_train, y_train,
                eval_set=(X_val, y_val),
                epochs=epochs,
                early_stopping_patience=5
            )

            # Evaluate
            y_pred = fold_model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)

            for key in results:
                results[key].append(metrics[key])

            logger.debug(f"Fold {fold + 1}: Accuracy = {metrics['accuracy']:.4f}")

        # Log average metrics
        for key in results:
            avg = np.mean(results[key])
            std = np.std(results[key])
            logger.info(f"CV {key}: {avg:.4f} (+/- {std:.4f})")

        return results

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 30,
        n_splits: int = 3,
        epochs_per_trial: int = 20
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X: Feature sequences of shape (n_samples, sequence_length, n_features).
            y: Target labels.
            n_trials: Number of optimization trials.
            n_splits: Number of CV splits for evaluation.
            epochs_per_trial: Training epochs per trial (reduced for speed).

        Returns:
            Dictionary with best parameters and score.
        """
        logger.info(f"Starting LSTM hyperparameter tuning with {n_trials} trials...")

        # Store original dimensions
        self.sequence_length = X.shape[1]
        self.n_features = X.shape[2]

        def objective(trial):
            # Suggest hyperparameters
            lstm_units_1 = trial.suggest_categorical("lstm_units_1", [32, 64, 128])
            lstm_units_2 = trial.suggest_categorical("lstm_units_2", [16, 32, 64])
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

            # Time series cross-validation
            fold_size = len(X) // (n_splits + 1)
            scores = []

            for fold in range(n_splits):
                train_end = (fold + 2) * fold_size
                val_start = train_end
                val_end = min(val_start + fold_size, len(X))

                if val_end <= val_start:
                    break

                X_train = X[:train_end]
                y_train = y[:train_end]
                X_val = X[val_start:val_end]
                y_val = y[val_start:val_end]

                # Normalize data
                X_train_norm, X_val_norm, _ = normalize_features(X_train, X_val)

                # Build model with trial parameters
                model = Sequential([
                    LSTM(lstm_units_1, return_sequences=True,
                         input_shape=(self.sequence_length, self.n_features)),
                    BatchNormalization(),
                    Dropout(dropout_rate),
                    LSTM(lstm_units_2, return_sequences=False),
                    BatchNormalization(),
                    Dropout(dropout_rate),
                    Dense(16, activation='relu'),
                    Dropout(dropout_rate / 2),
                    Dense(1, activation='sigmoid')
                ])

                model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )

                # Train with early stopping on accuracy
                early_stop = EarlyStopping(
                    monitor='val_accuracy',
                    mode='max',
                    patience=5,
                    restore_best_weights=True,
                    verbose=0
                )

                model.fit(
                    X_train_norm, y_train,
                    validation_data=(X_val_norm, y_val),
                    epochs=epochs_per_trial,
                    batch_size=batch_size,
                    callbacks=[early_stop],
                    verbose=0
                )

                # Evaluate
                y_pred = (model.predict(X_val_norm, verbose=0) > 0.5).astype(int).flatten()
                scores.append(accuracy_score(y_val, y_pred))

                # Clear session to prevent memory buildup
                keras.backend.clear_session()

            return np.mean(scores) if scores else 0.0

        # Create study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params
        best_lstm_units = (best_params["lstm_units_1"], best_params["lstm_units_2"])

        logger.info(f"Best CV accuracy: {study.best_value:.4f}")
        logger.info(f"Best parameters: lstm_units={best_lstm_units}, "
                   f"dropout={best_params['dropout_rate']:.3f}, "
                   f"lr={best_params['learning_rate']:.6f}, "
                   f"batch_size={best_params['batch_size']}")

        # Update model parameters
        self.lstm_units = best_lstm_units
        self.dropout_rate = best_params["dropout_rate"]
        self.learning_rate = best_params["learning_rate"]

        return {
            "best_params": {
                "lstm_units": best_lstm_units,
                "dropout_rate": best_params["dropout_rate"],
                "learning_rate": best_params["learning_rate"],
                "batch_size": best_params["batch_size"]
            },
            "best_score": study.best_value,
            "n_trials": n_trials
        }

    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            self.model = self.build_model()

        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        return "\n".join(summary_lines)

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
