"""
Transformer model for stock price direction prediction.
Uses multi-head self-attention to capture long-range dependencies in time series.
"""
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
import numpy as np

# Configure TensorFlow device (MPS for M2)
from src.ml.device_config import configure_tensorflow_device, log_device_info

# Configure device before importing TensorFlow
_DEVICE = configure_tensorflow_device()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config.settings import MODELS_DIR
from src.ml.sequence_utils import normalize_features, apply_normalization

logger = logging.getLogger(__name__)


class TransformerBlock(keras.layers.Layer):
    """
    A single Transformer encoder block with multi-head attention and feed-forward network.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def build(self, input_shape):
        self.att.build(input_shape, input_shape)
        self.ffn.build(input_shape)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)
        self.dropout1.build(input_shape)
        self.dropout2.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class PositionalEncoding(keras.layers.Layer):
    """
    Adds positional encoding to the input embeddings.
    """

    def __init__(self, sequence_length: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

        # Create positional encoding matrix
        position = np.arange(sequence_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))

        pe = np.zeros((sequence_length, embed_dim))
        pe[:, 0::2] = np.sin(position * div_term[:embed_dim//2 + embed_dim%2])
        pe[:, 1::2] = np.cos(position * div_term[:embed_dim//2])

        self.pos_encoding = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim,
        })
        return config


class TransformerModel:
    """
    Transformer classifier for predicting stock price direction.
    Uses multi-head self-attention to capture temporal patterns.
    """

    def __init__(
        self,
        sequence_length: int = 20,
        n_features: int = 75,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_transformer_blocks: int = 2,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001
    ):
        """
        Initialize Transformer model.

        Args:
            sequence_length: Number of time steps in each sequence.
            n_features: Number of features per time step.
            embed_dim: Dimension of the embedding/attention space.
            num_heads: Number of attention heads.
            ff_dim: Hidden dimension of feed-forward network.
            num_transformer_blocks: Number of transformer blocks.
            dropout_rate: Dropout rate for regularization.
            learning_rate: Learning rate for Adam optimizer.
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model: Optional[Model] = None
        self.normalization_stats: Optional[Dict] = None
        self.is_trained = False
        self.history = None
        self.feature_names: Optional[List[str]] = None

        logger.info(
            f"Initialized TransformerModel: seq_len={sequence_length}, "
            f"features={n_features}, embed_dim={embed_dim}, heads={num_heads}"
        )

    def build_model(self) -> Model:
        """
        Build the Transformer architecture.

        Returns:
            Compiled Keras Model.
        """
        inputs = Input(shape=(self.sequence_length, self.n_features))

        # Project input features to embedding dimension
        x = Dense(self.embed_dim)(inputs)

        # Add positional encoding
        x = PositionalEncoding(self.sequence_length, self.embed_dim)(x)
        x = Dropout(self.dropout_rate)(x)

        # Apply transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate
            )(x)

        # Global average pooling
        x = GlobalAveragePooling1D()(x)

        # Classification head
        x = Dense(32, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"Built Transformer model with {model.count_params():,} parameters (device: {_DEVICE})")

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
        Train the Transformer model.

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
        logger.info(f"Training Transformer with {len(X)} sequences")

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
        # increase even as accuracy improves due to calibration overshoot)
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
            verbose=2
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

    def save(self, name: str = "transformer_model") -> Path:
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

        # Save Keras model with .keras extension
        model_path = model_dir / "model.keras"
        self.model.save(model_path)

        # Save metadata and normalization stats
        metadata = {
            "model_type": "transformer",
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
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

    def load(self, name: str = "transformer_model") -> None:
        """
        Load model from disk.

        Args:
            name: Model name.
        """
        model_dir = MODELS_DIR / name

        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")

        # Register custom layers
        custom_objects = {
            "TransformerBlock": TransformerBlock,
            "PositionalEncoding": PositionalEncoding
        }

        # Load Keras model
        model_path = model_dir / "model.keras"
        if model_path.exists():
            self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
        else:
            self.model = keras.models.load_model(model_dir, custom_objects=custom_objects)

        # Load metadata
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.sequence_length = metadata["sequence_length"]
        self.n_features = metadata["n_features"]
        self.embed_dim = metadata["embed_dim"]
        self.num_heads = metadata["num_heads"]
        self.ff_dim = metadata["ff_dim"]
        self.num_transformer_blocks = metadata["num_transformer_blocks"]
        self.dropout_rate = metadata["dropout_rate"]
        self.learning_rate = metadata["learning_rate"]
        self.feature_names = metadata.get("feature_names")

        if metadata.get("normalization_stats"):
            self.normalization_stats = {
                "mean": np.array(metadata["normalization_stats"]["mean"]),
                "std": np.array(metadata["normalization_stats"]["std"])
            }

        self.is_trained = True
        logger.info(f"Model loaded from {model_dir}")

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 20,
        n_splits: int = 3,
        epochs_per_trial: int = 15
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
        logger.info(f"Starting Transformer hyperparameter tuning with {n_trials} trials...")

        # Store original dimensions
        self.sequence_length = X.shape[1]
        self.n_features = X.shape[2]

        def objective(trial):
            # Suggest hyperparameters
            embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
            num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
            ff_dim = trial.suggest_categorical("ff_dim", [64, 128, 256])
            num_blocks = trial.suggest_int("num_transformer_blocks", 1, 3)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

            # Ensure embed_dim is divisible by num_heads
            if embed_dim % num_heads != 0:
                return 0.0

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
                inputs = Input(shape=(self.sequence_length, self.n_features))
                x = Dense(embed_dim)(inputs)
                x = PositionalEncoding(self.sequence_length, embed_dim)(x)
                x = Dropout(dropout_rate)(x)

                for _ in range(num_blocks):
                    x = TransformerBlock(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        ff_dim=ff_dim,
                        dropout_rate=dropout_rate
                    )(x)

                x = GlobalAveragePooling1D()(x)
                x = Dense(32, activation='relu')(x)
                x = Dropout(dropout_rate)(x)
                outputs = Dense(1, activation='sigmoid')(x)

                model = Model(inputs=inputs, outputs=outputs)
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

        logger.info(f"Best CV accuracy: {study.best_value:.4f}")
        logger.info(f"Best parameters: embed_dim={best_params['embed_dim']}, "
                   f"heads={best_params['num_heads']}, "
                   f"ff_dim={best_params['ff_dim']}, "
                   f"blocks={best_params['num_transformer_blocks']}")

        # Update model parameters
        self.embed_dim = best_params["embed_dim"]
        self.num_heads = best_params["num_heads"]
        self.ff_dim = best_params["ff_dim"]
        self.num_transformer_blocks = best_params["num_transformer_blocks"]
        self.dropout_rate = best_params["dropout_rate"]
        self.learning_rate = best_params["learning_rate"]

        return {
            "best_params": {
                "embed_dim": best_params["embed_dim"],
                "num_heads": best_params["num_heads"],
                "ff_dim": best_params["ff_dim"],
                "num_transformer_blocks": best_params["num_transformer_blocks"],
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
