"""
Utilities for creating sequence data for LSTM models.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    sequence_length: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flat tabular data into sequences for LSTM.

    Args:
        X: Feature DataFrame of shape (n_samples, n_features)
        y: Target Series of shape (n_samples,)
        sequence_length: Number of time steps in each sequence

    Returns:
        X_seq: Array of shape (n_samples - sequence_length, sequence_length, n_features)
        y_seq: Array of shape (n_samples - sequence_length,)
    """
    X_values = X.values if hasattr(X, 'values') else np.asarray(X)
    y_values = y.values if hasattr(y, 'values') else np.asarray(y)

    n_samples = len(X_values) - sequence_length
    n_features = X_values.shape[1]

    if n_samples <= 0:
        raise ValueError(
            f"Not enough samples ({len(X_values)}) for sequence length {sequence_length}"
        )

    # Pre-allocate arrays
    X_seq = np.zeros((n_samples, sequence_length, n_features))
    y_seq = np.zeros(n_samples)

    for i in range(n_samples):
        X_seq[i] = X_values[i:i + sequence_length]
        y_seq[i] = y_values[i + sequence_length]

    logger.info(
        f"Created {n_samples} sequences of length {sequence_length} "
        f"with {n_features} features"
    )

    return X_seq, y_seq


def create_single_sequence(
    X: pd.DataFrame,
    sequence_length: int = 20
) -> np.ndarray:
    """
    Create a single sequence from the last N rows for prediction.

    Args:
        X: Feature DataFrame with at least sequence_length rows
        sequence_length: Number of time steps

    Returns:
        Array of shape (1, sequence_length, n_features) ready for prediction
    """
    if len(X) < sequence_length:
        raise ValueError(
            f"DataFrame has {len(X)} rows but needs {sequence_length} for sequence"
        )

    # Take the last sequence_length rows
    X_seq = X.iloc[-sequence_length:].values
    # Add batch dimension
    X_seq = np.expand_dims(X_seq, axis=0)

    return X_seq


def normalize_features(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Normalize features using training set statistics.

    Args:
        X_train: Training sequences (n_samples, seq_len, n_features)
        X_test: Optional test sequences

    Returns:
        X_train_norm: Normalized training data
        X_test_norm: Normalized test data (or None)
        stats: Dictionary with mean and std for each feature
    """
    # Reshape to 2D for computing stats
    n_samples, seq_len, n_features = X_train.shape
    X_flat = X_train.reshape(-1, n_features)

    # Compute statistics
    mean = X_flat.mean(axis=0)
    std = X_flat.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero

    stats = {"mean": mean, "std": std}

    # Normalize training data
    X_train_norm = (X_train - mean) / std

    # Normalize test data if provided
    X_test_norm = None
    if X_test is not None:
        X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm, stats


def apply_normalization(X: np.ndarray, stats: dict) -> np.ndarray:
    """
    Apply pre-computed normalization to new data.

    Args:
        X: Data to normalize
        stats: Dictionary with mean and std

    Returns:
        Normalized data
    """
    return (X - stats["mean"]) / stats["std"]


def split_sequences_time_series(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences maintaining time order (no shuffling).

    Args:
        X_seq: Sequence features
        y_seq: Sequence targets
        test_ratio: Fraction for test set

    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X_seq) * (1 - test_ratio))

    X_train = X_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]

    logger.info(f"Split: {len(X_train)} train, {len(X_test)} test sequences")

    return X_train, X_test, y_train, y_test
