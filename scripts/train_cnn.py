#!/usr/bin/env python3
"""
Script to train the 1D CNN trading model.
Much faster than LSTM on CPU.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import numpy as np
from config.settings import setup_logging, Settings
from src.data.data_fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer
from src.ml.models.cnn_model import CNNModel
from src.ml.sequence_utils import create_sequences, split_sequences_time_series

logger = setup_logging()


def prepare_data(
    symbols: list,
    period: str = "2y",
    prediction_horizon: int = 5,
    sequence_length: int = 20
):
    """
    Prepare sequential data for CNN training.
    """
    data_fetcher = DataFetcher()
    feature_engineer = FeatureEngineer()

    all_X = []
    all_y = []

    for symbol in symbols:
        logger.info(f"Processing {symbol}...")

        # Fetch data
        df = data_fetcher.fetch_historical(symbol, period=period)
        if df.empty:
            logger.warning(f"No data for {symbol}")
            continue

        # Add features
        df = feature_engineer.add_all_features(df)

        # Create labels (future returns)
        df["future_returns"] = df["close"].shift(-prediction_horizon) / df["close"] - 1
        df["label"] = (df["future_returns"] > 0).astype(int)

        # Drop rows with NaN
        df = df.dropna()

        if len(df) < sequence_length + 10:
            logger.warning(f"Not enough data for {symbol}")
            continue

        # Select feature columns
        exclude_cols = [
            "symbol", "date", "dividends", "stock_splits",
            "future_returns", "label", "label_binary", "label_3class"
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = df["label"]

        all_X.append(X)
        all_y.append(y)

    # Concatenate all symbols
    X_combined = np.concatenate([x.values for x in all_X], axis=0)
    y_combined = np.concatenate([y.values for y in all_y], axis=0)

    logger.info(f"Combined data: {len(X_combined)} samples, {X_combined.shape[1]} features")

    # Create sequences
    X_seq, y_seq = create_sequences(
        X=type('df', (), {'values': X_combined})(),  # Mock DataFrame
        y=type('s', (), {'values': y_combined})(),   # Mock Series
        sequence_length=sequence_length
    )

    return X_seq, y_seq, feature_cols


def main():
    """Train the CNN model."""
    parser = argparse.ArgumentParser(description="Train 1D CNN trading model")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--sequence-length", type=int, default=10, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    parser.add_argument("--filters", type=str, default="32,16", help="CNN filters per layer")

    args = parser.parse_args()

    # Load config
    config = Settings.load_trading_config()

    # Get symbols from args or config
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = config.get("trading", {}).get("symbols", [
            "AAPL", "MSFT", "GOOGL"
        ])

    # Parse filters
    filters = tuple(int(x) for x in args.filters.split(","))

    logger.info("=" * 60)
    logger.info("1D CNN MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Sequence length: {args.sequence_length}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Filters: {filters}")
    logger.info("=" * 60)

    # Prepare data
    logger.info("Preparing sequential data...")
    X_seq, y_seq, feature_cols = prepare_data(
        symbols=symbols,
        sequence_length=args.sequence_length
    )

    logger.info(f"Sequences: {X_seq.shape[0]}")
    logger.info(f"Sequence length: {X_seq.shape[1]}")
    logger.info(f"Features: {X_seq.shape[2]}")

    # Split data
    X_train, X_test, y_train, y_test = split_sequences_time_series(
        X_seq, y_seq, test_ratio=0.2
    )

    # Initialize model
    model = CNNModel(
        sequence_length=args.sequence_length,
        n_features=X_seq.shape[2],
        filters=filters
    )

    # Show model architecture
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(model.get_model_summary())
    print("=" * 60 + "\n")

    # Train model
    logger.info("Training model...")
    metrics = model.train(
        X_train, y_train,
        eval_set=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Evaluate on test set
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_test)

    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Training Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Training F1 Score:  {metrics['f1']:.4f}")
    print(f"Test Accuracy:      {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test F1 Score:      {f1_score(y_test, y_pred):.4f}")
    print("=" * 60)

    # Save model
    model.save("cnn_trading_model")
    logger.info("Model saved as 'cnn_trading_model'")

    # Class distribution
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    print(f"Training - Up: {y_train.sum():.0f} ({y_train.mean()*100:.1f}%), Down: {len(y_train)-y_train.sum():.0f}")
    print(f"Test     - Up: {y_test.sum():.0f} ({y_test.mean()*100:.1f}%), Down: {len(y_test)-y_test.sum():.0f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
