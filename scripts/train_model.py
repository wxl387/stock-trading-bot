#!/usr/bin/env python3
"""
Script to train the ML trading model.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from config.settings import setup_logging, Settings
from src.ml.training import ModelTrainer

logger = setup_logging()


def main():
    """Train the trading model."""
    # Load config
    config = Settings.load_trading_config()

    # Get symbols from config
    symbols = config.get("trading", {}).get("symbols", [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"
    ])

    # ML config
    ml_config = config.get("ml_model", {})
    prediction_horizon = ml_config.get("prediction_horizon", 5)
    train_window = ml_config.get("train_window_days", 252)

    logger.info(f"Training model with {len(symbols)} symbols")
    logger.info(f"Prediction horizon: {prediction_horizon} days")
    logger.info(f"Training window: {train_window} days")

    # Initialize trainer
    trainer = ModelTrainer(
        symbols=symbols,
        prediction_horizon=prediction_horizon,
        train_period_days=train_window
    )

    # Train model
    logger.info("Starting training...")
    metrics = trainer.train()

    # Print results
    print("\n" + "=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    print(f"Training Accuracy:  {metrics['train']['accuracy']:.4f}")
    print(f"Training F1 Score:  {metrics['train']['f1']:.4f}")
    print(f"Test Accuracy:      {metrics['test']['accuracy']:.4f}")
    print(f"Test F1 Score:      {metrics['test']['f1']:.4f}")
    print("=" * 50)

    # Cross-validation
    logger.info("Running cross-validation...")
    cv_results = trainer.cross_validate(n_splits=5)

    print("\nCROSS-VALIDATION RESULTS")
    print("=" * 50)
    for metric, values in cv_results.items():
        import numpy as np
        print(f"{metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")
    print("=" * 50)

    # Save model
    trainer.save_model("trading_model")
    logger.info("Model saved as 'trading_model'")

    # Print feature importance
    print("\nTOP 10 FEATURES")
    print("=" * 50)
    importance = trainer.model.get_feature_importance(top_n=10)
    for _, row in importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
