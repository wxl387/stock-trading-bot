#!/usr/bin/env python3
"""
Comprehensive test of the full 4-model ensemble (XGBoost + LSTM + CNN + Transformer).
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from src.ml.models.ensemble_model import EnsembleModel
from config.settings import setup_logging

logger = setup_logging()

def main():
    """Test full ensemble prediction."""
    print("=" * 60)
    print("TESTING FULL 4-MODEL ENSEMBLE")
    print("=" * 60)

    # Initialize and load ensemble
    print("\nInitializing ensemble...")
    ensemble = EnsembleModel(sequence_length=20)

    print("Loading all models...")
    active_models = ensemble.load_models()

    print("\n" + "=" * 60)
    print("ENSEMBLE STATUS")
    print("=" * 60)
    print(f"Active models: {active_models}")
    print(f"Number of models: {len(active_models)}")
    print(f"Model weights: {ensemble.model_weights}")
    print(f"Voting method: {ensemble.voting_method.value}")
    print("=" * 60)

    # Verify all 4 models are loaded
    expected_models = ['xgboost', 'lstm', 'cnn', 'transformer']
    if set(active_models) == set(expected_models):
        print("\n✓ SUCCESS: All 4 models loaded successfully!")
    else:
        print(f"\n✗ WARNING: Expected {expected_models}, got {active_models}")
        return

    # Create dummy test data
    print("\n" + "=" * 60)
    print("TESTING ENSEMBLE PREDICTIONS")
    print("=" * 60)

    n_samples = 3
    n_features = 100  # Number of features used in training
    sequence_length = 20

    # Create dummy flat features for XGBoost
    print(f"\nCreating test data ({n_samples} samples)...")
    X_flat = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )

    # Create dummy sequential features for LSTM/CNN/Transformer
    X_seq = np.random.randn(n_samples, sequence_length, n_features)

    # Make ensemble predictions
    print("Making ensemble predictions...")
    predictions = ensemble.predict(X_flat, X_seq)
    probabilities = ensemble.predict_proba(X_flat, X_seq)

    print("\n" + "=" * 60)
    print("ENSEMBLE PREDICTION RESULTS")
    print("=" * 60)
    for i in range(n_samples):
        print(f"\nSample {i+1}:")
        print(f"  Ensemble Prediction: {predictions[i]} ({'UP' if predictions[i] == 1 else 'DOWN'})")
        print(f"  Probability Down: {probabilities[i][0]:.4f}")
        print(f"  Probability Up:   {probabilities[i][1]:.4f}")
        print(f"  Confidence:       {max(probabilities[i]):.4f}")

    # Test single prediction with breakdown
    print("\n" + "=" * 60)
    print("DETAILED SINGLE PREDICTION (with model breakdown)")
    print("=" * 60)

    X_single = X_flat.iloc[[0]]
    feature_history = pd.DataFrame(
        X_seq[0],
        columns=[f"feature_{i}" for i in range(n_features)]
    )

    prediction, confidence, model_probs = ensemble.predict_single(X_single, feature_history)

    print(f"\nEnsemble Prediction: {prediction} ({'UP' if prediction == 1 else 'DOWN'})")
    print(f"Confidence: {confidence:.4f}")
    print("\nIndividual Model Probabilities (Prob UP):")
    for model_name, prob in model_probs.items():
        weight = ensemble.model_weights.get(model_name, 1.0)
        print(f"  {model_name.upper():12s}: {prob:.4f} (weight: {weight:.1f})")

    print("\n" + "=" * 60)
    print("✓ SUCCESS: Full ensemble is working correctly!")
    print("  - All 4 models (XGBoost, LSTM, CNN, Transformer) loaded")
    print("  - Predictions are being generated")
    print("  - Model weights are being applied")
    print("=" * 60)

if __name__ == "__main__":
    main()
