#!/usr/bin/env python3
"""
Quick test to verify transformer model can make predictions.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.ml.models.transformer_model import TransformerModel
from config.settings import setup_logging

logger = setup_logging()

def main():
    """Test transformer model prediction."""
    print("=" * 60)
    print("TESTING TRANSFORMER MODEL PREDICTION")
    print("=" * 60)

    # Load the model
    print("\nLoading transformer model...")
    model = TransformerModel()
    model.load("transformer_trading_model")

    print(f"Model loaded successfully!")
    print(f"- Sequence length: {model.sequence_length}")
    print(f"- Number of features: {model.n_features}")
    print(f"- Embed dimension: {model.embed_dim}")
    print(f"- Number of heads: {model.num_heads}")
    print(f"- Feed-forward dimension: {model.ff_dim}")
    print(f"- Number of transformer blocks: {model.num_transformer_blocks}")
    print(f"- Feature names: {len(model.feature_names)} features")

    # Create dummy test data
    print("\nCreating dummy test data...")
    n_samples = 5
    X_test = np.random.randn(n_samples, model.sequence_length, model.n_features)

    # Make predictions
    print(f"\nMaking predictions for {n_samples} samples...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    for i in range(n_samples):
        print(f"Sample {i+1}:")
        print(f"  Prediction: {predictions[i]} ({'UP' if predictions[i] == 1 else 'DOWN'})")
        print(f"  Probability Down: {probabilities[i][0]:.4f}")
        print(f"  Probability Up:   {probabilities[i][1]:.4f}")

    print("\n" + "=" * 60)
    print("SUCCESS: Transformer model is working correctly!")
    print("=" * 60)

if __name__ == "__main__":
    main()
