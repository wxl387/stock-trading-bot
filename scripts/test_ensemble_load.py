#!/usr/bin/env python3
"""
Quick test to verify ensemble can load all models including transformer.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.models.ensemble_model import EnsembleModel
from config.settings import setup_logging

logger = setup_logging()

def main():
    """Test ensemble model loading."""
    print("=" * 60)
    print("TESTING ENSEMBLE MODEL LOADING")
    print("=" * 60)

    # Initialize ensemble
    ensemble = EnsembleModel(sequence_length=20)

    # Load all models
    print("\nAttempting to load models...")
    active_models = ensemble.load_models()

    print("\n" + "=" * 60)
    print("ENSEMBLE MODEL INFO")
    print("=" * 60)
    info = ensemble.get_model_info()
    print(f"Active models: {info['active_models']}")
    print(f"Model weights: {info['model_weights']}")
    print(f"Voting method: {info['voting_method']}")
    print(f"Sequence length: {info['sequence_length']}")
    print(f"Is loaded: {info['is_loaded']}")
    print("=" * 60)

    # Check individual models
    print("\nModel Details:")
    if ensemble.xgboost_model:
        print(f"- XGBoost: Loaded, {len(ensemble.xgboost_model.feature_names)} features")

    if ensemble.lstm_model:
        print(f"- LSTM: Loaded, sequence_length={ensemble.lstm_model.sequence_length}, features={ensemble.lstm_model.n_features}")
        if hasattr(ensemble.lstm_model, 'feature_names') and ensemble.lstm_model.feature_names:
            print(f"  Feature names: {len(ensemble.lstm_model.feature_names)} features")

    if ensemble.cnn_model:
        print(f"- CNN: Loaded, sequence_length={ensemble.cnn_model.sequence_length}, features={ensemble.cnn_model.n_features}")
        if hasattr(ensemble.cnn_model, 'feature_names') and ensemble.cnn_model.feature_names:
            print(f"  Feature names: {len(ensemble.cnn_model.feature_names)} features")

    if ensemble.transformer_model:
        print(f"- Transformer: Loaded, sequence_length={ensemble.transformer_model.sequence_length}, features={ensemble.transformer_model.n_features}")
        if hasattr(ensemble.transformer_model, 'feature_names') and ensemble.transformer_model.feature_names:
            print(f"  Feature names: {len(ensemble.transformer_model.feature_names)} features")

    print("\n" + "=" * 60)
    if len(active_models) == 4:
        print("SUCCESS: All 4 models loaded successfully!")
    else:
        print(f"WARNING: Only {len(active_models)}/4 models loaded")
    print("=" * 60)

if __name__ == "__main__":
    main()
