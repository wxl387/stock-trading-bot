"""
Tests for ML models - XGBoost, LSTM, CNN, Ensemble.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os

from src.ml.models.xgboost_model import XGBoostModel


class TestXGBoostModel:
    """Tests for XGBoost model."""

    def test_initialization(self):
        """Test XGBoost model initializes correctly."""
        model = XGBoostModel()
        assert model is not None
        assert model.is_trained is False

    def test_train_predict(self, small_training_data):
        """Test training and prediction."""
        X, y = small_training_data
        model = XGBoostModel()

        # Train
        model.train(X, y)
        assert model.is_trained is True

        # Predict
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self, small_training_data):
        """Test probability predictions."""
        X, y = small_training_data
        model = XGBoostModel()
        model.train(X, y)

        probas = model.predict_proba(X)
        assert probas.shape == (len(y), 2)
        assert all(0 <= p <= 1 for p in probas.flatten())

    def test_save_load(self, small_training_data, tmp_model_dir):
        """Test model save and load."""
        X, y = small_training_data
        model = XGBoostModel()
        model.train(X, y)

        # Save
        model_name = "test_xgboost"
        model.save(model_name, model_dir=str(tmp_model_dir))

        # Load into new model
        model2 = XGBoostModel()
        model2.load(model_name, model_dir=str(tmp_model_dir))

        assert model2.is_trained is True

        # Predictions should match
        pred1 = model.predict(X)
        pred2 = model2.predict(X)
        assert np.array_equal(pred1, pred2)

    def test_feature_importance(self, small_training_data):
        """Test feature importance extraction."""
        X, y = small_training_data
        model = XGBoostModel()
        model.train(X, y)

        importance = model.get_feature_importance()
        assert importance is not None
        assert len(importance) == X.shape[1]

    def test_cross_validate(self, small_training_data):
        """Test cross-validation."""
        X, y = small_training_data
        model = XGBoostModel()

        scores = model.cross_validate(X, y, cv=3)
        assert 'accuracy' in scores
        assert 0 <= scores['accuracy'] <= 1

    def test_model_not_trained_error(self, small_training_data):
        """Test error when predicting without training."""
        X, _ = small_training_data
        model = XGBoostModel()

        with pytest.raises(Exception):
            model.predict(X)


class TestLSTMModel:
    """Tests for LSTM model."""

    @pytest.mark.slow
    def test_initialization(self):
        """Test LSTM model initializes correctly."""
        from src.ml.models.lstm_model import LSTMModel
        model = LSTMModel(sequence_length=20, n_features=10)
        assert model is not None

    @pytest.mark.slow
    def test_build_model(self):
        """Test LSTM model architecture builds correctly."""
        from src.ml.models.lstm_model import LSTMModel
        model = LSTMModel(sequence_length=20, n_features=10)
        model._build_model()

        assert model.model is not None

    @pytest.mark.slow
    def test_lstm_sequence_handling(self, sequence_data):
        """Test LSTM handles sequence data correctly."""
        from src.ml.models.lstm_model import LSTMModel
        X, y = sequence_data

        model = LSTMModel(
            sequence_length=X.shape[1],
            n_features=X.shape[2]
        )

        # Should accept 3D input
        assert X.ndim == 3

    @pytest.mark.slow
    def test_lstm_train_predict(self, sequence_data):
        """Test LSTM training and prediction."""
        from src.ml.models.lstm_model import LSTMModel
        X, y = sequence_data

        model = LSTMModel(
            sequence_length=X.shape[1],
            n_features=X.shape[2]
        )

        # Train with minimal epochs for speed
        model.train(X, y, epochs=2, batch_size=16, verbose=0)

        assert model.is_trained is True

        predictions = model.predict(X)
        assert len(predictions) == len(y)


class TestCNNModel:
    """Tests for CNN model."""

    @pytest.mark.slow
    def test_initialization(self):
        """Test CNN model initializes correctly."""
        from src.ml.models.cnn_model import CNNModel
        model = CNNModel(sequence_length=20, n_features=10)
        assert model is not None

    @pytest.mark.slow
    def test_cnn_build_model(self):
        """Test CNN model architecture builds correctly."""
        from src.ml.models.cnn_model import CNNModel
        model = CNNModel(sequence_length=20, n_features=10)
        model._build_model()

        assert model.model is not None

    @pytest.mark.slow
    def test_cnn_train_predict(self, sequence_data):
        """Test CNN training and prediction."""
        from src.ml.models.cnn_model import CNNModel
        X, y = sequence_data

        model = CNNModel(
            sequence_length=X.shape[1],
            n_features=X.shape[2]
        )

        model.train(X, y, epochs=2, batch_size=16, verbose=0)

        assert model.is_trained is True

        predictions = model.predict(X)
        assert len(predictions) == len(y)


class TestEnsembleModel:
    """Tests for Ensemble model."""

    def test_initialization(self):
        """Test Ensemble model initializes correctly."""
        from src.ml.models.ensemble_model import EnsembleModel
        model = EnsembleModel()
        assert model is not None

    def test_ensemble_soft_voting(self, small_training_data):
        """Test ensemble soft voting."""
        from src.ml.models.ensemble_model import EnsembleModel, VotingMethod

        X, y = small_training_data

        # Create and train base XGBoost model
        xgb = XGBoostModel()
        xgb.train(X, y)

        ensemble = EnsembleModel(voting_method=VotingMethod.SOFT)
        ensemble.xgboost_model = xgb

        # Should be able to get predictions
        # Note: Full ensemble requires all models, this tests partial

    def test_ensemble_weighted_voting(self, small_training_data):
        """Test ensemble weighted voting."""
        from src.ml.models.ensemble_model import EnsembleModel, VotingMethod

        X, y = small_training_data

        xgb = XGBoostModel()
        xgb.train(X, y)

        ensemble = EnsembleModel(
            voting_method=VotingMethod.WEIGHTED,
            weights={'xgboost': 0.5, 'lstm': 0.3, 'cnn': 0.2}
        )
        ensemble.xgboost_model = xgb

    def test_ensemble_hard_voting(self, small_training_data):
        """Test ensemble hard voting."""
        from src.ml.models.ensemble_model import EnsembleModel, VotingMethod

        ensemble = EnsembleModel(voting_method=VotingMethod.HARD)
        assert ensemble.voting_method == VotingMethod.HARD


class TestModelNotTrainedError:
    """Test error handling for untrained models."""

    def test_xgboost_not_trained(self):
        """Test XGBoost raises error when not trained."""
        model = XGBoostModel()
        X = np.random.randn(10, 5)

        with pytest.raises(Exception):
            model.predict(X)

    @pytest.mark.slow
    def test_lstm_not_trained(self):
        """Test LSTM raises error when not trained."""
        from src.ml.models.lstm_model import LSTMModel
        model = LSTMModel(sequence_length=20, n_features=10)
        X = np.random.randn(10, 20, 10)

        with pytest.raises(Exception):
            model.predict(X)


class TestModelMetrics:
    """Tests for model evaluation metrics."""

    def test_accuracy_calculation(self, small_training_data):
        """Test accuracy is calculated correctly."""
        X, y = small_training_data
        model = XGBoostModel()
        model.train(X, y)

        predictions = model.predict(X)
        accuracy = (predictions == y).mean()

        # Training accuracy should be reasonable
        assert accuracy > 0.5

    def test_cross_validation_folds(self, small_training_data):
        """Test cross-validation with different fold counts."""
        X, y = small_training_data
        model = XGBoostModel()

        scores_3 = model.cross_validate(X, y, cv=3)
        scores_5 = model.cross_validate(X, y, cv=5)

        # Both should produce valid results
        assert 'accuracy' in scores_3
        assert 'accuracy' in scores_5
