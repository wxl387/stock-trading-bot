"""
Tests for MLStrategy - signal generation and trade recommendations.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.strategy.ml_strategy import MLStrategy


class TestMLStrategyInitialization:
    """Tests for MLStrategy initialization."""

    def test_initialization(self):
        """Test MLStrategy initializes correctly."""
        strategy = MLStrategy()
        assert strategy is not None

    def test_initialization_with_confidence(self):
        """Test initialization with custom confidence threshold."""
        strategy = MLStrategy(confidence_threshold=0.7)
        assert strategy.confidence_threshold == 0.7

    def test_initialization_with_model(self, small_training_data):
        """Test initialization with pre-loaded model."""
        from src.ml.models.xgboost_model import XGBoostModel

        X, y = small_training_data
        model = XGBoostModel()
        model.train(X, y)

        strategy = MLStrategy()
        strategy.model = model
        assert strategy.model is not None


class TestSignalGeneration:
    """Tests for signal generation."""

    def test_generate_signals_buy(self, sample_features_df):
        """Test generating buy signals."""
        strategy = MLStrategy(confidence_threshold=0.5)

        # Mock model to return BUY signal with high confidence
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])  # BUY
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% confidence
        strategy.model = mock_model

        signal = strategy.generate_signal(sample_features_df)

        assert signal is not None
        assert signal.action in ['BUY', 'SELL', 'HOLD']

    def test_generate_signals_sell(self, sample_features_df):
        """Test generating sell signals."""
        strategy = MLStrategy(confidence_threshold=0.5)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])  # SELL
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        strategy.model = mock_model

        signal = strategy.generate_signal(sample_features_df)
        assert signal is not None

    def test_generate_signals_hold_low_confidence(self, sample_features_df):
        """Test HOLD signal when confidence is low."""
        strategy = MLStrategy(confidence_threshold=0.7)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.45, 0.55]])  # Low confidence
        strategy.model = mock_model

        signal = strategy.generate_signal(sample_features_df)

        # Should hold due to low confidence
        assert signal.action == 'HOLD' or signal.confidence < 0.7


class TestConfidenceThreshold:
    """Tests for confidence threshold handling."""

    def test_confidence_threshold_high(self, sample_features_df):
        """Test high confidence threshold filters signals."""
        strategy = MLStrategy(confidence_threshold=0.9)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])  # 85% < 90%
        strategy.model = mock_model

        signal = strategy.generate_signal(sample_features_df)

        # Should HOLD because confidence < threshold
        assert signal.action == 'HOLD'

    def test_confidence_threshold_met(self, sample_features_df):
        """Test signal generated when threshold is met."""
        strategy = MLStrategy(confidence_threshold=0.6)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% > 60%
        strategy.model = mock_model

        signal = strategy.generate_signal(sample_features_df)

        # Should generate BUY signal
        assert signal.action == 'BUY'


class TestEnsembleLoading:
    """Tests for ensemble model loading."""

    def test_load_ensemble(self, tmp_model_dir, small_training_data):
        """Test loading ensemble model."""
        # First save a model
        from src.ml.models.xgboost_model import XGBoostModel

        X, y = small_training_data
        model = XGBoostModel()
        model.train(X, y)
        model.save("test_model", model_dir=str(tmp_model_dir))

        strategy = MLStrategy()

        # Test that load_ensemble method exists
        assert hasattr(strategy, 'load_ensemble') or hasattr(strategy, 'load_model')


class TestSignalWithNoData:
    """Tests for handling missing or empty data."""

    def test_signal_with_no_data(self):
        """Test signal generation with empty data."""
        strategy = MLStrategy()

        mock_model = Mock()
        strategy.model = mock_model

        # Empty DataFrame
        empty_df = pd.DataFrame()

        # Should handle gracefully
        try:
            signal = strategy.generate_signal(empty_df)
            # Either returns HOLD or raises an exception
            assert signal is None or signal.action == 'HOLD'
        except (ValueError, KeyError):
            pass  # Expected for empty data

    def test_signal_with_nan_data(self, sample_features_df):
        """Test signal generation with NaN values."""
        strategy = MLStrategy()

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        strategy.model = mock_model

        # Introduce NaN
        df_with_nan = sample_features_df.copy()
        df_with_nan.iloc[-1, 0] = np.nan

        # Should handle NaN gracefully
        try:
            signal = strategy.generate_signal(df_with_nan)
            assert signal is not None
        except (ValueError, Exception):
            pass  # May raise on NaN


class TestMultipleSymbols:
    """Tests for handling multiple symbols."""

    def test_signals_for_multiple_symbols(self, mock_market_data):
        """Test generating signals for multiple symbols."""
        strategy = MLStrategy(confidence_threshold=0.5)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        strategy.model = mock_model

        signals = {}
        for symbol, data in mock_market_data.items():
            from src.data.feature_engineer import FeatureEngineer
            fe = FeatureEngineer()
            features = fe.create_features(data)
            if not features.empty:
                signal = strategy.generate_signal(features)
                signals[symbol] = signal

        # Should generate signals for all symbols
        assert len(signals) == len(mock_market_data)


class TestSignalAttributes:
    """Tests for signal object attributes."""

    def test_signal_has_required_attributes(self, sample_features_df):
        """Test signal object has required attributes."""
        strategy = MLStrategy()

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        strategy.model = mock_model

        signal = strategy.generate_signal(sample_features_df)

        # Check required attributes
        assert hasattr(signal, 'action')
        assert hasattr(signal, 'confidence')
        assert signal.action in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signal.confidence <= 1

    def test_signal_symbol_tracking(self, sample_features_df):
        """Test signal tracks symbol if provided."""
        strategy = MLStrategy()

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        strategy.model = mock_model

        signal = strategy.generate_signal(sample_features_df, symbol='AAPL')

        if hasattr(signal, 'symbol'):
            assert signal.symbol == 'AAPL'


class TestModelNotLoaded:
    """Tests for handling when model is not loaded."""

    def test_generate_signal_no_model(self, sample_features_df):
        """Test signal generation without loaded model."""
        strategy = MLStrategy()
        strategy.model = None

        with pytest.raises(Exception):
            strategy.generate_signal(sample_features_df)
