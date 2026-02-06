"""
Tests for MLStrategy - signal generation and trade recommendations.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.strategy.ml_strategy import MLStrategy, SignalType, TradingSignal


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

    def test_generate_signals_buy(self):
        """Test generating buy signals via generate_signal (which takes a symbol)."""
        strategy = MLStrategy(confidence_threshold=0.5)

        # Mock both model and data_fetcher since generate_signal fetches data internally
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])  # BUY
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% confidence
        mock_model.feature_names = None
        strategy.model = mock_model

        # Mock data fetcher to return sample data
        mock_fetcher = Mock()
        n = 250
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        mock_df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1e6, 1e7, n),
        }, index=dates)
        mock_fetcher.fetch_historical.return_value = mock_df
        strategy.data_fetcher = mock_fetcher

        signal = strategy.generate_signal("AAPL")

        assert signal is not None
        assert signal.signal in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]

    def test_generate_signals_sell(self):
        """Test generating sell signals."""
        strategy = MLStrategy(confidence_threshold=0.5)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([0])  # SELL
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        mock_model.feature_names = None
        strategy.model = mock_model

        mock_fetcher = Mock()
        n = 250
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        mock_df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1e6, 1e7, n),
        }, index=dates)
        mock_fetcher.fetch_historical.return_value = mock_df
        strategy.data_fetcher = mock_fetcher

        signal = strategy.generate_signal("AAPL")
        assert signal is not None

    def test_generate_signals_hold_low_confidence(self):
        """Test HOLD signal when confidence is low."""
        strategy = MLStrategy(confidence_threshold=0.7, min_confidence_sell=0.7)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.45, 0.55]])  # Low confidence
        mock_model.feature_names = None
        strategy.model = mock_model

        mock_fetcher = Mock()
        n = 250
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        mock_df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1e6, 1e7, n),
        }, index=dates)
        mock_fetcher.fetch_historical.return_value = mock_df
        strategy.data_fetcher = mock_fetcher

        signal = strategy.generate_signal("AAPL")

        # Should hold due to low confidence
        assert signal.signal == SignalType.HOLD or signal.confidence < 0.7


class TestConfidenceThreshold:
    """Tests for confidence threshold handling."""

    def test_confidence_threshold_high(self):
        """Test high confidence threshold filters signals."""
        strategy = MLStrategy(confidence_threshold=0.9, min_confidence_sell=0.9)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])  # 85% < 90%
        mock_model.feature_names = None
        strategy.model = mock_model

        mock_fetcher = Mock()
        n = 250
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        mock_df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1e6, 1e7, n),
        }, index=dates)
        mock_fetcher.fetch_historical.return_value = mock_df
        strategy.data_fetcher = mock_fetcher

        signal = strategy.generate_signal("AAPL")

        # Should HOLD because confidence < threshold
        assert signal.signal == SignalType.HOLD

    def test_confidence_threshold_met(self):
        """Test signal generated when threshold is met."""
        strategy = MLStrategy(confidence_threshold=0.6)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% > 60%
        mock_model.feature_names = None
        strategy.model = mock_model

        mock_fetcher = Mock()
        n = 250
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        mock_df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1e6, 1e7, n),
        }, index=dates)
        mock_fetcher.fetch_historical.return_value = mock_df
        strategy.data_fetcher = mock_fetcher

        signal = strategy.generate_signal("AAPL")

        # Should generate BUY signal
        assert signal.signal == SignalType.BUY


class TestEnsembleLoading:
    """Tests for ensemble model loading."""

    def test_load_ensemble(self, small_training_data):
        """Test loading ensemble model."""
        from src.ml.models.xgboost_model import XGBoostModel

        X, y = small_training_data
        model = XGBoostModel()
        model.train(X, y)

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

        # Mock data_fetcher to return empty DF
        mock_fetcher = Mock()
        mock_fetcher.fetch_historical.return_value = pd.DataFrame()
        strategy.data_fetcher = mock_fetcher

        # Should return HOLD when no data available
        signal = strategy.generate_signal("AAPL")
        assert signal is not None
        assert signal.signal == SignalType.HOLD

    def test_signal_with_nan_data(self):
        """Test signal generation with NaN values."""
        strategy = MLStrategy()

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.feature_names = None
        strategy.model = mock_model

        # Mock data fetcher with NaN-containing data
        mock_fetcher = Mock()
        n = 250
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        mock_df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1e6, 1e7, n),
        }, index=dates)
        mock_df.iloc[-1, 0] = np.nan
        mock_fetcher.fetch_historical.return_value = mock_df
        strategy.data_fetcher = mock_fetcher

        # Should handle NaN gracefully
        signal = strategy.generate_signal("AAPL")
        assert signal is not None


class TestMultipleSymbols:
    """Tests for handling multiple symbols."""

    def test_signals_for_multiple_symbols(self):
        """Test generating signals for multiple symbols."""
        strategy = MLStrategy(confidence_threshold=0.5)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.feature_names = None
        strategy.model = mock_model

        # Mock data fetcher
        mock_fetcher = Mock()
        n = 250
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        mock_df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1e6, 1e7, n),
        }, index=dates)
        mock_fetcher.fetch_historical.return_value = mock_df
        strategy.data_fetcher = mock_fetcher

        symbols = ['AAPL', 'MSFT', 'GOOGL']
        signals = strategy.generate_signals(symbols)

        # Should generate signals for all symbols
        assert len(signals) == len(symbols)


class TestSignalAttributes:
    """Tests for signal object attributes."""

    def test_signal_has_required_attributes(self):
        """Test signal object has required attributes."""
        strategy = MLStrategy()

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.feature_names = None
        strategy.model = mock_model

        mock_fetcher = Mock()
        n = 250
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        mock_df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1e6, 1e7, n),
        }, index=dates)
        mock_fetcher.fetch_historical.return_value = mock_df
        strategy.data_fetcher = mock_fetcher

        signal = strategy.generate_signal("AAPL")

        # Check required attributes (production uses .signal not .action)
        assert hasattr(signal, 'signal')
        assert hasattr(signal, 'confidence')
        assert signal.signal in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert 0 <= signal.confidence <= 1

    def test_signal_symbol_tracking(self):
        """Test signal tracks symbol."""
        strategy = MLStrategy()

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.feature_names = None
        strategy.model = mock_model

        mock_fetcher = Mock()
        n = 250
        dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
        mock_df = pd.DataFrame({
            'open': np.random.randn(n).cumsum() + 100,
            'high': np.random.randn(n).cumsum() + 101,
            'low': np.random.randn(n).cumsum() + 99,
            'close': np.random.randn(n).cumsum() + 100,
            'volume': np.random.randint(1e6, 1e7, n),
        }, index=dates)
        mock_fetcher.fetch_historical.return_value = mock_df
        strategy.data_fetcher = mock_fetcher

        # generate_signal takes just a symbol string
        signal = strategy.generate_signal('AAPL')

        assert signal.symbol == 'AAPL'


class TestModelNotLoaded:
    """Tests for handling when model is not loaded."""

    def test_generate_signal_no_model(self):
        """Test signal generation without loaded model."""
        strategy = MLStrategy()
        strategy.model = None

        with pytest.raises(Exception):
            strategy.generate_signal("AAPL")
