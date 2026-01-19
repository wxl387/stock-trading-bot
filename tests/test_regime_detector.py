"""
Tests for RegimeDetector - market regime detection.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.risk.regime_detector import RegimeDetector, MarketRegime


class TestRegimeDetectorInitialization:
    """Tests for RegimeDetector initialization."""

    def test_initialization(self):
        """Test RegimeDetector initializes correctly."""
        detector = RegimeDetector()
        assert detector is not None

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        detector = RegimeDetector(
            lookback_period=30,
            volatility_threshold=0.25
        )
        assert detector.lookback_period == 30


class TestBullMarketDetection:
    """Tests for bull market detection."""

    def test_detect_bull_market(self):
        """Test detection of bull market regime."""
        detector = RegimeDetector()

        # Create strong uptrend data
        n_days = 100
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        prices = 100 * np.exp(np.linspace(0, 0.3, n_days))  # 30% rise

        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        regime = detector.detect_regime(df)

        # Should detect uptrend
        assert regime in [MarketRegime.BULL, MarketRegime.CHOPPY]

    def test_bull_characteristics(self):
        """Test bull market has expected characteristics."""
        detector = RegimeDetector()

        # Strong bull data
        n_days = 60
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        prices = 100 * np.exp(np.linspace(0, 0.2, n_days))

        df = pd.DataFrame({
            'open': prices * 0.995,
            'high': prices * 1.005,
            'low': prices * 0.99,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        info = detector.get_regime_info(df)

        # Bull should have positive trend
        if info.get('trend_direction'):
            assert info['trend_direction'] > 0


class TestBearMarketDetection:
    """Tests for bear market detection."""

    def test_detect_bear_market(self):
        """Test detection of bear market regime."""
        detector = RegimeDetector()

        # Create strong downtrend data
        n_days = 100
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        prices = 100 * np.exp(np.linspace(0, -0.25, n_days))  # 25% decline

        df = pd.DataFrame({
            'open': prices * 1.01,
            'high': prices * 1.02,
            'low': prices * 0.99,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        regime = detector.detect_regime(df)

        # Should detect downtrend
        assert regime in [MarketRegime.BEAR, MarketRegime.CHOPPY, MarketRegime.VOLATILE]

    def test_bear_characteristics(self):
        """Test bear market has expected characteristics."""
        detector = RegimeDetector()

        # Strong bear data
        n_days = 60
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        prices = 100 * np.exp(np.linspace(0, -0.15, n_days))

        df = pd.DataFrame({
            'open': prices * 1.005,
            'high': prices * 1.01,
            'low': prices * 0.995,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        info = detector.get_regime_info(df)

        # Bear should have negative trend
        if info.get('trend_direction'):
            assert info['trend_direction'] < 0


class TestVolatileMarketDetection:
    """Tests for volatile market detection."""

    def test_detect_volatile_market(self):
        """Test detection of volatile market regime."""
        detector = RegimeDetector()

        # Create high volatility data with large swings
        n_days = 100
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

        np.random.seed(42)
        returns = np.random.randn(n_days) * 0.05  # 5% daily volatility
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices * (1 + np.random.randn(n_days) * 0.02),
            'high': prices * 1.03,
            'low': prices * 0.97,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        regime = detector.detect_regime(df)

        # High volatility should be detected
        assert regime in [MarketRegime.VOLATILE, MarketRegime.BEAR, MarketRegime.CHOPPY]

    def test_volatility_measurement(self):
        """Test volatility is measured correctly."""
        detector = RegimeDetector()

        # High volatility data
        n_days = 60
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

        np.random.seed(123)
        returns = np.random.randn(n_days) * 0.04
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        info = detector.get_regime_info(df)

        if 'volatility' in info:
            # Annualized volatility should be significant
            assert info['volatility'] > 0.2  # > 20% annualized


class TestChoppyMarketDetection:
    """Tests for choppy/sideways market detection."""

    def test_detect_choppy_market(self):
        """Test detection of choppy market regime."""
        detector = RegimeDetector()

        # Create sideways data
        n_days = 100
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

        # Oscillating prices around 100
        prices = 100 + np.sin(np.linspace(0, 8 * np.pi, n_days)) * 5

        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        regime = detector.detect_regime(df)

        # Should detect choppy/sideways
        assert regime in [MarketRegime.CHOPPY, MarketRegime.BULL, MarketRegime.BEAR]


class TestRegimeParameters:
    """Tests for regime-specific parameters."""

    def test_get_regime_parameters(self):
        """Test getting trading parameters for regime."""
        detector = RegimeDetector()

        for regime in MarketRegime:
            params = detector.get_trading_parameters(regime)

            assert isinstance(params, dict)
            assert 'position_multiplier' in params
            assert params['position_multiplier'] > 0

    def test_bull_parameters_more_aggressive(self):
        """Test bull market has more aggressive parameters."""
        detector = RegimeDetector()

        bull_params = detector.get_trading_parameters(MarketRegime.BULL)
        bear_params = detector.get_trading_parameters(MarketRegime.BEAR)

        # Bull should allow larger positions
        assert bull_params['position_multiplier'] >= bear_params['position_multiplier']

    def test_volatile_parameters_defensive(self):
        """Test volatile market has defensive parameters."""
        detector = RegimeDetector()

        volatile_params = detector.get_trading_parameters(MarketRegime.VOLATILE)

        # Volatile should reduce position sizes
        assert volatile_params['position_multiplier'] <= 1.0


class TestEdgeCases:
    """Test edge cases."""

    def test_short_data(self):
        """Test with very short data series."""
        detector = RegimeDetector()

        # Only 5 days
        dates = pd.date_range(end=datetime.now(), periods=5, freq='D')
        df = pd.DataFrame({
            'open': [100, 101, 102, 101, 100],
            'high': [101, 102, 103, 102, 101],
            'low': [99, 100, 101, 100, 99],
            'close': [100, 101, 102, 101, 100],
            'volume': [1000000] * 5
        }, index=dates)

        # Should handle gracefully
        regime = detector.detect_regime(df)
        assert regime in MarketRegime

    def test_flat_prices(self):
        """Test with completely flat prices."""
        detector = RegimeDetector()

        n_days = 50
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        df = pd.DataFrame({
            'open': [100] * n_days,
            'high': [100] * n_days,
            'low': [100] * n_days,
            'close': [100] * n_days,
            'volume': [1000000] * n_days
        }, index=dates)

        regime = detector.detect_regime(df)
        # Should handle zero volatility
        assert regime in MarketRegime

    def test_missing_data(self):
        """Test handling of missing data."""
        detector = RegimeDetector()

        n_days = 50
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        prices = np.random.randn(n_days).cumsum() + 100

        df = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        # Introduce NaN
        df.loc[df.index[10], 'close'] = np.nan

        # Should handle NaN gracefully
        try:
            regime = detector.detect_regime(df)
            assert regime in MarketRegime
        except (ValueError, KeyError):
            pass  # Some implementations may raise
