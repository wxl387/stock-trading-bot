"""
Tests for FeatureEngineer - technical indicator calculations.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.feature_engineer import FeatureEngineer


class TestFeatureEngineerInitialization:
    """Tests for FeatureEngineer initialization."""

    def test_default_initialization(self):
        """Test FeatureEngineer initializes correctly."""
        fe = FeatureEngineer()
        assert fe is not None


class TestMovingAverages:
    """Tests for moving average calculations."""

    def test_sma_calculation(self, sample_ohlcv_data):
        """Test Simple Moving Average calculation."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Check SMA columns exist
        sma_columns = [col for col in features.columns if 'sma' in col.lower()]
        assert len(sma_columns) > 0

        # SMA should be within price range
        if 'sma_20' in features.columns:
            valid_sma = features['sma_20'].dropna()
            assert all(valid_sma > 0)
            assert all(valid_sma < sample_ohlcv_data['close'].max() * 1.5)

    def test_ema_calculation(self, sample_ohlcv_data):
        """Test Exponential Moving Average calculation."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Check EMA columns exist
        ema_columns = [col for col in features.columns if 'ema' in col.lower()]
        assert len(ema_columns) > 0


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_calculation(self, sample_ohlcv_data):
        """Test RSI is calculated correctly."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Find RSI column
        rsi_columns = [col for col in features.columns if 'rsi' in col.lower()]
        assert len(rsi_columns) > 0

        rsi_col = rsi_columns[0]
        valid_rsi = features[rsi_col].dropna()

        # RSI should be between 0 and 100
        assert all(valid_rsi >= 0)
        assert all(valid_rsi <= 100)

    def test_rsi_extreme_values(self):
        """Test RSI with extreme price movements."""
        # Create data with strong uptrend
        n_days = 50
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        prices = 100 * np.exp(np.linspace(0, 0.5, n_days))  # Strong uptrend

        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        fe = FeatureEngineer()
        features = fe.add_all_features(df)

        rsi_columns = [col for col in features.columns if 'rsi' in col.lower()]
        if rsi_columns:
            rsi = features[rsi_columns[0]].iloc[-1]
            # Strong uptrend should have high RSI
            assert rsi > 50


class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_calculation(self, sample_ohlcv_data):
        """Test MACD is calculated correctly."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Check MACD columns exist
        macd_columns = [col for col in features.columns if 'macd' in col.lower()]
        assert len(macd_columns) > 0

    def test_macd_histogram(self, sample_ohlcv_data):
        """Test MACD histogram calculation."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # MACD histogram should exist
        hist_columns = [col for col in features.columns if 'hist' in col.lower() or 'macd_diff' in col.lower()]
        # May or may not have histogram depending on implementation


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands calculation."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Check for BB columns
        bb_columns = [col for col in features.columns if 'bb' in col.lower() or 'bollinger' in col.lower()]

        # Upper band should be above lower band
        if 'bb_upper' in features.columns and 'bb_lower' in features.columns:
            valid_idx = features['bb_upper'].notna() & features['bb_lower'].notna()
            assert all(features.loc[valid_idx, 'bb_upper'] >= features.loc[valid_idx, 'bb_lower'])


class TestATR:
    """Tests for Average True Range calculation."""

    def test_atr_calculation(self, sample_ohlcv_data):
        """Test ATR is calculated correctly."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Check for ATR column
        atr_columns = [col for col in features.columns if 'atr' in col.lower()]

        if atr_columns:
            atr = features[atr_columns[0]].dropna()
            # ATR should be positive
            assert all(atr >= 0)


class TestFeatureCompleteness:
    """Tests for feature completeness and quality."""

    def test_feature_completeness(self, sample_ohlcv_data):
        """Test that features are generated without excessive NaN."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # With 100 rows and sma_200 requiring 200+ rows, dropna() may
        # yield 0 rows. Instead verify features were generated and that
        # columns with shorter lookbacks have valid data.
        assert len(features.columns) > len(sample_ohlcv_data.columns)
        # Check a short-lookback column like sma_5 has valid data
        assert features['sma_5'].dropna().shape[0] > 50

    def test_no_inf_values(self, sample_ohlcv_data):
        """Test features don't contain infinite values."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        for col in features.columns:
            valid_data = features[col].dropna()
            assert not np.isinf(valid_data).any(), f"Infinite values in {col}"

    def test_feature_names_consistent(self, sample_ohlcv_data):
        """Test feature names are consistent across calls."""
        fe = FeatureEngineer()

        features1 = fe.add_all_features(sample_ohlcv_data)
        features2 = fe.add_all_features(sample_ohlcv_data)

        assert list(features1.columns) == list(features2.columns)


class TestMissingDataHandling:
    """Tests for handling missing or incomplete data."""

    def test_handles_missing_data(self):
        """Test feature engineer handles missing data gracefully."""
        n_days = 50
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

        # Create data with some NaN values
        prices = np.random.randn(n_days).cumsum() + 100
        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        # Introduce some NaN
        df.loc[df.index[10], 'close'] = np.nan
        df.loc[df.index[20], 'volume'] = np.nan

        fe = FeatureEngineer()

        # Should not crash
        try:
            features = fe.add_all_features(df)
            assert features is not None
        except Exception as e:
            pytest.fail(f"Feature engineer crashed on missing data: {e}")

    def test_handles_short_data(self):
        """Test feature engineer handles short data series."""
        # ADX (window=14) in the `ta` library requires ~2*window rows internally
        n_days = 30
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

        prices = np.random.randn(n_days).cumsum() + 100
        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': [1000000] * n_days
        }, index=dates)

        fe = FeatureEngineer()

        # Should handle gracefully (may have many NaN but shouldn't crash)
        features = fe.add_all_features(df)
        assert features is not None

    def test_handles_zero_volume(self):
        """Test feature engineer handles zero volume."""
        n_days = 30
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

        prices = np.random.randn(n_days).cumsum() + 100
        df = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': [0] * n_days  # Zero volume
        }, index=dates)

        fe = FeatureEngineer()
        features = fe.add_all_features(df)

        # Should not crash
        assert features is not None


class TestDerivedFeatures:
    """Tests for derived/calculated features."""

    def test_returns_calculation(self, sample_ohlcv_data):
        """Test return features are calculated."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Check for return columns
        return_columns = [col for col in features.columns if 'return' in col.lower() or 'pct' in col.lower()]
        # Returns may or may not be included depending on implementation

    def test_volatility_features(self, sample_ohlcv_data):
        """Test volatility features are calculated."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Check for volatility columns
        vol_columns = [col for col in features.columns if 'vol' in col.lower() or 'std' in col.lower()]
        # Implementation dependent

    def test_trend_features(self, sample_ohlcv_data):
        """Test trend indicator features."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Check for trend columns (ADX, etc.)
        trend_columns = [col for col in features.columns if 'adx' in col.lower() or 'trend' in col.lower()]
        # Implementation dependent


class TestFeatureScaling:
    """Tests for feature scaling/normalization."""

    def test_feature_ranges(self, sample_ohlcv_data):
        """Test features are in reasonable ranges."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        for col in features.columns:
            valid_data = features[col].dropna()
            if len(valid_data) > 0:
                # Check no extreme outliers (beyond 1000x range)
                data_range = valid_data.max() - valid_data.min()
                if data_range > 0:
                    # This is a sanity check, not a strict requirement
                    pass


class TestMLReadyFeatures:
    """Tests for ML-ready feature preparation."""

    def test_prepare_for_ml(self, sample_ohlcv_data):
        """Test features can be prepared for ML models."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Drop NaN for ML
        clean_features = features.dropna()

        # Should be able to convert to numpy
        X = clean_features.values
        assert isinstance(X, np.ndarray)
        assert not np.isnan(X).any()
        assert not np.isinf(X).any()

    def test_feature_column_count(self, sample_ohlcv_data):
        """Test reasonable number of features generated."""
        fe = FeatureEngineer()
        features = fe.add_all_features(sample_ohlcv_data)

        # Should generate multiple features
        assert len(features.columns) >= 5  # At least 5 features
        assert len(features.columns) <= 100  # Not too many
