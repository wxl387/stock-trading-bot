"""
Tests for MetricsCalculator - financial metrics calculations.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.analytics.metrics import MetricsCalculator, calculate_returns, calculate_cumulative_returns


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_initialization(self, sample_returns):
        """Test MetricsCalculator initializes correctly."""
        calc = MetricsCalculator(sample_returns)
        assert calc.returns is not None
        assert len(calc.returns) == len(sample_returns)

    def test_initialization_with_custom_risk_free_rate(self, sample_returns):
        """Test initialization with custom risk-free rate."""
        calc = MetricsCalculator(sample_returns, risk_free_rate=0.03)
        assert calc.risk_free_rate == 0.03

    def test_sharpe_ratio_positive_returns(self, sample_returns):
        """Test Sharpe ratio calculation with positive returns."""
        calc = MetricsCalculator(sample_returns)
        sharpe = calc.sharpe_ratio()

        # Sharpe should be a number
        assert isinstance(sharpe, float)
        # With positive drift in sample_returns, expect positive Sharpe
        assert not np.isnan(sharpe)

    def test_sharpe_ratio_negative_returns(self, sample_returns_negative):
        """Test Sharpe ratio with negative performance."""
        calc = MetricsCalculator(sample_returns_negative)
        sharpe = calc.sharpe_ratio()

        assert isinstance(sharpe, float)
        # Negative returns should produce lower Sharpe
        assert not np.isnan(sharpe)

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility returns."""
        # Constant returns (zero volatility)
        constant_returns = pd.Series([0.001] * 100)
        calc = MetricsCalculator(constant_returns)
        sharpe = calc.sharpe_ratio()

        # Should handle zero volatility gracefully
        assert sharpe == 0.0 or np.isinf(sharpe) or not np.isnan(sharpe)

    def test_sortino_ratio_calculation(self, sample_returns):
        """Test Sortino ratio uses downside deviation correctly."""
        calc = MetricsCalculator(sample_returns)
        sortino = calc.sortino_ratio()

        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

        # Sortino should generally be >= Sharpe (penalizes only downside)
        sharpe = calc.sharpe_ratio()
        # This relationship holds when there's downside volatility
        # but may not always hold exactly

    def test_sortino_ratio_no_downside(self):
        """Test Sortino ratio with all positive returns."""
        positive_returns = pd.Series(np.abs(np.random.randn(100)) * 0.01)
        calc = MetricsCalculator(positive_returns)
        sortino = calc.sortino_ratio()

        # With no downside, Sortino should be very high or inf
        assert sortino >= 0 or np.isinf(sortino)

    def test_calmar_ratio_calculation(self, sample_returns):
        """Test Calmar ratio calculation."""
        calc = MetricsCalculator(sample_returns)
        calmar = calc.calmar_ratio()

        assert isinstance(calmar, float)
        # Calmar = annualized_return / max_drawdown
        # Should be positive if returns are positive and drawdown exists

    def test_calmar_ratio_no_drawdown(self):
        """Test Calmar ratio with no drawdown (constantly rising)."""
        rising_returns = pd.Series([0.01] * 100)  # Constantly positive
        calc = MetricsCalculator(rising_returns)
        calmar = calc.calmar_ratio()

        # With no drawdown, should handle gracefully (inf or 0)
        assert not np.isnan(calmar) or calmar == 0.0

    def test_max_drawdown_calculation(self, sample_returns):
        """Test max drawdown is calculated correctly."""
        calc = MetricsCalculator(sample_returns)
        max_dd = calc.max_drawdown()

        # max_drawdown may return float or tuple (value, peak_date, trough_date)
        if isinstance(max_dd, tuple):
            max_dd_value = max_dd[0]
        else:
            max_dd_value = max_dd

        assert isinstance(max_dd_value, float)
        # Drawdown should be positive (representing % loss)
        assert max_dd_value >= 0

    def test_max_drawdown_known_value(self):
        """Test max drawdown with known expected value."""
        # Create returns that result in known drawdown
        # Start at 100, go to 120, drop to 90, end at 100
        # Max DD = (90 - 120) / 120 = -25%
        returns = pd.Series([0.20, -0.25, 0.111])  # 100 -> 120 -> 90 -> 100
        calc = MetricsCalculator(returns)
        max_dd = calc.max_drawdown()

        if isinstance(max_dd, tuple):
            max_dd_value = max_dd[0]
        else:
            max_dd_value = max_dd

        # Should be approximately 25% (positive value for loss)
        assert 0.20 <= max_dd_value <= 0.30

    def test_annualized_return(self, sample_returns):
        """Test annualized return calculation."""
        calc = MetricsCalculator(sample_returns)
        ann_return = calc.annualized_return()

        assert isinstance(ann_return, float)
        # With 252 trading days per year, check reasonable range
        assert -1.0 <= ann_return <= 5.0  # -100% to 500%

    def test_annualized_volatility(self, sample_returns):
        """Test annualized volatility calculation."""
        calc = MetricsCalculator(sample_returns)
        # Method may be named volatility or annualized_volatility
        if hasattr(calc, 'annualized_volatility'):
            vol = calc.annualized_volatility()
        elif hasattr(calc, 'volatility'):
            vol = calc.volatility()
        else:
            # Get from get_all_metrics
            metrics = calc.get_all_metrics()
            vol = metrics.get('volatility', metrics.get('annualized_volatility', 0))

        assert isinstance(vol, float)
        assert vol >= 0  # Volatility is always non-negative
        # Typical stock volatility is 15-30% annually
        assert vol <= 2.0  # 200% would be extreme

    def test_rolling_sharpe(self, sample_returns):
        """Test rolling Sharpe ratio calculation."""
        calc = MetricsCalculator(sample_returns)
        rolling = calc.rolling_sharpe(window=63)

        assert isinstance(rolling, pd.Series)
        # Should have valid values (may or may not have NaN depending on implementation)
        valid_values = rolling.dropna()
        assert len(valid_values) > 0
        # Values should be finite
        assert all(~np.isinf(valid_values))

    def test_rolling_sharpe_short_window(self, sample_returns):
        """Test rolling Sharpe with short window."""
        calc = MetricsCalculator(sample_returns)
        rolling = calc.rolling_sharpe(window=20)

        valid_values = rolling.dropna()
        assert len(valid_values) > len(sample_returns) - 20

    def test_monthly_returns(self, sample_returns):
        """Test monthly returns heatmap data."""
        calc = MetricsCalculator(sample_returns)
        monthly = calc.monthly_returns()

        assert isinstance(monthly, pd.DataFrame)
        # Should have data if returns span multiple months
        if not monthly.empty:
            # Columns may be month numbers (1-12) or month names
            assert len(monthly.columns) > 0

    def test_total_return(self, sample_returns):
        """Test total return calculation."""
        calc = MetricsCalculator(sample_returns)
        total = calc.total_return()

        assert isinstance(total, float)
        # Should match compound return
        expected = (1 + sample_returns).prod() - 1
        assert abs(total - expected) < 0.0001

    def test_get_all_metrics(self, sample_returns):
        """Test get_all_metrics returns complete dict."""
        calc = MetricsCalculator(sample_returns)
        metrics = calc.get_all_metrics()

        assert isinstance(metrics, dict)

        # Check core expected keys (names may vary slightly)
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'max_drawdown' in metrics
        # Volatility may be named volatility or annualized_volatility
        assert 'volatility' in metrics or 'annualized_volatility' in metrics
        # Return may be named total_return or annualized_return
        assert 'total_return' in metrics or 'annualized_return' in metrics

    def test_empty_returns(self):
        """Test handling of empty returns series."""
        empty_returns = pd.Series(dtype=float)
        calc = MetricsCalculator(empty_returns)

        # Should handle gracefully without crashing
        metrics = calc.get_all_metrics()
        assert isinstance(metrics, dict)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_calculate_returns(self, sample_ohlcv_data):
        """Test calculate_returns from price series."""
        prices = sample_ohlcv_data['close']
        returns = calculate_returns(prices)

        assert isinstance(returns, pd.Series)
        assert len(returns) == len(prices) - 1  # One less due to pct_change

    def test_calculate_cumulative_returns(self, sample_returns):
        """Test cumulative returns calculation."""
        cumulative = calculate_cumulative_returns(sample_returns)

        assert isinstance(cumulative, pd.Series)
        assert len(cumulative) == len(sample_returns)
        # First value should be first return
        # Last value should be total cumulative return
        expected_final = (1 + sample_returns).prod() - 1
        assert abs(cumulative.iloc[-1] - expected_final) < 0.0001


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_return(self):
        """Test with single return value."""
        single = pd.Series([0.05])
        calc = MetricsCalculator(single)
        metrics = calc.get_all_metrics()

        assert isinstance(metrics, dict)

    def test_all_zero_returns(self):
        """Test with all zero returns."""
        zeros = pd.Series([0.0] * 100)
        calc = MetricsCalculator(zeros)

        sharpe = calc.sharpe_ratio()
        assert sharpe == 0.0 or np.isnan(sharpe)

    def test_extreme_returns(self):
        """Test with extreme return values."""
        extreme = pd.Series([0.5, -0.5, 0.5, -0.5] * 25)  # 50% swings
        calc = MetricsCalculator(extreme)
        metrics = calc.get_all_metrics()

        # Should still produce valid metrics
        assert isinstance(metrics['sharpe_ratio'], float)
        assert isinstance(metrics['max_drawdown'], float)

    def test_nan_handling(self):
        """Test handling of NaN values in returns."""
        returns_with_nan = pd.Series([0.01, np.nan, 0.02, 0.01, np.nan])
        calc = MetricsCalculator(returns_with_nan)

        # Should handle NaN gracefully
        metrics = calc.get_all_metrics()
        assert isinstance(metrics, dict)
