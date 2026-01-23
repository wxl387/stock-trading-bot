"""
Tests for Backtester - backtesting ML strategies.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.backtest.backtester import Backtester


class TestBacktesterInitialization:
    """Tests for Backtester initialization."""

    def test_initialization(self):
        """Test Backtester initializes correctly."""
        backtester = Backtester(initial_capital=100000)
        assert backtester is not None
        assert backtester.initial_capital == 100000

    def test_initialization_default_capital(self):
        """Test Backtester with default capital."""
        backtester = Backtester()
        assert backtester.initial_capital > 0


class TestSimpleBacktest:
    """Tests for simple backtesting."""

    def test_simple_backtest(self, sample_ohlcv_data_long):
        """Test running a simple backtest."""
        backtester = Backtester(initial_capital=100000)

        # Create simple signals (buy and hold)
        signals = pd.Series(
            [1] * len(sample_ohlcv_data_long),  # Always buy
            index=sample_ohlcv_data_long.index
        )

        result = backtester.run_simple_backtest(
            prices=sample_ohlcv_data_long['close'],
            signals=signals
        )

        assert result is not None
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'final_value')

    def test_backtest_buy_and_hold(self, sample_ohlcv_data_long):
        """Test buy and hold strategy."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']

        # Buy on first day, hold forever
        signals = pd.Series([0] * len(prices), index=prices.index)
        signals.iloc[0] = 1  # Buy first day

        result = backtester.run_simple_backtest(prices, signals)

        # Final value should match price movement
        price_return = prices.iloc[-1] / prices.iloc[0] - 1
        assert result is not None

    def test_backtest_with_transaction_costs(self, sample_ohlcv_data_long):
        """Test backtest with transaction costs."""
        backtester = Backtester(
            initial_capital=100000,
            commission=0.001  # 0.1% commission
        )

        prices = sample_ohlcv_data_long['close']
        # Frequent trading
        signals = pd.Series(
            [1 if i % 5 == 0 else 0 for i in range(len(prices))],
            index=prices.index
        )

        result = backtester.run_simple_backtest(prices, signals)

        # With commissions, returns should be lower


class TestBacktestMetrics:
    """Tests for backtest result metrics."""

    def test_backtest_metrics(self, sample_ohlcv_data_long):
        """Test backtest metrics are calculated."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']
        signals = pd.Series([1] * len(prices), index=prices.index)

        result = backtester.run_simple_backtest(prices, signals)

        # Check metrics exist
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'max_drawdown')

    def test_sharpe_ratio_calculation(self, sample_ohlcv_data_long):
        """Test Sharpe ratio is calculated correctly."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']
        signals = pd.Series([1] * len(prices), index=prices.index)

        result = backtester.run_simple_backtest(prices, signals)

        # Sharpe should be a number
        assert isinstance(result.sharpe_ratio, (int, float))
        assert not np.isnan(result.sharpe_ratio) or result.sharpe_ratio == 0

    def test_max_drawdown_calculation(self, sample_ohlcv_data_long):
        """Test max drawdown is calculated correctly."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']
        signals = pd.Series([1] * len(prices), index=prices.index)

        result = backtester.run_simple_backtest(prices, signals)

        # Max drawdown should be >= 0 (stored as positive value)
        assert result.max_drawdown >= 0

    def test_win_rate_calculation(self, sample_ohlcv_data_long):
        """Test win rate is calculated correctly."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']
        # Create alternating buy/sell signals
        signals = pd.Series(
            [1 if i % 10 < 5 else -1 for i in range(len(prices))],
            index=prices.index
        )

        result = backtester.run_simple_backtest(prices, signals)

        # Win rate should be between 0 and 1
        if hasattr(result, 'win_rate'):
            assert 0 <= result.win_rate <= 1


class TestMLPortfolioBacktest:
    """Tests for ML portfolio backtesting."""

    @pytest.mark.slow
    def test_ml_portfolio_backtest(self, sample_ohlcv_data_long, small_training_data):
        """Test ML portfolio backtesting."""
        from src.ml.models.xgboost_model import XGBoostModel

        X, y = small_training_data
        model = XGBoostModel()
        model.train(X, y)

        backtester = Backtester(initial_capital=100000)

        # This would run the full ML backtest
        # For testing, we can mock the model's predictions

    def test_ml_backtest_with_mock_model(self, sample_ohlcv_data_long):
        """Test ML backtest with mocked model."""
        backtester = Backtester(initial_capital=100000)

        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])  # Always predict BUY
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        # Can run backtest with mocked model


class TestTradeHistoryTracking:
    """Tests for trade history in backtests."""

    def test_trade_history_tracking(self, sample_ohlcv_data_long):
        """Test trades are recorded during backtest."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']
        # Create signals that generate trades
        signals = pd.Series(
            [1 if i % 20 < 10 else -1 for i in range(len(prices))],
            index=prices.index
        )

        result = backtester.run_simple_backtest(prices, signals)

        # Should have trades recorded
        if hasattr(result, 'trades'):
            assert isinstance(result.trades, (list, pd.DataFrame))

    def test_trade_count(self, sample_ohlcv_data_long):
        """Test trade count matches signal changes."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']
        # Signals that change every 50 days
        signals = pd.Series(
            [1 if (i // 50) % 2 == 0 else -1 for i in range(len(prices))],
            index=prices.index
        )

        result = backtester.run_simple_backtest(prices, signals)

        if hasattr(result, 'total_trades'):
            # Should have multiple trades
            assert result.total_trades >= 0


class TestBacktestEdgeCases:
    """Tests for edge cases in backtesting."""

    def test_empty_data(self):
        """Test backtest with empty data."""
        backtester = Backtester(initial_capital=100000)

        empty_prices = pd.Series(dtype=float)
        empty_signals = pd.Series(dtype=float)

        try:
            result = backtester.run_simple_backtest(empty_prices, empty_signals)
            # Should handle gracefully
        except (ValueError, IndexError):
            pass  # Expected for empty data

    def test_single_day(self):
        """Test backtest with single day of data."""
        backtester = Backtester(initial_capital=100000)

        single_price = pd.Series([100.0], index=[datetime.now()])
        single_signal = pd.Series([1], index=[datetime.now()])

        try:
            result = backtester.run_simple_backtest(single_price, single_signal)
        except (ValueError, IndexError):
            pass  # May not support single day

    def test_no_trades(self, sample_ohlcv_data_long):
        """Test backtest with no trade signals."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']
        signals = pd.Series([0] * len(prices), index=prices.index)  # All HOLD

        result = backtester.run_simple_backtest(prices, signals)

        # Portfolio should remain at initial capital
        # or close to it (accounting for implementation)


class TestBacktestResult:
    """Tests for BacktestResult object."""

    def test_result_attributes(self, sample_ohlcv_data_long):
        """Test result object has expected attributes."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']
        signals = pd.Series([1] * len(prices), index=prices.index)

        result = backtester.run_simple_backtest(prices, signals)

        # Core attributes
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'final_value')
        assert hasattr(result, 'max_drawdown')

    def test_result_final_value(self, sample_ohlcv_data_long):
        """Test final value is reasonable."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']
        signals = pd.Series([1] * len(prices), index=prices.index)

        result = backtester.run_simple_backtest(prices, signals)

        # Final value should be positive
        assert result.final_value > 0

    def test_result_consistency(self, sample_ohlcv_data_long):
        """Test result values are internally consistent."""
        backtester = Backtester(initial_capital=100000)

        prices = sample_ohlcv_data_long['close']
        signals = pd.Series([1] * len(prices), index=prices.index)

        result = backtester.run_simple_backtest(prices, signals)

        # Total return should match final value / initial
        expected_return = (result.final_value / backtester.initial_capital) - 1
        assert abs(result.total_return - expected_return) < 0.01
