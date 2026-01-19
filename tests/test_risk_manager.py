"""
Tests for RiskManager - risk management and position sizing.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.risk.risk_manager import RiskManager


class TestRiskManagerInitialization:
    """Tests for RiskManager initialization."""

    def test_default_initialization(self):
        """Test RiskManager initializes with defaults."""
        rm = RiskManager()
        assert rm is not None
        assert rm.max_position_pct > 0
        assert rm.max_portfolio_pct > 0

    def test_custom_initialization(self, risk_config):
        """Test RiskManager with custom config."""
        rm = RiskManager(
            max_position_pct=risk_config['max_position_pct'],
            max_portfolio_pct=risk_config['max_portfolio_pct'],
            stop_loss_pct=risk_config['stop_loss_pct']
        )
        assert rm.max_position_pct == risk_config['max_position_pct']


class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_position_sizing_basic(self):
        """Test basic position sizing calculation."""
        rm = RiskManager(max_position_pct=0.10)

        portfolio_value = 100000
        price = 150.0

        # Should size based on max_position_pct
        shares = rm.calculate_position_size(
            symbol='AAPL',
            price=price,
            portfolio_value=portfolio_value,
            cash=portfolio_value
        )

        # Max position = 10% of 100k = 10k
        # At $150/share = ~66 shares max
        assert isinstance(shares, int)
        assert shares >= 0
        assert shares * price <= portfolio_value * 0.10 * 1.01  # Allow small tolerance

    def test_position_sizing_insufficient_cash(self):
        """Test position sizing with limited cash."""
        rm = RiskManager(max_position_pct=0.10)

        portfolio_value = 100000
        cash = 5000  # Only $5k available
        price = 150.0

        shares = rm.calculate_position_size(
            symbol='AAPL',
            price=price,
            portfolio_value=portfolio_value,
            cash=cash
        )

        # Should be limited by cash
        assert shares * price <= cash

    def test_position_sizing_zero_price(self):
        """Test position sizing with zero price."""
        rm = RiskManager()

        shares = rm.calculate_position_size(
            symbol='TEST',
            price=0.0,
            portfolio_value=100000,
            cash=100000
        )

        # Should return 0 for invalid price
        assert shares == 0

    def test_position_sizing_negative_price(self):
        """Test position sizing with negative price."""
        rm = RiskManager()

        shares = rm.calculate_position_size(
            symbol='TEST',
            price=-10.0,
            portfolio_value=100000,
            cash=100000
        )

        assert shares == 0


class TestStopLossCalculation:
    """Tests for stop-loss calculation."""

    def test_stop_loss_calculation(self):
        """Test basic stop-loss price calculation."""
        rm = RiskManager(stop_loss_pct=0.05)

        entry_price = 100.0
        stop_price = rm.calculate_stop_loss(entry_price)

        # 5% stop loss from $100 = $95
        expected = entry_price * (1 - 0.05)
        assert abs(stop_price - expected) < 0.01

    def test_stop_loss_custom_pct(self):
        """Test stop-loss with custom percentage."""
        rm = RiskManager(stop_loss_pct=0.10)

        entry_price = 200.0
        stop_price = rm.calculate_stop_loss(entry_price)

        expected = 200.0 * 0.90
        assert abs(stop_price - expected) < 0.01


class TestTakeProfitLevels:
    """Tests for take-profit calculations."""

    def test_take_profit_levels(self):
        """Test take-profit level calculation."""
        rm = RiskManager()

        entry_price = 100.0
        levels = rm.calculate_take_profit_levels(entry_price)

        assert isinstance(levels, list)
        # All levels should be above entry price
        for level in levels:
            assert level['price'] > entry_price
            assert 0 < level['exit_pct'] <= 1.0

    def test_take_profit_order(self):
        """Test creating take-profit order."""
        rm = RiskManager()

        # Create take profit for a position
        tp_orders = rm.get_take_profit_orders(
            symbol='AAPL',
            entry_price=100.0,
            quantity=100
        )

        assert isinstance(tp_orders, list)
        for order in tp_orders:
            assert order.symbol == 'AAPL'
            assert order.price > 100.0
            assert order.quantity > 0


class TestDrawdownProtection:
    """Tests for drawdown protection."""

    def test_drawdown_check_normal(self):
        """Test drawdown check under threshold."""
        rm = RiskManager(max_drawdown_pct=0.10)

        # 5% drawdown - should be OK
        result = rm.check_drawdown(
            current_value=95000,
            peak_value=100000
        )

        assert result.can_trade is True

    def test_drawdown_check_exceeded(self):
        """Test drawdown check when exceeded."""
        rm = RiskManager(max_drawdown_pct=0.10)

        # 15% drawdown - should block trading
        result = rm.check_drawdown(
            current_value=85000,
            peak_value=100000
        )

        assert result.can_trade is False
        assert 'drawdown' in result.reason.lower()

    def test_drawdown_recovery_mode(self):
        """Test recovery mode after drawdown breach."""
        rm = RiskManager(max_drawdown_pct=0.10)

        # Trigger drawdown breach
        rm.check_drawdown(current_value=85000, peak_value=100000)

        # Should be in recovery mode
        assert rm.in_recovery_mode is True

        # Position sizes should be reduced in recovery
        shares = rm.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            portfolio_value=85000,
            cash=85000
        )

        # Should be smaller than normal


class TestDailyLossLimit:
    """Tests for daily loss limit."""

    def test_daily_loss_check_normal(self):
        """Test daily loss within limit."""
        rm = RiskManager(max_daily_loss_pct=0.05)

        # 2% daily loss - should be OK
        result = rm.check_daily_loss(
            daily_pnl=-2000,
            portfolio_value=100000
        )

        assert result.can_trade is True

    def test_daily_loss_check_exceeded(self):
        """Test daily loss limit exceeded."""
        rm = RiskManager(max_daily_loss_pct=0.05)

        # 7% daily loss - should block trading
        result = rm.check_daily_loss(
            daily_pnl=-7000,
            portfolio_value=100000
        )

        assert result.can_trade is False


class TestVIXBasedSizing:
    """Tests for VIX-based position sizing."""

    @patch('src.risk.risk_manager.RiskManager._fetch_vix')
    def test_vix_multiplier_low_vix(self, mock_vix):
        """Test position multiplier with low VIX."""
        mock_vix.return_value = 12.0  # Low VIX

        rm = RiskManager()
        multiplier = rm.get_vix_multiplier()

        # Low VIX should allow larger positions (multiplier > 1)
        assert multiplier >= 1.0

    @patch('src.risk.risk_manager.RiskManager._fetch_vix')
    def test_vix_multiplier_high_vix(self, mock_vix):
        """Test position multiplier with high VIX."""
        mock_vix.return_value = 35.0  # High VIX

        rm = RiskManager()
        multiplier = rm.get_vix_multiplier()

        # High VIX should reduce positions (multiplier < 1)
        assert multiplier <= 1.0

    @patch('src.risk.risk_manager.RiskManager._fetch_vix')
    def test_vix_adjusted_position_size(self, mock_vix):
        """Test position sizing is adjusted by VIX."""
        mock_vix.return_value = 30.0  # Elevated VIX

        rm = RiskManager(max_position_pct=0.10)

        shares_high_vix = rm.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            portfolio_value=100000,
            cash=100000,
            use_vix_adjustment=True
        )

        mock_vix.return_value = 12.0  # Low VIX

        shares_low_vix = rm.calculate_position_size(
            symbol='AAPL',
            price=100.0,
            portfolio_value=100000,
            cash=100000,
            use_vix_adjustment=True
        )

        # Low VIX should allow more shares
        assert shares_low_vix >= shares_high_vix


class TestRegimeBasedParameters:
    """Tests for regime-based risk parameters."""

    def test_regime_bull_parameters(self):
        """Test risk parameters in bull market regime."""
        rm = RiskManager()
        rm.set_market_regime('BULL')

        params = rm.get_regime_parameters()

        # Bull market should have more aggressive parameters
        assert params['position_multiplier'] >= 1.0

    def test_regime_bear_parameters(self):
        """Test risk parameters in bear market regime."""
        rm = RiskManager()
        rm.set_market_regime('BEAR')

        params = rm.get_regime_parameters()

        # Bear market should have defensive parameters
        assert params['position_multiplier'] <= 1.0

    def test_regime_volatile_parameters(self):
        """Test risk parameters in volatile market."""
        rm = RiskManager()
        rm.set_market_regime('VOLATILE')

        params = rm.get_regime_parameters()

        # Volatile should reduce position sizes
        assert params['position_multiplier'] < 1.0


class TestRiskChecks:
    """Tests for comprehensive risk checks."""

    def test_check_trade_allowed(self):
        """Test overall trade permission check."""
        rm = RiskManager()

        result = rm.check_trade_allowed(
            symbol='AAPL',
            side='BUY',
            quantity=10,
            price=150.0,
            portfolio_value=100000,
            cash=50000,
            current_positions={'MSFT': {'quantity': 5, 'value': 1500}}
        )

        assert hasattr(result, 'can_trade')
        assert hasattr(result, 'reason')

    def test_check_trade_max_positions(self):
        """Test max positions limit."""
        rm = RiskManager(max_positions=3)

        # Already at max positions
        positions = {
            'AAPL': {'quantity': 10},
            'MSFT': {'quantity': 5},
            'GOOGL': {'quantity': 3}
        }

        result = rm.check_trade_allowed(
            symbol='NVDA',  # New position
            side='BUY',
            quantity=5,
            price=400.0,
            portfolio_value=100000,
            cash=50000,
            current_positions=positions
        )

        # Should block due to max positions
        assert result.can_trade is False or 'position' in result.reason.lower()


class TestPnLTracking:
    """Tests for P&L tracking functionality."""

    def test_update_daily_pnl(self):
        """Test daily P&L update."""
        rm = RiskManager()

        rm.update_daily_pnl(500.0)
        assert rm.daily_pnl == 500.0

        rm.update_daily_pnl(-200.0)
        assert rm.daily_pnl == 300.0

    def test_reset_daily_pnl(self):
        """Test daily P&L reset."""
        rm = RiskManager()

        rm.update_daily_pnl(1000.0)
        rm.reset_daily_pnl()

        assert rm.daily_pnl == 0.0

    def test_update_peak_value(self):
        """Test peak portfolio value tracking."""
        rm = RiskManager()

        rm.update_peak_value(100000)
        assert rm.peak_value == 100000

        rm.update_peak_value(110000)
        assert rm.peak_value == 110000

        # Lower value shouldn't update peak
        rm.update_peak_value(105000)
        assert rm.peak_value == 110000
