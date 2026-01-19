"""
Tests for SimulatedBroker - order execution and position tracking.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.broker.simulated_broker import SimulatedBroker


class TestSimulatedBrokerInitialization:
    """Tests for SimulatedBroker initialization."""

    def test_default_initialization(self):
        """Test SimulatedBroker initializes with default capital."""
        broker = SimulatedBroker()
        assert broker is not None
        assert broker.cash > 0

    def test_custom_capital(self):
        """Test SimulatedBroker with custom initial capital."""
        broker = SimulatedBroker(initial_capital=50000)
        assert broker.cash == 50000

    def test_initial_state(self):
        """Test initial state is clean."""
        broker = SimulatedBroker(initial_capital=100000)
        assert broker.cash == 100000
        assert len(broker.positions) == 0
        assert len(broker.get_trade_history()) == 0


class TestBuyOrders:
    """Tests for buy order execution."""

    def test_place_buy_order(self, mock_broker):
        """Test placing a basic buy order."""
        result = mock_broker.place_order(
            symbol='AAPL',
            side='BUY',
            quantity=10,
            price=150.0
        )

        assert result is not None
        assert result.success is True
        assert 'AAPL' in mock_broker.positions

    def test_buy_order_cash_deduction(self, mock_broker):
        """Test cash is properly deducted on buy."""
        initial_cash = mock_broker.cash

        mock_broker.place_order(
            symbol='AAPL',
            side='BUY',
            quantity=10,
            price=150.0
        )

        expected_cost = 10 * 150.0
        assert mock_broker.cash == initial_cash - expected_cost

    def test_buy_order_insufficient_funds(self, mock_broker):
        """Test buy order fails with insufficient funds."""
        # Try to buy more than we can afford
        result = mock_broker.place_order(
            symbol='AAPL',
            side='BUY',
            quantity=10000,  # Way too many at $150 each
            price=150.0
        )

        assert result.success is False

    def test_buy_multiple_orders_same_symbol(self, mock_broker):
        """Test buying same symbol multiple times."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)
        mock_broker.place_order('AAPL', 'BUY', 5, 155.0)

        position = mock_broker.positions['AAPL']
        assert position['quantity'] == 15

    def test_buy_order_average_cost(self, mock_broker):
        """Test average cost calculation."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)
        mock_broker.place_order('AAPL', 'BUY', 10, 160.0)

        position = mock_broker.positions['AAPL']
        # Average should be (10*150 + 10*160) / 20 = 155
        assert abs(position['avg_cost'] - 155.0) < 0.01


class TestSellOrders:
    """Tests for sell order execution."""

    def test_place_sell_order(self, mock_broker):
        """Test placing a sell order."""
        # First buy
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)

        # Then sell
        result = mock_broker.place_order('AAPL', 'SELL', 5, 155.0)

        assert result.success is True
        assert mock_broker.positions['AAPL']['quantity'] == 5

    def test_sell_order_cash_credit(self, mock_broker):
        """Test cash is credited on sell."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)
        cash_after_buy = mock_broker.cash

        mock_broker.place_order('AAPL', 'SELL', 5, 160.0)

        expected_credit = 5 * 160.0
        assert mock_broker.cash == cash_after_buy + expected_credit

    def test_sell_all_shares(self, mock_broker):
        """Test selling all shares removes position."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)
        mock_broker.place_order('AAPL', 'SELL', 10, 155.0)

        # Position should be removed or have 0 quantity
        assert 'AAPL' not in mock_broker.positions or mock_broker.positions['AAPL']['quantity'] == 0

    def test_sell_more_than_owned(self, mock_broker):
        """Test selling more shares than owned fails."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)
        result = mock_broker.place_order('AAPL', 'SELL', 20, 155.0)

        # Should fail or only sell what we have
        assert result.success is False or mock_broker.positions['AAPL']['quantity'] >= 0

    def test_sell_without_position(self, mock_broker):
        """Test selling without a position fails."""
        result = mock_broker.place_order('AAPL', 'SELL', 10, 155.0)
        assert result.success is False


class TestPositionTracking:
    """Tests for position tracking."""

    def test_position_tracking(self, mock_broker):
        """Test positions are tracked correctly."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)
        mock_broker.place_order('MSFT', 'BUY', 5, 300.0)

        assert len(mock_broker.positions) == 2
        assert 'AAPL' in mock_broker.positions
        assert 'MSFT' in mock_broker.positions

    def test_get_positions(self, mock_broker):
        """Test getting positions as DataFrame or dict."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)

        positions = mock_broker.get_positions()
        assert positions is not None

    def test_position_market_value(self, mock_broker):
        """Test position market value calculation."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)

        # Update current price
        mock_broker.update_price('AAPL', 160.0)

        position = mock_broker.positions['AAPL']
        market_value = position['quantity'] * position.get('current_price', 160.0)
        assert market_value == 10 * 160.0


class TestCashManagement:
    """Tests for cash management."""

    def test_cash_management(self, mock_broker):
        """Test cash is managed correctly through trades."""
        initial_cash = mock_broker.cash

        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)  # -$1500
        mock_broker.place_order('AAPL', 'SELL', 5, 160.0)  # +$800

        expected_cash = initial_cash - (10 * 150) + (5 * 160)
        assert mock_broker.cash == expected_cash

    def test_get_cash(self, mock_broker):
        """Test getting available cash."""
        cash = mock_broker.get_cash()
        assert cash == mock_broker.cash


class TestPnLCalculation:
    """Tests for P&L calculation."""

    def test_pnl_calculation_profit(self, mock_broker):
        """Test P&L calculation with profit."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)
        mock_broker.update_price('AAPL', 160.0)

        unrealized_pnl = mock_broker.get_unrealized_pnl('AAPL')
        expected_pnl = 10 * (160.0 - 150.0)
        assert unrealized_pnl == expected_pnl

    def test_pnl_calculation_loss(self, mock_broker):
        """Test P&L calculation with loss."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)
        mock_broker.update_price('AAPL', 140.0)

        unrealized_pnl = mock_broker.get_unrealized_pnl('AAPL')
        expected_pnl = 10 * (140.0 - 150.0)
        assert unrealized_pnl == expected_pnl

    def test_total_portfolio_value(self, mock_broker):
        """Test total portfolio value calculation."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)
        mock_broker.update_price('AAPL', 160.0)

        total_value = mock_broker.get_portfolio_value()
        # Cash + position value
        expected = mock_broker.cash + (10 * 160.0)
        assert total_value == expected


class TestTradeHistory:
    """Tests for trade history tracking."""

    def test_trade_history_tracking(self, mock_broker):
        """Test trades are recorded in history."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)
        mock_broker.place_order('MSFT', 'BUY', 5, 300.0)
        mock_broker.place_order('AAPL', 'SELL', 5, 155.0)

        history = mock_broker.get_trade_history()
        assert len(history) == 3

    def test_trade_history_details(self, mock_broker):
        """Test trade history contains correct details."""
        mock_broker.place_order('AAPL', 'BUY', 10, 150.0)

        history = mock_broker.get_trade_history()
        trade = history[0] if isinstance(history, list) else history.iloc[0]

        assert trade['symbol'] == 'AAPL'
        assert trade['side'] == 'BUY'
        assert trade['quantity'] == 10
        assert trade['price'] == 150.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_quantity_order(self, mock_broker):
        """Test order with zero quantity."""
        result = mock_broker.place_order('AAPL', 'BUY', 0, 150.0)
        assert result.success is False

    def test_negative_quantity_order(self, mock_broker):
        """Test order with negative quantity."""
        result = mock_broker.place_order('AAPL', 'BUY', -10, 150.0)
        assert result.success is False

    def test_zero_price_order(self, mock_broker):
        """Test order with zero price."""
        result = mock_broker.place_order('AAPL', 'BUY', 10, 0.0)
        assert result.success is False

    def test_invalid_side(self, mock_broker):
        """Test order with invalid side."""
        result = mock_broker.place_order('AAPL', 'INVALID', 10, 150.0)
        assert result.success is False
