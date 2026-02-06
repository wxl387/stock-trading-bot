"""
Tests for SimulatedBroker - order execution and position tracking.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.broker.simulated_broker import SimulatedBroker
from src.broker.base_broker import OrderSide, OrderType, OrderStatus


# All tests mock _get_current_price (avoids yfinance calls) and _save_state/_load_state (avoids file I/O).
MOCK_PRICE = 150.0


@pytest.fixture
def broker():
    """Create SimulatedBroker with mocked I/O."""
    with patch.object(SimulatedBroker, '_load_state'), \
         patch.object(SimulatedBroker, '_save_state'), \
         patch.object(SimulatedBroker, '_get_current_price', return_value=MOCK_PRICE):
        b = SimulatedBroker(initial_capital=100000)
        yield b


class TestSimulatedBrokerInitialization:
    """Tests for SimulatedBroker initialization."""

    def test_default_initialization(self):
        """Test SimulatedBroker initializes with default capital."""
        with patch.object(SimulatedBroker, '_load_state'), \
             patch.object(SimulatedBroker, '_save_state'):
            broker = SimulatedBroker()
            assert broker is not None
            assert broker.cash > 0

    def test_custom_capital(self):
        """Test SimulatedBroker with custom initial capital."""
        with patch.object(SimulatedBroker, '_load_state'), \
             patch.object(SimulatedBroker, '_save_state'):
            broker = SimulatedBroker(initial_capital=50000)
            assert broker.cash == 50000

    def test_initial_state(self, broker):
        """Test initial state is clean."""
        assert broker.cash == 100000
        assert len(broker.positions) == 0
        assert len(broker.trades) == 0


class TestBuyOrders:
    """Tests for buy order execution."""

    def test_place_buy_order(self, broker):
        """Test placing a basic buy order."""
        result = broker.place_order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=150.0
        )

        assert result is not None
        assert result.is_filled()
        assert 'AAPL' in broker.positions

    def test_buy_order_cash_deduction(self, broker):
        """Test cash is properly deducted on buy."""
        initial_cash = broker.cash

        broker.place_order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=150.0
        )

        expected_cost = 10 * 150.0
        assert broker.cash == initial_cash - expected_cost

    def test_buy_order_insufficient_funds(self, broker):
        """Test buy order fails with insufficient funds."""
        result = broker.place_order(
            symbol='AAPL',
            side=OrderSide.BUY,
            quantity=10000,
            order_type=OrderType.LIMIT,
            price=150.0
        )

        assert result.status == OrderStatus.REJECTED

    def test_buy_multiple_orders_same_symbol(self, broker):
        """Test buying same symbol multiple times."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)
        broker.place_order('AAPL', OrderSide.BUY, 5, OrderType.LIMIT, price=155.0)

        position = broker.positions['AAPL']
        assert position['quantity'] == 15

    def test_buy_order_average_cost(self, broker):
        """Test average cost calculation."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=160.0)

        position = broker.positions['AAPL']
        # Average should be (10*150 + 10*160) / 20 = 155
        assert abs(position['avg_cost'] - 155.0) < 0.01


class TestSellOrders:
    """Tests for sell order execution."""

    def test_place_sell_order(self, broker):
        """Test placing a sell order."""
        # First buy
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)

        # Then sell
        result = broker.place_order('AAPL', OrderSide.SELL, 5, OrderType.LIMIT, price=155.0)

        assert result.is_filled()
        assert broker.positions['AAPL']['quantity'] == 5

    def test_sell_order_cash_credit(self, broker):
        """Test cash is credited on sell."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)
        cash_after_buy = broker.cash

        broker.place_order('AAPL', OrderSide.SELL, 5, OrderType.LIMIT, price=160.0)

        expected_credit = 5 * 160.0
        assert broker.cash == cash_after_buy + expected_credit

    def test_sell_all_shares(self, broker):
        """Test selling all shares removes position."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)
        broker.place_order('AAPL', OrderSide.SELL, 10, OrderType.LIMIT, price=155.0)

        # Position should be removed (production deletes on full close)
        assert 'AAPL' not in broker.positions

    def test_sell_more_than_owned(self, broker):
        """Test selling more shares than owned fails."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)
        result = broker.place_order('AAPL', OrderSide.SELL, 20, OrderType.LIMIT, price=155.0)

        assert result.status == OrderStatus.REJECTED

    def test_sell_without_position(self, broker):
        """Test selling without a position fails."""
        result = broker.place_order('AAPL', OrderSide.SELL, 10, OrderType.LIMIT, price=155.0)
        assert result.status == OrderStatus.REJECTED


class TestPositionTracking:
    """Tests for position tracking."""

    def test_position_tracking(self, broker):
        """Test positions are tracked correctly."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)
        broker.place_order('MSFT', OrderSide.BUY, 5, OrderType.LIMIT, price=300.0)

        assert len(broker.positions) == 2
        assert 'AAPL' in broker.positions
        assert 'MSFT' in broker.positions

    def test_get_positions(self, broker):
        """Test getting positions as list of Position objects."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)

        positions = broker.get_positions()
        assert positions is not None
        assert len(positions) == 1
        assert positions[0].symbol == 'AAPL'

    def test_position_market_value(self, broker):
        """Test position market value calculation."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)

        # _get_current_price is mocked to return MOCK_PRICE (150.0)
        positions = broker.get_positions()
        assert positions[0].market_value == 10 * MOCK_PRICE


class TestCashManagement:
    """Tests for cash management."""

    def test_cash_management(self, broker):
        """Test cash is managed correctly through trades."""
        initial_cash = broker.cash

        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)   # -$1500
        broker.place_order('AAPL', OrderSide.SELL, 5, OrderType.LIMIT, price=160.0)    # +$800

        expected_cash = initial_cash - (10 * 150) + (5 * 160)
        assert broker.cash == expected_cash

    def test_cash_attribute(self, broker):
        """Test cash is accessible."""
        assert broker.cash == 100000


class TestPnLCalculation:
    """Tests for P&L calculation."""

    def test_realized_pnl_profit(self, broker):
        """Test realized P&L calculation with profit."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)
        broker.place_order('AAPL', OrderSide.SELL, 10, OrderType.LIMIT, price=160.0)

        # Realized P&L = 10 * (160 - 150) = 100
        assert broker.realized_pnl == 100.0

    def test_realized_pnl_loss(self, broker):
        """Test realized P&L calculation with loss."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)
        broker.place_order('AAPL', OrderSide.SELL, 10, OrderType.LIMIT, price=140.0)

        # Realized P&L = 10 * (140 - 150) = -100
        assert broker.realized_pnl == -100.0

    def test_portfolio_summary(self, broker):
        """Test portfolio summary returns expected keys."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)

        summary = broker.get_portfolio_summary()
        assert 'portfolio_value' in summary
        assert 'cash' in summary
        assert 'realized_pnl' in summary
        assert 'unrealized_pnl' in summary


class TestTradeHistory:
    """Tests for trade history tracking."""

    def test_trade_history_tracking(self, broker):
        """Test trades are recorded in history."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)
        broker.place_order('MSFT', OrderSide.BUY, 5, OrderType.LIMIT, price=300.0)
        broker.place_order('AAPL', OrderSide.SELL, 5, OrderType.LIMIT, price=155.0)

        assert len(broker.trades) == 3

    def test_trade_history_details(self, broker):
        """Test trade history contains correct details."""
        broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)

        assert len(broker.trades) == 1
        trade = broker.trades[0]

        assert trade['symbol'] == 'AAPL'
        assert trade['side'] == 'BUY'
        assert trade['quantity'] == 10


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_quantity_order(self, broker):
        """Test order with zero quantity."""
        # Production fills 0-quantity orders (no explicit validation).
        # The order goes through _execute_fill with quantity=0, which is a no-op.
        result = broker.place_order('AAPL', OrderSide.BUY, 0, OrderType.LIMIT, price=150.0)
        # Just verify it doesn't crash and returns an Order
        assert result is not None

    def test_sell_without_holding(self, broker):
        """Test selling a symbol we don't own is rejected."""
        result = broker.place_order('AAPL', OrderSide.SELL, 10, OrderType.LIMIT, price=150.0)
        assert result.status == OrderStatus.REJECTED

    def test_order_has_order_id(self, broker):
        """Test that orders are assigned unique IDs."""
        result1 = broker.place_order('AAPL', OrderSide.BUY, 10, OrderType.LIMIT, price=150.0)
        result2 = broker.place_order('MSFT', OrderSide.BUY, 5, OrderType.LIMIT, price=300.0)

        assert result1.order_id != result2.order_id
