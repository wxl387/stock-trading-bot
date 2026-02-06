"""
Comprehensive unit tests for SimulatedBroker.

Tests cover:
- Connection / disconnection
- BUY orders: cash deducted, position created, correct avg_cost
- SELL orders: cash increased, position removed, realized P&L tracked
- Partial sells: position quantity reduced, avg_cost preserved
- Order rejection: insufficient funds, selling more than held
- Multiple buys of same symbol: avg_cost averaging
- Account info: portfolio_value = cash + positions market value
- Position info: get_position, get_positions
- State persistence: save_state / load_state round-trip (via tmp_path)
- Market price fetching: mock _get_current_price
- Order history tracking
- Portfolio summary
- Quote generation
- Order cap / trade cap trimming
- Cancel order
- Reset
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.broker.simulated_broker import SimulatedBroker
from src.broker.base_broker import (
    OrderSide, OrderType, OrderStatus, TimeInForce,
    Order, Position, AccountInfo, Quote,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_PRICE = 150.0  # default mock price for all symbols


def _make_broker(initial_capital: float = 100_000.0) -> SimulatedBroker:
    """Create a SimulatedBroker with _load_state and _save_state mocked out
    so that no real disk I/O happens and yfinance is never called."""
    with patch.object(SimulatedBroker, "_load_state"):
        broker = SimulatedBroker(initial_capital=initial_capital)
    # Ensure the broker starts completely clean
    broker.cash = initial_capital
    broker.initial_capital = initial_capital
    broker.positions = {}
    broker.orders = {}
    broker.trades = []
    broker.realized_pnl = 0.0
    broker._connected = False
    return broker


@pytest.fixture
def broker():
    """Fixture returning a fresh broker with _get_current_price mocked."""
    b = _make_broker()
    b._get_current_price = MagicMock(return_value=MOCK_PRICE)
    b._save_state = MagicMock()  # no-op for persistence during tests
    return b


# ===================================================================
# Connection / Disconnection
# ===================================================================

class TestConnection:

    def test_connect_returns_true(self, broker):
        assert broker.connect() is True

    def test_is_connected_after_connect(self, broker):
        broker.connect()
        assert broker.is_connected() is True

    def test_not_connected_initially(self, broker):
        assert broker.is_connected() is False

    def test_disconnect_sets_flag(self, broker):
        broker.connect()
        broker.disconnect()
        assert broker.is_connected() is False

    def test_disconnect_saves_state(self, broker):
        broker.connect()
        broker.disconnect()
        broker._save_state.assert_called()


# ===================================================================
# BUY Orders
# ===================================================================

class TestBuyOrders:

    def test_buy_deducts_cash(self, broker):
        qty = 10
        broker.place_order("AAPL", OrderSide.BUY, qty)
        expected_cash = 100_000.0 - MOCK_PRICE * qty
        assert broker.cash == pytest.approx(expected_cash)

    def test_buy_creates_position(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        assert "AAPL" in broker.positions
        assert broker.positions["AAPL"]["quantity"] == 10

    def test_buy_correct_avg_cost(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        assert broker.positions["AAPL"]["avg_cost"] == pytest.approx(MOCK_PRICE)

    def test_buy_order_is_filled(self, broker):
        order = broker.place_order("AAPL", OrderSide.BUY, 5)
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 5
        assert order.filled_price == pytest.approx(MOCK_PRICE)

    def test_buy_order_has_correct_metadata(self, broker):
        order = broker.place_order("AAPL", OrderSide.BUY, 5, OrderType.MARKET)
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 5

    def test_buy_limit_order_uses_limit_price(self, broker):
        limit_price = 145.0
        order = broker.place_order(
            "AAPL", OrderSide.BUY, 10,
            order_type=OrderType.LIMIT, price=limit_price,
        )
        assert order.filled_price == pytest.approx(limit_price)
        expected_cash = 100_000.0 - limit_price * 10
        assert broker.cash == pytest.approx(expected_cash)


# ===================================================================
# SELL Orders
# ===================================================================

class TestSellOrders:

    def test_sell_increases_cash(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        cash_after_buy = broker.cash
        broker.place_order("AAPL", OrderSide.SELL, 10)
        expected_cash = cash_after_buy + MOCK_PRICE * 10
        assert broker.cash == pytest.approx(expected_cash)

    def test_sell_removes_full_position(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker.place_order("AAPL", OrderSide.SELL, 10)
        # Position dict entry is removed when quantity reaches 0
        assert "AAPL" not in broker.positions

    def test_sell_tracks_realized_pnl_profit(self, broker):
        # Buy at 150, sell at 160 -> profit = 10 * 10 = 100
        broker._get_current_price.return_value = 150.0
        broker.place_order("AAPL", OrderSide.BUY, 10)

        broker._get_current_price.return_value = 160.0
        broker.place_order("AAPL", OrderSide.SELL, 10)

        expected_pnl = (160.0 - 150.0) * 10
        assert broker.realized_pnl == pytest.approx(expected_pnl)

    def test_sell_tracks_realized_pnl_loss(self, broker):
        # Buy at 150, sell at 140 -> loss = -10 * 10 = -100
        broker._get_current_price.return_value = 150.0
        broker.place_order("AAPL", OrderSide.BUY, 10)

        broker._get_current_price.return_value = 140.0
        broker.place_order("AAPL", OrderSide.SELL, 10)

        expected_pnl = (140.0 - 150.0) * 10
        assert broker.realized_pnl == pytest.approx(expected_pnl)

    def test_sell_order_is_filled(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        order = broker.place_order("AAPL", OrderSide.SELL, 10)
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10


# ===================================================================
# Partial Sells
# ===================================================================

class TestPartialSells:

    def test_partial_sell_reduces_quantity(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 20)
        broker.place_order("AAPL", OrderSide.SELL, 5)
        assert broker.positions["AAPL"]["quantity"] == 15

    def test_partial_sell_preserves_avg_cost(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 20)
        avg_before = broker.positions["AAPL"]["avg_cost"]
        broker.place_order("AAPL", OrderSide.SELL, 5)
        assert broker.positions["AAPL"]["avg_cost"] == pytest.approx(avg_before)

    def test_partial_sell_tracks_pnl(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 20)

        broker._get_current_price.return_value = 110.0
        broker.place_order("AAPL", OrderSide.SELL, 5)

        expected_pnl = (110.0 - 100.0) * 5
        assert broker.realized_pnl == pytest.approx(expected_pnl)

    def test_multiple_partial_sells(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 30)

        broker._get_current_price.return_value = 110.0
        broker.place_order("AAPL", OrderSide.SELL, 10)

        broker._get_current_price.return_value = 120.0
        broker.place_order("AAPL", OrderSide.SELL, 10)

        # 10 * 10  +  10 * 20 = 300
        expected_pnl = (110.0 - 100.0) * 10 + (120.0 - 100.0) * 10
        assert broker.realized_pnl == pytest.approx(expected_pnl)
        assert broker.positions["AAPL"]["quantity"] == 10


# ===================================================================
# Order Rejection
# ===================================================================

class TestOrderRejection:

    def test_reject_buy_insufficient_funds(self, broker):
        # Try to buy more than we can afford
        broker._get_current_price.return_value = 100_000.0  # very expensive
        order = broker.place_order("AAPL", OrderSide.BUY, 2)
        assert order.status == OrderStatus.REJECTED
        assert order.filled_quantity == 0
        assert order.filled_price is None

    def test_reject_buy_insufficient_funds_cash_unchanged(self, broker):
        broker._get_current_price.return_value = 100_000.0
        original_cash = broker.cash
        broker.place_order("AAPL", OrderSide.BUY, 2)
        assert broker.cash == pytest.approx(original_cash)

    def test_reject_sell_no_position(self, broker):
        order = broker.place_order("AAPL", OrderSide.SELL, 10)
        assert order.status == OrderStatus.REJECTED

    def test_reject_sell_more_than_held(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 5)
        order = broker.place_order("AAPL", OrderSide.SELL, 10)
        assert order.status == OrderStatus.REJECTED
        # Position unchanged
        assert broker.positions["AAPL"]["quantity"] == 5

    def test_rejected_order_stored_in_history(self, broker):
        broker._get_current_price.return_value = 200_000.0
        order = broker.place_order("AAPL", OrderSide.BUY, 1)
        assert order.order_id in broker.orders
        assert broker.orders[order.order_id].status == OrderStatus.REJECTED


# ===================================================================
# Multiple Buys (Average Cost)
# ===================================================================

class TestMultipleBuysAvgCost:

    def test_avg_cost_two_buys_same_price(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker.place_order("AAPL", OrderSide.BUY, 10)
        assert broker.positions["AAPL"]["quantity"] == 20
        assert broker.positions["AAPL"]["avg_cost"] == pytest.approx(MOCK_PRICE)

    def test_avg_cost_two_buys_different_prices(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 10)  # 10 @ 100

        broker._get_current_price.return_value = 200.0
        broker.place_order("AAPL", OrderSide.BUY, 10)  # 10 @ 200

        # avg = (10*100 + 10*200) / 20 = 150
        assert broker.positions["AAPL"]["quantity"] == 20
        assert broker.positions["AAPL"]["avg_cost"] == pytest.approx(150.0)

    def test_avg_cost_three_buys_unequal_quantities(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 10)  # 10 @ 100

        broker._get_current_price.return_value = 120.0
        broker.place_order("AAPL", OrderSide.BUY, 20)  # 20 @ 120

        broker._get_current_price.return_value = 80.0
        broker.place_order("AAPL", OrderSide.BUY, 10)  # 10 @ 80

        total_cost = 10 * 100.0 + 20 * 120.0 + 10 * 80.0
        total_qty = 40
        expected_avg = total_cost / total_qty
        assert broker.positions["AAPL"]["quantity"] == total_qty
        assert broker.positions["AAPL"]["avg_cost"] == pytest.approx(expected_avg)

    def test_cash_correct_after_multiple_buys(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 10)

        broker._get_current_price.return_value = 200.0
        broker.place_order("AAPL", OrderSide.BUY, 10)

        expected_cash = 100_000.0 - (10 * 100.0 + 10 * 200.0)
        assert broker.cash == pytest.approx(expected_cash)


# ===================================================================
# Account Info
# ===================================================================

class TestAccountInfo:

    def test_initial_account_info(self, broker):
        info = broker.get_account_info()
        assert isinstance(info, AccountInfo)
        assert info.cash == pytest.approx(100_000.0)
        assert info.portfolio_value == pytest.approx(100_000.0)
        assert info.positions_count == 0

    def test_portfolio_value_includes_positions(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        info = broker.get_account_info()
        # cash = 100_000 - 10*150 = 98_500
        # position value = 10 * 150 = 1_500
        # portfolio = 98_500 + 1_500 = 100_000
        assert info.portfolio_value == pytest.approx(100_000.0)

    def test_portfolio_value_with_price_change(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 10)
        # cash = 100_000 - 1_000 = 99_000

        # Price goes up; _get_current_price is now called for positions value
        broker._get_current_price.return_value = 120.0
        info = broker.get_account_info()
        # position market value = 10 * 120 = 1_200
        # portfolio = 99_000 + 1_200 = 100_200
        assert info.portfolio_value == pytest.approx(100_200.0)

    def test_positions_count(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 5)
        broker._get_current_price.return_value = 50.0
        broker.place_order("MSFT", OrderSide.BUY, 10)
        info = broker.get_account_info()
        assert info.positions_count == 2

    def test_buying_power_equals_cash(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        info = broker.get_account_info()
        assert info.buying_power == pytest.approx(info.cash)


# ===================================================================
# Position Info
# ===================================================================

class TestPositionInfo:

    def test_get_position_returns_none_for_no_position(self, broker):
        assert broker.get_position("AAPL") is None

    def test_get_position_returns_position_object(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        pos = broker.get_position("AAPL")
        assert isinstance(pos, Position)
        assert pos.symbol == "AAPL"
        assert pos.quantity == 10
        assert pos.avg_cost == pytest.approx(MOCK_PRICE)

    def test_get_position_unrealized_pnl(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 10)

        broker._get_current_price.return_value = 110.0
        pos = broker.get_position("AAPL")
        assert pos.unrealized_pnl == pytest.approx(100.0)  # (110-100)*10
        assert pos.market_value == pytest.approx(1100.0)

    def test_get_position_returns_none_after_full_sell(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker.place_order("AAPL", OrderSide.SELL, 10)
        assert broker.get_position("AAPL") is None

    def test_get_positions_multiple_symbols(self, broker):
        broker._get_current_price.return_value = 150.0
        broker.place_order("AAPL", OrderSide.BUY, 5)
        broker._get_current_price.return_value = 50.0
        broker.place_order("MSFT", OrderSide.BUY, 10)

        positions = broker.get_positions()
        assert len(positions) == 2
        symbols = {p.symbol for p in positions}
        assert symbols == {"AAPL", "MSFT"}

    def test_get_positions_empty_initially(self, broker):
        assert broker.get_positions() == []

    def test_position_unrealized_pnl_pct(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 10)

        broker._get_current_price.return_value = 110.0
        pos = broker.get_position("AAPL")
        # unrealized_pnl_pct = (1100 - 1000) / 1000 * 100 = 10.0
        assert pos.unrealized_pnl_pct == pytest.approx(10.0)


# ===================================================================
# State Persistence
# ===================================================================

class TestStatePersistence:

    def test_save_and_load_round_trip(self, tmp_path):
        state_file = tmp_path / "broker_state.json"

        with patch.object(SimulatedBroker, "_load_state"):
            broker = SimulatedBroker(initial_capital=50_000.0)
        broker.cash = 50_000.0
        broker.initial_capital = 50_000.0
        broker.positions = {}
        broker.orders = {}
        broker.trades = []
        broker.realized_pnl = 0.0
        broker._connected = False

        # Point STATE_FILE to tmp_path
        broker.__class__.STATE_FILE = state_file
        broker._get_current_price = MagicMock(return_value=100.0)

        # Execute some trades
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker.place_order("MSFT", OrderSide.BUY, 5)

        # Manually save
        broker._save_state()

        # Verify file exists
        assert state_file.exists()

        # Load into new broker
        with patch.object(SimulatedBroker, "_load_state"):
            broker2 = SimulatedBroker(initial_capital=50_000.0)
        broker2.__class__.STATE_FILE = state_file
        broker2._load_state()

        assert broker2.cash == pytest.approx(broker.cash)
        assert broker2.positions == broker.positions
        assert broker2.realized_pnl == pytest.approx(broker.realized_pnl)

    def test_load_state_no_file(self, tmp_path):
        state_file = tmp_path / "nonexistent.json"
        with patch.object(SimulatedBroker, "_load_state"):
            broker = SimulatedBroker(initial_capital=100_000.0)
        broker.__class__.STATE_FILE = state_file
        # Should not raise
        broker._load_state()
        # Cash unchanged
        assert broker.cash == pytest.approx(100_000.0)

    def test_save_state_creates_directory(self, tmp_path):
        state_file = tmp_path / "broker_state.json"
        with patch.object(SimulatedBroker, "_load_state"):
            broker = SimulatedBroker(initial_capital=100_000.0)
        broker.__class__.STATE_FILE = state_file

        # Patch DATA_DIR to the tmp directory so mkdir works
        with patch("src.broker.simulated_broker.DATA_DIR", tmp_path):
            broker._save_state()
        assert state_file.exists()

    def test_state_preserves_account_id(self, tmp_path):
        state_file = tmp_path / "broker_state.json"
        with patch.object(SimulatedBroker, "_load_state"):
            broker = SimulatedBroker(initial_capital=100_000.0)
        broker.__class__.STATE_FILE = state_file
        original_id = broker._account_id

        with patch("src.broker.simulated_broker.DATA_DIR", tmp_path):
            broker._save_state()

        with patch.object(SimulatedBroker, "_load_state"):
            broker2 = SimulatedBroker(initial_capital=100_000.0)
        broker2.__class__.STATE_FILE = state_file
        broker2._load_state()

        assert broker2._account_id == original_id

    def test_state_preserves_trades(self, tmp_path):
        state_file = tmp_path / "broker_state.json"
        with patch.object(SimulatedBroker, "_load_state"):
            broker = SimulatedBroker(initial_capital=100_000.0)
        broker.cash = 100_000.0
        broker.positions = {}
        broker.orders = {}
        broker.trades = []
        broker.realized_pnl = 0.0
        broker.__class__.STATE_FILE = state_file
        broker._get_current_price = MagicMock(return_value=100.0)

        broker.place_order("AAPL", OrderSide.BUY, 5)

        with patch("src.broker.simulated_broker.DATA_DIR", tmp_path):
            broker._save_state()

        with patch.object(SimulatedBroker, "_load_state"):
            broker2 = SimulatedBroker(initial_capital=100_000.0)
        broker2.__class__.STATE_FILE = state_file
        broker2._load_state()

        assert len(broker2.trades) > 0
        assert broker2.trades[0]["symbol"] == "AAPL"


# ===================================================================
# Market Price Fetching
# ===================================================================

class TestMarketPrice:

    def test_get_current_price_called_on_buy(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 1)
        broker._get_current_price.assert_called_with("AAPL")

    def test_get_current_price_called_on_sell(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker._get_current_price.reset_mock()
        broker.place_order("AAPL", OrderSide.SELL, 10)
        broker._get_current_price.assert_called_with("AAPL")

    def test_get_current_price_fallback_to_position(self):
        """Test that _get_current_price falls back to last known price."""
        with patch.object(SimulatedBroker, "_load_state"):
            broker = SimulatedBroker(initial_capital=100_000.0)
        broker.positions = {"AAPL": {"quantity": 10, "avg_cost": 150.0,
                                      "last_price": 155.0, "realized_pnl": 0.0}}

        with patch("src.broker.simulated_broker.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = MagicMock(empty=True)
            mock_yf.Ticker.return_value = mock_ticker
            price = broker._get_current_price("AAPL")
            assert price == pytest.approx(155.0)

    def test_get_current_price_raises_if_no_data(self):
        """Test ValueError when no price is available at all."""
        with patch.object(SimulatedBroker, "_load_state"):
            broker = SimulatedBroker(initial_capital=100_000.0)
        broker.positions = {}

        with patch("src.broker.simulated_broker.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = MagicMock(empty=True)
            mock_yf.Ticker.return_value = mock_ticker
            with pytest.raises(ValueError, match="No price available"):
                broker._get_current_price("AAPL")


# ===================================================================
# Order History Tracking
# ===================================================================

class TestOrderHistory:

    def test_orders_stored(self, broker):
        order = broker.place_order("AAPL", OrderSide.BUY, 10)
        assert order.order_id in broker.orders

    def test_get_order_by_id(self, broker):
        order = broker.place_order("AAPL", OrderSide.BUY, 10)
        retrieved = broker.get_order(order.order_id)
        assert retrieved is not None
        assert retrieved.order_id == order.order_id

    def test_get_order_returns_none_for_unknown_id(self, broker):
        assert broker.get_order("NONEXISTENT") is None

    def test_trades_list_populated(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        assert len(broker.trades) == 1
        assert broker.trades[0]["symbol"] == "AAPL"
        assert broker.trades[0]["side"] == "BUY"
        assert broker.trades[0]["quantity"] == 10

    def test_multiple_trades_tracked(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker.place_order("AAPL", OrderSide.SELL, 10)
        assert len(broker.trades) == 2

    def test_rejected_orders_not_in_trades(self, broker):
        """Rejected orders should not create trade entries."""
        order = broker.place_order("AAPL", OrderSide.SELL, 10)  # no position
        assert order.status == OrderStatus.REJECTED
        assert len(broker.trades) == 0

    def test_get_open_orders_empty_after_fill(self, broker):
        """Market orders fill immediately, so open orders should be empty."""
        broker.place_order("AAPL", OrderSide.BUY, 10)
        assert broker.get_open_orders() == []


# ===================================================================
# Quote Generation
# ===================================================================

class TestQuotes:

    def test_get_quote_returns_quote(self, broker):
        quote = broker.get_quote("AAPL")
        assert isinstance(quote, Quote)
        assert quote.symbol == "AAPL"
        assert quote.last == pytest.approx(MOCK_PRICE)

    def test_get_quote_bid_ask_spread(self, broker):
        quote = broker.get_quote("AAPL")
        assert quote.bid < quote.last
        assert quote.ask > quote.last
        assert quote.bid == pytest.approx(MOCK_PRICE * 0.999)
        assert quote.ask == pytest.approx(MOCK_PRICE * 1.001)

    def test_get_quotes_multiple(self, broker):
        quotes = broker.get_quotes(["AAPL", "MSFT"])
        assert "AAPL" in quotes
        assert "MSFT" in quotes
        assert all(isinstance(q, Quote) for q in quotes.values())


# ===================================================================
# Portfolio Summary
# ===================================================================

class TestPortfolioSummary:

    def test_initial_summary(self, broker):
        summary = broker.get_portfolio_summary()
        assert summary["initial_capital"] == pytest.approx(100_000.0)
        assert summary["cash"] == pytest.approx(100_000.0)
        assert summary["total_pnl"] == pytest.approx(0.0)
        assert summary["total_trades"] == 0
        assert summary["open_positions"] == 0

    def test_summary_after_trades(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 10)

        broker._get_current_price.return_value = 110.0
        summary = broker.get_portfolio_summary()
        # positions_value = 10 * 110 = 1100
        assert summary["positions_value"] == pytest.approx(1100.0)
        assert summary["open_positions"] == 1
        assert summary["total_trades"] == 1

    def test_summary_realized_pnl(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker._get_current_price.return_value = 120.0
        broker.place_order("AAPL", OrderSide.SELL, 10)

        summary = broker.get_portfolio_summary()
        assert summary["realized_pnl"] == pytest.approx(200.0)


# ===================================================================
# Cancel Order
# ===================================================================

class TestCancelOrder:

    def test_cancel_nonexistent_order(self, broker):
        assert broker.cancel_order("FAKE-ID") is False

    def test_cancel_filled_order_returns_false(self, broker):
        order = broker.place_order("AAPL", OrderSide.BUY, 10)
        # Already filled, so cancel should fail
        assert broker.cancel_order(order.order_id) is False


# ===================================================================
# Reset
# ===================================================================

class TestReset:

    def test_reset_restores_cash(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker.reset()
        assert broker.cash == pytest.approx(100_000.0)

    def test_reset_clears_positions(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker.reset()
        assert broker.positions == {}

    def test_reset_clears_orders(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker.reset()
        assert broker.orders == {}

    def test_reset_clears_trades(self, broker):
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker.reset()
        assert broker.trades == []

    def test_reset_clears_realized_pnl(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker._get_current_price.return_value = 120.0
        broker.place_order("AAPL", OrderSide.SELL, 10)
        broker.reset()
        assert broker.realized_pnl == pytest.approx(0.0)


# ===================================================================
# Convenience Methods (buy / sell / close_position from BaseBroker)
# ===================================================================

class TestConvenienceMethods:

    def test_buy_method(self, broker):
        order = broker.buy("AAPL", 10)
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.FILLED

    def test_sell_method(self, broker):
        broker.buy("AAPL", 10)
        order = broker.sell("AAPL", 10)
        assert order.side == OrderSide.SELL
        assert order.status == OrderStatus.FILLED

    def test_close_position(self, broker):
        broker.buy("AAPL", 10)
        order = broker.close_position("AAPL")
        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.quantity == 10
        assert order.status == OrderStatus.FILLED

    def test_close_position_no_position(self, broker):
        result = broker.close_position("AAPL")
        assert result is None


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:

    def test_buy_exact_cash_amount(self, broker):
        """Buy shares that cost exactly all available cash."""
        broker._get_current_price.return_value = 100.0
        broker.cash = 1000.0
        order = broker.place_order("AAPL", OrderSide.BUY, 10)  # 10 * 100 = 1000
        assert order.status == OrderStatus.FILLED
        assert broker.cash == pytest.approx(0.0)

    def test_buy_one_cent_over_budget(self, broker):
        """Buy that costs one more share than cash allows is rejected."""
        broker._get_current_price.return_value = 100.0
        broker.cash = 999.99
        order = broker.place_order("AAPL", OrderSide.BUY, 10)  # 10 * 100 = 1000
        assert order.status == OrderStatus.REJECTED

    def test_multiple_symbols_independent(self, broker):
        broker._get_current_price.return_value = 100.0
        broker.place_order("AAPL", OrderSide.BUY, 10)
        broker._get_current_price.return_value = 50.0
        broker.place_order("MSFT", OrderSide.BUY, 20)

        assert broker.positions["AAPL"]["quantity"] == 10
        assert broker.positions["AAPL"]["avg_cost"] == pytest.approx(100.0)
        assert broker.positions["MSFT"]["quantity"] == 20
        assert broker.positions["MSFT"]["avg_cost"] == pytest.approx(50.0)

    def test_last_price_updated_on_trade(self, broker):
        broker._get_current_price.return_value = 155.0
        broker.place_order("AAPL", OrderSide.BUY, 10)
        assert broker.positions["AAPL"]["last_price"] == pytest.approx(155.0)
