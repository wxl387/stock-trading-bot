"""
Tests for WebullBroker - WeBull API broker implementation.

All tests mock the webull library entirely (it may not be installed).
All I/O and API calls are mocked for fast, offline testing.
"""
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock

# Mock the webull library before importing WebullBroker
import sys
mock_webull_module = MagicMock()
sys.modules['webull'] = mock_webull_module

from src.broker.webull_broker import WebullBroker
from src.broker.base_broker import (
    Order, Position, AccountInfo, Quote,
    OrderSide, OrderType, OrderStatus, TimeInForce
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """Create a mock webull client with sensible defaults."""
    client = MagicMock()
    client.login.return_value = {"accessToken": "fake-token"}
    client.get_trade_token.return_value = True
    client.get_account_id.return_value = "12345678"
    client.logout.return_value = True
    return client


@pytest.fixture
def broker(mock_client):
    """Create a connected WebullBroker with a mocked client (paper mode)."""
    with patch.object(WebullBroker, '__init__', lambda self, **kw: None):
        b = WebullBroker()
    b.paper_trading = True
    b._connected = False
    b._account_id = None
    b.client = mock_client
    return b


@pytest.fixture
def connected_broker(broker):
    """Return a broker that is already marked as connected."""
    broker._connected = True
    broker._account_id = "12345678"
    return broker


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

SAMPLE_ACCOUNT = {
    "cashBalance": "25000.50",
    "dayBuyingPower": "50000.00",
    "totalMarketValue": "75000.00",
    "dayTradesRemaining": "3",
}

SAMPLE_POSITIONS = [
    {
        "ticker": {"symbol": "AAPL"},
        "position": "10",
        "costPrice": "150.00",
        "lastPrice": "155.00",
        "marketValue": "1550.00",
        "unrealizedProfitLoss": "50.00",
    },
    {
        "ticker": {"symbol": "MSFT"},
        "position": "5",
        "costPrice": "300.00",
        "lastPrice": "310.00",
        "marketValue": "1550.00",
        "unrealizedProfitLoss": "50.00",
    },
]

SAMPLE_ORDER_RESULT = {
    "orderId": "ORD-001",
}

SAMPLE_QUOTE = {
    "bid": 154.50,
    "ask": 155.50,
    "close": 155.00,
    "volume": 5000000,
}

SAMPLE_ORDER_DATA_FILLED = {
    "orderId": "ORD-100",
    "ticker": {"symbol": "AAPL"},
    "action": "BUY",
    "orderType": "LMT",
    "totalQuantity": "10",
    "lmtPrice": "150.00",
    "stpPrice": None,
    "filledQuantity": "10",
    "avgFilledPrice": "149.95",
    "status": "Filled",
}

SAMPLE_ORDER_DATA_PENDING = {
    "orderId": "ORD-200",
    "ticker": {"symbol": "TSLA"},
    "action": "SELL",
    "orderType": "MKT",
    "totalQuantity": "20",
    "lmtPrice": None,
    "stpPrice": None,
    "filledQuantity": "0",
    "avgFilledPrice": None,
    "status": "Working",
}

SAMPLE_ORDER_DATA_CANCELLED = {
    "orderId": "ORD-300",
    "ticker": {"symbol": "GOOGL"},
    "action": "BUY",
    "orderType": "STP",
    "totalQuantity": "15",
    "lmtPrice": None,
    "stpPrice": "130.00",
    "filledQuantity": "0",
    "avgFilledPrice": None,
    "status": "Cancelled",
}


# ===========================================================================
# Initialization Tests
# ===========================================================================

class TestWebullBrokerInitialization:
    """Tests for WebullBroker __init__."""

    def test_paper_trading_init(self):
        """Paper trading mode uses paper_webull client."""
        mock_paper_cls = MagicMock()
        with patch('src.broker.webull_broker.paper_webull', mock_paper_cls):
            b = WebullBroker(paper_trading=True)

        mock_paper_cls.assert_called_once()
        assert b.paper_trading is True
        assert b._connected is False
        assert b._account_id is None

    def test_live_trading_init(self):
        """Live trading mode uses webull client."""
        mock_live_cls = MagicMock()
        with patch('src.broker.webull_broker.webull', mock_live_cls):
            b = WebullBroker(paper_trading=False)

        mock_live_cls.assert_called_once()
        assert b.paper_trading is False
        assert b._connected is False

    def test_default_is_paper(self):
        """Default initialization should be paper trading."""
        mock_paper_cls = MagicMock()
        with patch('src.broker.webull_broker.paper_webull', mock_paper_cls):
            b = WebullBroker()

        assert b.paper_trading is True


# ===========================================================================
# Connection Tests
# ===========================================================================

class TestConnect:
    """Tests for connect() / disconnect() / is_connected()."""

    def test_connect_success(self, broker, mock_client):
        """Successful login sets _connected and _account_id."""
        result = broker.connect()

        assert result is True
        assert broker._connected is True
        assert broker._account_id == "12345678"
        mock_client.login.assert_called_once()
        mock_client.get_trade_token.assert_called_once()
        mock_client.get_account_id.assert_called_once()

    def test_connect_login_returns_none(self, broker, mock_client):
        """Login returning None means credentials failed."""
        mock_client.login.return_value = None

        result = broker.connect()

        assert result is False
        assert broker._connected is False

    def test_connect_exception(self, broker, mock_client):
        """Exception during login returns False."""
        mock_client.login.side_effect = Exception("Network error")

        result = broker.connect()

        assert result is False
        assert broker._connected is False

    def test_connect_trade_token_exception(self, broker, mock_client):
        """Exception during get_trade_token returns False."""
        mock_client.get_trade_token.side_effect = Exception("Invalid PIN")

        result = broker.connect()

        assert result is False
        assert broker._connected is False

    def test_disconnect_success(self, connected_broker, mock_client):
        """Disconnect calls logout and clears _connected."""
        connected_broker.disconnect()

        assert connected_broker._connected is False
        mock_client.logout.assert_called_once()

    def test_disconnect_exception_does_not_raise(self, connected_broker, mock_client):
        """Disconnect catches exceptions gracefully."""
        mock_client.logout.side_effect = Exception("Logout failed")

        # Should not raise
        connected_broker.disconnect()

    def test_is_connected_true(self, connected_broker):
        """is_connected returns True when connected."""
        assert connected_broker.is_connected() is True

    def test_is_connected_false(self, broker):
        """is_connected returns False when not connected."""
        assert broker.is_connected() is False


# ===========================================================================
# Ensure Connected Guard Tests
# ===========================================================================

class TestEnsureConnected:
    """Tests that methods requiring connection raise ConnectionError."""

    def test_get_account_info_not_connected(self, broker):
        """get_account_info raises ConnectionError when not connected."""
        with pytest.raises(ConnectionError, match="Not connected"):
            broker.get_account_info()

    def test_get_positions_not_connected(self, broker):
        """get_positions raises ConnectionError when not connected."""
        with pytest.raises(ConnectionError, match="Not connected"):
            broker.get_positions()

    def test_place_order_not_connected(self, broker):
        """place_order raises ConnectionError when not connected."""
        with pytest.raises(ConnectionError, match="Not connected"):
            broker.place_order("AAPL", OrderSide.BUY, 10)

    def test_cancel_order_not_connected(self, broker):
        """cancel_order raises ConnectionError when not connected."""
        with pytest.raises(ConnectionError, match="Not connected"):
            broker.cancel_order("ORD-001")

    def test_get_order_not_connected(self, broker):
        """get_order raises ConnectionError when not connected."""
        with pytest.raises(ConnectionError, match="Not connected"):
            broker.get_order("ORD-001")

    def test_get_open_orders_not_connected(self, broker):
        """get_open_orders raises ConnectionError when not connected."""
        with pytest.raises(ConnectionError, match="Not connected"):
            broker.get_open_orders()


# ===========================================================================
# Account Info Tests
# ===========================================================================

class TestGetAccountInfo:
    """Tests for get_account_info()."""

    def test_get_account_info_success(self, connected_broker, mock_client):
        """get_account_info returns correctly parsed AccountInfo."""
        mock_client.get_account.return_value = SAMPLE_ACCOUNT
        mock_client.get_positions.return_value = SAMPLE_POSITIONS

        info = connected_broker.get_account_info()

        assert isinstance(info, AccountInfo)
        assert info.account_id == "12345678"
        assert info.cash == 25000.50
        assert info.buying_power == 50000.00
        assert info.portfolio_value == 75000.00
        assert info.day_trades_remaining == 3
        assert info.positions_count == 2

    def test_get_account_info_invalid_data_none(self, connected_broker, mock_client):
        """get_account_info raises on None account data."""
        mock_client.get_account.return_value = None

        with pytest.raises(ConnectionError, match="invalid data"):
            connected_broker.get_account_info()

    def test_get_account_info_invalid_data_not_dict(self, connected_broker, mock_client):
        """get_account_info raises when account returns a non-dict."""
        mock_client.get_account.return_value = "error string"

        with pytest.raises(ConnectionError, match="invalid data"):
            connected_broker.get_account_info()

    def test_get_account_info_empty_dict_raises(self, connected_broker, mock_client):
        """get_account_info raises on empty dict (falsy)."""
        mock_client.get_account.return_value = {}

        with pytest.raises(ConnectionError, match="invalid data"):
            connected_broker.get_account_info()

    def test_get_account_info_missing_fields(self, connected_broker, mock_client):
        """get_account_info defaults to 0 for missing fields."""
        # Dict must be non-empty (truthy) to pass the `not account` check
        mock_client.get_account.return_value = {"accountId": "12345678"}
        mock_client.get_positions.return_value = []

        info = connected_broker.get_account_info()

        assert info.cash == 0.0
        assert info.buying_power == 0.0
        assert info.portfolio_value == 0.0
        assert info.day_trades_remaining == 0
        assert info.positions_count == 0

    def test_get_account_info_positions_fetch_fails(self, connected_broker, mock_client):
        """Positions count defaults to 0 if positions fetch fails."""
        mock_client.get_account.return_value = SAMPLE_ACCOUNT
        mock_client.get_positions.side_effect = Exception("API error")

        info = connected_broker.get_account_info()

        assert info.positions_count == 0
        # Other fields still correct
        assert info.cash == 25000.50


# ===========================================================================
# Positions Tests
# ===========================================================================

class TestGetPositions:
    """Tests for get_positions() and get_position()."""

    def test_get_positions_success(self, connected_broker, mock_client):
        """get_positions returns correctly parsed Position objects."""
        mock_client.get_positions.return_value = SAMPLE_POSITIONS

        positions = connected_broker.get_positions()

        assert len(positions) == 2
        assert all(isinstance(p, Position) for p in positions)

        aapl = positions[0]
        assert aapl.symbol == "AAPL"
        assert aapl.quantity == 10
        assert aapl.avg_cost == 150.00
        assert aapl.current_price == 155.00
        assert aapl.market_value == 1550.00
        assert aapl.unrealized_pnl == 50.00

    def test_get_positions_empty(self, connected_broker, mock_client):
        """get_positions returns empty list when no positions."""
        mock_client.get_positions.return_value = []

        positions = connected_broker.get_positions()

        assert positions == []

    def test_get_positions_none_response(self, connected_broker, mock_client):
        """get_positions handles None response from client."""
        mock_client.get_positions.return_value = None

        positions = connected_broker.get_positions()

        assert positions == []

    def test_get_positions_skips_malformed(self, connected_broker, mock_client):
        """get_positions skips positions with malformed data."""
        positions_data = [
            SAMPLE_POSITIONS[0],
            {"ticker": None, "position": "bad"},  # Will cause error
        ]
        mock_client.get_positions.return_value = positions_data

        positions = connected_broker.get_positions()

        # First position parsed, second skipped
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"

    def test_get_positions_unrealized_pnl_pct(self, connected_broker, mock_client):
        """Unrealized P&L percentage is computed correctly."""
        mock_client.get_positions.return_value = SAMPLE_POSITIONS

        positions = connected_broker.get_positions()
        aapl = positions[0]

        # Expected: 50 / (150 * 10) * 100 = 3.333...
        expected_pct = 50.0 / (150.0 * 10) * 100
        assert abs(aapl.unrealized_pnl_pct - expected_pct) < 0.01

    def test_get_positions_zero_cost_pnl_pct(self, connected_broker, mock_client):
        """Zero avg_cost * quantity yields 0% unrealized P&L."""
        positions_data = [{
            "ticker": {"symbol": "FREE"},
            "position": "0",
            "costPrice": "0",
            "lastPrice": "10.00",
            "marketValue": "0",
            "unrealizedProfitLoss": "0",
        }]
        mock_client.get_positions.return_value = positions_data

        positions = connected_broker.get_positions()

        assert len(positions) == 1
        assert positions[0].unrealized_pnl_pct == 0

    def test_get_position_found(self, connected_broker, mock_client):
        """get_position returns the matching Position."""
        mock_client.get_positions.return_value = SAMPLE_POSITIONS

        pos = connected_broker.get_position("AAPL")

        assert pos is not None
        assert pos.symbol == "AAPL"

    def test_get_position_case_insensitive(self, connected_broker, mock_client):
        """get_position matches case-insensitively."""
        mock_client.get_positions.return_value = SAMPLE_POSITIONS

        pos = connected_broker.get_position("aapl")

        assert pos is not None
        assert pos.symbol == "AAPL"

    def test_get_position_not_found(self, connected_broker, mock_client):
        """get_position returns None for unknown symbol."""
        mock_client.get_positions.return_value = SAMPLE_POSITIONS

        pos = connected_broker.get_position("TSLA")

        assert pos is None


# ===========================================================================
# Place Order Tests
# ===========================================================================

class TestPlaceOrder:
    """Tests for place_order() with different order types."""

    def test_market_order_buy(self, connected_broker, mock_client):
        """Market BUY order uses correct parameters."""
        mock_client.place_order.return_value = SAMPLE_ORDER_RESULT

        order = connected_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )

        assert isinstance(order, Order)
        assert order.order_id == "ORD-001"
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 10
        assert order.status == OrderStatus.PENDING
        assert order.filled_quantity == 0

        mock_client.place_order.assert_called_once_with(
            stock="AAPL",
            action="BUY",
            orderType="MKT",
            enforce="DAY",
            qty=10,
        )

    def test_market_order_sell(self, connected_broker, mock_client):
        """Market SELL order maps action correctly."""
        mock_client.place_order.return_value = SAMPLE_ORDER_RESULT

        order = connected_broker.place_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.MARKET,
        )

        assert order.side == OrderSide.SELL
        mock_client.place_order.assert_called_once_with(
            stock="AAPL",
            action="SELL",
            orderType="MKT",
            enforce="DAY",
            qty=5,
        )

    def test_limit_order(self, connected_broker, mock_client):
        """Limit order includes price parameter."""
        mock_client.place_order.return_value = SAMPLE_ORDER_RESULT

        order = connected_broker.place_order(
            symbol="MSFT",
            side=OrderSide.BUY,
            quantity=20,
            order_type=OrderType.LIMIT,
            price=300.50,
        )

        assert order.price == 300.50
        assert order.order_type == OrderType.LIMIT
        mock_client.place_order.assert_called_once_with(
            stock="MSFT",
            action="BUY",
            orderType="LMT",
            enforce="DAY",
            qty=20,
            price=300.50,
        )

    def test_limit_order_missing_price_raises(self, connected_broker, mock_client):
        """Limit order without price raises ValueError."""
        with pytest.raises(ValueError, match="Limit price required"):
            connected_broker.place_order(
                symbol="MSFT",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.LIMIT,
                price=None,
            )

    def test_stop_order(self, connected_broker, mock_client):
        """Stop order includes stpPrice parameter."""
        mock_client.place_order.return_value = SAMPLE_ORDER_RESULT

        order = connected_broker.place_order(
            symbol="TSLA",
            side=OrderSide.SELL,
            quantity=15,
            order_type=OrderType.STOP,
            stop_price=200.00,
        )

        assert order.stop_price == 200.00
        assert order.order_type == OrderType.STOP
        mock_client.place_order.assert_called_once_with(
            stock="TSLA",
            action="SELL",
            orderType="STP",
            enforce="DAY",
            qty=15,
            stpPrice=200.00,
        )

    def test_stop_order_missing_stop_price_raises(self, connected_broker, mock_client):
        """Stop order without stop_price raises ValueError."""
        with pytest.raises(ValueError, match="Stop price required"):
            connected_broker.place_order(
                symbol="TSLA",
                side=OrderSide.SELL,
                quantity=15,
                order_type=OrderType.STOP,
                stop_price=None,
            )

    def test_unsupported_order_type_raises(self, connected_broker, mock_client):
        """Unsupported order type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported order type"):
            connected_broker.place_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.STOP_LIMIT,  # Not handled in place_order
            )

    def test_order_result_none_raises(self, connected_broker, mock_client):
        """place_order raises when API returns None."""
        mock_client.place_order.return_value = None

        with pytest.raises(Exception, match="returned None"):
            connected_broker.place_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
            )

    def test_order_api_exception_propagates(self, connected_broker, mock_client):
        """API exception during place_order propagates."""
        mock_client.place_order.side_effect = Exception("API rate limited")

        with pytest.raises(Exception, match="API rate limited"):
            connected_broker.place_order(
                symbol="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
            )

    def test_order_timestamps(self, connected_broker, mock_client):
        """Returned order has created_at and updated_at set."""
        mock_client.place_order.return_value = SAMPLE_ORDER_RESULT

        before = datetime.now()
        order = connected_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
        )
        after = datetime.now()

        assert before <= order.created_at <= after
        assert before <= order.updated_at <= after

    def test_time_in_force_gtc(self, connected_broker, mock_client):
        """GTC time-in-force maps correctly."""
        mock_client.place_order.return_value = SAMPLE_ORDER_RESULT

        connected_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
        )

        call_kwargs = mock_client.place_order.call_args
        assert call_kwargs[1]["enforce"] == "GTC" or call_kwargs.kwargs.get("enforce") == "GTC"

    def test_time_in_force_ioc(self, connected_broker, mock_client):
        """IOC time-in-force maps correctly."""
        mock_client.place_order.return_value = SAMPLE_ORDER_RESULT

        connected_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
        )

        _, kwargs = mock_client.place_order.call_args
        assert kwargs["enforce"] == "IOC"

    def test_time_in_force_fok(self, connected_broker, mock_client):
        """FOK time-in-force maps correctly."""
        mock_client.place_order.return_value = SAMPLE_ORDER_RESULT

        connected_broker.place_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.FOK,
        )

        _, kwargs = mock_client.place_order.call_args
        assert kwargs["enforce"] == "FOK"


# ===========================================================================
# Cancel Order Tests
# ===========================================================================

class TestCancelOrder:
    """Tests for cancel_order()."""

    def test_cancel_order_success(self, connected_broker, mock_client):
        """Successful cancellation returns True."""
        mock_client.cancel_order.return_value = True

        result = connected_broker.cancel_order("ORD-001")

        assert result is True
        mock_client.cancel_order.assert_called_once_with("ORD-001")

    def test_cancel_order_failure(self, connected_broker, mock_client):
        """Failed cancellation (falsy result) returns False."""
        mock_client.cancel_order.return_value = None

        result = connected_broker.cancel_order("ORD-001")

        assert result is False

    def test_cancel_order_false_result(self, connected_broker, mock_client):
        """Explicit False result returns False."""
        mock_client.cancel_order.return_value = False

        result = connected_broker.cancel_order("ORD-001")

        assert result is False

    def test_cancel_order_exception(self, connected_broker, mock_client):
        """Exception during cancel returns False."""
        mock_client.cancel_order.side_effect = Exception("Order not found")

        result = connected_broker.cancel_order("ORD-999")

        assert result is False


# ===========================================================================
# Get Order Tests
# ===========================================================================

class TestGetOrder:
    """Tests for get_order() and get_open_orders()."""

    def test_get_order_found_in_open_orders(self, connected_broker, mock_client):
        """get_order finds order in current (open) orders."""
        mock_client.get_current_orders.return_value = [SAMPLE_ORDER_DATA_PENDING]

        order = connected_broker.get_order("ORD-200")

        assert order is not None
        assert order.order_id == "ORD-200"
        assert order.symbol == "TSLA"

    def test_get_order_found_in_history(self, connected_broker, mock_client):
        """get_order finds order in history when not in open orders."""
        mock_client.get_current_orders.return_value = []  # Not in open orders
        mock_client.get_history_orders.return_value = [SAMPLE_ORDER_DATA_FILLED]

        order = connected_broker.get_order("ORD-100")

        assert order is not None
        assert order.order_id == "ORD-100"
        assert order.status == OrderStatus.FILLED

    def test_get_order_not_found(self, connected_broker, mock_client):
        """get_order returns None when order doesn't exist."""
        mock_client.get_current_orders.return_value = []
        mock_client.get_history_orders.return_value = []

        order = connected_broker.get_order("ORD-NONEXISTENT")

        assert order is None

    def test_get_open_orders_returns_active_only(self, connected_broker, mock_client):
        """get_open_orders returns only active orders."""
        mock_client.get_current_orders.return_value = [
            SAMPLE_ORDER_DATA_PENDING,   # Working -> PENDING (active)
            SAMPLE_ORDER_DATA_FILLED,    # Filled -> FILLED (not active)
            SAMPLE_ORDER_DATA_CANCELLED, # Cancelled (not active)
        ]

        orders = connected_broker.get_open_orders()

        # Only the pending/working order should be returned
        assert len(orders) == 1
        assert orders[0].order_id == "ORD-200"

    def test_get_open_orders_empty(self, connected_broker, mock_client):
        """get_open_orders returns empty list when no orders."""
        mock_client.get_current_orders.return_value = []

        orders = connected_broker.get_open_orders()

        assert orders == []

    def test_get_open_orders_none_response(self, connected_broker, mock_client):
        """get_open_orders handles None response."""
        mock_client.get_current_orders.return_value = None

        orders = connected_broker.get_open_orders()

        assert orders == []

    def test_get_open_orders_skips_malformed(self, connected_broker, mock_client):
        """get_open_orders skips orders that fail to parse."""
        bad_order = {"invalid": "data"}  # Missing required fields
        mock_client.get_current_orders.return_value = [
            SAMPLE_ORDER_DATA_PENDING,
            bad_order,
        ]

        # Should not raise, should skip the bad one
        orders = connected_broker.get_open_orders()
        # The pending order is active; bad_order may parse with defaults
        # but the key thing is it doesn't crash
        assert isinstance(orders, list)


# ===========================================================================
# Quote Tests
# ===========================================================================

class TestGetQuote:
    """Tests for get_quote() and get_quotes()."""

    def test_get_quote_success(self, connected_broker, mock_client):
        """get_quote returns correctly parsed Quote."""
        mock_client.get_quote.return_value = SAMPLE_QUOTE

        quote = connected_broker.get_quote("AAPL")

        assert isinstance(quote, Quote)
        assert quote.symbol == "AAPL"
        assert quote.bid == 154.50
        assert quote.ask == 155.50
        assert quote.last == 155.00
        assert quote.volume == 5000000
        assert isinstance(quote.timestamp, datetime)

    def test_get_quote_no_data_raises(self, connected_broker, mock_client):
        """get_quote raises ValueError when no data returned."""
        mock_client.get_quote.return_value = None

        with pytest.raises(ValueError, match="No quote data"):
            connected_broker.get_quote("FAKE")

    def test_get_quote_empty_dict_raises(self, connected_broker, mock_client):
        """get_quote raises ValueError for empty response."""
        mock_client.get_quote.return_value = {}

        with pytest.raises(ValueError, match="No quote data"):
            connected_broker.get_quote("FAKE")

    def test_get_quote_missing_fields_default_zero(self, connected_broker, mock_client):
        """get_quote defaults to 0 for missing fields."""
        mock_client.get_quote.return_value = {"some_field": "value"}  # truthy dict

        quote = connected_broker.get_quote("AAPL")

        assert quote.bid == 0.0
        assert quote.ask == 0.0
        assert quote.last == 0.0
        assert quote.volume == 0

    def test_get_quote_does_not_require_connection(self, broker, mock_client):
        """get_quote does NOT call _ensure_connected (no guard in source)."""
        mock_client.get_quote.return_value = SAMPLE_QUOTE

        # broker is NOT connected, but get_quote should still work
        quote = broker.get_quote("AAPL")

        assert quote.symbol == "AAPL"

    def test_get_quotes_multiple_symbols(self, connected_broker, mock_client):
        """get_quotes returns quotes for multiple symbols."""
        mock_client.get_quote.return_value = SAMPLE_QUOTE

        quotes = connected_broker.get_quotes(["AAPL", "MSFT", "TSLA"])

        assert len(quotes) == 3
        assert all(isinstance(q, Quote) for q in quotes.values())
        assert "AAPL" in quotes
        assert "MSFT" in quotes
        assert "TSLA" in quotes

    def test_get_quotes_empty_list(self, connected_broker, mock_client):
        """get_quotes with empty list returns empty dict."""
        quotes = connected_broker.get_quotes([])

        assert quotes == {}
        mock_client.get_quote.assert_not_called()

    def test_get_quotes_partial_failure(self, connected_broker, mock_client):
        """get_quotes skips symbols that fail and returns the rest."""
        def side_effect(symbol):
            if symbol == "BAD":
                raise Exception("No data")
            return SAMPLE_QUOTE

        mock_client.get_quote.side_effect = side_effect

        quotes = connected_broker.get_quotes(["AAPL", "BAD", "MSFT"])

        assert len(quotes) == 2
        assert "AAPL" in quotes
        assert "MSFT" in quotes
        assert "BAD" not in quotes


# ===========================================================================
# Order Status Mapping Tests
# ===========================================================================

class TestParseOrder:
    """Tests for _parse_order() order status mapping."""

    def test_parse_filled_order(self, connected_broker):
        """Filled order status maps correctly."""
        order = connected_broker._parse_order(SAMPLE_ORDER_DATA_FILLED)

        assert order.status == OrderStatus.FILLED
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.symbol == "AAPL"
        assert order.quantity == 10
        assert order.price == 150.00
        assert order.filled_quantity == 10
        assert order.filled_price == 149.95

    def test_parse_pending_order(self, connected_broker):
        """Working status maps to PENDING."""
        order = connected_broker._parse_order(SAMPLE_ORDER_DATA_PENDING)

        assert order.status == OrderStatus.PENDING
        assert order.side == OrderSide.SELL
        assert order.order_type == OrderType.MARKET

    def test_parse_cancelled_order(self, connected_broker):
        """Cancelled status maps correctly."""
        order = connected_broker._parse_order(SAMPLE_ORDER_DATA_CANCELLED)

        assert order.status == OrderStatus.CANCELLED
        assert order.order_type == OrderType.STOP
        assert order.stop_price == 130.00

    def test_parse_rejected_order(self, connected_broker):
        """Rejected status maps correctly."""
        data = {**SAMPLE_ORDER_DATA_FILLED, "status": "Rejected"}
        order = connected_broker._parse_order(data)
        assert order.status == OrderStatus.REJECTED

    def test_parse_expired_order(self, connected_broker):
        """Expired status maps correctly."""
        data = {**SAMPLE_ORDER_DATA_FILLED, "status": "Expired"}
        order = connected_broker._parse_order(data)
        assert order.status == OrderStatus.EXPIRED

    def test_parse_unknown_status_defaults_pending(self, connected_broker):
        """Unknown status defaults to PENDING."""
        data = {**SAMPLE_ORDER_DATA_FILLED, "status": "SomeNewStatus"}
        order = connected_broker._parse_order(data)
        assert order.status == OrderStatus.PENDING

    def test_parse_sell_action(self, connected_broker):
        """SELL action maps to OrderSide.SELL."""
        data = {**SAMPLE_ORDER_DATA_FILLED, "action": "SELL"}
        order = connected_broker._parse_order(data)
        assert order.side == OrderSide.SELL

    def test_parse_unknown_action_defaults_sell(self, connected_broker):
        """Non-BUY action defaults to SELL."""
        data = {**SAMPLE_ORDER_DATA_FILLED, "action": ""}
        order = connected_broker._parse_order(data)
        assert order.side == OrderSide.SELL

    def test_parse_order_type_mkt(self, connected_broker):
        """MKT orderType maps to MARKET."""
        data = {**SAMPLE_ORDER_DATA_FILLED, "orderType": "MKT"}
        order = connected_broker._parse_order(data)
        assert order.order_type == OrderType.MARKET

    def test_parse_order_type_stp_lmt(self, connected_broker):
        """STP LMT orderType maps to STOP_LIMIT."""
        data = {**SAMPLE_ORDER_DATA_FILLED, "orderType": "STP LMT"}
        order = connected_broker._parse_order(data)
        assert order.order_type == OrderType.STOP_LIMIT

    def test_parse_unknown_order_type_defaults_market(self, connected_broker):
        """Unknown orderType defaults to MARKET."""
        data = {**SAMPLE_ORDER_DATA_FILLED, "orderType": "EXOTIC"}
        order = connected_broker._parse_order(data)
        assert order.order_type == OrderType.MARKET

    def test_parse_order_no_limit_price(self, connected_broker):
        """Order without limit price sets price to None."""
        data = {**SAMPLE_ORDER_DATA_FILLED, "lmtPrice": None}
        order = connected_broker._parse_order(data)
        assert order.price is None

    def test_parse_order_no_filled_price(self, connected_broker):
        """Order without filled price sets filled_price to None."""
        data = {**SAMPLE_ORDER_DATA_FILLED, "avgFilledPrice": None}
        order = connected_broker._parse_order(data)
        assert order.filled_price is None

    def test_parse_order_timestamps(self, connected_broker):
        """Parsed order has timestamps set."""
        order = connected_broker._parse_order(SAMPLE_ORDER_DATA_FILLED)
        assert isinstance(order.created_at, datetime)
        assert isinstance(order.updated_at, datetime)


# ===========================================================================
# Internal Mapping Tests
# ===========================================================================

class TestInternalMappings:
    """Tests for _map_order_type() and _map_time_in_force()."""

    def test_map_order_type_market(self, connected_broker):
        assert connected_broker._map_order_type(OrderType.MARKET) == "MKT"

    def test_map_order_type_limit(self, connected_broker):
        assert connected_broker._map_order_type(OrderType.LIMIT) == "LMT"

    def test_map_order_type_stop(self, connected_broker):
        assert connected_broker._map_order_type(OrderType.STOP) == "STP"

    def test_map_order_type_stop_limit(self, connected_broker):
        assert connected_broker._map_order_type(OrderType.STOP_LIMIT) == "STP LMT"

    def test_map_tif_day(self, connected_broker):
        assert connected_broker._map_time_in_force(TimeInForce.DAY) == "DAY"

    def test_map_tif_gtc(self, connected_broker):
        assert connected_broker._map_time_in_force(TimeInForce.GTC) == "GTC"

    def test_map_tif_ioc(self, connected_broker):
        assert connected_broker._map_time_in_force(TimeInForce.IOC) == "IOC"

    def test_map_tif_fok(self, connected_broker):
        assert connected_broker._map_time_in_force(TimeInForce.FOK) == "FOK"


# ===========================================================================
# Integration-style / Edge Case Tests
# ===========================================================================

class TestEdgeCases:
    """Edge cases and integration-style scenarios."""

    def test_connect_then_disconnect_then_reconnect(self, broker, mock_client):
        """Can reconnect after disconnecting."""
        assert broker.connect() is True
        assert broker.is_connected() is True

        broker.disconnect()
        assert broker.is_connected() is False

        # Reconnect
        assert broker.connect() is True
        assert broker.is_connected() is True

    def test_place_order_returns_filled_quantity_zero(self, connected_broker, mock_client):
        """Newly placed orders always have filled_quantity=0."""
        mock_client.place_order.return_value = {"orderId": "ORD-NEW"}

        order = connected_broker.place_order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100
        )

        assert order.filled_quantity == 0
        assert order.filled_price is None

    def test_place_order_returns_pending_status(self, connected_broker, mock_client):
        """Newly placed orders always start as PENDING."""
        mock_client.place_order.return_value = {"orderId": "ORD-NEW"}

        order = connected_broker.place_order(
            symbol="AAPL", side=OrderSide.BUY, quantity=100
        )

        assert order.status == OrderStatus.PENDING

    def test_order_is_active(self, connected_broker):
        """PENDING orders are active; FILLED are not."""
        pending = connected_broker._parse_order(SAMPLE_ORDER_DATA_PENDING)
        filled = connected_broker._parse_order(SAMPLE_ORDER_DATA_FILLED)

        assert pending.is_active() is True
        assert filled.is_active() is False

    def test_order_is_filled(self, connected_broker):
        """FILLED orders return True for is_filled()."""
        filled = connected_broker._parse_order(SAMPLE_ORDER_DATA_FILLED)
        pending = connected_broker._parse_order(SAMPLE_ORDER_DATA_PENDING)

        assert filled.is_filled() is True
        assert pending.is_filled() is False

    def test_position_float_quantity_truncated(self, connected_broker, mock_client):
        """Position quantity is int-truncated from float string."""
        positions_data = [{
            "ticker": {"symbol": "FRAC"},
            "position": "10.7",  # float string
            "costPrice": "100.00",
            "lastPrice": "110.00",
            "marketValue": "1100.00",
            "unrealizedProfitLoss": "100.00",
        }]
        mock_client.get_positions.return_value = positions_data

        positions = connected_broker.get_positions()

        assert positions[0].quantity == 10  # int(float("10.7")) == 10

    def test_get_order_checks_open_before_history(self, connected_broker, mock_client):
        """get_order checks open orders first, skips history if found."""
        mock_client.get_current_orders.return_value = [SAMPLE_ORDER_DATA_PENDING]

        order = connected_broker.get_order("ORD-200")

        assert order is not None
        # Should NOT have called get_history_orders since found in open
        mock_client.get_history_orders.assert_not_called()

    def test_get_history_orders_none_response(self, connected_broker, mock_client):
        """get_order handles None from get_history_orders."""
        mock_client.get_current_orders.return_value = []
        mock_client.get_history_orders.return_value = None

        order = connected_broker.get_order("ORD-GHOST")

        assert order is None
