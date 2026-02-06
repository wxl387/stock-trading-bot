"""
WeBull broker implementation.
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict
from webull import webull, paper_webull

from config.settings import settings
from .base_broker import (
    BaseBroker, Order, Position, AccountInfo, Quote,
    OrderSide, OrderType, OrderStatus, TimeInForce
)

logger = logging.getLogger(__name__)


class WebullBroker(BaseBroker):
    """
    WeBull broker implementation using the unofficial webull library.
    Supports both live and paper trading modes.
    """

    def __init__(self, paper_trading: bool = True):
        """
        Initialize WeBull broker.

        Args:
            paper_trading: If True, use paper trading account.
        """
        self.paper_trading = paper_trading
        self._connected = False
        self._account_id = None

        # Initialize appropriate client
        if paper_trading:
            self.client = paper_webull()
            logger.info("Initialized WeBull paper trading client")
        else:
            self.client = webull()
            logger.info("Initialized WeBull live trading client")

    def connect(self) -> bool:
        """
        Connect to WeBull and authenticate.

        Returns:
            True if connection successful.
        """
        try:
            # Login with credentials
            login_result = self.client.login(
                username=settings.WEBULL_EMAIL,
                password=settings.WEBULL_PASSWORD
            )

            if login_result is None:
                logger.error("WeBull login failed - check credentials")
                return False

            # Handle MFA if required
            # Note: MFA code would need to be obtained interactively
            # For automated trading, device ID should be pre-configured

            # Get trade token for order placement
            self.client.get_trade_token(settings.WEBULL_TRADE_PIN)

            # Get account ID
            self._account_id = self.client.get_account_id()
            self._connected = True

            logger.info(f"Connected to WeBull {'paper' if self.paper_trading else 'live'} account")
            return True

        except Exception as e:
            logger.error(f"WeBull connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from WeBull."""
        try:
            self.client.logout()
            self._connected = False
            logger.info("Disconnected from WeBull")
        except Exception as e:
            logger.warning(f"Error during disconnect: {e}")

    def is_connected(self) -> bool:
        """Check if connected to WeBull."""
        return self._connected

    def get_account_info(self) -> AccountInfo:
        """Get account information."""
        self._ensure_connected()

        account = self.client.get_account()
        if not account or not isinstance(account, dict):
            raise ConnectionError("WeBull get_account() returned invalid data")

        try:
            positions_count = len(self.client.get_positions() or [])
        except Exception as e:
            logger.warning(f"Failed to fetch positions count: {e}")
            positions_count = 0

        return AccountInfo(
            account_id=str(self._account_id),
            cash=float(account.get("cashBalance", 0)),
            buying_power=float(account.get("dayBuyingPower", 0)),
            portfolio_value=float(account.get("totalMarketValue", 0)),
            day_trades_remaining=int(account.get("dayTradesRemaining", 0)),
            positions_count=positions_count
        )

    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        self._ensure_connected()

        positions_data = self.client.get_positions() or []
        positions = []

        for pos in positions_data:
            try:
                quantity = int(float(pos.get("position", 0)))
                avg_cost = float(pos.get("costPrice", 0))
                current_price = float(pos.get("lastPrice", 0))
                market_value = float(pos.get("marketValue", 0))
                unrealized_pnl = float(pos.get("unrealizedProfitLoss", 0))

                positions.append(Position(
                    symbol=pos.get("ticker", {}).get("symbol", ""),
                    quantity=quantity,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=(unrealized_pnl / (avg_cost * quantity) * 100) if avg_cost * quantity != 0 else 0,
                    realized_pnl=0  # Would need separate tracking
                ))
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Error parsing position: {e}")
                continue

        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        positions = self.get_positions()
        for pos in positions:
            if pos.symbol.upper() == symbol.upper():
                return pos
        return None

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Order:
        """Place a new order."""
        self._ensure_connected()

        # Map order type to WeBull format
        wb_order_type = self._map_order_type(order_type)
        wb_action = "BUY" if side == OrderSide.BUY else "SELL"
        wb_enforce = self._map_time_in_force(time_in_force)

        try:
            # Place the order
            if order_type == OrderType.MARKET:
                result = self.client.place_order(
                    stock=symbol,
                    action=wb_action,
                    orderType="MKT",
                    enforce=wb_enforce,
                    qty=quantity
                )
            elif order_type == OrderType.LIMIT:
                if price is None:
                    raise ValueError("Limit price required for limit orders")
                result = self.client.place_order(
                    stock=symbol,
                    action=wb_action,
                    orderType="LMT",
                    enforce=wb_enforce,
                    qty=quantity,
                    price=price
                )
            elif order_type == OrderType.STOP:
                if stop_price is None:
                    raise ValueError("Stop price required for stop orders")
                result = self.client.place_order(
                    stock=symbol,
                    action=wb_action,
                    orderType="STP",
                    enforce=wb_enforce,
                    qty=quantity,
                    stpPrice=stop_price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")

            if result is None:
                raise Exception("Order placement returned None")

            # Parse order result
            order_id = str(result.get("orderId", ""))

            logger.info(f"Placed {side.value} order for {quantity} {symbol}: {order_id}")

            return Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                filled_quantity=0,
                filled_price=None,
                time_in_force=time_in_force,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        self._ensure_connected()

        try:
            result = self.client.cancel_order(order_id)
            if result:
                logger.info(f"Cancelled order: {order_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        self._ensure_connected()

        # Check in open orders first
        open_orders = self.get_open_orders()
        for order in open_orders:
            if order.order_id == order_id:
                return order

        # Check in history
        history = self.client.get_history_orders() or []
        for order_data in history:
            if str(order_data.get("orderId")) == order_id:
                return self._parse_order(order_data)

        return None

    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        self._ensure_connected()

        orders_data = self.client.get_current_orders() or []
        orders = []

        for order_data in orders_data:
            try:
                order = self._parse_order(order_data)
                if order.is_active():
                    orders.append(order)
            except Exception as e:
                logger.warning(f"Error parsing order: {e}")

        return orders

    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote for a symbol."""
        quote_data = self.client.get_quote(symbol)

        if not quote_data:
            raise ValueError(f"No quote data for {symbol}")

        return Quote(
            symbol=symbol,
            bid=float(quote_data.get("bid", 0)),
            ask=float(quote_data.get("ask", 0)),
            last=float(quote_data.get("close", 0)),
            volume=int(quote_data.get("volume", 0)),
            timestamp=datetime.now()
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get real-time quotes for multiple symbols."""
        quotes = {}
        for symbol in symbols:
            try:
                quotes[symbol] = self.get_quote(symbol)
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
        return quotes

    def _ensure_connected(self):
        """Ensure broker is connected."""
        if not self._connected:
            raise ConnectionError("Not connected to WeBull. Call connect() first.")

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map OrderType to WeBull format."""
        mapping = {
            OrderType.MARKET: "MKT",
            OrderType.LIMIT: "LMT",
            OrderType.STOP: "STP",
            OrderType.STOP_LIMIT: "STP LMT"
        }
        return mapping.get(order_type, "MKT")

    def _map_time_in_force(self, tif: TimeInForce) -> str:
        """Map TimeInForce to WeBull format."""
        mapping = {
            TimeInForce.DAY: "DAY",
            TimeInForce.GTC: "GTC",
            TimeInForce.IOC: "IOC",
            TimeInForce.FOK: "FOK"
        }
        return mapping.get(tif, "DAY")

    def _parse_order(self, order_data: dict) -> Order:
        """Parse WeBull order data to Order object."""
        # Map status
        status_str = order_data.get("status", "").lower()
        status_mapping = {
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
            "pending": OrderStatus.PENDING,
            "working": OrderStatus.PENDING,
        }
        status = status_mapping.get(status_str, OrderStatus.PENDING)

        # Map side
        action = order_data.get("action", "").upper()
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL

        # Map order type
        order_type_str = order_data.get("orderType", "MKT")
        order_type_mapping = {
            "MKT": OrderType.MARKET,
            "LMT": OrderType.LIMIT,
            "STP": OrderType.STOP,
            "STP LMT": OrderType.STOP_LIMIT
        }
        order_type = order_type_mapping.get(order_type_str, OrderType.MARKET)

        return Order(
            order_id=str(order_data.get("orderId", "")),
            symbol=order_data.get("ticker", {}).get("symbol", ""),
            side=side,
            order_type=order_type,
            quantity=int(float(order_data.get("totalQuantity", 0))),
            price=float(order_data.get("lmtPrice", 0)) if order_data.get("lmtPrice") else None,
            stop_price=float(order_data.get("stpPrice", 0)) if order_data.get("stpPrice") else None,
            status=status,
            filled_quantity=int(float(order_data.get("filledQuantity", 0))),
            filled_price=float(order_data.get("avgFilledPrice", 0)) if order_data.get("avgFilledPrice") else None,
            time_in_force=TimeInForce.DAY,  # Default
            created_at=datetime.now(),  # Would need parsing
            updated_at=datetime.now()
        )
