"""
Abstract base class for broker implementations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


class OrderSide(Enum):
    """Order side enum."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enum."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force enum."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


@dataclass
class Order:
    """Order data class."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: int
    filled_price: Optional[float]
    time_in_force: TimeInForce
    created_at: datetime
    updated_at: datetime

    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]


@dataclass
class Position:
    """Position data class."""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class AccountInfo:
    """Account information data class."""
    account_id: str
    cash: float
    buying_power: float
    portfolio_value: float
    day_trades_remaining: int
    positions_count: int


@dataclass
class Quote:
    """Real-time quote data class."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime


class BaseBroker(ABC):
    """
    Abstract base class for broker implementations.

    All broker implementations (WeBull, paper trading, etc.)
    should inherit from this class.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the broker.

        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to the broker."""
        pass

    @abstractmethod
    def get_account_info(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            AccountInfo with current account state.
        """
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Returns:
            List of Position objects.
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Position if exists, None otherwise.
        """
        pass

    @abstractmethod
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
        """
        Place a new order.

        Args:
            symbol: Stock ticker symbol.
            side: Buy or sell.
            quantity: Number of shares.
            order_type: Market, limit, stop, etc.
            price: Limit price (required for limit orders).
            stop_price: Stop price (required for stop orders).
            time_in_force: Order duration.

        Returns:
            Order object with order details.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancelled successfully.
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order ID.

        Returns:
            Order if exists, None otherwise.
        """
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """
        Get all open orders.

        Returns:
            List of open Order objects.
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Quote with current price data.
        """
        pass

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get real-time quotes for multiple symbols.

        Args:
            symbols: List of stock ticker symbols.

        Returns:
            Dictionary mapping symbol to Quote.
        """
        pass

    def buy(
        self,
        symbol: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None
    ) -> Order:
        """Convenience method to place a buy order."""
        return self.place_order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=order_type,
            price=price
        )

    def sell(
        self,
        symbol: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None
    ) -> Order:
        """Convenience method to place a sell order."""
        return self.place_order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=quantity,
            order_type=order_type,
            price=price
        )

    def close_position(self, symbol: str) -> Optional[Order]:
        """
        Close entire position for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Order if position closed, None if no position.
        """
        position = self.get_position(symbol)
        if position is None or position.quantity == 0:
            return None

        if position.is_long:
            return self.sell(symbol, abs(position.quantity))
        else:
            return self.buy(symbol, abs(position.quantity))
