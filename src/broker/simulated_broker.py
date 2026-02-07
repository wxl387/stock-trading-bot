"""
Simulated broker for offline paper trading without real broker credentials.
"""
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import yfinance as yf

from src.broker.base_broker import (
    BaseBroker, Order, Position, AccountInfo, Quote,
    OrderSide, OrderType, OrderStatus, TimeInForce
)
from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


class SimulatedBroker(BaseBroker):
    """
    Simulated broker for offline paper trading.

    Uses yfinance for real-time price data and simulates
    order execution, position tracking, and P&L calculation.
    """

    STATE_FILE = DATA_DIR / "simulated_broker_state.json"

    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize simulated broker.

        Args:
            initial_capital: Starting cash balance.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position data
        self.orders: Dict[str, Order] = {}  # order_id -> Order
        self.trades: List[Dict[str, Any]] = []  # Trade history
        self.realized_pnl = 0.0
        self._connected = False
        self._account_id = f"SIM-{uuid.uuid4().hex[:8].upper()}"

        # Try to load existing state
        self._load_state()

        logger.info(f"Initialized SimulatedBroker with ${initial_capital:,.2f} capital")

    def connect(self) -> bool:
        """Connect to simulated broker (always succeeds)."""
        self._connected = True
        logger.info("SimulatedBroker connected (offline mode)")
        return True

    def disconnect(self) -> None:
        """Disconnect and save state."""
        self._save_state()
        self._connected = False
        logger.info("SimulatedBroker disconnected, state saved")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def get_account_info(self) -> AccountInfo:
        """Get simulated account information."""
        portfolio_value = self.cash + self._get_positions_value()

        return AccountInfo(
            account_id=self._account_id,
            cash=self.cash,
            buying_power=self.cash,  # Simplified - no margin
            portfolio_value=portfolio_value,
            day_trades_remaining=3,  # PDT rule simulation
            positions_count=len([p for p in self.positions.values() if p["quantity"] != 0])
        )

    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        positions = []
        for symbol, pos_data in self.positions.items():
            if pos_data["quantity"] != 0:
                try:
                    current_price = self._get_current_price(symbol)
                except ValueError as e:
                    logger.warning(f"Skipping position {symbol}: {e}")
                    continue
                positions.append(self._make_position(symbol, pos_data, current_price))
        return positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        if symbol not in self.positions or self.positions[symbol]["quantity"] == 0:
            return None

        pos_data = self.positions[symbol]
        try:
            current_price = self._get_current_price(symbol)
        except ValueError as e:
            logger.warning(f"Cannot get position {symbol}: {e}")
            return None
        return self._make_position(symbol, pos_data, current_price)

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
        Place and immediately execute a simulated order.

        Market orders are filled immediately at current price.
        Limit orders are also filled immediately for simplicity.
        """
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        current_price = self._get_current_price(symbol)

        # Determine fill price
        if order_type == OrderType.MARKET:
            fill_price = current_price
        elif order_type == OrderType.LIMIT:
            fill_price = price if price else current_price
        else:
            fill_price = current_price

        # Check if we have enough cash for buy orders
        if side == OrderSide.BUY:
            required_cash = fill_price * quantity
            if required_cash > self.cash:
                # Reject order - insufficient funds
                order = Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    filled_price=None,
                    time_in_force=time_in_force,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                self.orders[order_id] = order
                logger.warning(f"Order rejected: insufficient funds. Need ${required_cash:,.2f}, have ${self.cash:,.2f}")
                return order

        # Check if we have enough shares for sell orders
        if side == OrderSide.SELL:
            current_qty = self.positions.get(symbol, {}).get("quantity", 0)
            if current_qty < quantity:
                order = Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0,
                    filled_price=None,
                    time_in_force=time_in_force,
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                self.orders[order_id] = order
                logger.warning(f"Order rejected: insufficient shares. Have {current_qty}, trying to sell {quantity}")
                return order

        # Execute order
        self._execute_fill(symbol, side, quantity, fill_price)

        # Create filled order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=fill_price,
            time_in_force=time_in_force,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        self.orders[order_id] = order

        # Cap orders dict to prevent unbounded growth
        if len(self.orders) > 1000:
            sorted_ids = sorted(self.orders.keys(), key=lambda k: self.orders[k].created_at)
            for old_id in sorted_ids[:500]:
                del self.orders[old_id]

        # Record trade (cap in-memory list to prevent unbounded growth)
        self.trades.append({
            "order_id": order_id,
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "price": fill_price,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.trades) > 1000:
            self.trades = self.trades[-500:]

        # Save state after trade
        self._save_state()

        logger.info(f"Order filled: {side.value} {quantity} {symbol} @ ${fill_price:.2f}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order (simulated orders are filled immediately, so this is a no-op)."""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        """Get all open orders (usually empty since orders fill immediately)."""
        return [o for o in self.orders.values() if o.is_active()]

    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote using yfinance."""
        price = self._get_current_price(symbol)

        return Quote(
            symbol=symbol,
            bid=price * 0.999,  # Simulate spread
            ask=price * 1.001,
            last=price,
            volume=0,
            timestamp=datetime.now()
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """Get quotes for multiple symbols."""
        quotes = {}
        for symbol in symbols:
            try:
                quotes[symbol] = self.get_quote(symbol)
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
        return quotes

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get a summary of portfolio performance."""
        portfolio_value = self.cash + self._get_positions_value()
        total_pnl = portfolio_value - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0

        return {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "positions_value": self._get_positions_value(),
            "portfolio_value": portfolio_value,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self._get_unrealized_pnl(),
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "total_trades": len(self.trades),
            "open_positions": len([p for p in self.positions.values() if p["quantity"] > 0])
        }

    def reset(self) -> None:
        """Reset broker to initial state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.orders = {}
        self.trades = []
        self.realized_pnl = 0.0
        self._save_state()
        logger.info("SimulatedBroker reset to initial state")

    # Private methods

    def _get_current_price(self, symbol: str) -> float:
        """Get current price from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")

        # Fallback: use last known price from position
        if symbol in self.positions:
            last = self.positions[symbol].get("last_price")
            if last and last > 0:
                logger.warning(f"Using last known price for {symbol}: ${last:.2f}")
                return last
            avg = self.positions[symbol].get("avg_cost")
            if avg and avg > 0:
                logger.warning(f"Using avg cost as fallback price for {symbol}: ${avg:.2f}")
                return avg
        logger.error(f"No price available for {symbol}, cannot execute order")
        raise ValueError(f"No price available for {symbol}")

    def _execute_fill(self, symbol: str, side: OrderSide, quantity: int, price: float) -> None:
        """Execute a fill and update positions/cash."""
        if symbol not in self.positions:
            self.positions[symbol] = {
                "quantity": 0,
                "avg_cost": 0.0,
                "realized_pnl": 0.0,
                "last_price": price
            }

        pos = self.positions[symbol]

        if side == OrderSide.BUY:
            # Calculate new average cost
            total_cost = pos["avg_cost"] * pos["quantity"] + price * quantity
            new_quantity = pos["quantity"] + quantity
            pos["avg_cost"] = total_cost / new_quantity if new_quantity > 0 else 0
            pos["quantity"] = new_quantity
            pos["last_price"] = price

            # Deduct cash
            self.cash -= price * quantity

        else:  # SELL
            # Calculate realized P&L
            cost_basis = pos["avg_cost"] * quantity
            proceeds = price * quantity
            trade_pnl = proceeds - cost_basis

            pos["realized_pnl"] += trade_pnl
            self.realized_pnl += trade_pnl
            pos["quantity"] -= quantity

            # Add cash
            self.cash += proceeds

            pos["last_price"] = price

            # Remove position entry if fully closed
            if pos["quantity"] == 0:
                del self.positions[symbol]

    def _get_positions_value(self) -> float:
        """Get total value of all positions."""
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos["quantity"] > 0:
                current_price = self._get_current_price(symbol)
                total += current_price * pos["quantity"]
        return total

    def _get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos["quantity"] > 0:
                current_price = self._get_current_price(symbol)
                cost_basis = pos["avg_cost"] * pos["quantity"]
                market_value = current_price * pos["quantity"]
                total += market_value - cost_basis
        return total

    def _make_position(self, symbol: str, pos_data: Dict, current_price: float) -> Position:
        """Create Position object from position data."""
        quantity = pos_data["quantity"]
        avg_cost = pos_data["avg_cost"]
        market_value = current_price * quantity
        cost_basis = avg_cost * quantity
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

        return Position(
            symbol=symbol,
            quantity=quantity,
            avg_cost=avg_cost,
            current_price=current_price,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            realized_pnl=pos_data.get("realized_pnl", 0.0)
        )

    def _save_state(self) -> None:
        """Save broker state to JSON file."""
        try:
            DATA_DIR.mkdir(exist_ok=True)

            state = {
                "account_id": self._account_id,
                "initial_capital": self.initial_capital,
                "cash": self.cash,
                "positions": self.positions,
                "realized_pnl": self.realized_pnl,
                "trades": self.trades[-100:],  # Keep last 100 trades
                "saved_at": datetime.now().isoformat()
            }

            fd, tmp_path = tempfile.mkstemp(
                dir=str(DATA_DIR), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp_path, self.STATE_FILE)
            except BaseException:
                os.unlink(tmp_path)
                raise

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _load_state(self) -> None:
        """Load broker state from JSON file."""
        try:
            if self.STATE_FILE.exists():
                with open(self.STATE_FILE, "r") as f:
                    state = json.load(f)

                self._account_id = state.get("account_id", self._account_id)
                self.initial_capital = state.get("initial_capital", self.initial_capital)
                self.cash = state.get("cash", self.initial_capital)
                self.positions = state.get("positions", {})
                self.realized_pnl = state.get("realized_pnl", 0.0)
                self.trades = state.get("trades", [])

                logger.info(f"Loaded state from {self.STATE_FILE}")
                logger.info(f"  Cash: ${self.cash:,.2f}, Positions: {len(self.positions)}")

        except Exception as e:
            logger.error(f"Error loading state: {e}")
