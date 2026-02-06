"""
Risk management module.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, TYPE_CHECKING
from enum import Enum
from datetime import datetime, timedelta

from config.settings import settings

logger = logging.getLogger(__name__)


# VIX-based position sizing thresholds
DEFAULT_VIX_THRESHOLDS = {
    "low": 15,      # VIX < 15: calm market
    "normal": 25,   # VIX 15-25: normal volatility
    "high": 35,     # VIX 25-35: high volatility
    "extreme": 50   # VIX > 35: extreme volatility
}

DEFAULT_VIX_MULTIPLIERS = {
    "low": 1.2,     # Increase size in calm markets
    "normal": 1.0,  # Normal sizing
    "high": 0.7,    # Reduce size in high vol
    "extreme": 0.5  # Half size in extreme vol
}


class StopLossType(Enum):
    """Stop loss type enum."""
    FIXED = "fixed"
    ATR = "atr"
    TRAILING = "trailing"


@dataclass
class StopLoss:
    """Stop loss data class."""
    symbol: str
    stop_type: StopLossType
    stop_price: float
    entry_price: float
    trailing_distance: Optional[float] = None


@dataclass
class TakeProfitLevel:
    """Single take-profit level."""
    target_price: float
    target_pct: float  # Percentage gain (e.g., 0.05 for 5%)
    exit_pct: float    # Percentage of position to exit (e.g., 0.5 for 50%)
    triggered: bool = False


@dataclass
class TakeProfitOrder:
    """Take-profit order with multiple levels."""
    symbol: str
    entry_price: float
    total_quantity: int
    levels: List[TakeProfitLevel]
    quantity_remaining: int = 0

    def __post_init__(self):
        if self.quantity_remaining == 0:
            self.quantity_remaining = self.total_quantity


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    approved: bool
    reason: str
    adjusted_quantity: Optional[int] = None


class RiskManager:
    """
    Manages trading risk including position sizing, stop losses, and exposure limits.
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_daily_loss_pct: float = 0.05,
        max_total_exposure: float = 0.80,
        max_sector_exposure: float = 0.30,
        max_positions: int = 20,
        max_daily_trades: int = 50,
        pause_after_consecutive_losses: int = 3
    ):
        """
        Initialize RiskManager.

        Args:
            max_position_pct: Maximum portfolio percentage per position.
            max_daily_loss_pct: Maximum daily loss percentage.
            max_total_exposure: Maximum total portfolio exposure.
            max_sector_exposure: Maximum sector exposure.
            max_positions: Maximum number of positions.
            max_daily_trades: Maximum trades per day.
            pause_after_consecutive_losses: Pause after N consecutive losses.
        """
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_total_exposure = max_total_exposure
        self.max_sector_exposure = max_sector_exposure
        self.max_positions = max_positions
        self.max_daily_trades = max_daily_trades
        self.pause_after_consecutive_losses = pause_after_consecutive_losses

        # Daily tracking
        self.daily_pnl = 0.0
        self.starting_equity = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.trading_halted = False

        # Stop losses
        self.stop_losses: Dict[str, StopLoss] = {}

        # Take profits
        self.take_profits: Dict[str, TakeProfitOrder] = {}

        # VIX-based sizing
        self.vix_sizing_enabled = True
        self.vix_thresholds = DEFAULT_VIX_THRESHOLDS.copy()
        self.vix_multipliers = DEFAULT_VIX_MULTIPLIERS.copy()
        self._cached_vix: Optional[float] = None
        self._vix_cache_time: Optional[datetime] = None
        self._vix_cache_ttl = timedelta(hours=1)  # Cache VIX for 1 hour

        # Max drawdown protection
        self.max_drawdown_pct = 0.10  # 10% max drawdown
        self.peak_portfolio_value = 0.0
        self.current_drawdown_pct = 0.0
        self.drawdown_recovery_mode = False
        self.recovery_threshold = 0.95  # Exit recovery when at 95% of peak
        self.recovery_size_multiplier = 0.5  # 50% position size in recovery

    def reset_daily_limits(self, current_equity: float) -> None:
        """
        Reset daily tracking (call at market open).

        Args:
            current_equity: Current portfolio value.
        """
        self.daily_pnl = 0.0
        self.starting_equity = current_equity
        self.daily_trades = 0
        self.trading_halted = False

        # Seed peak_portfolio_value on first call so drawdown tracking is correct
        if self.peak_portfolio_value == 0.0 and current_equity > 0:
            self.peak_portfolio_value = current_equity
            logger.info(f"Peak portfolio value initialized to ${current_equity:,.2f}")

        logger.info(f"Daily limits reset. Starting equity: ${current_equity:,.2f}")

    def check_can_trade(self) -> RiskCheckResult:
        """
        Check if trading is allowed.

        Returns:
            RiskCheckResult with approval status.
        """
        if self.trading_halted:
            return RiskCheckResult(
                approved=False,
                reason="Trading halted due to daily loss limit"
            )

        if self.consecutive_losses >= self.pause_after_consecutive_losses:
            return RiskCheckResult(
                approved=False,
                reason=f"Paused after {self.consecutive_losses} consecutive losses"
            )

        if self.daily_trades >= self.max_daily_trades:
            return RiskCheckResult(
                approved=False,
                reason=f"Daily trade limit reached ({self.max_daily_trades})"
            )

        return RiskCheckResult(approved=True, reason="OK")

    def check_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, float],
        sector: Optional[str] = None
    ) -> RiskCheckResult:
        """
        Check if a new position passes risk checks.

        Args:
            symbol: Stock symbol.
            quantity: Number of shares.
            price: Entry price.
            portfolio_value: Total portfolio value.
            current_positions: Dict of symbol -> market value.
            sector: Optional sector for concentration check.

        Returns:
            RiskCheckResult with approval status.
        """
        # Check if trading allowed
        trade_check = self.check_can_trade()
        if not trade_check.approved:
            return trade_check

        position_value = quantity * price

        # Guard against zero portfolio value
        if portfolio_value <= 0:
            return RiskCheckResult(
                approved=False,
                reason=f"Invalid portfolio value: ${portfolio_value:,.2f}",
                adjusted_quantity=0
            )

        # Check position size
        position_pct = position_value / portfolio_value
        if position_pct > self.max_position_pct:
            max_shares = int((portfolio_value * self.max_position_pct) / price)
            return RiskCheckResult(
                approved=False,
                reason=f"Position size {position_pct:.1%} exceeds max {self.max_position_pct:.1%}",
                adjusted_quantity=max_shares
            )

        # Check total exposure
        total_invested = sum(current_positions.values()) + position_value
        exposure_pct = total_invested / portfolio_value
        if exposure_pct > self.max_total_exposure:
            return RiskCheckResult(
                approved=False,
                reason=f"Total exposure {exposure_pct:.1%} would exceed max {self.max_total_exposure:.1%}"
            )

        # Check number of positions
        if len(current_positions) >= self.max_positions and symbol not in current_positions:
            return RiskCheckResult(
                approved=False,
                reason=f"Maximum positions ({self.max_positions}) reached"
            )

        return RiskCheckResult(approved=True, reason="OK")

    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        risk_per_trade: float = 0.02,
        apply_vix_sizing: bool = True
    ) -> int:
        """
        Calculate position size based on risk and VIX volatility.

        Args:
            portfolio_value: Total portfolio value.
            entry_price: Entry price per share.
            stop_loss_price: Stop loss price.
            risk_per_trade: Risk per trade as fraction of portfolio.
            apply_vix_sizing: Whether to apply VIX-based multiplier.

        Returns:
            Number of shares to buy.
        """
        # Maximum position based on portfolio percentage
        max_position_value = portfolio_value * self.max_position_pct
        max_shares_by_value = int(max_position_value / entry_price)

        # If no stop loss, use maximum position
        if stop_loss_price is None:
            base_shares = max_shares_by_value
        else:
            # Position size based on risk per trade
            risk_amount = portfolio_value * risk_per_trade
            risk_per_share = abs(entry_price - stop_loss_price)

            if risk_per_share == 0:
                base_shares = max_shares_by_value
            else:
                shares_by_risk = int(risk_amount / risk_per_share)
                base_shares = min(max_shares_by_value, shares_by_risk)

        # Apply VIX-based multiplier
        if apply_vix_sizing and self.vix_sizing_enabled:
            multiplier = self.calculate_volatility_multiplier()
            adjusted_shares = int(base_shares * multiplier)
            if adjusted_shares != base_shares:
                logger.debug(f"VIX multiplier {multiplier:.2f} applied: {base_shares} -> {adjusted_shares} shares")
            return max(1, adjusted_shares) if base_shares > 0 else 0

        return base_shares

    def get_current_vix(self, use_cache: bool = True) -> Optional[float]:
        """
        Get current VIX value.

        Args:
            use_cache: Whether to use cached value.

        Returns:
            Current VIX value or None if unavailable.
        """
        # Check cache
        if use_cache and self._cached_vix is not None and self._vix_cache_time is not None:
            if datetime.now() - self._vix_cache_time < self._vix_cache_ttl:
                return self._cached_vix

        try:
            from src.data.macro_fetcher import get_macro_fetcher
            fetcher = get_macro_fetcher()

            vix_df = fetcher.fetch_indicator(
                "vix",
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now(),
                use_cache=use_cache
            )

            if not vix_df.empty:
                vix_value = float(vix_df.iloc[-1].iloc[0])
                self._cached_vix = vix_value
                self._vix_cache_time = datetime.now()
                logger.info(f"Current VIX: {vix_value:.2f}")
                return vix_value

        except Exception as e:
            logger.warning(f"Failed to fetch VIX: {e}")

        # Return cached value only if not too stale (max 4 hours)
        if self._cached_vix is not None and self._vix_cache_time is not None:
            staleness = datetime.now() - self._vix_cache_time
            if staleness < timedelta(hours=4):
                logger.info(f"Using stale VIX cache ({staleness.total_seconds()/3600:.1f}h old): {self._cached_vix:.2f}")
                return self._cached_vix
            else:
                logger.warning(f"VIX cache too stale ({staleness.total_seconds()/3600:.1f}h old), discarding")
                return None

        return None

    def calculate_volatility_multiplier(self, vix: Optional[float] = None) -> float:
        """
        Calculate position size multiplier based on VIX.

        Args:
            vix: VIX value (fetched if not provided).

        Returns:
            Position size multiplier (0.5 to 1.2).
        """
        if vix is None:
            vix = self.get_current_vix()

        if vix is None:
            return 1.0  # Default to normal sizing if VIX unavailable

        thresholds = self.vix_thresholds
        multipliers = self.vix_multipliers

        if vix < thresholds["low"]:
            return multipliers["low"]
        elif vix < thresholds["normal"]:
            return multipliers["normal"]
        elif vix < thresholds["high"]:
            return multipliers["high"]
        else:
            return multipliers["extreme"]

    def set_vix_sizing(
        self,
        enabled: bool = True,
        thresholds: Optional[Dict[str, float]] = None,
        multipliers: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Configure VIX-based position sizing.

        Args:
            enabled: Whether to enable VIX sizing.
            thresholds: Custom VIX thresholds.
            multipliers: Custom position multipliers.
        """
        self.vix_sizing_enabled = enabled

        if thresholds:
            self.vix_thresholds.update(thresholds)
        if multipliers:
            self.vix_multipliers.update(multipliers)

        logger.info(f"VIX sizing {'enabled' if enabled else 'disabled'}")

    def update_portfolio_value(self, current_value: float) -> Dict:
        """
        Update portfolio value and check drawdown.

        Args:
            current_value: Current portfolio value.

        Returns:
            Dict with drawdown status.
        """
        # Update peak
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value

            # Check if we can exit recovery mode
            if self.drawdown_recovery_mode:
                logger.info("Drawdown recovery: new peak reached, exiting recovery mode")
                self.drawdown_recovery_mode = False

        # Calculate current drawdown
        if self.peak_portfolio_value > 0:
            self.current_drawdown_pct = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value
        else:
            self.current_drawdown_pct = 0.0

        # Check if we should enter recovery mode
        if self.current_drawdown_pct >= self.max_drawdown_pct and not self.drawdown_recovery_mode:
            self.drawdown_recovery_mode = True
            logger.warning(f"Max drawdown {self.max_drawdown_pct:.0%} breached! "
                          f"Current drawdown: {self.current_drawdown_pct:.1%}. Entering recovery mode.")

        # Check if we can exit recovery mode (recovered to threshold)
        if self.drawdown_recovery_mode and self.peak_portfolio_value > 0:
            recovery_target = self.peak_portfolio_value * self.recovery_threshold
            if current_value >= recovery_target:
                logger.info(f"Portfolio recovered to {self.recovery_threshold:.0%} of peak. Exiting recovery mode.")
                self.drawdown_recovery_mode = False

        return {
            "peak_value": self.peak_portfolio_value,
            "current_value": current_value,
            "drawdown_pct": self.current_drawdown_pct,
            "recovery_mode": self.drawdown_recovery_mode,
            "max_drawdown_breached": self.current_drawdown_pct >= self.max_drawdown_pct
        }

    def check_drawdown(self, portfolio_value: float) -> RiskCheckResult:
        """
        Check if trading is allowed based on drawdown.

        Args:
            portfolio_value: Current portfolio value.

        Returns:
            RiskCheckResult with approval status.
        """
        # Update drawdown tracking
        self.update_portfolio_value(portfolio_value)

        # If max drawdown breached, block new trades
        if self.current_drawdown_pct >= self.max_drawdown_pct:
            return RiskCheckResult(
                approved=False,
                reason=f"Max drawdown {self.max_drawdown_pct:.0%} breached. "
                       f"Current: {self.current_drawdown_pct:.1%}"
            )

        return RiskCheckResult(approved=True, reason="OK")

    def get_drawdown_adjusted_size(self, base_shares: int) -> int:
        """
        Adjust position size based on recovery mode.

        Args:
            base_shares: Original position size.

        Returns:
            Adjusted position size.
        """
        if self.drawdown_recovery_mode:
            adjusted = int(base_shares * self.recovery_size_multiplier)
            logger.debug(f"Recovery mode: adjusted size from {base_shares} to {adjusted}")
            return max(1, adjusted)
        return base_shares

    def set_drawdown_protection(
        self,
        max_drawdown_pct: float = 0.10,
        recovery_threshold: float = 0.95,
        recovery_size_multiplier: float = 0.5
    ) -> None:
        """
        Configure drawdown protection parameters.

        Args:
            max_drawdown_pct: Maximum allowed drawdown (e.g., 0.10 for 10%).
            recovery_threshold: Portfolio recovery level to exit recovery mode.
            recovery_size_multiplier: Position size multiplier in recovery mode.
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.recovery_threshold = recovery_threshold
        self.recovery_size_multiplier = recovery_size_multiplier

        logger.info(f"Drawdown protection set: max {max_drawdown_pct:.0%}, "
                   f"recovery at {recovery_threshold:.0%}, "
                   f"recovery sizing {recovery_size_multiplier:.0%}")

    def calculate_stop_loss(
        self,
        entry_price: float,
        stop_type: StopLossType = StopLossType.FIXED,
        fixed_pct: float = 0.05,
        atr: Optional[float] = None,
        atr_multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price.
            stop_type: Type of stop loss.
            fixed_pct: Percentage for fixed stop loss.
            atr: Average True Range value.
            atr_multiplier: Multiplier for ATR-based stop.

        Returns:
            Stop loss price.
        """
        if stop_type == StopLossType.FIXED:
            stop_price = entry_price * (1 - fixed_pct)

        elif stop_type == StopLossType.ATR:
            if atr is None or atr <= 0:
                logger.warning("ATR not provided or invalid, using fixed stop loss")
                stop_price = entry_price * (1 - fixed_pct)
            else:
                stop_price = entry_price - (atr * atr_multiplier)

        elif stop_type == StopLossType.TRAILING:
            stop_price = entry_price * (1 - fixed_pct)

        else:
            stop_price = entry_price * (1 - fixed_pct)

        # Validate stop is below entry price (for long positions)
        if stop_price >= entry_price:
            fallback = entry_price * 0.95  # 5% default
            logger.warning(f"Invalid stop ${stop_price:.2f} >= entry ${entry_price:.2f}, clamping to ${fallback:.2f}")
            stop_price = fallback

        # Prevent negative stop prices (can happen with large ATR * multiplier)
        if stop_price <= 0:
            stop_price = entry_price * 0.95
            logger.warning(f"Negative stop price, clamping to ${stop_price:.2f}")

        return stop_price

    def set_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        stop_type: StopLossType = StopLossType.FIXED,
        stop_price: Optional[float] = None,
        atr: Optional[float] = None
    ) -> StopLoss:
        """
        Set stop loss for a position.

        Args:
            symbol: Stock symbol.
            entry_price: Entry price.
            stop_type: Type of stop loss.
            stop_price: Explicit stop price (optional).
            atr: ATR value for ATR-based stops.

        Returns:
            StopLoss object.
        """
        if stop_price is None:
            stop_price = self.calculate_stop_loss(
                entry_price=entry_price,
                stop_type=stop_type,
                atr=atr
            )

        trailing_distance = None
        if stop_type == StopLossType.TRAILING:
            trailing_distance = entry_price - stop_price

        stop_loss = StopLoss(
            symbol=symbol,
            stop_type=stop_type,
            stop_price=stop_price,
            entry_price=entry_price,
            trailing_distance=trailing_distance
        )

        self.stop_losses[symbol] = stop_loss
        logger.info(f"Set {stop_type.value} stop loss for {symbol}: ${stop_price:.2f}")

        return stop_loss

    def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[float]:
        """
        Update trailing stop loss.

        Args:
            symbol: Stock symbol.
            current_price: Current price.

        Returns:
            New stop price if updated, None otherwise.
        """
        if symbol not in self.stop_losses:
            return None

        stop = self.stop_losses[symbol]
        if stop.stop_type != StopLossType.TRAILING:
            return None

        if stop.trailing_distance is None:
            return None

        # Calculate new stop based on current price
        new_stop = current_price - stop.trailing_distance

        # Only update if higher (for long positions)
        if new_stop > stop.stop_price:
            stop.stop_price = new_stop
            logger.debug(f"Updated trailing stop for {symbol}: ${new_stop:.2f}")
            return new_stop

        return None

    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Check which positions have hit stop loss.

        Args:
            current_prices: Dict of symbol -> current price.

        Returns:
            List of symbols that hit stop loss.
        """
        triggered = []

        for symbol, stop in self.stop_losses.items():
            if symbol in current_prices:
                if current_prices[symbol] <= stop.stop_price:
                    logger.warning(f"Stop loss triggered for {symbol} at ${current_prices[symbol]:.2f}")
                    triggered.append(symbol)

        return triggered

    def update_pnl(self, pnl: float, is_loss: bool = False) -> bool:
        """
        Update daily P&L after a trade.

        Args:
            pnl: Trade P&L amount.
            is_loss: Whether this trade was a loss.

        Returns:
            True if trading should continue, False if halted.
        """
        self.daily_pnl += pnl
        self.daily_trades += 1

        # Track consecutive losses
        if is_loss:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Check daily loss limit
        if self.starting_equity > 0:
            daily_loss_pct = -self.daily_pnl / self.starting_equity
            if daily_loss_pct >= self.max_daily_loss_pct:
                self.trading_halted = True
                logger.warning(f"Daily loss limit reached: {daily_loss_pct:.1%}")
                return False

        return True

    def remove_stop_loss(self, symbol: str) -> None:
        """Remove stop loss for a symbol."""
        if symbol in self.stop_losses:
            del self.stop_losses[symbol]

    def set_take_profit(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        tp_levels: Optional[List[Tuple[float, float]]] = None
    ) -> TakeProfitOrder:
        """
        Set take-profit levels for a position.

        Args:
            symbol: Stock symbol.
            entry_price: Entry price per share.
            quantity: Total position quantity.
            tp_levels: List of (target_pct, exit_pct) tuples.
                       Default: [(0.05, 0.33), (0.10, 0.50), (0.15, 1.0)]
                       - TP1: 5% gain, sell 33% of position
                       - TP2: 10% gain, sell 50% of remaining
                       - TP3: 15% gain, sell rest

        Returns:
            TakeProfitOrder object.
        """
        if tp_levels is None:
            tp_levels = [(0.05, 0.33), (0.10, 0.50), (0.15, 1.0)]

        levels = []
        for target_pct, exit_pct in tp_levels:
            target_price = entry_price * (1 + target_pct)
            levels.append(TakeProfitLevel(
                target_price=target_price,
                target_pct=target_pct,
                exit_pct=exit_pct,
                triggered=False
            ))

        tp_order = TakeProfitOrder(
            symbol=symbol,
            entry_price=entry_price,
            total_quantity=quantity,
            levels=levels,
            quantity_remaining=quantity
        )

        self.take_profits[symbol] = tp_order
        logger.info(f"Set take-profit for {symbol}: {len(levels)} levels, "
                   f"targets at {', '.join([f'{l.target_pct:.0%}' for l in levels])}")

        return tp_order

    def check_take_profits(
        self,
        current_prices: Dict[str, float]
    ) -> List[Tuple[str, int, float]]:
        """
        Check which positions have hit take-profit levels.

        Args:
            current_prices: Dict of symbol -> current price.

        Returns:
            List of (symbol, quantity_to_sell, target_price) tuples.
        """
        triggered_sales = []

        for symbol, tp in list(self.take_profits.items()):
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            for level in tp.levels:
                if level.triggered:
                    continue

                if current_price >= level.target_price:
                    # Calculate quantity to sell at this level
                    sell_qty = int(tp.quantity_remaining * level.exit_pct)

                    # Ensure at least 1 share if we have any
                    if sell_qty == 0 and tp.quantity_remaining > 0:
                        sell_qty = tp.quantity_remaining

                    if sell_qty > 0:
                        level.triggered = True
                        tp.quantity_remaining -= sell_qty
                        triggered_sales.append((symbol, sell_qty, current_price))

                        logger.info(f"Take-profit hit for {symbol}: {level.target_pct:.0%} target, "
                                   f"selling {sell_qty} shares at ${current_price:.2f}")

                        # Remove TP order if fully exited
                        if tp.quantity_remaining <= 0:
                            del self.take_profits[symbol]
                            break

        return triggered_sales

    def remove_take_profit(self, symbol: str) -> None:
        """Remove take-profit for a symbol."""
        if symbol in self.take_profits:
            del self.take_profits[symbol]

    def get_risk_summary(self, portfolio_value: float) -> Dict:
        """
        Get current risk summary.

        Args:
            portfolio_value: Current portfolio value.

        Returns:
            Dictionary with risk metrics.
        """
        # Update drawdown tracking
        self.update_portfolio_value(portfolio_value)

        return {
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": (self.daily_pnl / self.starting_equity * 100) if self.starting_equity > 0 else 0,
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "trading_halted": self.trading_halted,
            "active_stop_losses": len(self.stop_losses),
            "active_take_profits": len(self.take_profits),
            "remaining_trades": max(0, self.max_daily_trades - self.daily_trades),
            # Drawdown metrics
            "peak_portfolio_value": self.peak_portfolio_value,
            "current_drawdown_pct": self.current_drawdown_pct * 100,
            "max_drawdown_pct": self.max_drawdown_pct * 100,
            "recovery_mode": self.drawdown_recovery_mode,
            # VIX metrics
            "vix_sizing_enabled": self.vix_sizing_enabled,
            "current_vix": self._cached_vix,
            "vix_multiplier": self.calculate_volatility_multiplier() if self._cached_vix else 1.0
        }
