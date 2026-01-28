"""
Portfolio rebalancing logic with threshold and calendar-based triggers.

Manages portfolio rebalancing decisions and generates minimal trade orders
to reach target allocation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.portfolio.transaction_costs import TransactionCostModel, TransactionCosts

logger = logging.getLogger(__name__)


class RebalanceTrigger(Enum):
    """Rebalancing trigger types."""
    THRESHOLD = "threshold"           # Drift exceeds threshold
    CALENDAR = "calendar"             # Time-based (weekly/monthly)
    COMBINED = "combined"             # Both threshold and calendar


@dataclass
class Position:
    """Portfolio position representation."""
    symbol: str
    shares: int
    price: float
    value: float
    weight: float


@dataclass
class RebalanceSignal:
    """Rebalancing recommendation."""
    should_rebalance: bool
    reason: str
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    drift_pct: float
    trades_needed: List[Dict] = field(default_factory=list)
    transaction_costs: Optional[TransactionCosts] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'should_rebalance': self.should_rebalance,
            'reason': self.reason,
            'current_weights': self.current_weights,
            'target_weights': self.target_weights,
            'drift_pct': self.drift_pct,
            'trades_needed': self.trades_needed,
            'timestamp': self.timestamp.isoformat()
        }

        if self.transaction_costs:
            result['transaction_costs'] = {
                'total_cost': self.transaction_costs.total_cost,
                'total_cost_pct': self.transaction_costs.total_cost_pct,
                'slippage_cost': self.transaction_costs.slippage_cost,
                'market_impact_cost': self.transaction_costs.market_impact_cost,
                'commission_cost': self.transaction_costs.commission_cost,
                'expected_trades': self.transaction_costs.expected_trades,
                'turnover_pct': self.transaction_costs.turnover_pct
            }

        return result


class PortfolioRebalancer:
    """
    Manages portfolio rebalancing logic.

    Features:
    - Threshold-based: Rebalance when drift > X%
    - Calendar-based: Rebalance weekly/monthly/quarterly
    - Combined: Rebalance if drift > X% AND time > Y
    - Generate minimal trade orders to reach target allocation
    """

    def __init__(
        self,
        drift_threshold: float = 0.05,      # 5% drift triggers rebalance
        min_trade_value: float = 100.0,     # Minimum $100 per trade
        calendar_frequency: str = "weekly", # weekly, monthly, quarterly
        day_of_week: str = "monday",        # For weekly: monday, tuesday, etc.
        day_of_month: int = 1,              # For monthly: 1-28
        trigger_type: RebalanceTrigger = RebalanceTrigger.COMBINED,
        max_trades_per_rebalance: int = 10,
        rebalance_slippage: float = 0.001   # 0.1% slippage assumption
    ):
        """
        Initialize rebalancer with trigger settings.

        Args:
            drift_threshold: Maximum drift before rebalancing (0.05 = 5%)
            min_trade_value: Minimum dollar value per trade
            calendar_frequency: Rebalancing frequency (weekly, monthly, quarterly)
            day_of_week: Day of week for weekly rebalancing
            day_of_month: Day of month for monthly rebalancing
            trigger_type: Type of rebalancing trigger
            max_trades_per_rebalance: Maximum number of trades per rebalance
            rebalance_slippage: Expected slippage per trade
        """
        self.drift_threshold = drift_threshold
        self.min_trade_value = min_trade_value
        self.calendar_frequency = calendar_frequency.lower()
        self.day_of_week = day_of_week.lower()
        self.day_of_month = day_of_month
        self.trigger_type = trigger_type
        self.max_trades_per_rebalance = max_trades_per_rebalance
        self.rebalance_slippage = rebalance_slippage

        self.last_rebalance: Optional[datetime] = None

        # Initialize transaction cost model
        self.cost_model = TransactionCostModel(
            base_slippage_bps=rebalance_slippage * 10000,  # Convert to basis points
            commission_per_trade=0.0,  # WeBull is commission-free
            market_impact_factor=0.1,
            min_trade_value=min_trade_value
        )

        logger.info(f"Initialized PortfolioRebalancer: trigger={trigger_type.value}, "
                   f"drift_threshold={drift_threshold:.1%}, "
                   f"frequency={calendar_frequency}")

    def check_rebalance_needed(
        self,
        current_positions: Dict[str, Position],
        target_weights: Dict[str, float],
        portfolio_value: float,
        current_prices: Optional[Dict[str, float]] = None
    ) -> RebalanceSignal:
        """
        Check if portfolio needs rebalancing.

        Args:
            current_positions: Current positions (symbol -> Position)
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            current_prices: Current prices for each symbol

        Returns:
            RebalanceSignal with recommendation and required trades
        """
        try:
            # Calculate current weights
            current_weights = {}
            for symbol, position in current_positions.items():
                if portfolio_value > 0:
                    current_weights[symbol] = position.value / portfolio_value
                else:
                    current_weights[symbol] = 0.0

            # Add symbols that are in target but not in current
            for symbol in target_weights:
                if symbol not in current_weights:
                    current_weights[symbol] = 0.0

            # Calculate drift
            drift = self.calculate_drift(current_weights, target_weights)

            # Check triggers
            threshold_triggered = self._should_rebalance_threshold(drift)
            calendar_triggered = self._should_rebalance_calendar()

            should_rebalance = False
            reason = ""

            if self.trigger_type == RebalanceTrigger.THRESHOLD:
                should_rebalance = threshold_triggered
                if should_rebalance:
                    reason = f"Drift threshold exceeded: {drift:.2%} > {self.drift_threshold:.2%}"
            elif self.trigger_type == RebalanceTrigger.CALENDAR:
                should_rebalance = calendar_triggered
                if should_rebalance:
                    reason = f"Calendar trigger: {self.calendar_frequency} rebalancing due"
            else:  # COMBINED
                should_rebalance = threshold_triggered and calendar_triggered
                if should_rebalance:
                    reason = f"Combined trigger: drift={drift:.2%}, calendar={self.calendar_frequency}"

            # Generate trades if rebalancing needed
            trades_needed = []
            transaction_costs = None

            if should_rebalance:
                # Get current prices
                if current_prices is None:
                    current_prices = {
                        symbol: position.price
                        for symbol, position in current_positions.items()
                    }

                trades_needed = self.generate_rebalance_orders(
                    current_positions,
                    target_weights,
                    portfolio_value,
                    current_prices
                )

                # Estimate transaction costs
                transaction_costs = self.cost_model.estimate_rebalancing_costs(
                    current_weights=current_weights,
                    target_weights=target_weights,
                    portfolio_value=portfolio_value,
                    current_prices=current_prices
                )

            signal = RebalanceSignal(
                should_rebalance=should_rebalance,
                reason=reason,
                current_weights=current_weights,
                target_weights=target_weights,
                drift_pct=drift,
                trades_needed=trades_needed,
                transaction_costs=transaction_costs
            )

            if should_rebalance:
                logger.info(f"Rebalancing recommended: {reason}")
                logger.info(f"  Drift: {drift:.2%}, Trades needed: {len(trades_needed)}")
                if transaction_costs:
                    logger.info(f"  Estimated costs: ${transaction_costs.total_cost:.2f} "
                              f"({transaction_costs.total_cost_pct:.3%}), "
                              f"Turnover: {transaction_costs.turnover_pct:.1f}%")
            else:
                logger.debug(f"No rebalancing needed: drift={drift:.2%}, "
                           f"threshold={threshold_triggered}, calendar={calendar_triggered}")

            return signal

        except Exception as e:
            logger.error(f"Error checking rebalance: {e}")
            return RebalanceSignal(
                should_rebalance=False,
                reason=f"Error: {e}",
                current_weights={},
                target_weights=target_weights,
                drift_pct=0.0
            )

    def generate_rebalance_orders(
        self,
        current_positions: Dict[str, Position],
        target_weights: Dict[str, float],
        portfolio_value: float,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Generate minimal trade orders to reach target allocation.

        Args:
            current_positions: Current positions
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            current_prices: Current prices for each symbol

        Returns:
            List of trade orders: [{"symbol": "AAPL", "action": "BUY", "shares": 10, "value": 1500}]
        """
        try:
            trades = []

            # Get all symbols (union of current and target)
            all_symbols = set(current_positions.keys()) | set(target_weights.keys())

            for symbol in all_symbols:
                # Get target value and current value
                target_weight = target_weights.get(symbol, 0.0)
                target_value = portfolio_value * target_weight

                current_position = current_positions.get(symbol)
                current_value = current_position.value if current_position else 0.0
                current_shares = current_position.shares if current_position else 0

                # Calculate value difference
                value_diff = target_value - current_value

                # Get current price
                price = current_prices.get(symbol)
                if price is None or price <= 0:
                    logger.warning(f"No valid price for {symbol}, skipping")
                    continue

                # Calculate shares to trade
                shares_to_trade = int(value_diff / price)

                # Skip if trade value is too small
                trade_value = abs(shares_to_trade * price)
                if trade_value < self.min_trade_value:
                    logger.debug(f"Skipping {symbol}: trade value ${trade_value:.2f} "
                               f"< min ${self.min_trade_value:.2f}")
                    continue

                # Determine action
                if shares_to_trade > 0:
                    action = "BUY"
                elif shares_to_trade < 0:
                    action = "SELL"
                    shares_to_trade = abs(shares_to_trade)
                else:
                    continue

                trades.append({
                    'symbol': symbol,
                    'action': action,
                    'shares': shares_to_trade,
                    'price': price,
                    'value': trade_value,
                    'current_weight': current_value / portfolio_value if portfolio_value > 0 else 0,
                    'target_weight': target_weight,
                    'weight_diff': target_weight - (current_value / portfolio_value if portfolio_value > 0 else 0)
                })

            # Sort by trade value (largest first)
            trades.sort(key=lambda x: x['value'], reverse=True)

            # Limit number of trades
            if len(trades) > self.max_trades_per_rebalance:
                logger.warning(f"Limiting trades from {len(trades)} to {self.max_trades_per_rebalance}")
                trades = trades[:self.max_trades_per_rebalance]

            logger.info(f"Generated {len(trades)} rebalancing orders")
            for trade in trades:
                logger.debug(f"  {trade['action']} {trade['shares']} {trade['symbol']} "
                           f"@ ${trade['price']:.2f} (${trade['value']:.2f})")

            return trades

        except Exception as e:
            logger.error(f"Error generating rebalance orders: {e}")
            return []

    def calculate_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio drift as maximum absolute deviation from target.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights

        Returns:
            Maximum absolute weight deviation
        """
        try:
            # Get all symbols
            all_symbols = set(current_weights.keys()) | set(target_weights.keys())

            max_drift = 0.0

            for symbol in all_symbols:
                current = current_weights.get(symbol, 0.0)
                target = target_weights.get(symbol, 0.0)
                drift = abs(current - target)
                max_drift = max(max_drift, drift)

            return max_drift

        except Exception as e:
            logger.error(f"Error calculating drift: {e}")
            return 0.0

    def record_rebalance(self, timestamp: Optional[datetime] = None):
        """
        Record that a rebalance occurred.

        Args:
            timestamp: Timestamp of rebalance (default: now)
        """
        self.last_rebalance = timestamp or datetime.now()
        logger.info(f"Recorded rebalance at {self.last_rebalance}")

    def _should_rebalance_threshold(self, drift: float) -> bool:
        """
        Check if drift exceeds threshold.

        Args:
            drift: Current portfolio drift

        Returns:
            True if drift exceeds threshold
        """
        return drift > self.drift_threshold

    def _should_rebalance_calendar(self) -> bool:
        """
        Check if calendar trigger fired.

        Returns:
            True if calendar trigger fired
        """
        now = datetime.now()

        # If never rebalanced, don't trigger on calendar
        if self.last_rebalance is None:
            return False

        if self.calendar_frequency == "daily":
            # Rebalance every day
            return (now - self.last_rebalance).days >= 1

        elif self.calendar_frequency == "weekly":
            # Rebalance on specified day of week
            days_since = (now - self.last_rebalance).days

            if days_since < 7:
                return False

            # Check if today matches the target day
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            target_day = day_map.get(self.day_of_week, 0)
            return now.weekday() == target_day

        elif self.calendar_frequency == "monthly":
            # Rebalance on specified day of month
            days_since = (now - self.last_rebalance).days

            if days_since < 28:  # At least 28 days between rebalances
                return False

            return now.day == self.day_of_month

        elif self.calendar_frequency == "quarterly":
            # Rebalance every 3 months
            months_since = (now.year - self.last_rebalance.year) * 12 + \
                          (now.month - self.last_rebalance.month)
            return months_since >= 3

        else:
            logger.warning(f"Unknown calendar frequency: {self.calendar_frequency}")
            return False

    def get_next_rebalance_date(self) -> Optional[datetime]:
        """
        Get the next calendar-based rebalance date.

        Returns:
            Next rebalance datetime or None
        """
        if self.last_rebalance is None:
            return None

        now = datetime.now()
        last = self.last_rebalance

        if self.calendar_frequency == "daily":
            return last + timedelta(days=1)

        elif self.calendar_frequency == "weekly":
            # Find next occurrence of target day
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            target_day = day_map.get(self.day_of_week, 0)

            # Start from next week
            next_date = last + timedelta(days=7)

            # Adjust to target day of week
            while next_date.weekday() != target_day:
                next_date += timedelta(days=1)

            return next_date

        elif self.calendar_frequency == "monthly":
            # Next month on target day
            if last.month == 12:
                next_month = 1
                next_year = last.year + 1
            else:
                next_month = last.month + 1
                next_year = last.year

            try:
                return datetime(next_year, next_month, self.day_of_month)
            except ValueError:
                # Day doesn't exist in month, use last day
                import calendar
                last_day = calendar.monthrange(next_year, next_month)[1]
                return datetime(next_year, next_month, last_day)

        elif self.calendar_frequency == "quarterly":
            # 3 months from last rebalance
            return last + timedelta(days=90)

        return None
