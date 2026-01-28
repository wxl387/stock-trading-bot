"""
Transaction cost modeling for portfolio optimization.

Models slippage, market impact, and incorporates costs into optimization.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TransactionCosts:
    """
    Transaction cost estimates for portfolio rebalancing.
    """
    total_cost: float                    # Total transaction cost ($)
    total_cost_pct: float                # Total cost as % of portfolio value
    slippage_cost: float                 # Cost due to slippage ($)
    market_impact_cost: float            # Cost due to market impact ($)
    commission_cost: float               # Commission costs ($)
    expected_trades: int                 # Number of expected trades
    turnover_pct: float                  # Portfolio turnover percentage
    cost_breakdown: Dict[str, float]     # Cost breakdown by symbol


class TransactionCostModel:
    """
    Models transaction costs for portfolio optimization.

    Features:
    - Slippage estimation based on symbol liquidity
    - Market impact modeling based on trade size
    - Commission costs (zero for most US brokers)
    - Turnover-aware optimization
    """

    def __init__(
        self,
        base_slippage_bps: float = 10.0,     # Base slippage in basis points
        commission_per_trade: float = 0.0,    # Commission per trade (WeBull is free)
        market_impact_factor: float = 0.1,   # Market impact multiplier
        min_trade_value: float = 100.0       # Minimum trade value to consider
    ):
        """
        Initialize transaction cost model.

        Args:
            base_slippage_bps: Base slippage in basis points (0.1% = 10 bps)
            commission_per_trade: Commission per trade (default 0 for commission-free brokers)
            market_impact_factor: Market impact scaling factor
            min_trade_value: Minimum trade value to include in cost calculation
        """
        self.base_slippage_bps = base_slippage_bps
        self.commission_per_trade = commission_per_trade
        self.market_impact_factor = market_impact_factor
        self.min_trade_value = min_trade_value

        logger.info(f"Initialized TransactionCostModel: "
                   f"slippage={base_slippage_bps}bps, "
                   f"commission=${commission_per_trade}")

    def estimate_rebalancing_costs(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        current_prices: Optional[Dict[str, float]] = None,
        volumes: Optional[Dict[str, float]] = None
    ) -> TransactionCosts:
        """
        Estimate transaction costs for rebalancing from current to target weights.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value ($)
            current_prices: Current prices per symbol (for market impact calculation)
            volumes: Average daily trading volumes (for liquidity estimation)

        Returns:
            TransactionCosts object with cost breakdown
        """
        try:
            # Get all symbols
            all_symbols = set(current_weights.keys()) | set(target_weights.keys())

            total_slippage = 0.0
            total_market_impact = 0.0
            total_commission = 0.0
            cost_breakdown = {}
            num_trades = 0
            total_turnover = 0.0

            for symbol in all_symbols:
                current_weight = current_weights.get(symbol, 0.0)
                target_weight = target_weights.get(symbol, 0.0)

                # Calculate weight change
                weight_change = abs(target_weight - current_weight)
                trade_value = weight_change * portfolio_value

                # Skip if trade is too small
                if trade_value < self.min_trade_value:
                    continue

                num_trades += 1
                total_turnover += weight_change

                # Estimate slippage
                slippage = self._estimate_slippage(
                    symbol, trade_value, current_prices, volumes
                )
                slippage_cost = trade_value * slippage

                # Estimate market impact
                market_impact = self._estimate_market_impact(
                    symbol, trade_value, portfolio_value, current_prices, volumes
                )
                impact_cost = trade_value * market_impact

                # Commission
                commission = self.commission_per_trade

                # Total cost for this symbol
                symbol_cost = slippage_cost + impact_cost + commission

                total_slippage += slippage_cost
                total_market_impact += impact_cost
                total_commission += commission
                cost_breakdown[symbol] = symbol_cost

            total_cost = total_slippage + total_market_impact + total_commission
            total_cost_pct = (total_cost / portfolio_value) if portfolio_value > 0 else 0.0

            costs = TransactionCosts(
                total_cost=total_cost,
                total_cost_pct=total_cost_pct,
                slippage_cost=total_slippage,
                market_impact_cost=total_market_impact,
                commission_cost=total_commission,
                expected_trades=num_trades,
                turnover_pct=total_turnover * 100,
                cost_breakdown=cost_breakdown
            )

            logger.debug(f"Estimated transaction costs: ${total_cost:.2f} "
                        f"({total_cost_pct:.3%}) for {num_trades} trades, "
                        f"turnover={total_turnover:.1%}")

            return costs

        except Exception as e:
            logger.error(f"Error estimating transaction costs: {e}")
            return TransactionCosts(
                total_cost=0.0,
                total_cost_pct=0.0,
                slippage_cost=0.0,
                market_impact_cost=0.0,
                commission_cost=0.0,
                expected_trades=0,
                turnover_pct=0.0,
                cost_breakdown={}
            )

    def _estimate_slippage(
        self,
        symbol: str,
        trade_value: float,
        prices: Optional[Dict[str, float]] = None,
        volumes: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Estimate slippage as percentage of trade value.

        Slippage depends on:
        - Symbol liquidity (volume)
        - Trade size relative to average volume
        - Time of day (we use base estimate)

        Args:
            symbol: Trading symbol
            trade_value: Value of trade ($)
            prices: Current prices (optional, for more accurate estimation)
            volumes: Average daily trading volumes (optional)

        Returns:
            Slippage as decimal (e.g., 0.001 = 0.1% = 10 bps)
        """
        # Base slippage in decimal
        base_slippage = self.base_slippage_bps / 10000.0

        # If we have volume data, adjust slippage based on liquidity
        if volumes and symbol in volumes and prices and symbol in prices:
            volume = volumes[symbol]
            price = prices[symbol]

            # Estimate number of shares traded
            shares = trade_value / price if price > 0 else 0

            # Adjust slippage based on trade size relative to daily volume
            # Higher percentage of volume → higher slippage
            volume_pct = shares / volume if volume > 0 else 0.0

            # Slippage scales with sqrt of volume percentage (market microstructure)
            liquidity_multiplier = 1.0 + (np.sqrt(volume_pct) * 10.0)
            adjusted_slippage = base_slippage * liquidity_multiplier

            return min(adjusted_slippage, 0.01)  # Cap at 1%

        # Default to base slippage if no volume data
        return base_slippage

    def _estimate_market_impact(
        self,
        symbol: str,
        trade_value: float,
        portfolio_value: float,
        prices: Optional[Dict[str, float]] = None,
        volumes: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Estimate market impact as percentage of trade value.

        Market impact is the price movement caused by the trade itself.
        Larger trades have higher impact (square-root law).

        Args:
            symbol: Trading symbol
            trade_value: Value of trade ($)
            portfolio_value: Total portfolio value ($)
            prices: Current prices (optional)
            volumes: Average daily trading volumes (optional)

        Returns:
            Market impact as decimal
        """
        # Market impact is generally smaller than slippage for typical retail trades
        base_impact = (self.base_slippage_bps / 2.0) / 10000.0  # Half of slippage

        # For small retail trades (<$100k), market impact is minimal
        # We scale by portfolio size
        if portfolio_value > 0:
            size_factor = (trade_value / portfolio_value) ** 0.5
            adjusted_impact = base_impact * size_factor * self.market_impact_factor
            return min(adjusted_impact, 0.005)  # Cap at 0.5%

        return base_impact

    def calculate_turnover_penalty(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        penalty_coefficient: float = 0.01
    ) -> float:
        """
        Calculate turnover penalty for optimization objective.

        This penalty discourages excessive trading and can be added to
        the optimization objective to reduce transaction costs.

        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            penalty_coefficient: Penalty coefficient (higher = more penalty)

        Returns:
            Turnover penalty value
        """
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        total_turnover = sum(
            abs(target_weights.get(sym, 0.0) - current_weights.get(sym, 0.0))
            for sym in all_symbols
        )

        penalty = penalty_coefficient * total_turnover

        logger.debug(f"Turnover penalty: {penalty:.6f} for {total_turnover:.3f} turnover")

        return penalty

    def get_cost_summary(self, costs: TransactionCosts) -> str:
        """
        Get human-readable summary of transaction costs.

        Args:
            costs: TransactionCosts object

        Returns:
            Formatted string summary
        """
        summary = f"""
Transaction Cost Estimate:
  Total Cost: ${costs.total_cost:.2f} ({costs.total_cost_pct:.3%} of portfolio)
  Slippage: ${costs.slippage_cost:.2f}
  Market Impact: ${costs.market_impact_cost:.2f}
  Commission: ${costs.commission_cost:.2f}
  Expected Trades: {costs.expected_trades}
  Portfolio Turnover: {costs.turnover_pct:.1f}%
"""
        return summary.strip()
