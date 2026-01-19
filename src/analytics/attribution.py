"""
Performance attribution analysis.
Analyzes which positions and trades contributed to portfolio returns.
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceAttribution:
    """
    Analyze which positions and trades contributed to portfolio performance.

    Provides:
    - Position contribution analysis
    - Winner/loser trade analysis
    - Realized vs unrealized P&L breakdown
    """

    def __init__(
        self,
        trades: pd.DataFrame,
        positions: Dict,
        total_pnl: float = 0.0
    ):
        """
        Initialize PerformanceAttribution.

        Args:
            trades: DataFrame of trade history.
            positions: Dict of current positions with P&L.
            total_pnl: Total portfolio P&L for contribution calculation.
        """
        self.trades = trades
        self.positions = positions
        self.total_pnl = total_pnl

    def position_contribution(self) -> pd.DataFrame:
        """
        Calculate P&L contribution per position.

        Returns:
            DataFrame with symbol, pnl, contribution_pct, sorted by contribution.
        """
        contributions = []

        # Get P&L from current positions (unrealized)
        for symbol, pos_data in self.positions.items():
            if isinstance(pos_data, dict):
                unrealized = pos_data.get('unrealized_pnl', 0)
                realized = pos_data.get('realized_pnl', 0)
                total_pos_pnl = unrealized + realized
            else:
                # Assume it's just unrealized if single value
                total_pos_pnl = pos_data

            contributions.append({
                'symbol': symbol,
                'pnl': total_pos_pnl,
                'type': 'position'
            })

        # Get closed trade P&L
        if not self.trades.empty and 'symbol' in self.trades.columns:
            trade_pnl = self._calculate_closed_trade_pnl()
            for symbol, pnl in trade_pnl.items():
                # Check if already in contributions
                existing = [c for c in contributions if c['symbol'] == symbol]
                if existing:
                    existing[0]['pnl'] += pnl
                else:
                    contributions.append({
                        'symbol': symbol,
                        'pnl': pnl,
                        'type': 'closed'
                    })

        if not contributions:
            return pd.DataFrame(columns=['symbol', 'pnl', 'contribution_pct'])

        df = pd.DataFrame(contributions)

        # Calculate contribution percentage
        total = abs(self.total_pnl) if self.total_pnl != 0 else df['pnl'].abs().sum()
        if total == 0:
            total = 1  # Avoid division by zero

        df['contribution_pct'] = df['pnl'] / total

        # Sort by absolute contribution
        df = df.sort_values('pnl', ascending=False, key=abs)

        return df[['symbol', 'pnl', 'contribution_pct']]

    def _calculate_closed_trade_pnl(self) -> Dict[str, float]:
        """Calculate realized P&L from closed trades."""
        if self.trades.empty:
            return {}

        pnl_by_symbol = {}

        # Group by symbol
        for symbol in self.trades['symbol'].unique():
            symbol_trades = self.trades[self.trades['symbol'] == symbol].copy()

            if 'timestamp' in symbol_trades.columns:
                symbol_trades = symbol_trades.sort_values('timestamp')

            buy_queue = []  # FIFO: (quantity, price)
            total_pnl = 0.0

            for _, trade in symbol_trades.iterrows():
                side = trade.get('side', '').upper()
                qty = trade.get('quantity', 0)
                price = trade.get('price', 0)

                if side == 'BUY':
                    buy_queue.append((qty, price))
                elif side == 'SELL' and buy_queue:
                    remaining = qty
                    while remaining > 0 and buy_queue:
                        buy_qty, buy_price = buy_queue[0]
                        matched = min(remaining, buy_qty)

                        trade_pnl = (price - buy_price) * matched
                        total_pnl += trade_pnl

                        remaining -= matched
                        if matched >= buy_qty:
                            buy_queue.pop(0)
                        else:
                            buy_queue[0] = (buy_qty - matched, buy_price)

            if total_pnl != 0:
                pnl_by_symbol[symbol] = total_pnl

        return pnl_by_symbol

    def winner_loser_analysis(self) -> Dict:
        """
        Analyze winning vs losing trades.

        Returns:
            Dict with trade statistics.
        """
        if self.trades.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0,
                'avg_win_loss_ratio': 0.0
            }

        # Calculate P&L for each completed trade
        trade_pnls = self._get_individual_trade_pnls()

        if not trade_pnls:
            return {
                'total_trades': len(self.trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0,
                'avg_win_loss_ratio': 0.0
            }

        pnl_series = pd.Series(trade_pnls)
        winners = pnl_series[pnl_series > 0]
        losers = pnl_series[pnl_series < 0]

        avg_win = winners.mean() if len(winners) > 0 else 0.0
        avg_loss = losers.mean() if len(losers) > 0 else 0.0

        gross_profit = winners.sum() if len(winners) > 0 else 0.0
        gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.0

        return {
            'total_trades': len(trade_pnls),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(trade_pnls) if trade_pnls else 0.0,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': winners.max() if len(winners) > 0 else 0.0,
            'largest_loss': losers.min() if len(losers) > 0 else 0.0,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'avg_win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'total_pnl': pnl_series.sum()
        }

    def _get_individual_trade_pnls(self) -> List[float]:
        """Get P&L for each completed trade (buy+sell pair)."""
        if self.trades.empty:
            return []

        pnls = []

        for symbol in self.trades['symbol'].unique():
            symbol_trades = self.trades[self.trades['symbol'] == symbol].copy()

            if 'timestamp' in symbol_trades.columns:
                symbol_trades = symbol_trades.sort_values('timestamp')

            buy_queue = []

            for _, trade in symbol_trades.iterrows():
                side = trade.get('side', '').upper()
                qty = trade.get('quantity', 0)
                price = trade.get('price', 0)

                if side == 'BUY':
                    buy_queue.append((qty, price))
                elif side == 'SELL' and buy_queue:
                    remaining = qty
                    while remaining > 0 and buy_queue:
                        buy_qty, buy_price = buy_queue[0]
                        matched = min(remaining, buy_qty)

                        trade_pnl = (price - buy_price) * matched
                        pnls.append(trade_pnl)

                        remaining -= matched
                        if matched >= buy_qty:
                            buy_queue.pop(0)
                        else:
                            buy_queue[0] = (buy_qty - matched, buy_price)

        return pnls

    def realized_vs_unrealized(self) -> Dict:
        """
        Break down total P&L into realized and unrealized.

        Returns:
            Dict with realized, unrealized, and total P&L.
        """
        realized = 0.0
        unrealized = 0.0

        # Unrealized from current positions
        for symbol, pos_data in self.positions.items():
            if isinstance(pos_data, dict):
                unrealized += pos_data.get('unrealized_pnl', 0)
                realized += pos_data.get('realized_pnl', 0)

        # Additional realized from closed trades not in positions
        closed_pnl = self._calculate_closed_trade_pnl()
        for symbol, pnl in closed_pnl.items():
            if symbol not in self.positions:
                realized += pnl

        return {
            'realized_pnl': realized,
            'unrealized_pnl': unrealized,
            'total_pnl': realized + unrealized,
            'realized_pct': realized / (realized + unrealized) if (realized + unrealized) != 0 else 0
        }

    def top_contributors(self, n: int = 5) -> pd.DataFrame:
        """
        Get top N contributors (positive and negative).

        Args:
            n: Number of top contributors.

        Returns:
            DataFrame with top winners and losers.
        """
        contrib = self.position_contribution()

        if contrib.empty:
            return pd.DataFrame()

        top_winners = contrib[contrib['pnl'] > 0].head(n)
        top_losers = contrib[contrib['pnl'] < 0].tail(n)

        return pd.concat([top_winners, top_losers])

    def get_summary(self) -> Dict:
        """
        Get complete attribution summary.

        Returns:
            Dict with all attribution metrics.
        """
        winner_loser = self.winner_loser_analysis()
        realized_unrealized = self.realized_vs_unrealized()
        top_contrib = self.top_contributors(5)

        return {
            'trade_analysis': winner_loser,
            'pnl_breakdown': realized_unrealized,
            'top_contributors': top_contrib.to_dict('records') if not top_contrib.empty else [],
            'position_count': len(self.positions),
            'trade_count': len(self.trades) if not self.trades.empty else 0
        }
