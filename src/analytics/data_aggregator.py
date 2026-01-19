"""
Data aggregator for portfolio analytics.
Prepares data from broker state and portfolio history for analysis.
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Aggregates and prepares data for portfolio analytics.

    Pulls data from:
    - Portfolio history (daily snapshots)
    - Trade history (from broker state)
    - Position data (current holdings)
    """

    def __init__(self, data_provider=None):
        """
        Initialize DataAggregator.

        Args:
            data_provider: Optional DashboardDataProvider instance.
        """
        self._data_provider = data_provider
        self._portfolio_history_file = DATA_DIR / "portfolio_history.json"
        self._broker_state_file = DATA_DIR / "simulated_broker_state.json"

    @property
    def data_provider(self):
        """Lazy load data provider."""
        if self._data_provider is None:
            try:
                from src.dashboard.data_provider import get_data_provider
                self._data_provider = get_data_provider()
            except Exception as e:
                logger.warning(f"Could not load data provider: {e}")
        return self._data_provider

    def get_equity_curve(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Get portfolio value time series.

        Args:
            start_date: Start date filter.
            end_date: End date filter.

        Returns:
            Series with datetime index and portfolio values.
        """
        try:
            if self.data_provider:
                history = self.data_provider.get_portfolio_history()
                if history:
                    df = pd.DataFrame(history)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp').sort_index()

                    # Filter by date range
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]

                    return df['portfolio_value']
        except Exception as e:
            logger.warning(f"Error getting equity curve from data provider: {e}")

        # Fallback: read directly from file
        return self._load_equity_from_file(start_date, end_date)

    def _load_equity_from_file(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """Load equity curve directly from portfolio history file."""
        import json

        if not self._portfolio_history_file.exists():
            logger.warning("Portfolio history file not found")
            return pd.Series(dtype=float)

        try:
            with open(self._portfolio_history_file, 'r') as f:
                history = json.load(f)

            if not history:
                return pd.Series(dtype=float)

            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            return df['portfolio_value']

        except Exception as e:
            logger.error(f"Error loading portfolio history: {e}")
            return pd.Series(dtype=float)

    def get_daily_returns(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Calculate daily returns from portfolio history.

        Args:
            start_date: Start date filter.
            end_date: End date filter.

        Returns:
            Series of daily returns.
        """
        equity = self.get_equity_curve(start_date, end_date)

        if len(equity) < 2:
            return pd.Series(dtype=float)

        # Resample to daily (take last value of each day)
        daily_equity = equity.resample('D').last().dropna()

        # Calculate returns
        returns = daily_equity.pct_change().dropna()

        return returns

    def get_trade_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get completed trades with P&L information.

        Args:
            start_date: Start date filter.
            end_date: End date filter.

        Returns:
            DataFrame with trade history.
        """
        try:
            if self.data_provider:
                trades = self.data_provider.get_trade_history()
                if trades:
                    df = pd.DataFrame(trades)
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])

                        if start_date:
                            df = df[df['timestamp'] >= start_date]
                        if end_date:
                            df = df[df['timestamp'] <= end_date]

                    return df
        except Exception as e:
            logger.warning(f"Error getting trade history: {e}")

        # Fallback to file
        return self._load_trades_from_file(start_date, end_date)

    def _load_trades_from_file(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load trades directly from broker state file."""
        import json

        if not self._broker_state_file.exists():
            return pd.DataFrame()

        try:
            with open(self._broker_state_file, 'r') as f:
                state = json.load(f)

            trades = state.get('trade_history', [])
            if not trades:
                return pd.DataFrame()

            df = pd.DataFrame(trades)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]

            return df

        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            return pd.DataFrame()

    def get_positions(self) -> Dict:
        """
        Get current positions with P&L.

        Returns:
            Dict of symbol -> position info.
        """
        if self.data_provider:
            return self.data_provider.get_positions()
        return {}

    def get_portfolio_metrics(self) -> Dict:
        """
        Get current portfolio metrics.

        Returns:
            Dict with portfolio summary.
        """
        if self.data_provider:
            return self.data_provider.get_portfolio_metrics()
        return {}

    def prepare_analytics_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Prepare all data needed for analytics.

        Args:
            start_date: Start date for analysis.
            end_date: End date for analysis.

        Returns:
            Dict with:
            - equity_curve: pd.Series
            - returns: pd.Series
            - trades: pd.DataFrame
            - positions: Dict
            - portfolio_metrics: Dict
        """
        equity = self.get_equity_curve(start_date, end_date)
        returns = self.get_daily_returns(start_date, end_date)
        trades = self.get_trade_history(start_date, end_date)
        positions = self.get_positions()
        metrics = self.get_portfolio_metrics()

        return {
            'equity_curve': equity,
            'returns': returns,
            'trades': trades,
            'positions': positions,
            'portfolio_metrics': metrics,
            'start_date': start_date or (returns.index.min() if len(returns) > 0 else None),
            'end_date': end_date or (returns.index.max() if len(returns) > 0 else None),
            'trading_days': len(returns)
        }

    def calculate_trade_pnl(self, trades: pd.DataFrame) -> pd.Series:
        """
        Calculate P&L for each completed trade pair (buy -> sell).

        Args:
            trades: DataFrame with trade history.

        Returns:
            Series of trade P&L values.
        """
        if trades.empty or 'symbol' not in trades.columns:
            return pd.Series(dtype=float)

        pnl_list = []

        # Group by symbol
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol].sort_values('timestamp')

            buy_queue = []  # FIFO queue of (quantity, price)

            for _, trade in symbol_trades.iterrows():
                side = trade.get('side', '').upper()
                qty = trade.get('quantity', 0)
                price = trade.get('price', 0)

                if side == 'BUY':
                    buy_queue.append((qty, price))
                elif side == 'SELL' and buy_queue:
                    # Match against buys (FIFO)
                    remaining = qty
                    while remaining > 0 and buy_queue:
                        buy_qty, buy_price = buy_queue[0]
                        matched = min(remaining, buy_qty)

                        # Calculate P&L for this match
                        trade_pnl = (price - buy_price) * matched
                        pnl_list.append(trade_pnl)

                        remaining -= matched
                        if matched >= buy_qty:
                            buy_queue.pop(0)
                        else:
                            buy_queue[0] = (buy_qty - matched, buy_price)

        return pd.Series(pnl_list)


# Singleton instance
_data_aggregator: Optional[DataAggregator] = None


def get_data_aggregator() -> DataAggregator:
    """Get singleton data aggregator instance."""
    global _data_aggregator
    if _data_aggregator is None:
        _data_aggregator = DataAggregator()
    return _data_aggregator
