"""
Core financial metrics calculations for portfolio analytics.
"""
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate financial performance metrics from returns series.

    Provides proper implementations of risk-adjusted return metrics
    including Sharpe, Sortino, Calmar, and rolling metrics.
    """

    def __init__(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252
    ):
        """
        Initialize MetricsCalculator.

        Args:
            returns: Daily returns series (as decimals, e.g., 0.01 for 1%).
            risk_free_rate: Annual risk-free rate (default 5%).
            periods_per_year: Trading periods per year (252 for daily).
        """
        self.returns = returns.dropna()
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.daily_rf = risk_free_rate / periods_per_year

    def total_return(self) -> float:
        """Calculate total cumulative return."""
        if len(self.returns) == 0:
            return 0.0
        return (1 + self.returns).prod() - 1

    def annualized_return(self) -> float:
        """Calculate annualized return."""
        if len(self.returns) == 0:
            return 0.0
        total = self.total_return()
        n_years = len(self.returns) / self.periods_per_year
        if n_years <= 0:
            return 0.0
        return (1 + total) ** (1 / n_years) - 1

    def volatility(self) -> float:
        """Calculate annualized volatility (standard deviation)."""
        if len(self.returns) < 2:
            return 0.0
        return self.returns.std() * np.sqrt(self.periods_per_year)

    def sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio.

        Sharpe = (annualized_return - risk_free_rate) / volatility
        """
        vol = self.volatility()
        if vol == 0:
            return 0.0
        return (self.annualized_return() - self.risk_free_rate) / vol

    def sortino_ratio(self) -> float:
        """
        Calculate Sortino ratio using proper downside deviation.

        Sortino = (annualized_return - risk_free_rate) / downside_deviation

        Only uses negative returns for denominator (downside risk).
        """
        # Calculate downside returns (below risk-free rate)
        excess_returns = self.returns - self.daily_rf
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float('inf') if self.annualized_return() > self.risk_free_rate else 0.0

        # Downside deviation (annualized)
        downside_dev = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(self.periods_per_year)

        if downside_dev == 0:
            return 0.0

        return (self.annualized_return() - self.risk_free_rate) / downside_dev

    def calmar_ratio(self, max_drawdown: Optional[float] = None) -> float:
        """
        Calculate Calmar ratio.

        Calmar = annualized_return / max_drawdown

        Args:
            max_drawdown: Maximum drawdown as positive decimal (e.g., 0.10 for 10%).
                         If None, calculated from returns.
        """
        if max_drawdown is None:
            max_drawdown, _, _ = self.max_drawdown()

        if max_drawdown == 0:
            return float('inf') if self.annualized_return() > 0 else 0.0

        return self.annualized_return() / abs(max_drawdown)

    def max_drawdown(self) -> Tuple[float, Optional[datetime], Optional[datetime]]:
        """
        Calculate maximum drawdown with peak and trough dates.

        Returns:
            Tuple of (max_drawdown, peak_date, trough_date).
            Max drawdown is returned as positive value (e.g., 0.10 for 10%).
        """
        if len(self.returns) == 0:
            return 0.0, None, None

        # Calculate cumulative returns
        cum_returns = (1 + self.returns).cumprod()

        # Running maximum
        running_max = cum_returns.expanding().max()

        # Drawdown series
        drawdowns = (cum_returns - running_max) / running_max

        # Find max drawdown
        max_dd = drawdowns.min()

        if max_dd == 0:
            return 0.0, None, None

        # Find trough date
        trough_idx = drawdowns.idxmin()

        # Find peak date (last peak before trough)
        peak_idx = cum_returns[:trough_idx].idxmax()

        return abs(max_dd), peak_idx, trough_idx

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """
        Calculate rolling Sharpe ratio.

        Args:
            window: Rolling window in periods (default 63 = ~3 months).

        Returns:
            Series of rolling Sharpe ratios.
        """
        if len(self.returns) < window:
            return pd.Series(dtype=float)

        rolling_mean = self.returns.rolling(window=window).mean()
        rolling_std = self.returns.rolling(window=window).std()

        # Annualize
        excess_return = rolling_mean - self.daily_rf
        annualized_excess = excess_return * self.periods_per_year
        annualized_std = rolling_std * np.sqrt(self.periods_per_year)

        rolling_sharpe = annualized_excess / annualized_std
        return rolling_sharpe.dropna()

    def monthly_returns(self) -> pd.DataFrame:
        """
        Aggregate returns by month for heatmap display.

        Returns:
            DataFrame with years as index, months as columns.
        """
        if len(self.returns) == 0:
            return pd.DataFrame()

        # Ensure datetime index
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            return pd.DataFrame()

        # Calculate monthly returns
        monthly = (1 + self.returns).resample('ME').prod() - 1

        # Create pivot table
        monthly_df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values
        })

        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                       7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        pivot.columns = [month_names[m] for m in pivot.columns]

        return pivot

    def win_rate(self, trades_pnl: Optional[pd.Series] = None) -> float:
        """
        Calculate win rate from trades or returns.

        Args:
            trades_pnl: Series of trade P&L values. If None, uses returns.
        """
        data = trades_pnl if trades_pnl is not None else self.returns
        if len(data) == 0:
            return 0.0
        return (data > 0).sum() / len(data)

    def profit_factor(self, trades_pnl: Optional[pd.Series] = None) -> float:
        """
        Calculate profit factor (gross profits / gross losses).

        Args:
            trades_pnl: Series of trade P&L values. If None, uses returns.
        """
        data = trades_pnl if trades_pnl is not None else self.returns

        gross_profit = data[data > 0].sum()
        gross_loss = abs(data[data < 0].sum())

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio vs benchmark.

        IR = (portfolio_return - benchmark_return) / tracking_error
        """
        # Align dates
        aligned = pd.concat([self.returns, benchmark_returns], axis=1, join='inner')
        aligned.columns = ['portfolio', 'benchmark']

        if len(aligned) < 2:
            return 0.0

        # Calculate excess returns
        excess = aligned['portfolio'] - aligned['benchmark']

        # Tracking error (annualized std of excess returns)
        tracking_error = excess.std() * np.sqrt(self.periods_per_year)

        if tracking_error == 0:
            return 0.0

        # Annualized excess return
        ann_excess = excess.mean() * self.periods_per_year

        return ann_excess / tracking_error

    def get_all_metrics(self, max_drawdown: Optional[float] = None) -> Dict:
        """
        Get all metrics as a dictionary.

        Args:
            max_drawdown: Pre-calculated max drawdown (if available).

        Returns:
            Dictionary with all metrics.
        """
        if max_drawdown is None:
            max_dd, peak_date, trough_date = self.max_drawdown()
        else:
            max_dd = max_drawdown
            peak_date = None
            trough_date = None

        return {
            'total_return': self.total_return(),
            'annualized_return': self.annualized_return(),
            'volatility': self.volatility(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'calmar_ratio': self.calmar_ratio(max_dd),
            'max_drawdown': max_dd,
            'max_drawdown_peak': peak_date,
            'max_drawdown_trough': trough_date,
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'trading_days': len(self.returns)
        }


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate returns from price series.

    Args:
        prices: Series of prices.

    Returns:
        Series of returns.
    """
    return prices.pct_change().dropna()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns from returns series.

    Args:
        returns: Series of returns.

    Returns:
        Series of cumulative returns (starting at 0).
    """
    return (1 + returns).cumprod() - 1
