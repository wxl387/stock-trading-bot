"""
Benchmark comparison for portfolio analytics.
Compares portfolio performance against market benchmarks (SPY).
"""
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BenchmarkComparison:
    """
    Compare portfolio performance against a benchmark (default: SPY).

    Calculates alpha, beta, information ratio, and cumulative comparison.
    """

    def __init__(
        self,
        portfolio_values: pd.Series,
        benchmark_symbol: str = "SPY",
        risk_free_rate: float = 0.05
    ):
        """
        Initialize BenchmarkComparison.

        Args:
            portfolio_values: Series of portfolio values with datetime index.
            benchmark_symbol: Benchmark ticker symbol (default SPY).
            risk_free_rate: Annual risk-free rate for alpha calculation.
        """
        self.portfolio_values = portfolio_values
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate

        self._benchmark_data: Optional[pd.Series] = None
        self._aligned_data: Optional[pd.DataFrame] = None

    def fetch_benchmark_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.Series:
        """
        Fetch benchmark price data.

        Args:
            start_date: Start date (defaults to portfolio start).
            end_date: End date (defaults to portfolio end).

        Returns:
            Series of benchmark prices.
        """
        if start_date is None and len(self.portfolio_values) > 0:
            start_date = self.portfolio_values.index.min()
        if end_date is None and len(self.portfolio_values) > 0:
            end_date = self.portfolio_values.index.max()

        # Add buffer for alignment
        if start_date:
            start_date = start_date - timedelta(days=7)

        try:
            from src.data.data_fetcher import DataFetcher
            fetcher = DataFetcher()

            df = fetcher.fetch_stock_data(
                symbol=self.benchmark_symbol,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and not df.empty:
                if 'close' in df.columns:
                    self._benchmark_data = df['close']
                else:
                    self._benchmark_data = df.iloc[:, 0]

                logger.info(f"Fetched {len(self._benchmark_data)} days of {self.benchmark_symbol} data")
                return self._benchmark_data

        except Exception as e:
            logger.error(f"Error fetching benchmark data: {e}")

        return pd.Series(dtype=float)

    def _align_data(self) -> pd.DataFrame:
        """Align portfolio and benchmark data by date."""
        if self._aligned_data is not None:
            return self._aligned_data

        if self._benchmark_data is None or len(self._benchmark_data) == 0:
            self.fetch_benchmark_data()

        if self._benchmark_data is None or len(self._benchmark_data) == 0:
            return pd.DataFrame()

        # Calculate returns
        portfolio_returns = self.portfolio_values.pct_change().dropna()
        benchmark_returns = self._benchmark_data.pct_change().dropna()

        # Align by date
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join='inner')
        aligned.columns = ['portfolio', 'benchmark']

        self._aligned_data = aligned.dropna()
        return self._aligned_data

    def calculate_beta(self) -> float:
        """
        Calculate portfolio beta relative to benchmark.

        Beta = Cov(portfolio, benchmark) / Var(benchmark)
        """
        aligned = self._align_data()
        if len(aligned) < 10:
            return 1.0

        cov = aligned['portfolio'].cov(aligned['benchmark'])
        var = aligned['benchmark'].var()

        if var == 0:
            return 1.0

        return cov / var

    def calculate_alpha(self) -> float:
        """
        Calculate Jensen's alpha (annualized).

        Alpha = portfolio_return - (risk_free + beta * (market_return - risk_free))
        """
        aligned = self._align_data()
        if len(aligned) < 10:
            return 0.0

        # Annualized returns
        portfolio_annual = (1 + aligned['portfolio'].mean()) ** 252 - 1
        benchmark_annual = (1 + aligned['benchmark'].mean()) ** 252 - 1

        beta = self.calculate_beta()

        # CAPM expected return
        expected_return = self.risk_free_rate + beta * (benchmark_annual - self.risk_free_rate)

        return portfolio_annual - expected_return

    def information_ratio(self) -> float:
        """
        Calculate information ratio.

        IR = (portfolio_return - benchmark_return) / tracking_error
        """
        aligned = self._align_data()
        if len(aligned) < 10:
            return 0.0

        # Excess returns
        excess = aligned['portfolio'] - aligned['benchmark']

        # Tracking error (annualized)
        tracking_error = excess.std() * np.sqrt(252)

        if tracking_error == 0:
            return 0.0

        # Annualized excess return
        ann_excess = excess.mean() * 252

        return ann_excess / tracking_error

    def tracking_error(self) -> float:
        """
        Calculate annualized tracking error.

        Tracking error = annualized std of return differences.
        """
        aligned = self._align_data()
        if len(aligned) < 2:
            return 0.0

        excess = aligned['portfolio'] - aligned['benchmark']
        return excess.std() * np.sqrt(252)

    def correlation(self) -> float:
        """Calculate correlation with benchmark."""
        aligned = self._align_data()
        if len(aligned) < 10:
            return 0.0

        return aligned['portfolio'].corr(aligned['benchmark'])

    def cumulative_comparison(self) -> pd.DataFrame:
        """
        Get cumulative returns comparison.

        Returns:
            DataFrame with cumulative returns for portfolio and benchmark.
        """
        aligned = self._align_data()
        if len(aligned) == 0:
            return pd.DataFrame()

        cumulative = pd.DataFrame({
            'portfolio': (1 + aligned['portfolio']).cumprod() - 1,
            'benchmark': (1 + aligned['benchmark']).cumprod() - 1
        })

        # Add relative performance
        cumulative['excess'] = cumulative['portfolio'] - cumulative['benchmark']

        return cumulative

    def relative_performance(self) -> float:
        """
        Calculate total relative performance vs benchmark.

        Returns:
            Portfolio total return - Benchmark total return.
        """
        aligned = self._align_data()
        if len(aligned) == 0:
            return 0.0

        portfolio_total = (1 + aligned['portfolio']).prod() - 1
        benchmark_total = (1 + aligned['benchmark']).prod() - 1

        return portfolio_total - benchmark_total

    def up_capture_ratio(self) -> float:
        """
        Calculate up-market capture ratio.

        Measures portfolio performance during positive benchmark periods.
        """
        aligned = self._align_data()
        if len(aligned) < 10:
            return 1.0

        up_market = aligned[aligned['benchmark'] > 0]
        if len(up_market) == 0:
            return 1.0

        portfolio_up = (1 + up_market['portfolio']).prod() - 1
        benchmark_up = (1 + up_market['benchmark']).prod() - 1

        if benchmark_up == 0:
            return 1.0

        return portfolio_up / benchmark_up

    def down_capture_ratio(self) -> float:
        """
        Calculate down-market capture ratio.

        Measures portfolio performance during negative benchmark periods.
        Lower is better (less downside captured).
        """
        aligned = self._align_data()
        if len(aligned) < 10:
            return 1.0

        down_market = aligned[aligned['benchmark'] < 0]
        if len(down_market) == 0:
            return 1.0

        portfolio_down = (1 + down_market['portfolio']).prod() - 1
        benchmark_down = (1 + down_market['benchmark']).prod() - 1

        if benchmark_down == 0:
            return 1.0

        return portfolio_down / benchmark_down

    def get_all_metrics(self) -> Dict:
        """
        Get all benchmark comparison metrics.

        Returns:
            Dictionary with all metrics.
        """
        aligned = self._align_data()

        if len(aligned) == 0:
            return {
                'alpha': 0.0,
                'beta': 1.0,
                'information_ratio': 0.0,
                'tracking_error': 0.0,
                'correlation': 0.0,
                'relative_performance': 0.0,
                'up_capture': 1.0,
                'down_capture': 1.0,
                'benchmark_symbol': self.benchmark_symbol,
                'trading_days': 0
            }

        return {
            'alpha': self.calculate_alpha(),
            'beta': self.calculate_beta(),
            'information_ratio': self.information_ratio(),
            'tracking_error': self.tracking_error(),
            'correlation': self.correlation(),
            'relative_performance': self.relative_performance(),
            'up_capture': self.up_capture_ratio(),
            'down_capture': self.down_capture_ratio(),
            'benchmark_symbol': self.benchmark_symbol,
            'trading_days': len(aligned)
        }
