"""
Data fetching module for historical and real-time market data.
"""
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
import yfinance as yf

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches historical and real-time market data from multiple sources.
    Primary source: Yahoo Finance (yfinance)
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize DataFetcher.

        Args:
            cache_dir: Directory for caching data. Defaults to data/cache.
        """
        self.cache_dir = cache_dir or DATA_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_historical(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol.
            start_date: Start date (YYYY-MM-DD). If None, uses period.
            end_date: End date (YYYY-MM-DD). If None, uses today.
            period: Period to fetch if dates not specified (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo).
            use_cache: Whether to use cached data.

        Returns:
            DataFrame with OHLCV data.
        """
        cache_key = f"{symbol}_{interval}_{start_date or period}_{end_date or 'latest'}"
        cache_file = self.cache_dir / f"{cache_key}.parquet"

        # Check cache
        if use_cache and cache_file.exists():
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age < timedelta(hours=1):  # Cache valid for 1 hour
                logger.debug(f"Loading {symbol} from cache")
                return pd.read_parquet(cache_file)

        # Fetch from Yahoo Finance
        logger.info(f"Fetching historical data for {symbol}")
        try:
            ticker = yf.Ticker(symbol)

            if start_date:
                df = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Standardize column names
            df = self._standardize_columns(df)

            # Add symbol column
            df["symbol"] = symbol

            # Cache the data
            if use_cache:
                df.to_parquet(cache_file)

            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise

    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            period: Period to fetch if dates not specified.
            interval: Data interval.

        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        data = {}
        for symbol in symbols:
            try:
                df = self.fetch_historical(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    period=period,
                    interval=interval
                )
                if not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue

        return data

    def fetch_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        days: int = 5
    ) -> pd.DataFrame:
        """
        Fetch intraday data for a symbol.

        Args:
            symbol: Stock ticker symbol.
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m).
            days: Number of days of intraday data.

        Returns:
            DataFrame with intraday OHLCV data.
        """
        period = f"{days}d"
        return self.fetch_historical(
            symbol=symbol,
            period=period,
            interval=interval,
            use_cache=False  # Don't cache intraday data
        )

    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Latest closing price.
        """
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        try:
            return info.last_price
        except (AttributeError, KeyError):
            try:
                return info.regular_market_previous_close
            except (AttributeError, KeyError):
                return 0.0

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get latest prices for multiple symbols.

        Args:
            symbols: List of stock ticker symbols.

        Returns:
            Dictionary mapping symbol to latest price.
        """
        prices = {}
        for symbol in symbols:
            try:
                prices[symbol] = self.get_latest_price(symbol)
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
        return prices

    def get_company_info(self, symbol: str) -> Dict:
        """
        Get company information.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Dictionary with company info.
        """
        ticker = yf.Ticker(symbol)
        return ticker.info

    def clear_cache(self, older_than_hours: int = 24):
        """
        Clear old cache files.

        Args:
            older_than_hours: Remove files older than this many hours.
        """
        now = datetime.now()
        for cache_file in self.cache_dir.glob("*.parquet"):
            file_age = now - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age > timedelta(hours=older_than_hours):
                cache_file.unlink()
                logger.debug(f"Removed old cache file: {cache_file}")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to lowercase.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with standardized column names.
        """
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits"
        }

        df = df.rename(columns=column_mapping)
        df.index.name = "date"

        return df
