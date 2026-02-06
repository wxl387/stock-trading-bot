"""
Fundamental data fetcher for financial metrics and company data.

Fetches financial data including valuation ratios, profitability metrics,
growth rates, balance sheet health, cash flow, and analyst data.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

import pandas as pd
import numpy as np
import yfinance as yf

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

# Module-level singleton
_fundamental_fetcher: Optional["FundamentalFetcher"] = None
_lock = threading.Lock()


def get_fundamental_fetcher() -> "FundamentalFetcher":
    """Get or create the singleton FundamentalFetcher instance."""
    global _fundamental_fetcher
    with _lock:
        if _fundamental_fetcher is None:
            _fundamental_fetcher = FundamentalFetcher()
        return _fundamental_fetcher


@dataclass
class FundamentalData:
    """Container for fundamental data of a stock."""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Valuation metrics
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    enterprise_value: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    ev_to_revenue: Optional[float] = None

    # Profitability metrics
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    roic: Optional[float] = None  # Return on Invested Capital

    # Growth metrics
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    revenue_growth_quarterly: Optional[float] = None
    earnings_growth_quarterly: Optional[float] = None

    # Balance sheet health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    total_debt: Optional[float] = None
    total_cash: Optional[float] = None
    book_value: Optional[float] = None

    # Cash flow metrics
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    fcf_per_share: Optional[float] = None

    # Dividend metrics
    dividend_yield: Optional[float] = None
    dividend_payout_ratio: Optional[float] = None

    # Analyst data
    analyst_rating: Optional[str] = None
    analyst_target_price: Optional[float] = None
    num_analysts: Optional[int] = None
    target_upside: Optional[float] = None

    # Company info
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None
    float_shares: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None

    # Earnings info
    next_earnings_date: Optional[datetime] = None
    days_to_earnings: Optional[int] = None
    last_earnings_date: Optional[datetime] = None
    earnings_surprise_pct: Optional[float] = None

    # Data quality flag
    fetch_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    def to_features(self) -> Dict[str, float]:
        """Convert to feature dictionary for ML models.

        If fetch_failed is True, returns NaN values so downstream code
        (e.g., pandas dropna) can properly handle missing data rather than
        treating API failures as zero-valued fundamentals.
        """
        fallback = float('nan') if self.fetch_failed else 0.0
        features = {
            "pe_ratio": self.pe_ratio if self.pe_ratio is not None else fallback,
            "forward_pe": self.forward_pe if self.forward_pe is not None else fallback,
            "peg_ratio": self.peg_ratio if self.peg_ratio is not None else fallback,
            "price_to_book": self.price_to_book if self.price_to_book is not None else fallback,
            "price_to_sales": self.price_to_sales if self.price_to_sales is not None else fallback,
            "profit_margin": self.profit_margin if self.profit_margin is not None else fallback,
            "operating_margin": self.operating_margin if self.operating_margin is not None else fallback,
            "gross_margin": self.gross_margin if self.gross_margin is not None else fallback,
            "roe": self.roe if self.roe is not None else fallback,
            "roa": self.roa if self.roa is not None else fallback,
            "revenue_growth_yoy": self.revenue_growth_yoy if self.revenue_growth_yoy is not None else fallback,
            "earnings_growth_yoy": self.earnings_growth_yoy if self.earnings_growth_yoy is not None else fallback,
            "debt_to_equity": self.debt_to_equity if self.debt_to_equity is not None else fallback,
            "current_ratio": self.current_ratio if self.current_ratio is not None else fallback,
            "quick_ratio": self.quick_ratio if self.quick_ratio is not None else fallback,
            "dividend_yield": self.dividend_yield if self.dividend_yield is not None else fallback,
            "days_to_earnings": float(self.days_to_earnings) if self.days_to_earnings is not None else fallback,
            "target_upside": self.target_upside if self.target_upside is not None else fallback,
        }
        return features


class FundamentalFetcher:
    """
    Fetches fundamental financial data for stocks.

    Uses Yahoo Finance (yfinance) as the primary data source.
    Supports caching to reduce API calls.
    """

    DEFAULT_CACHE_HOURS = 24  # Fundamentals don't change frequently

    def __init__(self, cache_dir: Optional[Path] = None, cache_hours: int = DEFAULT_CACHE_HOURS):
        """
        Initialize FundamentalFetcher.

        Args:
            cache_dir: Directory for caching data
            cache_hours: Hours to cache fundamental data
        """
        self.cache_dir = cache_dir or DATA_DIR / "cache" / "fundamentals"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hours = cache_hours

        # In-memory cache
        self._cache: Dict[str, FundamentalData] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    def fetch_fundamentals(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> FundamentalData:
        """
        Fetch fundamental data for a symbol.

        Args:
            symbol: Stock ticker symbol
            use_cache: Whether to use cached data

        Returns:
            FundamentalData object with all available metrics
        """
        # Check memory cache
        if use_cache and self._is_cache_valid(symbol):
            logger.debug(f"Returning cached fundamentals for {symbol}")
            return self._cache[symbol]

        # Check file cache
        if use_cache:
            cached_data = self._load_from_file_cache(symbol)
            if cached_data:
                self._cache[symbol] = cached_data
                self._cache_timestamps[symbol] = datetime.now()
                return cached_data

        # Fetch from API
        logger.info(f"Fetching fundamental data for {symbol}")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Create FundamentalData object
            data = FundamentalData(symbol=symbol)

            # Valuation metrics
            data.pe_ratio = self._safe_get(info, "trailingPE")
            data.forward_pe = self._safe_get(info, "forwardPE")
            data.peg_ratio = self._safe_get(info, "pegRatio")
            data.price_to_book = self._safe_get(info, "priceToBook")
            data.price_to_sales = self._safe_get(info, "priceToSalesTrailing12Months")
            data.enterprise_value = self._safe_get(info, "enterpriseValue")
            data.ev_to_ebitda = self._safe_get(info, "enterpriseToEbitda")
            data.ev_to_revenue = self._safe_get(info, "enterpriseToRevenue")

            # Profitability metrics
            data.profit_margin = self._safe_get(info, "profitMargins")
            data.operating_margin = self._safe_get(info, "operatingMargins")
            data.gross_margin = self._safe_get(info, "grossMargins")
            data.roe = self._safe_get(info, "returnOnEquity")
            data.roa = self._safe_get(info, "returnOnAssets")

            # Growth metrics
            data.revenue_growth_yoy = self._safe_get(info, "revenueGrowth")
            data.earnings_growth_yoy = self._safe_get(info, "earningsGrowth")
            data.revenue_growth_quarterly = self._safe_get(info, "revenueQuarterlyGrowth")
            data.earnings_growth_quarterly = self._safe_get(info, "earningsQuarterlyGrowth")

            # Balance sheet health
            data.debt_to_equity = self._safe_get(info, "debtToEquity")
            if data.debt_to_equity is not None:
                data.debt_to_equity = data.debt_to_equity / 100  # Convert from percentage
            data.current_ratio = self._safe_get(info, "currentRatio")
            data.quick_ratio = self._safe_get(info, "quickRatio")
            data.total_debt = self._safe_get(info, "totalDebt")
            data.total_cash = self._safe_get(info, "totalCash")
            data.book_value = self._safe_get(info, "bookValue")

            # Cash flow metrics
            data.free_cash_flow = self._safe_get(info, "freeCashflow")
            data.operating_cash_flow = self._safe_get(info, "operatingCashflow")

            # Calculate FCF per share
            shares = self._safe_get(info, "sharesOutstanding")
            if data.free_cash_flow and shares:
                data.fcf_per_share = data.free_cash_flow / shares

            # Dividend metrics
            data.dividend_yield = self._safe_get(info, "dividendYield")
            data.dividend_payout_ratio = self._safe_get(info, "payoutRatio")

            # Analyst data
            data.analyst_rating = self._safe_get(info, "recommendationKey")
            data.analyst_target_price = self._safe_get(info, "targetMeanPrice")
            data.num_analysts = self._safe_get(info, "numberOfAnalystOpinions")

            # Calculate target upside
            current_price = self._safe_get(info, "currentPrice") or self._safe_get(info, "regularMarketPrice")
            if data.analyst_target_price and current_price:
                data.target_upside = (data.analyst_target_price - current_price) / current_price

            # Company info
            data.market_cap = self._safe_get(info, "marketCap")
            data.shares_outstanding = shares
            data.float_shares = self._safe_get(info, "floatShares")
            data.sector = self._safe_get(info, "sector")
            data.industry = self._safe_get(info, "industry")

            # Earnings info
            self._fetch_earnings_info(ticker, data)

            # Update cache
            self._cache[symbol] = data
            self._cache_timestamps[symbol] = datetime.now()

            # Save to file cache
            self._save_to_file_cache(symbol, data)

            return data

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            # Return empty data object marked as failed so to_features() returns NaN
            return FundamentalData(symbol=symbol, fetch_failed=True)

    def fetch_multiple(
        self,
        symbols: List[str],
        use_cache: bool = True
    ) -> Dict[str, FundamentalData]:
        """
        Fetch fundamental data for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            use_cache: Whether to use cached data

        Returns:
            Dictionary mapping symbol to FundamentalData
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.fetch_fundamentals(symbol, use_cache)
            except Exception as e:
                logger.error(f"Failed to fetch fundamentals for {symbol}: {e}")
                results[symbol] = FundamentalData(symbol=symbol, fetch_failed=True)
        return results

    def get_market_percentiles(
        self,
        symbols: List[str],
        metric: str = "pe_ratio"
    ) -> Dict[str, float]:
        """
        Calculate percentile rank of each symbol for a given metric.

        Args:
            symbols: List of symbols to compare
            metric: Metric to calculate percentiles for

        Returns:
            Dictionary mapping symbol to percentile (0-100)
        """
        # Fetch all fundamentals
        fundamentals = self.fetch_multiple(symbols)

        # Extract metric values
        values = {}
        for symbol, data in fundamentals.items():
            value = getattr(data, metric, None)
            if value is not None and not np.isnan(value):
                values[symbol] = value

        if not values:
            return {s: 50.0 for s in symbols}  # Default to median

        # Calculate percentiles
        all_values = list(values.values())
        percentiles = {}

        for symbol in symbols:
            if symbol in values:
                # Calculate percentile rank
                value = values[symbol]
                rank = sum(1 for v in all_values if v <= value)
                percentiles[symbol] = (rank / len(all_values)) * 100
            else:
                percentiles[symbol] = 50.0  # Default to median

        return percentiles

    def get_fundamental_features(
        self,
        symbol: str,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Add fundamental features to a price DataFrame.

        Args:
            symbol: Stock symbol
            df: DataFrame with price data (index should be datetime)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with added fundamental features
        """
        df = df.copy()

        # Fetch fundamentals
        fundamentals = self.fetch_fundamentals(symbol, use_cache)
        features = fundamentals.to_features()

        # Add features as constant columns
        for feature_name, value in features.items():
            df[feature_name] = value

        return df

    def _fetch_earnings_info(self, ticker: yf.Ticker, data: FundamentalData) -> None:
        """Fetch earnings calendar information."""
        try:
            calendar = ticker.calendar
            if calendar is not None and not calendar.empty:
                # Handle different calendar formats
                if isinstance(calendar, pd.DataFrame):
                    # yfinance sometimes returns DataFrame
                    if 'Earnings Date' in calendar.columns:
                        earnings_dates = calendar['Earnings Date'].values
                        if len(earnings_dates) > 0:
                            next_date = pd.Timestamp(earnings_dates[0])
                            if pd.notna(next_date):
                                data.next_earnings_date = next_date.to_pydatetime()
                    elif 0 in calendar.columns:
                        # Sometimes it's indexed differently
                        try:
                            next_date = calendar.loc['Earnings Date', 0]
                            if pd.notna(next_date):
                                data.next_earnings_date = pd.Timestamp(next_date).to_pydatetime()
                        except (KeyError, TypeError):
                            pass
                elif isinstance(calendar, dict):
                    # Dictionary format
                    if 'Earnings Date' in calendar:
                        dates = calendar['Earnings Date']
                        if isinstance(dates, list) and len(dates) > 0:
                            next_date = dates[0]
                            if pd.notna(next_date):
                                data.next_earnings_date = pd.Timestamp(next_date).to_pydatetime()

            # Calculate days to earnings
            if data.next_earnings_date:
                days = (data.next_earnings_date - datetime.now()).days
                data.days_to_earnings = max(0, days)

            # Get earnings history for surprise data
            try:
                earnings_history = ticker.earnings_history
                if earnings_history is not None and not earnings_history.empty:
                    if 'Surprise(%)' in earnings_history.columns:
                        latest_surprise = earnings_history['Surprise(%)'].iloc[-1]
                        if pd.notna(latest_surprise):
                            data.earnings_surprise_pct = latest_surprise / 100

                    # Get last earnings date
                    if 'Earnings Date' in earnings_history.columns:
                        last_date = earnings_history['Earnings Date'].iloc[-1]
                        if pd.notna(last_date):
                            data.last_earnings_date = pd.Timestamp(last_date).to_pydatetime()
            except Exception:
                pass

        except Exception as e:
            logger.debug(f"Could not fetch earnings info: {e}")

    def _safe_get(self, info: Dict, key: str) -> Optional[float]:
        """Safely get a value from info dict, handling various edge cases."""
        try:
            value = info.get(key)
            if value is None:
                return None
            if isinstance(value, str):
                return None
            if np.isnan(value) or np.isinf(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid."""
        if symbol not in self._cache or symbol not in self._cache_timestamps:
            return False

        cache_age = datetime.now() - self._cache_timestamps[symbol]
        return cache_age < timedelta(hours=self.cache_hours)

    def _load_from_file_cache(self, symbol: str) -> Optional[FundamentalData]:
        """Load cached data from file."""
        cache_file = self.cache_dir / f"{symbol}_fundamentals.json"

        if not cache_file.exists():
            return None

        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > timedelta(hours=self.cache_hours):
            return None

        try:
            import json
            with open(cache_file, 'r') as f:
                data = json.load(f)

            # Convert back to FundamentalData
            fundamental = FundamentalData(symbol=symbol)
            for key, value in data.items():
                if key == 'timestamp' and value:
                    setattr(fundamental, key, datetime.fromisoformat(value))
                elif key in ['next_earnings_date', 'last_earnings_date'] and value:
                    setattr(fundamental, key, datetime.fromisoformat(value))
                elif hasattr(fundamental, key):
                    setattr(fundamental, key, value)

            return fundamental

        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
            return None

    def _save_to_file_cache(self, symbol: str, data: FundamentalData) -> None:
        """Save data to file cache."""
        cache_file = self.cache_dir / f"{symbol}_fundamentals.json"

        try:
            import json
            with open(cache_file, 'w') as f:
                json.dump(data.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {e}")

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached data.

        Args:
            symbol: Specific symbol to clear, or None to clear all
        """
        if symbol:
            self._cache.pop(symbol, None)
            self._cache_timestamps.pop(symbol, None)
            cache_file = self.cache_dir / f"{symbol}_fundamentals.json"
            if cache_file.exists():
                cache_file.unlink()
        else:
            self._cache.clear()
            self._cache_timestamps.clear()
            for cache_file in self.cache_dir.glob("*_fundamentals.json"):
                cache_file.unlink()
