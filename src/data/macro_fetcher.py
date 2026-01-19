"""
Macroeconomic data fetcher using FRED API.
Fetches VIX, unemployment, CPI, and other economic indicators.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import pandas as pd
import numpy as np

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


class MacroFetcher:
    """
    Fetches macroeconomic indicators from FRED API.

    Available indicators:
    - VIX: Volatility Index (CBOE)
    - UNRATE: Unemployment Rate
    - CPIAUCSL: Consumer Price Index
    - GDP: Gross Domestic Product
    - DGS10: 10-Year Treasury Rate
    - FEDFUNDS: Federal Funds Rate
    """

    # FRED series IDs
    SERIES = {
        "vix": "VIXCLS",           # VIX daily
        "unemployment": "UNRATE",   # Monthly
        "cpi": "CPIAUCSL",          # Monthly
        "gdp": "GDP",               # Quarterly
        "treasury_10y": "DGS10",    # Daily
        "fed_funds": "FEDFUNDS",    # Monthly
        "sp500": "SP500",           # Daily
        "industrial_production": "INDPRO",  # Monthly
    }

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        cache_days: int = 1
    ):
        """
        Initialize macro fetcher.

        Args:
            fred_api_key: FRED API key (from env FRED_API_KEY)
            cache_days: Days to cache data
        """
        self.fred_api_key = fred_api_key or os.getenv("FRED_API_KEY")
        self.cache_days = cache_days

        # Cache directory
        self.cache_dir = DATA_DIR / "macro_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Session for API calls
        self.session = requests.Session()

        # Check if fredapi is available
        self.fredapi_available = False
        try:
            import fredapi
            if self.fred_api_key:
                self.fred = fredapi.Fred(api_key=self.fred_api_key)
                self.fredapi_available = True
        except ImportError:
            pass

        logger.info(f"MacroFetcher initialized. "
                   f"FRED API: {'available' if self.fred_api_key else 'N/A'}, "
                   f"fredapi: {'installed' if self.fredapi_available else 'N/A'}")

    def fetch_indicator(
        self,
        indicator: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch a single macroeconomic indicator.

        Args:
            indicator: Indicator name (vix, unemployment, cpi, etc.)
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data

        Returns:
            DataFrame with date index and indicator value
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        series_id = self.SERIES.get(indicator.lower())
        if series_id is None:
            logger.warning(f"Unknown indicator: {indicator}")
            return pd.DataFrame()

        # Check cache
        if use_cache:
            cached = self._load_cache(indicator)
            if cached is not None:
                # Filter to date range
                cached.index = pd.to_datetime(cached.index)
                return cached[(cached.index >= start_date) & (cached.index <= end_date)]

        # Fetch from FRED
        if self.fredapi_available:
            try:
                data = self._fetch_with_fredapi(series_id, start_date, end_date)
                if data is not None and len(data) > 0:
                    self._save_cache(indicator, data)
                    return data
            except Exception as e:
                logger.warning(f"fredapi fetch failed for {indicator}: {e}")

        # Fallback to direct API
        if self.fred_api_key:
            try:
                data = self._fetch_with_api(series_id, start_date, end_date)
                if data is not None and len(data) > 0:
                    self._save_cache(indicator, data)
                    return data
            except Exception as e:
                logger.warning(f"FRED API fetch failed for {indicator}: {e}")

        logger.warning(f"Could not fetch {indicator}")
        return pd.DataFrame()

    def fetch_all_indicators(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        indicators: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch multiple indicators and combine into single DataFrame.

        Args:
            start_date: Start date
            end_date: End date
            indicators: List of indicators (default: common ones)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with date index and all indicators
        """
        if indicators is None:
            indicators = ["vix", "unemployment", "cpi", "treasury_10y", "fed_funds"]

        all_data = {}

        for indicator in indicators:
            try:
                data = self.fetch_indicator(indicator, start_date, end_date, use_cache)
                if not data.empty:
                    all_data[indicator] = data.iloc[:, 0]  # First column
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {e}")

        if not all_data:
            return pd.DataFrame()

        # Combine all series
        combined = pd.DataFrame(all_data)

        # Forward fill (for monthly/quarterly data)
        combined = combined.ffill()

        return combined

    def _fetch_with_fredapi(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch using fredapi library."""
        data = self.fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date
        )

        if data is None or len(data) == 0:
            return None

        df = pd.DataFrame(data, columns=['value'])
        df.index.name = 'date'
        return df

    def _fetch_with_api(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch using direct FRED API."""
        url = "https://api.stlouisfed.org/fred/series/observations"

        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "observation_start": start_date.strftime("%Y-%m-%d"),
            "observation_end": end_date.strftime("%Y-%m-%d")
        }

        response = self.session.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        if "observations" not in data:
            return None

        observations = data["observations"]
        if not observations:
            return None

        records = []
        for obs in observations:
            try:
                value = float(obs["value"])
                records.append({
                    "date": obs["date"],
                    "value": value
                })
            except (ValueError, KeyError):
                continue

        if not records:
            return None

        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        return df

    def _load_cache(self, indicator: str) -> Optional[pd.DataFrame]:
        """Load cached indicator data."""
        cache_file = self.cache_dir / f"{indicator}_macro.csv"

        if not cache_file.exists():
            return None

        # Check age
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age.total_seconds() > self.cache_days * 86400:
            return None

        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        except Exception:
            return None

    def _save_cache(self, indicator: str, df: pd.DataFrame) -> None:
        """Save indicator data to cache."""
        cache_file = self.cache_dir / f"{indicator}_macro.csv"
        df.to_csv(cache_file)

    def get_macro_features(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Add macroeconomic features to a price DataFrame.

        Args:
            df: Price DataFrame with date index or column
            indicators: List of indicators to add
            use_cache: Whether to use cached macro data

        Returns:
            DataFrame with added macro columns
        """
        if indicators is None:
            indicators = ["vix", "unemployment", "cpi", "treasury_10y"]

        # Ensure date is accessible
        if 'date' not in df.columns and df.index.name != 'date':
            df = df.reset_index()

        # Get date range
        if 'date' in df.columns:
            min_date = pd.to_datetime(df['date'].min())
            max_date = pd.to_datetime(df['date'].max())
        else:
            min_date = pd.to_datetime(df.index.min())
            max_date = pd.to_datetime(df.index.max())

        # Fetch macro data
        macro_df = self.fetch_all_indicators(
            start_date=min_date - timedelta(days=90),  # Extra for rolling
            end_date=max_date,
            indicators=indicators,
            use_cache=use_cache
        )

        if macro_df.empty:
            # Add zero features
            for ind in indicators:
                df[ind] = 0.0
                df[f'{ind}_ma20'] = 0.0
            return df

        # Resample to daily (forward fill for monthly data)
        macro_df = macro_df.resample('D').ffill()

        # Merge with price data
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            merged = df.join(macro_df, how='left')
            merged = merged.reset_index()
        else:
            df.index = pd.to_datetime(df.index)
            merged = df.join(macro_df, how='left')

        # Fill missing and add rolling features
        for ind in indicators:
            if ind in merged.columns:
                merged[ind] = merged[ind].ffill().fillna(0)
                merged[f'{ind}_ma20'] = merged[ind].rolling(20, min_periods=1).mean()
            else:
                merged[ind] = 0.0
                merged[f'{ind}_ma20'] = 0.0

        return merged


# Singleton instance
_macro_fetcher: Optional[MacroFetcher] = None


def get_macro_fetcher() -> MacroFetcher:
    """Get singleton macro fetcher instance."""
    global _macro_fetcher
    if _macro_fetcher is None:
        _macro_fetcher = MacroFetcher()
    return _macro_fetcher
