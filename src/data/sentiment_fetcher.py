"""
Sentiment analysis fetcher for stock news.
Fetches and aggregates news sentiment from multiple sources.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import pandas as pd
import numpy as np
from functools import lru_cache

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


class SentimentFetcher:
    """
    Fetches news sentiment for stocks.

    Sources (in order of preference):
    1. Finnhub API - Company news with sentiment
    2. Alpha Vantage - News sentiment API
    3. Fallback to neutral sentiment if APIs unavailable
    """

    def __init__(
        self,
        finnhub_api_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None,
        cache_hours: int = 4
    ):
        """
        Initialize sentiment fetcher.

        Args:
            finnhub_api_key: Finnhub API key (from env FINNHUB_API_KEY)
            alpha_vantage_key: Alpha Vantage API key (from env ALPHA_VANTAGE_API_KEY)
            cache_hours: Hours to cache sentiment data
        """
        self.finnhub_api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.cache_hours = cache_hours

        # Cache directory
        self.cache_dir = DATA_DIR / "sentiment_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Session for API calls
        self.session = requests.Session()

        logger.info(f"SentimentFetcher initialized. "
                   f"Finnhub: {'available' if self.finnhub_api_key else 'N/A'}, "
                   f"Alpha Vantage: {'available' if self.alpha_vantage_key else 'N/A'}")

    def fetch_sentiment(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch sentiment data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for sentiment data
            end_date: End date for sentiment data
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: date, sentiment_score, sentiment_label, article_count
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Check cache
        if use_cache:
            cached = self._load_cache(symbol)
            if cached is not None and len(cached) > 0:
                # Filter to date range
                cached = cached[(cached['date'] >= start_date.strftime('%Y-%m-%d')) &
                               (cached['date'] <= end_date.strftime('%Y-%m-%d'))]
                if len(cached) > 0:
                    return cached

        # Try Finnhub first
        if self.finnhub_api_key:
            try:
                sentiment_df = self._fetch_finnhub_sentiment(symbol, start_date, end_date)
                if sentiment_df is not None and len(sentiment_df) > 0:
                    self._save_cache(symbol, sentiment_df)
                    return sentiment_df
            except Exception as e:
                logger.warning(f"Finnhub sentiment fetch failed: {e}")

        # Try Alpha Vantage
        if self.alpha_vantage_key:
            try:
                sentiment_df = self._fetch_alpha_vantage_sentiment(symbol)
                if sentiment_df is not None and len(sentiment_df) > 0:
                    self._save_cache(symbol, sentiment_df)
                    return sentiment_df
            except Exception as e:
                logger.warning(f"Alpha Vantage sentiment fetch failed: {e}")

        # Return empty DataFrame if no API available
        logger.warning(f"No sentiment data available for {symbol}")
        return self._create_neutral_sentiment(start_date, end_date)

    def _fetch_finnhub_sentiment(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch sentiment from Finnhub API."""
        url = "https://finnhub.io/api/v1/company-news"

        params = {
            "symbol": symbol,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "token": self.finnhub_api_key
        }

        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()

        news_items = response.json()

        if not news_items:
            return None

        # Aggregate sentiment by date
        daily_sentiment = {}

        for item in news_items:
            # Finnhub returns timestamp in seconds
            date = datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d")

            # Finnhub doesn't provide direct sentiment, so we use a simple heuristic
            # based on headline keywords (in production, use NLP)
            headline = item.get("headline", "").lower()
            sentiment = self._analyze_headline_sentiment(headline)

            if date not in daily_sentiment:
                daily_sentiment[date] = {"scores": [], "count": 0}

            daily_sentiment[date]["scores"].append(sentiment)
            daily_sentiment[date]["count"] += 1

        # Convert to DataFrame
        records = []
        for date, data in sorted(daily_sentiment.items()):
            avg_sentiment = np.mean(data["scores"]) if data["scores"] else 0
            records.append({
                "date": date,
                "sentiment_score": avg_sentiment,
                "sentiment_label": self._score_to_label(avg_sentiment),
                "article_count": data["count"]
            })

        return pd.DataFrame(records)

    def _fetch_alpha_vantage_sentiment(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment from Alpha Vantage News API."""
        url = "https://www.alphavantage.co/query"

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "apikey": self.alpha_vantage_key,
            "limit": 200
        }

        response = self.session.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        if "feed" not in data:
            return None

        # Aggregate by date
        daily_sentiment = {}

        for article in data["feed"]:
            # Parse date from time_published
            time_str = article.get("time_published", "")
            if len(time_str) >= 8:
                date = f"{time_str[:4]}-{time_str[4:6]}-{time_str[6:8]}"
            else:
                continue

            # Get ticker-specific sentiment
            ticker_sentiment = None
            for ticker_data in article.get("ticker_sentiment", []):
                if ticker_data.get("ticker") == symbol:
                    ticker_sentiment = float(ticker_data.get("ticker_sentiment_score", 0))
                    break

            if ticker_sentiment is None:
                # Use overall sentiment
                ticker_sentiment = float(article.get("overall_sentiment_score", 0))

            if date not in daily_sentiment:
                daily_sentiment[date] = {"scores": [], "count": 0}

            daily_sentiment[date]["scores"].append(ticker_sentiment)
            daily_sentiment[date]["count"] += 1

        # Convert to DataFrame
        records = []
        for date, data in sorted(daily_sentiment.items()):
            avg_sentiment = np.mean(data["scores"]) if data["scores"] else 0
            records.append({
                "date": date,
                "sentiment_score": avg_sentiment,
                "sentiment_label": self._score_to_label(avg_sentiment),
                "article_count": data["count"]
            })

        return pd.DataFrame(records)

    def _analyze_headline_sentiment(self, headline: str) -> float:
        """
        Simple rule-based sentiment analysis for headlines.
        Returns score between -1 (bearish) and 1 (bullish).
        """
        bullish_words = [
            "surge", "soar", "rally", "jump", "gain", "rise", "profit",
            "beat", "exceed", "bullish", "upgrade", "buy", "strong",
            "growth", "positive", "record", "high", "boom", "success"
        ]

        bearish_words = [
            "drop", "fall", "crash", "plunge", "decline", "loss", "miss",
            "bearish", "downgrade", "sell", "weak", "negative", "low",
            "concern", "worry", "risk", "fail", "warning", "cut"
        ]

        bullish_count = sum(1 for word in bullish_words if word in headline)
        bearish_count = sum(1 for word in bearish_words if word in headline)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0

        return (bullish_count - bearish_count) / total

    def _score_to_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.2:
            return "bullish"
        elif score < -0.2:
            return "bearish"
        else:
            return "neutral"

    def _create_neutral_sentiment(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Create neutral sentiment DataFrame when no data available."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame({
            "date": [d.strftime("%Y-%m-%d") for d in dates],
            "sentiment_score": [0.0] * len(dates),
            "sentiment_label": ["neutral"] * len(dates),
            "article_count": [0] * len(dates)
        })

    def _load_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached sentiment data."""
        cache_file = self.cache_dir / f"{symbol}_sentiment.csv"

        if not cache_file.exists():
            return None

        # Check age
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age.total_seconds() > self.cache_hours * 3600:
            return None

        try:
            return pd.read_csv(cache_file)
        except Exception:
            return None

    def _save_cache(self, symbol: str, df: pd.DataFrame) -> None:
        """Save sentiment data to cache."""
        cache_file = self.cache_dir / f"{symbol}_sentiment.csv"
        df.to_csv(cache_file, index=False)

    def get_sentiment_features(
        self,
        symbol: str,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Add sentiment features to a price DataFrame.

        Args:
            symbol: Stock symbol
            df: Price DataFrame with date index
            use_cache: Whether to use cached sentiment

        Returns:
            DataFrame with added sentiment columns
        """
        # Ensure date column exists
        if 'date' not in df.columns and df.index.name != 'date':
            df = df.reset_index()

        # Get date range from price data
        if 'date' in df.columns:
            min_date = pd.to_datetime(df['date'].min())
            max_date = pd.to_datetime(df['date'].max())
        else:
            min_date = pd.to_datetime(df.index.min())
            max_date = pd.to_datetime(df.index.max())

        # Fetch sentiment
        sentiment_df = self.fetch_sentiment(
            symbol,
            start_date=min_date - timedelta(days=30),  # Extra for rolling
            end_date=max_date,
            use_cache=use_cache
        )

        if sentiment_df.empty:
            # Add neutral features
            df['sentiment_score'] = 0.0
            df['sentiment_ma5'] = 0.0
            df['sentiment_volatility'] = 0.0
            df['article_count'] = 0
            return df

        # Merge with price data
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            merged = df.merge(
                sentiment_df[['date', 'sentiment_score', 'article_count']],
                on='date',
                how='left'
            )
        else:
            df.index = pd.to_datetime(df.index)
            sentiment_df = sentiment_df.set_index('date')
            merged = df.join(
                sentiment_df[['sentiment_score', 'article_count']],
                how='left'
            )

        # Fill missing values
        merged['sentiment_score'] = merged['sentiment_score'].fillna(0)
        merged['article_count'] = merged['article_count'].fillna(0)

        # Add rolling features
        merged['sentiment_ma5'] = merged['sentiment_score'].rolling(5, min_periods=1).mean()
        merged['sentiment_volatility'] = merged['sentiment_score'].rolling(10, min_periods=1).std().fillna(0)

        return merged


# Singleton instance
_sentiment_fetcher: Optional[SentimentFetcher] = None


def get_sentiment_fetcher() -> SentimentFetcher:
    """Get singleton sentiment fetcher instance."""
    global _sentiment_fetcher
    if _sentiment_fetcher is None:
        _sentiment_fetcher = SentimentFetcher()
    return _sentiment_fetcher
