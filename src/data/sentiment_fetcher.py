"""
Sentiment analysis fetcher for stock news.
Uses BlueSky/viaNexus API (MT Newswires) + FinBERT for NLP-based sentiment.
Falls back to Finnhub/Alpha Vantage if BlueSky unavailable.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import pandas as pd
import numpy as np

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)


class SentimentFetcher:
    """
    Fetches news sentiment for stocks using FinBERT NLP analysis.

    Sources (in order of preference):
    1. BlueSky/viaNexus API (MT Newswires) + FinBERT local NLP
    2. Finnhub API + FinBERT
    3. Alpha Vantage News Sentiment API
    4. Fallback to neutral sentiment if all APIs unavailable
    """

    def __init__(
        self,
        bluesky_api_token: Optional[str] = None,
        finnhub_api_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None,
        cache_hours: int = 4
    ):
        self.bluesky_api_token = bluesky_api_token or os.getenv("BLUESKY_API_TOKEN")
        self.finnhub_api_key = finnhub_api_key or os.getenv("FINNHUB_API_KEY")
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.cache_hours = cache_hours

        # Cache directory
        self.cache_dir = DATA_DIR / "sentiment_cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Session for API calls
        self.session = requests.Session()

        # Lazy-loaded components
        self._news_fetcher = None
        self._analyzer = None

        logger.info(f"SentimentFetcher initialized. "
                   f"BlueSky: {'available' if self.bluesky_api_token else 'N/A'}, "
                   f"Finnhub: {'available' if self.finnhub_api_key else 'N/A'}, "
                   f"Alpha Vantage: {'available' if self.alpha_vantage_key else 'N/A'}")

    def _get_news_fetcher(self):
        """Lazy-load NewsFetcher."""
        if self._news_fetcher is None:
            from src.data.news_fetcher import NewsFetcher
            self._news_fetcher = NewsFetcher(api_token=self.bluesky_api_token)
        return self._news_fetcher

    def _get_analyzer(self):
        """Lazy-load FinBERT analyzer."""
        if self._analyzer is None:
            from src.ml.finbert_analyzer import FinBERTAnalyzer
            self._analyzer = FinBERTAnalyzer()
        return self._analyzer

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
            DataFrame with columns: date, sentiment_score, article_count,
            sentiment_dispersion, positive_ratio, sentiment_confidence
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Check cache
        if use_cache:
            cached = self._load_cache(symbol)
            if cached is not None and len(cached) > 0:
                cached = cached[(cached['date'] >= start_date.strftime('%Y-%m-%d')) &
                               (cached['date'] <= end_date.strftime('%Y-%m-%d'))]
                if len(cached) > 0:
                    return cached

        # Try BlueSky + FinBERT first
        if self.bluesky_api_token:
            try:
                sentiment_df = self._fetch_bluesky_finbert(symbol)
                if sentiment_df is not None and len(sentiment_df) > 0:
                    self._save_cache(symbol, sentiment_df)
                    return sentiment_df
            except Exception as e:
                logger.warning(f"BlueSky+FinBERT sentiment failed for {symbol}: {e}")

        # Try Finnhub + FinBERT
        if self.finnhub_api_key:
            try:
                sentiment_df = self._fetch_finnhub_finbert(symbol, start_date, end_date)
                if sentiment_df is not None and len(sentiment_df) > 0:
                    self._save_cache(symbol, sentiment_df)
                    return sentiment_df
            except Exception as e:
                logger.warning(f"Finnhub+FinBERT sentiment failed for {symbol}: {e}")

        # Try Alpha Vantage (has its own sentiment scores)
        if self.alpha_vantage_key:
            try:
                sentiment_df = self._fetch_alpha_vantage_sentiment(symbol)
                if sentiment_df is not None and len(sentiment_df) > 0:
                    self._save_cache(symbol, sentiment_df)
                    return sentiment_df
            except Exception as e:
                logger.warning(f"Alpha Vantage sentiment failed for {symbol}: {e}")

        # Return empty DataFrame if no API available
        logger.warning(f"No sentiment data available for {symbol}")
        return self._create_neutral_sentiment(start_date, end_date)

    def _fetch_bluesky_finbert(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch news from BlueSky API and analyze with FinBERT."""
        news_fetcher = self._get_news_fetcher()
        articles = news_fetcher.fetch_news(symbol, last=50)

        if not articles:
            return None

        # Analyze with FinBERT
        analyzer = self._get_analyzer()
        daily_df = analyzer.analyze_articles(articles)

        if daily_df.empty:
            return None

        # Rename to match expected schema
        result = pd.DataFrame({
            "date": daily_df["date"],
            "sentiment_score": daily_df["sentiment_score"],
            "article_count": daily_df["article_count"],
            "sentiment_dispersion": daily_df["sentiment_dispersion"],
            "positive_ratio": daily_df["positive_ratio"],
            "sentiment_confidence": daily_df["sentiment_confidence"],
        })

        logger.info(f"BlueSky+FinBERT: {len(result)} days of sentiment for {symbol}")
        return result

    def _fetch_finnhub_finbert(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch news from Finnhub and analyze with FinBERT."""
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

        # Convert to article format expected by FinBERT analyzer
        articles = []
        for item in news_items:
            articles.append({
                "headline": item.get("headline", ""),
                "summary": item.get("summary", ""),
                "date": datetime.fromtimestamp(
                    item.get("datetime", 0)
                ).strftime("%Y-%m-%d"),
            })

        # Analyze with FinBERT
        analyzer = self._get_analyzer()
        daily_df = analyzer.analyze_articles(articles)

        if daily_df.empty:
            return None

        result = pd.DataFrame({
            "date": daily_df["date"],
            "sentiment_score": daily_df["sentiment_score"],
            "article_count": daily_df["article_count"],
            "sentiment_dispersion": daily_df["sentiment_dispersion"],
            "positive_ratio": daily_df["positive_ratio"],
            "sentiment_confidence": daily_df["sentiment_confidence"],
        })

        logger.info(f"Finnhub+FinBERT: {len(result)} days of sentiment for {symbol}")
        return result

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
                ticker_sentiment = float(article.get("overall_sentiment_score", 0))

            if date not in daily_sentiment:
                daily_sentiment[date] = {"scores": [], "count": 0}

            daily_sentiment[date]["scores"].append(ticker_sentiment)
            daily_sentiment[date]["count"] += 1

        # Convert to DataFrame
        records = []
        for date, data in sorted(daily_sentiment.items()):
            scores = data["scores"]
            avg_sentiment = np.mean(scores) if scores else 0
            records.append({
                "date": date,
                "sentiment_score": avg_sentiment,
                "article_count": data["count"],
                "sentiment_dispersion": float(np.std(scores)) if len(scores) > 1 else 0.0,
                "positive_ratio": float(np.mean([s > 0.1 for s in scores])) if scores else 0.0,
                "sentiment_confidence": float(np.mean([abs(s) for s in scores])) if scores else 0.0,
            })

        return pd.DataFrame(records) if records else None

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
            "article_count": [0] * len(dates),
            "sentiment_dispersion": [0.0] * len(dates),
            "positive_ratio": [0.0] * len(dates),
            "sentiment_confidence": [0.0] * len(dates),
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
            start_date=min_date - timedelta(days=30),
            end_date=max_date,
            use_cache=use_cache
        )

        if sentiment_df.empty:
            df['sentiment_score'] = 0.0
            df['sentiment_ma5'] = 0.0
            df['sentiment_volatility'] = 0.0
            df['article_count'] = 0
            df['sentiment_momentum'] = 0.0
            df['sentiment_dispersion'] = 0.0
            df['positive_ratio'] = 0.0
            df['news_intensity'] = 0.0
            return df

        # Fix timezone: ensure both sides are tz-naive for merge
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.tz_localize(None)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            merge_cols = ['date', 'sentiment_score', 'article_count',
                         'sentiment_dispersion', 'positive_ratio']
            available_cols = [c for c in merge_cols if c in sentiment_df.columns]
            merged = df.merge(
                sentiment_df[available_cols],
                on='date',
                how='left'
            )
        else:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            sentiment_df = sentiment_df.set_index('date')
            merge_cols = ['sentiment_score', 'article_count',
                         'sentiment_dispersion', 'positive_ratio']
            available_cols = [c for c in merge_cols if c in sentiment_df.columns]
            merged = df.join(
                sentiment_df[available_cols],
                how='left'
            )

        # Fill missing values
        merged['sentiment_score'] = merged['sentiment_score'].fillna(0)
        merged['article_count'] = merged['article_count'].fillna(0)
        merged['sentiment_dispersion'] = merged.get('sentiment_dispersion', pd.Series(0.0, index=merged.index)).fillna(0)
        merged['positive_ratio'] = merged.get('positive_ratio', pd.Series(0.0, index=merged.index)).fillna(0)

        # Derived rolling features
        merged['sentiment_ma5'] = merged['sentiment_score'].rolling(5, min_periods=1).mean()
        merged['sentiment_volatility'] = merged['sentiment_score'].rolling(10, min_periods=1).std().fillna(0)
        merged['sentiment_momentum'] = merged['sentiment_score'].diff(1).fillna(0)
        merged['news_intensity'] = merged['article_count'] / merged['article_count'].rolling(20, min_periods=1).mean().replace(0, 1)

        return merged


# Singleton instance
_sentiment_fetcher: Optional[SentimentFetcher] = None


def get_sentiment_fetcher() -> SentimentFetcher:
    """Get singleton sentiment fetcher instance."""
    global _sentiment_fetcher
    if _sentiment_fetcher is None:
        _sentiment_fetcher = SentimentFetcher()
    return _sentiment_fetcher
