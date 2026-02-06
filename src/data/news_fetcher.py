"""
News fetcher supporting Finnhub and BlueSky APIs.
Fetches financial news articles for sentiment analysis.
"""
import logging
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache"


class NewsFetcher:
    """
    Fetches financial news from Finnhub (preferred) or BlueSky API.
    """

    def __init__(self, provider: Optional[str] = None, api_token: Optional[str] = None):
        """
        Initialize news fetcher.

        Args:
            provider: 'finnhub' or 'bluesky'. Auto-detects if not specified.
            api_token: API token. Auto-loads from environment if not specified.
        """
        self.cache_hours = 4
        self._request_delay = 0.5  # Rate limiting
        self._session = requests.Session()

        # Auto-detect provider based on available API keys
        finnhub_key = api_token if provider == "finnhub" else os.getenv("FINNHUB_API_KEY")
        bluesky_key = api_token if provider == "bluesky" else os.getenv("BLUESKY_API_TOKEN")

        if provider == "finnhub" or (finnhub_key and provider != "bluesky"):
            self.provider = "finnhub"
            self.token = finnhub_key
            self.base_url = "https://finnhub.io/api/v1/company-news"
            if self.token:
                logger.info("NewsFetcher initialized with Finnhub API")
            else:
                logger.warning("No Finnhub API key found. News fetching disabled.")
        elif provider == "bluesky" or bluesky_key:
            self.provider = "bluesky"
            self.token = bluesky_key
            self.base_url = "https://api.blueskyapi.com/v1/data/core/news"
            if self.token:
                logger.info("NewsFetcher initialized with BlueSky API")
            else:
                logger.warning("No BlueSky API token found. News fetching disabled.")
        else:
            self.provider = None
            self.token = None
            self.base_url = None
            logger.warning("No API keys configured. News fetching disabled.")

    def fetch_news(self, symbol: str, last: int = 50) -> List[Dict]:
        """
        Fetch recent news articles for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            last: Number of articles to fetch

        Returns:
            List of article dicts with keys: datetime, headline, summary, source
        """
        if not self.token:
            logger.debug(f"No API token - skipping news fetch for {symbol}")
            return []

        # Check cache first
        cached = self._load_cache(symbol)
        if cached is not None:
            return cached

        try:
            if self.provider == "finnhub":
                articles = self._fetch_finnhub(symbol, last)
            else:
                articles = self._fetch_bluesky(symbol, last)

            # Cache results
            if articles:
                self._save_cache(symbol, articles)
                logger.info(f"Fetched {len(articles)} news articles for {symbol} via {self.provider}")

            time.sleep(self._request_delay)  # Rate limiting
            return articles

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch news for {symbol}: {e}")
            return []
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse news response for {symbol}: {e}")
            return []

    def _fetch_finnhub(self, symbol: str, last: int) -> List[Dict]:
        """Fetch news from Finnhub API."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last 7 days

        params = {
            "symbol": symbol.upper(),
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "token": self.token
        }

        response = self._session.get(self.base_url, params=params, timeout=10)
        response.raise_for_status()

        articles = response.json()
        if not isinstance(articles, list):
            articles = [articles] if articles else []

        # Limit to requested count
        articles = articles[:last]

        # Normalize to common format
        normalized = []
        for article in articles:
            normalized.append({
                "datetime": article.get("datetime", 0) * 1000,  # Convert to ms
                "date": datetime.fromtimestamp(
                    article.get("datetime", 0)
                ).strftime("%Y-%m-%d") if article.get("datetime") else None,
                "headline": article.get("headline", ""),
                "summary": article.get("summary", ""),
                "source": article.get("source", ""),
                "symbol": symbol.upper(),
                "url": article.get("url", ""),
                "related": article.get("related", symbol),
            })

        return normalized

    def _fetch_bluesky(self, symbol: str, last: int) -> List[Dict]:
        """Fetch news from BlueSky API."""
        url = f"{self.base_url}/{symbol.lower()}"
        params = {"token": self.token, "last": last}

        response = self._session.get(url, params=params, timeout=10)
        response.raise_for_status()

        articles = response.json()
        if not isinstance(articles, list):
            articles = [articles] if articles else []

        # Normalize article format
        normalized = []
        for article in articles:
            normalized.append({
                "datetime": article.get("datetime", 0),
                "date": datetime.fromtimestamp(
                    article.get("datetime", 0) / 1000
                ).strftime("%Y-%m-%d") if article.get("datetime") else None,
                "headline": article.get("headline", ""),
                "summary": article.get("summary", ""),
                "source": article.get("source", ""),
                "symbol": symbol.upper(),
                "related": article.get("related", ""),
            })

        return normalized

    def fetch_news_batch(self, symbols: List[str], last: int = 50) -> Dict[str, List[Dict]]:
        """
        Fetch news for multiple symbols.

        Args:
            symbols: List of stock tickers
            last: Number of articles per symbol

        Returns:
            Dict mapping symbol to list of articles
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.fetch_news(symbol, last=last)
        return results

    def _cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / f"news_{symbol.upper()}.json"

    def _load_cache(self, symbol: str) -> Optional[List[Dict]]:
        """Load cached news if still fresh."""
        cache_file = self._cache_path(symbol)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)

            cached_time = datetime.fromisoformat(cached.get("timestamp", ""))
            if datetime.now() - cached_time < timedelta(hours=self.cache_hours):
                logger.debug(f"Using cached news for {symbol}")
                return cached.get("articles", [])
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        return None

    def _save_cache(self, symbol: str, articles: List[Dict]) -> None:
        """Save articles to cache."""
        cache_file = self._cache_path(symbol)

        try:
            with open(cache_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol.upper(),
                    "articles": articles
                }, f)
        except IOError as e:
            logger.warning(f"Failed to cache news for {symbol}: {e}")
