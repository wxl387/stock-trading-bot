"""
News fetcher for BlueSky/viaNexus API (MT Newswires).
Fetches financial news articles for sentiment analysis.
"""
import logging
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
    Fetches financial news from BlueSky/viaNexus API.
    """

    def __init__(self, api_token: Optional[str] = None):
        self.base_url = "https://api.blueskyapi.com/v1/data/core/news"
        self.token = api_token or self._load_token()
        self.cache_hours = 4
        self._request_delay = 0.5  # Rate limiting: 500ms between requests

        if not self.token:
            logger.warning("No BlueSky API token configured. News fetching disabled.")

    def _load_token(self) -> Optional[str]:
        """Load API token from environment."""
        import os
        return os.environ.get("BLUESKY_API_TOKEN")

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
            url = f"{self.base_url}/{symbol.lower()}"
            params = {"token": self.token, "last": last}

            response = requests.get(url, params=params, timeout=10)
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

            # Cache results
            self._save_cache(symbol, normalized)

            logger.info(f"Fetched {len(normalized)} news articles for {symbol}")
            time.sleep(self._request_delay)  # Rate limiting

            return normalized

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch news for {symbol}: {e}")
            return []
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse news response for {symbol}: {e}")
            return []

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
