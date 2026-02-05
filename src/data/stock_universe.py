"""
Stock Universe Manager

Manages the universe of tradeable stocks including S&P 500 and NASDAQ 100 constituents.
Provides filtering by market cap, volume, and price criteria.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import threading
import json

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

# Module-level singleton
_stock_universe: Optional["StockUniverse"] = None
_lock = threading.Lock()


def get_stock_universe(config: Optional[Dict] = None) -> "StockUniverse":
    """Get or create the singleton StockUniverse instance."""
    global _stock_universe
    with _lock:
        if _stock_universe is None:
            _stock_universe = StockUniverse(config or {})
        return _stock_universe


@dataclass
class StockInfo:
    """Basic stock information."""
    symbol: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: float = 0.0
    avg_volume: float = 0.0
    price: float = 0.0
    in_sp500: bool = False
    in_nasdaq100: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "sector": self.sector,
            "industry": self.industry,
            "market_cap": self.market_cap,
            "avg_volume": self.avg_volume,
            "price": self.price,
            "in_sp500": self.in_sp500,
            "in_nasdaq100": self.in_nasdaq100,
        }


@dataclass
class UniverseFilter:
    """Filter criteria for stock universe."""
    min_market_cap: float = 10_000_000_000  # $10B
    min_avg_volume: float = 1_000_000  # 1M shares
    min_price: float = 5.0
    max_price: float = 10_000.0
    exclude_sectors: List[str] = field(default_factory=list)
    include_sectors: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=lambda: ["sp500", "nasdaq100"])


class StockUniverse:
    """
    Manages the universe of tradeable stocks.

    Features:
    - Fetches S&P 500 and NASDAQ 100 constituents
    - Filters by market cap, volume, and price
    - Caches data with weekly refresh
    - Provides sector breakdown
    """

    # Wikipedia URLs for index constituents
    SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

    # Default cache duration
    CACHE_DAYS = 7

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize StockUniverse.

        Args:
            config: Configuration dictionary with universe settings
        """
        self.config = config
        self.cache_dir = DATA_DIR / "cache" / "universe"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load filter settings from config
        universe_config = config.get("universe", {})
        self.default_filter = UniverseFilter(
            min_market_cap=universe_config.get("min_market_cap", 10_000_000_000),
            min_avg_volume=universe_config.get("min_avg_volume", 1_000_000),
            min_price=universe_config.get("min_price", 5.0),
            sources=universe_config.get("sources", ["sp500", "nasdaq100"]),
        )

        # In-memory cache
        self._sp500_symbols: Optional[Set[str]] = None
        self._nasdaq100_symbols: Optional[Set[str]] = None
        self._stock_info: Dict[str, StockInfo] = {}
        self._last_refresh: Optional[datetime] = None

    def get_universe(
        self,
        filter_criteria: Optional[UniverseFilter] = None,
        use_cache: bool = True
    ) -> List[str]:
        """
        Get filtered universe of symbols.

        Args:
            filter_criteria: Filter to apply (uses default if None)
            use_cache: Whether to use cached data

        Returns:
            List of symbols meeting criteria
        """
        if filter_criteria is None:
            filter_criteria = self.default_filter

        # Ensure data is loaded
        self._ensure_data_loaded(use_cache)

        # Get all symbols from requested sources
        all_symbols = set()
        if "sp500" in filter_criteria.sources and self._sp500_symbols:
            all_symbols.update(self._sp500_symbols)
        if "nasdaq100" in filter_criteria.sources and self._nasdaq100_symbols:
            all_symbols.update(self._nasdaq100_symbols)

        # Apply filters
        filtered = []
        for symbol in all_symbols:
            info = self._stock_info.get(symbol)
            if info and self._passes_filter(info, filter_criteria):
                filtered.append(symbol)

        # Sort by market cap (largest first)
        filtered.sort(key=lambda s: self._stock_info.get(s, StockInfo(s)).market_cap, reverse=True)

        logger.info(f"Universe: {len(filtered)} symbols (from {len(all_symbols)} total)")
        return filtered

    def get_sp500_symbols(self, use_cache: bool = True) -> List[str]:
        """Get S&P 500 constituent symbols."""
        self._ensure_data_loaded(use_cache)
        return list(self._sp500_symbols) if self._sp500_symbols else []

    def get_nasdaq100_symbols(self, use_cache: bool = True) -> List[str]:
        """Get NASDAQ 100 constituent symbols."""
        self._ensure_data_loaded(use_cache)
        return list(self._nasdaq100_symbols) if self._nasdaq100_symbols else []

    def get_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """Get info for a specific symbol."""
        self._ensure_data_loaded()
        return self._stock_info.get(symbol)

    def get_sector_breakdown(self, symbols: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Get breakdown of symbols by sector.

        Args:
            symbols: List of symbols to analyze (uses full universe if None)

        Returns:
            Dictionary mapping sector to list of symbols
        """
        self._ensure_data_loaded()

        if symbols is None:
            symbols = list(self._stock_info.keys())

        sectors: Dict[str, List[str]] = {}
        for symbol in symbols:
            info = self._stock_info.get(symbol)
            if info and info.sector:
                if info.sector not in sectors:
                    sectors[info.sector] = []
                sectors[info.sector].append(symbol)

        return sectors

    def get_symbols_by_sector(self, sector: str, limit: Optional[int] = None) -> List[str]:
        """
        Get symbols in a specific sector.

        Args:
            sector: Sector name
            limit: Maximum number to return (sorted by market cap)

        Returns:
            List of symbols in the sector
        """
        self._ensure_data_loaded()

        symbols = []
        for symbol, info in self._stock_info.items():
            if info.sector and info.sector.lower() == sector.lower():
                symbols.append((symbol, info.market_cap))

        # Sort by market cap
        symbols.sort(key=lambda x: x[1], reverse=True)
        result = [s[0] for s in symbols]

        if limit:
            result = result[:limit]

        return result

    def refresh(self, force: bool = False) -> bool:
        """
        Refresh universe data.

        Args:
            force: Force refresh even if cache is valid

        Returns:
            True if refresh was performed
        """
        if not force and self._is_cache_valid():
            logger.info("Universe cache is still valid")
            return False

        logger.info("Refreshing stock universe...")

        try:
            # Fetch index constituents
            self._sp500_symbols = self._fetch_sp500_constituents()
            self._nasdaq100_symbols = self._fetch_nasdaq100_constituents()

            # Combine all symbols
            all_symbols = set()
            if self._sp500_symbols:
                all_symbols.update(self._sp500_symbols)
            if self._nasdaq100_symbols:
                all_symbols.update(self._nasdaq100_symbols)

            # Fetch stock info for all symbols
            self._fetch_stock_info(list(all_symbols))

            # Update last refresh time
            self._last_refresh = datetime.now()

            # Save to cache
            self._save_to_cache()

            logger.info(f"Universe refreshed: {len(all_symbols)} total symbols")
            return True

        except Exception as e:
            logger.error(f"Error refreshing universe: {e}")
            # Try to load from cache as fallback
            self._load_from_cache()
            return False

    def _ensure_data_loaded(self, use_cache: bool = True) -> None:
        """Ensure universe data is loaded."""
        if self._sp500_symbols is not None and self._nasdaq100_symbols is not None:
            return

        if use_cache and self._load_from_cache():
            return

        self.refresh(force=True)

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._last_refresh is None:
            return False

        age = datetime.now() - self._last_refresh
        return age < timedelta(days=self.CACHE_DAYS)

    def _passes_filter(self, info: StockInfo, filter_criteria: UniverseFilter) -> bool:
        """Check if stock passes filter criteria."""
        # Market cap filter
        if info.market_cap < filter_criteria.min_market_cap:
            return False

        # Volume filter
        if info.avg_volume < filter_criteria.min_avg_volume:
            return False

        # Price filter
        if info.price < filter_criteria.min_price or info.price > filter_criteria.max_price:
            return False

        # Sector filter (include)
        if filter_criteria.include_sectors:
            if info.sector not in filter_criteria.include_sectors:
                return False

        # Sector filter (exclude)
        if filter_criteria.exclude_sectors:
            if info.sector in filter_criteria.exclude_sectors:
                return False

        return True

    def _fetch_sp500_constituents(self) -> Set[str]:
        """Fetch S&P 500 constituent symbols from Wikipedia."""
        try:
            tables = pd.read_html(self.SP500_URL)
            # The first table contains the constituents
            df = tables[0]

            # Handle different column names
            symbol_col = None
            for col in ["Symbol", "Ticker", "Ticker symbol"]:
                if col in df.columns:
                    symbol_col = col
                    break

            if symbol_col is None:
                # Try first column
                symbol_col = df.columns[0]

            symbols = set(df[symbol_col].str.replace(".", "-").str.strip().tolist())
            logger.info(f"Fetched {len(symbols)} S&P 500 constituents")
            return symbols

        except Exception as e:
            logger.error(f"Error fetching S&P 500 constituents: {e}")
            # Return cached or default list
            return self._get_default_sp500()

    def _fetch_nasdaq100_constituents(self) -> Set[str]:
        """Fetch NASDAQ 100 constituent symbols from Wikipedia."""
        try:
            tables = pd.read_html(self.NASDAQ100_URL)
            # Find the table with the constituents
            df = None
            for table in tables:
                if "Ticker" in table.columns or "Symbol" in table.columns:
                    df = table
                    break

            if df is None:
                # Try the fourth table (common location)
                df = tables[4] if len(tables) > 4 else tables[0]

            # Handle different column names
            symbol_col = None
            for col in ["Ticker", "Symbol", "Ticker symbol"]:
                if col in df.columns:
                    symbol_col = col
                    break

            if symbol_col is None:
                symbol_col = df.columns[0]

            symbols = set(df[symbol_col].str.replace(".", "-").str.strip().tolist())
            logger.info(f"Fetched {len(symbols)} NASDAQ 100 constituents")
            return symbols

        except Exception as e:
            logger.error(f"Error fetching NASDAQ 100 constituents: {e}")
            return self._get_default_nasdaq100()

    def _fetch_stock_info(self, symbols: List[str]) -> None:
        """Fetch detailed info for symbols using yfinance."""
        if not HAS_YFINANCE:
            logger.warning("yfinance not available, using minimal stock info")
            for symbol in symbols:
                self._stock_info[symbol] = StockInfo(
                    symbol=symbol,
                    in_sp500=symbol in (self._sp500_symbols or set()),
                    in_nasdaq100=symbol in (self._nasdaq100_symbols or set()),
                )
            return

        logger.info(f"Fetching info for {len(symbols)} symbols...")

        # Batch fetch using yfinance
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                tickers = yf.Tickers(" ".join(batch))

                for symbol in batch:
                    try:
                        ticker = tickers.tickers.get(symbol)
                        if ticker is None:
                            continue

                        info = ticker.info
                        if not info:
                            continue

                        self._stock_info[symbol] = StockInfo(
                            symbol=symbol,
                            name=info.get("shortName", info.get("longName", "")),
                            sector=info.get("sector", ""),
                            industry=info.get("industry", ""),
                            market_cap=info.get("marketCap", 0) or 0,
                            avg_volume=info.get("averageVolume", 0) or 0,
                            price=info.get("currentPrice", info.get("regularMarketPrice", 0)) or 0,
                            in_sp500=symbol in (self._sp500_symbols or set()),
                            in_nasdaq100=symbol in (self._nasdaq100_symbols or set()),
                        )
                    except Exception as e:
                        logger.debug(f"Error fetching info for {symbol}: {e}")
                        self._stock_info[symbol] = StockInfo(
                            symbol=symbol,
                            in_sp500=symbol in (self._sp500_symbols or set()),
                            in_nasdaq100=symbol in (self._nasdaq100_symbols or set()),
                        )

            except Exception as e:
                logger.error(f"Error fetching batch info: {e}")

        logger.info(f"Fetched info for {len(self._stock_info)} symbols")

    def _save_to_cache(self) -> None:
        """Save universe data to cache file."""
        cache_file = self.cache_dir / "universe_cache.json"

        try:
            data = {
                "timestamp": self._last_refresh.isoformat() if self._last_refresh else None,
                "sp500": list(self._sp500_symbols) if self._sp500_symbols else [],
                "nasdaq100": list(self._nasdaq100_symbols) if self._nasdaq100_symbols else [],
                "stock_info": {
                    symbol: info.to_dict()
                    for symbol, info in self._stock_info.items()
                },
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Universe cache saved to {cache_file}")

        except Exception as e:
            logger.error(f"Error saving universe cache: {e}")

    def _load_from_cache(self) -> bool:
        """Load universe data from cache file."""
        cache_file = self.cache_dir / "universe_cache.json"

        if not cache_file.exists():
            return False

        try:
            # Check cache age
            file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if file_age > timedelta(days=self.CACHE_DAYS):
                logger.info("Universe cache is too old")
                return False

            with open(cache_file, 'r') as f:
                data = json.load(f)

            self._last_refresh = datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None
            self._sp500_symbols = set(data.get("sp500", []))
            self._nasdaq100_symbols = set(data.get("nasdaq100", []))

            # Reconstruct stock info
            self._stock_info = {}
            for symbol, info_dict in data.get("stock_info", {}).items():
                self._stock_info[symbol] = StockInfo(
                    symbol=info_dict.get("symbol", symbol),
                    name=info_dict.get("name", ""),
                    sector=info_dict.get("sector", ""),
                    industry=info_dict.get("industry", ""),
                    market_cap=info_dict.get("market_cap", 0),
                    avg_volume=info_dict.get("avg_volume", 0),
                    price=info_dict.get("price", 0),
                    in_sp500=info_dict.get("in_sp500", False),
                    in_nasdaq100=info_dict.get("in_nasdaq100", False),
                )

            logger.info(f"Universe loaded from cache: {len(self._stock_info)} symbols")
            return True

        except Exception as e:
            logger.error(f"Error loading universe cache: {e}")
            return False

    def _get_default_sp500(self) -> Set[str]:
        """Return a default subset of S&P 500 symbols."""
        # Top 100 by market cap as fallback
        return {
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH",
            "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
            "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "WMT", "MCD", "CSCO", "ACN",
            "DHR", "ABT", "NEE", "VZ", "ADBE", "CMCSA", "NKE", "PM", "WFC", "TXN",
            "LIN", "BMY", "QCOM", "RTX", "UPS", "COP", "MS", "HON", "ORCL", "SCHW",
            "LOW", "SPGI", "IBM", "SBUX", "INTC", "CAT", "DE", "ELV", "GS", "INTU",
            "AMD", "AXP", "BKNG", "PLD", "ISRG", "GILD", "MDLZ", "BLK", "ADI", "CVS",
            "SYK", "VRTX", "TJX", "C", "ADP", "TMUS", "AMT", "MMC", "REGN", "PGR",
            "SO", "CI", "MO", "LRCX", "ZTS", "DUK", "EOG", "CB", "NOW", "BDX",
            "FIS", "ITW", "MU", "CL", "BSX", "NOC", "EQIX", "SLB", "APD", "ATVI",
        }

    def _get_default_nasdaq100(self) -> Set[str]:
        """Return a default subset of NASDAQ 100 symbols."""
        return {
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "COST",
            "PEP", "ADBE", "CSCO", "NFLX", "CMCSA", "AMD", "INTC", "TXN", "QCOM", "INTU",
            "AMGN", "HON", "AMAT", "BKNG", "SBUX", "ISRG", "GILD", "ADI", "ADP", "MDLZ",
            "REGN", "VRTX", "LRCX", "MU", "PYPL", "SNPS", "KLAC", "PANW", "CDNS", "MAR",
            "ABNB", "MRVL", "ORLY", "FTNT", "ASML", "AZN", "CTAS", "MNST", "ADSK", "WDAY",
            "MELI", "CHTR", "PCAR", "PAYX", "KDP", "AEP", "DXCM", "CPRT", "NXPI", "IDXX",
            "LULU", "FAST", "ROST", "KHC", "EA", "CTSH", "VRSK", "EBAY", "XEL", "EXC",
            "BIIB", "CSGP", "WBD", "ODFL", "DLTR", "ILMN", "ZS", "ANSS", "TEAM", "CRWD",
            "DDOG", "ZM", "DOCU", "OKTA", "SPLK", "SGEN", "MTCH", "ALGN", "SIRI", "CEG",
            "LCID", "RIVN", "TTD", "COIN", "ROKU", "MRNA", "PDD", "JD", "BIDU", "NTES",
        }

    def get_status(self) -> Dict[str, Any]:
        """Get universe status."""
        return {
            "sp500_count": len(self._sp500_symbols) if self._sp500_symbols else 0,
            "nasdaq100_count": len(self._nasdaq100_symbols) if self._nasdaq100_symbols else 0,
            "total_with_info": len(self._stock_info),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
            "cache_valid": self._is_cache_valid(),
        }
