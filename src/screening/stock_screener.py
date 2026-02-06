"""
Stock Screener

Multi-factor scoring and ranking system for stock selection.
Combines fundamental, technical, momentum, sentiment, and quality factors.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import threading

import pandas as pd
import numpy as np

from config.settings import DATA_DIR
from .screening_strategies import (
    ScreeningStrategy,
    BalancedStrategy,
    get_strategy,
    StrategyWeights,
)

logger = logging.getLogger(__name__)

# Module-level singleton
_stock_screener: Optional["StockScreener"] = None
_lock = threading.Lock()


def get_stock_screener(config: Optional[Dict] = None) -> "StockScreener":
    """Get or create the singleton StockScreener instance."""
    global _stock_screener
    with _lock:
        if _stock_screener is None:
            _stock_screener = StockScreener(config or {})
        return _stock_screener


@dataclass
class ScoreBreakdown:
    """Breakdown of individual score components."""
    fundamental: float = 0.0
    technical: float = 0.0
    momentum: float = 0.0
    sentiment: float = 0.0
    quality: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "fundamental": self.fundamental,
            "technical": self.technical,
            "momentum": self.momentum,
            "sentiment": self.sentiment,
            "quality": self.quality,
        }


@dataclass
class StockScore:
    """Score for a single stock."""
    symbol: str
    total_score: float  # 0-100
    rank: int = 0
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    timestamp: datetime = field(default_factory=datetime.now)

    # Additional context
    sector: str = ""
    market_cap: float = 0.0
    price: float = 0.0

    # Key metrics used in scoring
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    roe: Optional[float] = None
    earnings_growth: Optional[float] = None
    revenue_growth: Optional[float] = None
    rsi: Optional[float] = None
    return_1m: Optional[float] = None
    return_3m: Optional[float] = None
    analyst_rating: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "total_score": self.total_score,
            "rank": self.rank,
            "breakdown": self.breakdown.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "sector": self.sector,
            "market_cap": self.market_cap,
            "price": self.price,
            "pe_ratio": self.pe_ratio,
            "peg_ratio": self.peg_ratio,
            "roe": self.roe,
            "earnings_growth": self.earnings_growth,
            "revenue_growth": self.revenue_growth,
            "rsi": self.rsi,
            "return_1m": self.return_1m,
            "return_3m": self.return_3m,
            "analyst_rating": self.analyst_rating,
        }


class StockScreener:
    """
    Multi-factor stock screening and ranking system.

    Scoring Components (0-100 each):
    - Fundamental Score (25%): P/E percentile, earnings growth, ROE
    - Technical Score (25%): RSI zone, MACD signal, trend
    - Momentum Score (20%): 1/3/6 month returns vs sector
    - Sentiment Score (15%): News sentiment, analyst rating
    - Quality Score (15%): Margins, consistency, low volatility
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize StockScreener.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cache_dir = DATA_DIR / "cache" / "screening"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Default strategy
        strategy_name = config.get("screening", {}).get("strategy", "balanced")
        self.default_strategy = get_strategy(strategy_name)

        # Lazy-loaded components
        self._data_fetcher = None
        self._fundamental_fetcher = None
        self._feature_engineer = None
        self._sentiment_fetcher = None
        self._llm_client = None

        # Score cache
        self._score_cache: Dict[str, StockScore] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_hours = config.get("screening", {}).get("cache_hours", 4)

    @property
    def data_fetcher(self):
        """Lazy load data fetcher."""
        if self._data_fetcher is None:
            try:
                from src.data.data_fetcher import DataFetcher
                self._data_fetcher = DataFetcher()
            except ImportError as e:
                logger.error(f"Failed to import DataFetcher: {e}")
        return self._data_fetcher

    @property
    def fundamental_fetcher(self):
        """Lazy load fundamental fetcher."""
        if self._fundamental_fetcher is None:
            try:
                from src.data.fundamental_fetcher import get_fundamental_fetcher
                self._fundamental_fetcher = get_fundamental_fetcher()
            except ImportError as e:
                logger.error(f"Failed to import FundamentalFetcher: {e}")
        return self._fundamental_fetcher

    @property
    def feature_engineer(self):
        """Lazy load feature engineer."""
        if self._feature_engineer is None:
            try:
                from src.data.feature_engineer import FeatureEngineer
                self._feature_engineer = FeatureEngineer()
            except ImportError as e:
                logger.error(f"Failed to import FeatureEngineer: {e}")
        return self._feature_engineer

    @property
    def sentiment_fetcher(self):
        """Lazy load sentiment fetcher."""
        if self._sentiment_fetcher is None:
            try:
                from src.data.sentiment_fetcher import get_sentiment_fetcher
                self._sentiment_fetcher = get_sentiment_fetcher()
            except ImportError as e:
                logger.warning(f"Sentiment fetcher not available: {e}")
        return self._sentiment_fetcher

    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None:
            try:
                from src.agents.llm_client import get_llm_client
                self._llm_client = get_llm_client()
            except ImportError as e:
                logger.warning(f"LLM client not available: {e}")
        return self._llm_client

    def score_stocks(
        self,
        symbols: List[str],
        strategy: Optional[ScreeningStrategy] = None,
        use_cache: bool = True
    ) -> List[StockScore]:
        """
        Score a list of stocks.

        Args:
            symbols: List of stock symbols to score
            strategy: Screening strategy to use (uses default if None)
            use_cache: Whether to use cached scores

        Returns:
            List of StockScore objects sorted by total_score descending
        """
        if strategy is None:
            strategy = self.default_strategy

        logger.info(f"Scoring {len(symbols)} stocks with {strategy.get_name()} strategy")

        # Check cache
        if use_cache and self._is_cache_valid():
            cached_scores = [
                self._score_cache[s] for s in symbols
                if s in self._score_cache
            ]
            if len(cached_scores) == len(symbols):
                logger.info("Returning cached scores")
                return sorted(cached_scores, key=lambda x: x.total_score, reverse=True)

        # Fetch data for all symbols
        fundamentals = self._fetch_fundamentals(symbols)
        technicals = self._fetch_technicals(symbols)
        sentiments = self._fetch_sentiments(symbols)

        # Calculate market-wide metrics for percentile calculations
        market_metrics = self._calculate_market_metrics(fundamentals)

        # Score each stock
        scores = []
        for symbol in symbols:
            try:
                score = self._score_single_stock(
                    symbol=symbol,
                    fundamentals=fundamentals.get(symbol, {}),
                    technicals=technicals.get(symbol, {}),
                    sentiments=sentiments.get(symbol, {}),
                    market_metrics=market_metrics,
                    strategy=strategy,
                )
                scores.append(score)
                self._score_cache[symbol] = score
            except Exception as e:
                logger.error(f"Error scoring {symbol}: {e}")
                # Add with zero score
                scores.append(StockScore(symbol=symbol, total_score=0.0))

        # Sort by score and assign ranks
        scores.sort(key=lambda x: x.total_score, reverse=True)
        for i, score in enumerate(scores):
            score.rank = i + 1

        self._cache_timestamp = datetime.now()

        if scores:
            logger.info(f"Scored {len(scores)} stocks. Top: {scores[0].symbol} ({scores[0].total_score:.1f})")
        else:
            logger.warning("No stocks scored successfully")
        return scores

    def get_top_stocks(
        self,
        n: int = 20,
        strategy: Optional[str] = None,
        universe: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get top N stocks by score.

        Args:
            n: Number of stocks to return
            strategy: Strategy name (growth, value, momentum, quality, balanced)
            universe: List of symbols to screen (fetches from StockUniverse if None)
            exclude: Symbols to exclude from results

        Returns:
            List of top N symbols
        """
        # Get universe if not provided
        if universe is None:
            try:
                from src.data.stock_universe import get_stock_universe
                universe_mgr = get_stock_universe()
                universe = universe_mgr.get_universe()
            except Exception as e:
                logger.error(f"Failed to get universe: {e}")
                return []

        # Get strategy
        strat = get_strategy(strategy) if strategy else self.default_strategy

        # Score stocks
        scores = self.score_stocks(universe, strat)

        # Filter out excluded symbols
        if exclude:
            exclude_set = set(exclude)
            scores = [s for s in scores if s.symbol not in exclude_set]

        # Return top N symbols
        return [s.symbol for s in scores[:n]]

    def ai_screen(
        self,
        symbols: List[str],
        prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[StockScore]:
        """
        AI-enhanced stock screening using LLM.

        Args:
            symbols: List of symbols to screen
            prompt: Custom prompt for LLM analysis
            context: Additional context (market conditions, portfolio, etc.)

        Returns:
            List of scored stocks with AI-enhanced rankings
        """
        if not self.llm_client or not self.llm_client.is_available():
            logger.warning("LLM not available, falling back to standard screening")
            return self.score_stocks(symbols)

        # First, get quantitative scores
        scores = self.score_stocks(symbols)
        top_candidates = scores[:30]  # Narrow down to top 30

        # Prepare data for LLM
        candidates_data = []
        for score in top_candidates:
            candidates_data.append({
                "symbol": score.symbol,
                "score": score.total_score,
                "breakdown": score.breakdown.to_dict(),
                "sector": score.sector,
                "pe_ratio": score.pe_ratio,
                "earnings_growth": score.earnings_growth,
                "rsi": score.rsi,
                "return_3m": score.return_3m,
            })

        # Build prompt
        if prompt is None:
            prompt = self._build_default_ai_prompt(candidates_data, context)

        try:
            # Get LLM analysis
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self._get_screening_system_prompt(),
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for more consistent rankings
            )

            if response:
                # Parse LLM response to adjust rankings
                adjusted_scores = self._parse_llm_rankings(response, top_candidates)
                return adjusted_scores

        except Exception as e:
            logger.error(f"LLM screening failed: {e}")

        return scores

    def _score_single_stock(
        self,
        symbol: str,
        fundamentals: Dict[str, Any],
        technicals: Dict[str, Any],
        sentiments: Dict[str, Any],
        market_metrics: Dict[str, Any],
        strategy: ScreeningStrategy
    ) -> StockScore:
        """Score a single stock."""
        weights = strategy.get_weights()

        # Calculate component scores (0-100)
        fundamental_score = self._calculate_fundamental_score(
            fundamentals, market_metrics, strategy
        )
        technical_score = self._calculate_technical_score(
            technicals, strategy
        )
        momentum_score = self._calculate_momentum_score(
            technicals, market_metrics, strategy
        )
        sentiment_score = self._calculate_sentiment_score(
            sentiments, fundamentals
        )
        quality_score = self._calculate_quality_score(
            fundamentals, technicals
        )

        # Calculate weighted total score
        total_score = (
            fundamental_score * weights.fundamental +
            technical_score * weights.technical +
            momentum_score * weights.momentum +
            sentiment_score * weights.sentiment +
            quality_score * weights.quality
        )

        # Ensure score is in 0-100 range
        total_score = max(0, min(100, total_score))

        return StockScore(
            symbol=symbol,
            total_score=total_score,
            breakdown=ScoreBreakdown(
                fundamental=fundamental_score,
                technical=technical_score,
                momentum=momentum_score,
                sentiment=sentiment_score,
                quality=quality_score,
            ),
            sector=fundamentals.get("sector", ""),
            market_cap=fundamentals.get("market_cap", 0),
            price=technicals.get("price", 0),
            pe_ratio=fundamentals.get("pe_ratio"),
            peg_ratio=fundamentals.get("peg_ratio"),
            roe=fundamentals.get("roe"),
            earnings_growth=fundamentals.get("earnings_growth_yoy"),
            revenue_growth=fundamentals.get("revenue_growth_yoy"),
            rsi=technicals.get("rsi_14"),
            return_1m=technicals.get("return_1m"),
            return_3m=technicals.get("return_3m"),
            analyst_rating=fundamentals.get("analyst_rating"),
        )

    def _calculate_fundamental_score(
        self,
        fundamentals: Dict[str, Any],
        market_metrics: Dict[str, Any],
        strategy: ScreeningStrategy
    ) -> float:
        """Calculate fundamental score (0-100)."""
        criteria = strategy.get_fundamental_criteria()
        scores = []

        # P/E Score (inverted - lower is better for value)
        pe = fundamentals.get("pe_ratio")
        if pe is not None and pe > 0:
            pe_percentile = self._percentile_rank(pe, market_metrics.get("pe_values", []))
            if criteria.pe_preference == "low":
                pe_score = 100 - pe_percentile  # Lower P/E = higher score
            elif criteria.pe_preference == "high":
                pe_score = pe_percentile  # Higher P/E = higher score (growth)
            else:
                # Moderate - prefer middle range
                pe_score = 100 - abs(pe_percentile - 50) * 2
            scores.append(pe_score)

        # PEG Score
        peg = fundamentals.get("peg_ratio")
        if peg is not None and peg > 0:
            if peg <= 1.0:
                peg_score = 100
            elif peg <= criteria.peg_max:
                peg_score = 100 - ((peg - 1.0) / (criteria.peg_max - 1.0)) * 50
            else:
                peg_score = max(0, 50 - (peg - criteria.peg_max) * 20)
            scores.append(peg_score)

        # Earnings Growth Score
        eg = fundamentals.get("earnings_growth_yoy")
        if eg is not None:
            if eg >= criteria.min_earnings_growth:
                eg_score = min(100, 50 + eg * 200)  # 25% growth = 100
            else:
                eg_score = max(0, 50 + eg * 200)
            scores.append(eg_score)

        # ROE Score
        roe = fundamentals.get("roe")
        if roe is not None:
            if roe >= criteria.min_roe:
                roe_score = min(100, 50 + roe * 250)  # 20% ROE = 100
            else:
                roe_score = max(0, roe / criteria.min_roe * 50) if criteria.min_roe > 0 else 50
            scores.append(roe_score)

        # Debt/Equity Score (lower is better)
        de = fundamentals.get("debt_to_equity")
        if de is not None:
            if de <= criteria.max_debt_to_equity:
                de_score = 100 - (de / criteria.max_debt_to_equity) * 50
            else:
                de_score = max(0, 50 - (de - criteria.max_debt_to_equity) * 25)
            scores.append(de_score)

        return np.mean(scores) if scores else 50.0

    def _calculate_technical_score(
        self,
        technicals: Dict[str, Any],
        strategy: ScreeningStrategy
    ) -> float:
        """Calculate technical score (0-100)."""
        criteria = strategy.get_technical_criteria()
        scores = []

        # RSI Score
        rsi = technicals.get("rsi_14")
        if rsi is not None:
            if rsi < criteria.rsi_oversold:
                rsi_score = 80  # Oversold = bullish
            elif rsi > criteria.rsi_overbought:
                rsi_score = 30  # Overbought = bearish
            else:
                # Neutral zone - prefer middle
                rsi_score = 50 + (50 - rsi) * 0.5
            scores.append(rsi_score)

        # SMA Position Score
        price = technicals.get("close") or technicals.get("price", 0)
        if price > 0:
            sma_scores = []

            sma20 = technicals.get("sma_20")
            if sma20 and sma20 > 0:
                if price > sma20:
                    sma_scores.append(70 if criteria.require_above_sma20 else 60)
                else:
                    sma_scores.append(30 if criteria.require_above_sma20 else 40)

            sma50 = technicals.get("sma_50")
            if sma50 and sma50 > 0:
                if price > sma50:
                    sma_scores.append(75 if criteria.require_above_sma50 else 65)
                else:
                    sma_scores.append(25 if criteria.require_above_sma50 else 35)

            sma200 = technicals.get("sma_200")
            if sma200 and sma200 > 0:
                if price > sma200:
                    sma_scores.append(80 if criteria.require_above_sma200 else 70)
                else:
                    sma_scores.append(20 if criteria.require_above_sma200 else 30)

            if sma_scores:
                scores.append(np.mean(sma_scores))

        # MACD Score
        macd = technicals.get("macd")
        macd_signal = technicals.get("macd_signal")
        if macd is not None and macd_signal is not None:
            if macd > 0 and macd > macd_signal:
                macd_score = 80  # Bullish
            elif macd > 0:
                macd_score = 60  # Positive but no crossover
            elif macd > macd_signal:
                macd_score = 50  # Negative but improving
            else:
                macd_score = 30  # Bearish
            scores.append(macd_score)

        return np.mean(scores) if scores else 50.0

    def _calculate_momentum_score(
        self,
        technicals: Dict[str, Any],
        market_metrics: Dict[str, Any],
        strategy: ScreeningStrategy
    ) -> float:
        """Calculate momentum score (0-100)."""
        criteria = strategy.get_momentum_criteria()
        scores = []

        # 1-Month Return Score
        ret_1m = technicals.get("return_1m")
        if ret_1m is not None:
            if ret_1m >= criteria.min_return_1m:
                # Score based on return magnitude
                ret_1m_score = min(100, 50 + ret_1m * 200)
            else:
                ret_1m_score = max(0, 25 + ret_1m * 100)
            scores.append(ret_1m_score)

        # 3-Month Return Score
        ret_3m = technicals.get("return_3m")
        if ret_3m is not None:
            if ret_3m >= criteria.min_return_3m:
                ret_3m_score = min(100, 50 + ret_3m * 150)
            else:
                ret_3m_score = max(0, 25 + ret_3m * 75)
            scores.append(ret_3m_score)

        # 6-Month Return Score
        ret_6m = technicals.get("return_6m")
        if ret_6m is not None:
            if ret_6m >= criteria.min_return_6m:
                ret_6m_score = min(100, 50 + ret_6m * 100)
            else:
                ret_6m_score = max(0, 25 + ret_6m * 50)
            scores.append(ret_6m_score)

        # Relative Strength vs Market
        if criteria.compare_to_market:
            market_return = market_metrics.get("market_return_3m", 0)
            if ret_3m is not None:
                relative_strength = ret_3m - market_return
                if relative_strength >= criteria.min_relative_strength:
                    rs_score = min(100, 50 + relative_strength * 200)
                else:
                    rs_score = max(0, 50 + relative_strength * 200)
                scores.append(rs_score)

        return np.mean(scores) if scores else 50.0

    def _calculate_sentiment_score(
        self,
        sentiments: Dict[str, Any],
        fundamentals: Dict[str, Any]
    ) -> float:
        """Calculate sentiment score (0-100)."""
        scores = []

        # News Sentiment Score
        sentiment = sentiments.get("sentiment_score")
        if sentiment is not None:
            # sentiment is typically -1 to +1
            sent_score = 50 + sentiment * 50
            scores.append(sent_score)

        # Article Volume Score (more coverage = more interest)
        article_count = sentiments.get("article_count", 0)
        if article_count > 0:
            # More articles = better (up to a point)
            vol_score = min(100, 50 + article_count * 2)
            scores.append(vol_score)

        # Analyst Rating Score
        rating = fundamentals.get("analyst_rating")
        if rating:
            rating_map = {
                "strong_buy": 100,
                "buy": 80,
                "hold": 50,
                "sell": 20,
                "strong_sell": 0,
            }
            rating_score = rating_map.get(rating.lower(), 50)
            scores.append(rating_score)

        # Target Upside Score
        target_upside = fundamentals.get("target_upside")
        if target_upside is not None:
            # 20% upside = 100 score
            upside_score = min(100, max(0, 50 + target_upside * 250))
            scores.append(upside_score)

        return np.mean(scores) if scores else 50.0

    def _calculate_quality_score(
        self,
        fundamentals: Dict[str, Any],
        technicals: Dict[str, Any]
    ) -> float:
        """Calculate quality score (0-100)."""
        scores = []

        # Profit Margin Score
        margin = fundamentals.get("profit_margin")
        if margin is not None:
            # 20% margin = 100 score
            margin_score = min(100, max(0, margin * 500))
            scores.append(margin_score)

        # Operating Margin Score
        op_margin = fundamentals.get("operating_margin")
        if op_margin is not None:
            op_margin_score = min(100, max(0, op_margin * 400))
            scores.append(op_margin_score)

        # Low Volatility Score (lower = better)
        volatility = technicals.get("volatility_20d")
        if volatility is not None and volatility > 0:
            # 20% annual vol = 100 score, 50% vol = 40 score
            vol_score = max(0, 100 - volatility * 200)
            scores.append(vol_score)

        # Consistent Returns Score
        ret_std = technicals.get("returns_std_20d")
        if ret_std is not None and ret_std > 0:
            # Lower standard deviation = more consistent
            consistency_score = max(0, 100 - ret_std * 1000)
            scores.append(consistency_score)

        # Low Debt Score
        de = fundamentals.get("debt_to_equity")
        if de is not None:
            if de <= 0.5:
                debt_score = 100
            elif de <= 1.0:
                debt_score = 80
            elif de <= 2.0:
                debt_score = 50
            else:
                debt_score = max(0, 50 - de * 10)
            scores.append(debt_score)

        return np.mean(scores) if scores else 50.0

    def _fetch_fundamentals(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch fundamental data for symbols."""
        result = {}

        if self.fundamental_fetcher:
            try:
                fundamentals = self.fundamental_fetcher.fetch_multiple(symbols)
                for symbol, data in fundamentals.items():
                    result[symbol] = data.to_features()
                    result[symbol]["sector"] = data.sector
                    result[symbol]["market_cap"] = data.market_cap
                    result[symbol]["analyst_rating"] = data.analyst_rating
                    result[symbol]["target_upside"] = data.target_upside
            except Exception as e:
                logger.error(f"Error fetching fundamentals: {e}")

        return result

    def _fetch_technicals(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch technical data for symbols."""
        result = {}

        if self.data_fetcher and self.feature_engineer:
            for symbol in symbols:
                try:
                    # Fetch price data
                    df = self.data_fetcher.fetch_historical(symbol, period="1y")
                    if df.empty:
                        continue

                    # Add technical indicators
                    df = self.feature_engineer.add_all_features(df)

                    # Get latest values
                    latest = df.iloc[-1]
                    result[symbol] = {
                        "close": latest.get("close", 0),
                        "price": latest.get("close", 0),
                        "rsi_14": latest.get("rsi_14"),
                        "macd": latest.get("macd"),
                        "macd_signal": latest.get("macd_signal"),
                        "sma_20": latest.get("sma_20"),
                        "sma_50": latest.get("sma_50"),
                        "sma_200": latest.get("sma_200"),
                        "volatility_20d": latest.get("volatility_20d"),
                        "returns_std_20d": latest.get("returns_std_20d"),
                    }

                    # Calculate returns
                    if len(df) >= 21:
                        result[symbol]["return_1m"] = (df["close"].iloc[-1] / df["close"].iloc[-21] - 1)
                    if len(df) >= 63:
                        result[symbol]["return_3m"] = (df["close"].iloc[-1] / df["close"].iloc[-63] - 1)
                    if len(df) >= 126:
                        result[symbol]["return_6m"] = (df["close"].iloc[-1] / df["close"].iloc[-126] - 1)

                except Exception as e:
                    logger.debug(f"Error fetching technicals for {symbol}: {e}")

        return result

    def _fetch_sentiments(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch sentiment data for symbols."""
        result = {}

        if self.sentiment_fetcher:
            for symbol in symbols:
                try:
                    # This returns sentiment features
                    sentiment = self.sentiment_fetcher.get_sentiment(symbol)
                    if sentiment:
                        result[symbol] = {
                            "sentiment_score": sentiment.get("sentiment_score", 0),
                            "article_count": sentiment.get("article_count", 0),
                        }
                except Exception as e:
                    logger.debug(f"Error fetching sentiment for {symbol}: {e}")

        return result

    def _calculate_market_metrics(self, fundamentals: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate market-wide metrics for percentile calculations."""
        pe_values = [
            f.get("pe_ratio") for f in fundamentals.values()
            if f.get("pe_ratio") is not None and f.get("pe_ratio") > 0
        ]

        # Get market return (SPY)
        market_return_3m = 0.0
        if self.data_fetcher:
            try:
                spy_df = self.data_fetcher.fetch_historical("SPY", period="6mo")
                if len(spy_df) >= 63:
                    market_return_3m = spy_df["close"].iloc[-1] / spy_df["close"].iloc[-63] - 1
            except Exception:
                pass

        return {
            "pe_values": pe_values,
            "market_return_3m": market_return_3m,
        }

    def _percentile_rank(self, value: float, values: List[float]) -> float:
        """Calculate percentile rank of value in list."""
        if not values:
            return 50.0

        below = sum(1 for v in values if v <= value)
        return (below / len(values)) * 100

    def _is_cache_valid(self) -> bool:
        """Check if score cache is still valid."""
        if self._cache_timestamp is None:
            return False

        age = datetime.now() - self._cache_timestamp
        return age < timedelta(hours=self._cache_ttl_hours)

    def _get_screening_system_prompt(self) -> str:
        """Get system prompt for LLM screening."""
        return """You are a quantitative stock analyst AI assistant.
Your role is to analyze stock candidates and provide investment rankings.
Consider both quantitative metrics and qualitative factors.
Be data-driven and provide clear reasoning for your rankings.
Format your response as a numbered list of stock recommendations."""

    def _build_default_ai_prompt(
        self,
        candidates: List[Dict],
        context: Optional[Dict] = None
    ) -> str:
        """Build default prompt for AI screening."""
        candidates_str = "\n".join([
            f"- {c['symbol']}: Score={c['score']:.1f}, Sector={c['sector']}, "
            f"P/E={c['pe_ratio']}, Growth={c['earnings_growth']}, RSI={c['rsi']}"
            for c in candidates[:20]  # Limit to top 20
        ])

        context_str = ""
        if context:
            if context.get("market_regime"):
                context_str += f"\nMarket Regime: {context['market_regime']}"
            if context.get("portfolio"):
                context_str += f"\nCurrent Portfolio: {context['portfolio']}"

        return f"""Analyze these stock candidates and rank the top 10 for investment:

## Candidates (sorted by quantitative score)
{candidates_str}

{context_str}

Please provide:
1. Your top 10 stock recommendations in order of preference
2. Brief reasoning for each (1-2 sentences)
3. Any stocks to avoid from this list and why

Focus on:
- Sector diversification
- Risk/reward balance
- Current market conditions
- Growth vs value characteristics"""

    def _parse_llm_rankings(
        self,
        response: str,
        original_scores: List[StockScore]
    ) -> List[StockScore]:
        """Parse LLM response to adjust rankings."""
        # Simple parsing - look for stock symbols in order
        adjusted = original_scores.copy()

        # Extract mentioned symbols in order
        mentioned_order = []
        for score in original_scores:
            if score.symbol in response:
                mentioned_order.append(score.symbol)

        # Boost scores for early-mentioned stocks
        boost_map = {}
        for i, symbol in enumerate(mentioned_order[:10]):
            # First mentioned gets +20, decaying
            boost_map[symbol] = max(0, 20 - i * 2)

        for score in adjusted:
            if score.symbol in boost_map:
                score.total_score = min(100, score.total_score + boost_map[score.symbol])

        # Re-sort and re-rank
        adjusted.sort(key=lambda x: x.total_score, reverse=True)
        for i, score in enumerate(adjusted):
            score.rank = i + 1

        return adjusted

    def get_status(self) -> Dict[str, Any]:
        """Get screener status."""
        return {
            "cache_valid": self._is_cache_valid(),
            "cached_symbols": len(self._score_cache),
            "cache_timestamp": self._cache_timestamp.isoformat() if self._cache_timestamp else None,
            "default_strategy": self.default_strategy.get_name(),
        }
