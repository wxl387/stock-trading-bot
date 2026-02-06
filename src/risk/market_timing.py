"""
Market Timing Module

Analyzes market conditions to determine optimal timing for adding or reducing exposure.
Uses VIX, market breadth, SPY vs moving averages, and economic indicators.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
import threading

import pandas as pd
import numpy as np

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

# Module-level singleton
_market_timer: Optional["MarketTimer"] = None
_lock = threading.Lock()


def get_market_timer(config: Optional[Dict] = None) -> "MarketTimer":
    """Get or create the singleton MarketTimer instance."""
    global _market_timer
    with _lock:
        if _market_timer is None:
            _market_timer = MarketTimer(config or {})
        return _market_timer


class TimingSignal(Enum):
    """Market timing signals."""
    ADD_EXPOSURE = "add_exposure"
    REDUCE_EXPOSURE = "reduce_exposure"
    HOLD = "hold"


@dataclass
class MarketConditions:
    """Container for current market conditions."""
    timestamp: datetime = field(default_factory=datetime.now)

    # VIX levels
    vix_level: float = 0.0
    vix_percentile: float = 50.0  # Historical percentile (0-100)
    vix_trend: str = "neutral"  # rising, falling, neutral

    # Market breadth
    advance_decline_ratio: float = 1.0
    new_highs_lows_ratio: float = 1.0
    pct_above_200ma: float = 50.0  # % of S&P 500 stocks above 200-day MA

    # SPY levels
    spy_price: float = 0.0
    spy_vs_sma20: float = 0.0  # % above/below
    spy_vs_sma50: float = 0.0
    spy_vs_sma200: float = 0.0
    spy_rsi: float = 50.0

    # Economic indicators
    treasury_10y: float = 0.0
    unemployment_trend: str = "neutral"

    # Overall assessment
    overall_condition: str = "neutral"  # bullish, bearish, neutral, volatile

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "vix_level": self.vix_level,
            "vix_percentile": self.vix_percentile,
            "vix_trend": self.vix_trend,
            "advance_decline_ratio": self.advance_decline_ratio,
            "new_highs_lows_ratio": self.new_highs_lows_ratio,
            "pct_above_200ma": self.pct_above_200ma,
            "spy_price": self.spy_price,
            "spy_vs_sma20": self.spy_vs_sma20,
            "spy_vs_sma50": self.spy_vs_sma50,
            "spy_vs_sma200": self.spy_vs_sma200,
            "spy_rsi": self.spy_rsi,
            "treasury_10y": self.treasury_10y,
            "unemployment_trend": self.unemployment_trend,
            "overall_condition": self.overall_condition,
        }


@dataclass
class MarketTimingSignal:
    """Result of market timing analysis."""
    signal: TimingSignal
    confidence: float  # 0-1
    timestamp: datetime = field(default_factory=datetime.now)
    conditions: MarketConditions = field(default_factory=MarketConditions)
    reasons: List[str] = field(default_factory=list)
    recommended_exposure: float = 1.0  # Multiplier (0.5 = half exposure, 1.5 = increase)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal": self.signal.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "conditions": self.conditions.to_dict(),
            "reasons": self.reasons,
            "recommended_exposure": self.recommended_exposure,
        }


class MarketTimer:
    """
    Analyzes market conditions to determine optimal timing for exposure changes.

    Factors analyzed:
    - VIX levels and trends
    - Market breadth (advance/decline, new highs/lows)
    - SPY position vs moving averages
    - Economic indicators

    Returns:
    - ADD_EXPOSURE: Market conditions favorable for increasing positions
    - REDUCE_EXPOSURE: Market conditions suggest caution
    - HOLD: Neutral conditions, maintain current exposure
    """

    # VIX thresholds
    VIX_LOW = 15
    VIX_ELEVATED = 25
    VIX_HIGH = 30
    VIX_EXTREME = 40

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MarketTimer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cache_dir = DATA_DIR / "cache" / "timing"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache
        self._last_conditions: Optional[MarketConditions] = None
        self._last_signal: Optional[MarketTimingSignal] = None
        self._last_update: Optional[datetime] = None
        self._cache_ttl_minutes = config.get("timing", {}).get("cache_minutes", 60)

        # Lazy-loaded components
        self._data_fetcher = None
        self._macro_fetcher = None

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
    def macro_fetcher(self):
        """Lazy load macro fetcher."""
        if self._macro_fetcher is None:
            try:
                from src.data.macro_fetcher import get_macro_fetcher
                self._macro_fetcher = get_macro_fetcher()
            except ImportError as e:
                logger.warning(f"Macro fetcher not available: {e}")
        return self._macro_fetcher

    def get_timing_signal(self, use_cache: bool = True) -> MarketTimingSignal:
        """
        Get current market timing signal.

        Args:
            use_cache: Whether to use cached signal

        Returns:
            MarketTimingSignal with recommendation
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            return self._last_signal

        # Gather market conditions
        conditions = self._gather_market_conditions()

        # Analyze and determine signal
        signal = self._analyze_conditions(conditions)

        # Update cache
        self._last_conditions = conditions
        self._last_signal = signal
        self._last_update = datetime.now()

        return signal

    def get_market_conditions(self, use_cache: bool = True) -> MarketConditions:
        """
        Get current market conditions without signal analysis.

        Args:
            use_cache: Whether to use cached conditions

        Returns:
            MarketConditions object
        """
        if use_cache and self._last_conditions and self._is_cache_valid():
            return self._last_conditions

        return self._gather_market_conditions()

    def _gather_market_conditions(self) -> MarketConditions:
        """Gather all market condition data."""
        conditions = MarketConditions()

        try:
            # Fetch VIX data
            self._fetch_vix_data(conditions)

            # Fetch SPY data
            self._fetch_spy_data(conditions)

            # Fetch economic indicators
            self._fetch_economic_data(conditions)

            # Determine overall condition
            conditions.overall_condition = self._determine_overall_condition(conditions)

        except Exception as e:
            logger.error(f"Error gathering market conditions: {e}")

        return conditions

    def _fetch_vix_data(self, conditions: MarketConditions) -> None:
        """Fetch VIX-related data."""
        if not self.data_fetcher:
            return

        try:
            # Fetch VIX historical data
            vix_df = self.data_fetcher.fetch_historical("^VIX", period="1y")

            if vix_df.empty:
                return

            # Current VIX level
            conditions.vix_level = vix_df["close"].iloc[-1]

            # VIX percentile (where current level sits in past year)
            current = conditions.vix_level
            all_values = vix_df["close"].values
            percentile = (all_values < current).sum() / len(all_values) * 100
            conditions.vix_percentile = percentile

            # VIX trend (comparing 5-day MA to 20-day MA)
            vix_sma5 = vix_df["close"].rolling(5).mean().iloc[-1]
            vix_sma20 = vix_df["close"].rolling(20).mean().iloc[-1]

            if pd.notna(vix_sma5) and pd.notna(vix_sma20):
                if vix_sma5 > vix_sma20 * 1.05:
                    conditions.vix_trend = "rising"
                elif vix_sma5 < vix_sma20 * 0.95:
                    conditions.vix_trend = "falling"
                else:
                    conditions.vix_trend = "neutral"

        except Exception as e:
            logger.warning(f"Error fetching VIX data: {e}")

    def _fetch_spy_data(self, conditions: MarketConditions) -> None:
        """Fetch SPY-related data."""
        if not self.data_fetcher:
            return

        try:
            spy_df = self.data_fetcher.fetch_historical("SPY", period="1y")

            if spy_df.empty:
                return

            # Current price
            conditions.spy_price = spy_df["close"].iloc[-1]

            # Calculate moving averages
            sma20 = spy_df["close"].rolling(20).mean().iloc[-1]
            sma50 = spy_df["close"].rolling(50).mean().iloc[-1]
            sma200 = spy_df["close"].rolling(200).mean().iloc[-1]

            # Position vs MAs (as percentage) - guard against NaN
            if pd.notna(sma20) and sma20 != 0:
                conditions.spy_vs_sma20 = (conditions.spy_price - sma20) / sma20 * 100
            if pd.notna(sma50) and sma50 != 0:
                conditions.spy_vs_sma50 = (conditions.spy_price - sma50) / sma50 * 100
            if pd.notna(sma200) and sma200 != 0:
                conditions.spy_vs_sma200 = (conditions.spy_price - sma200) / sma200 * 100

            # Calculate RSI
            conditions.spy_rsi = self._calculate_rsi(spy_df["close"], 14)

        except Exception as e:
            logger.warning(f"Error fetching SPY data: {e}")

    def _fetch_economic_data(self, conditions: MarketConditions) -> None:
        """Fetch economic indicator data."""
        if not self.macro_fetcher:
            return

        try:
            # Get treasury rates
            macro_data = self.macro_fetcher.get_macro_features(
                pd.DataFrame(index=[datetime.now()]),
                indicators=["treasury_10y"]
            )
            if "treasury_10y" in macro_data.columns:
                conditions.treasury_10y = macro_data["treasury_10y"].iloc[-1]

        except Exception as e:
            logger.warning(f"Error fetching economic data: {e}")

    def _determine_overall_condition(self, conditions: MarketConditions) -> str:
        """Determine overall market condition based on all factors."""
        bullish_signals = 0
        bearish_signals = 0

        # VIX assessment
        if conditions.vix_level < self.VIX_LOW:
            bullish_signals += 2
        elif conditions.vix_level < self.VIX_ELEVATED:
            bullish_signals += 1
        elif conditions.vix_level > self.VIX_EXTREME:
            bearish_signals += 3
        elif conditions.vix_level > self.VIX_HIGH:
            bearish_signals += 2
        elif conditions.vix_level > self.VIX_ELEVATED:
            bearish_signals += 1

        # VIX trend
        if conditions.vix_trend == "falling":
            bullish_signals += 1
        elif conditions.vix_trend == "rising":
            bearish_signals += 1

        # SPY vs moving averages
        if conditions.spy_vs_sma200 > 0:
            bullish_signals += 2
        else:
            bearish_signals += 2

        if conditions.spy_vs_sma50 > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1

        if conditions.spy_vs_sma20 > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1

        # RSI assessment
        if conditions.spy_rsi > 70:
            bearish_signals += 1  # Overbought
        elif conditions.spy_rsi < 30:
            bullish_signals += 1  # Oversold (contrarian)
        elif conditions.spy_rsi > 50:
            bullish_signals += 0.5

        # Determine overall
        if conditions.vix_level > self.VIX_EXTREME:
            return "volatile"

        net_signal = bullish_signals - bearish_signals

        if net_signal >= 3:
            return "bullish"
        elif net_signal <= -3:
            return "bearish"
        else:
            return "neutral"

    def _analyze_conditions(self, conditions: MarketConditions) -> MarketTimingSignal:
        """Analyze conditions and determine timing signal."""
        signal = MarketTimingSignal(
            signal=TimingSignal.HOLD,
            confidence=0.5,
            conditions=conditions,
            reasons=[],
            recommended_exposure=1.0,
        )

        add_score = 0  # Positive = add, negative = reduce
        reasons = []

        # === VIX Analysis ===
        if conditions.vix_level < self.VIX_LOW:
            add_score += 2
            reasons.append(f"VIX very low ({conditions.vix_level:.1f}) - calm market")
        elif conditions.vix_level < self.VIX_ELEVATED:
            add_score += 1
            reasons.append(f"VIX normal ({conditions.vix_level:.1f})")
        elif conditions.vix_level > self.VIX_EXTREME:
            add_score -= 3
            reasons.append(f"VIX extreme ({conditions.vix_level:.1f}) - high fear")
        elif conditions.vix_level > self.VIX_HIGH:
            add_score -= 2
            reasons.append(f"VIX high ({conditions.vix_level:.1f}) - elevated fear")
        elif conditions.vix_level > self.VIX_ELEVATED:
            add_score -= 1
            reasons.append(f"VIX elevated ({conditions.vix_level:.1f})")

        # VIX trend
        if conditions.vix_trend == "falling":
            add_score += 1
            reasons.append("VIX trending down - fear decreasing")
        elif conditions.vix_trend == "rising":
            add_score -= 1
            reasons.append("VIX trending up - fear increasing")

        # === SPY Analysis ===
        if conditions.spy_vs_sma200 > 5:
            add_score += 2
            reasons.append(f"SPY well above 200 MA (+{conditions.spy_vs_sma200:.1f}%)")
        elif conditions.spy_vs_sma200 > 0:
            add_score += 1
            reasons.append(f"SPY above 200 MA (+{conditions.spy_vs_sma200:.1f}%)")
        elif conditions.spy_vs_sma200 < -5:
            add_score -= 2
            reasons.append(f"SPY well below 200 MA ({conditions.spy_vs_sma200:.1f}%)")
        else:
            add_score -= 1
            reasons.append(f"SPY below 200 MA ({conditions.spy_vs_sma200:.1f}%)")

        # Short-term trend
        if conditions.spy_vs_sma20 > 0 and conditions.spy_vs_sma50 > 0:
            add_score += 1
            reasons.append("Short-term trend positive")
        elif conditions.spy_vs_sma20 < 0 and conditions.spy_vs_sma50 < 0:
            add_score -= 1
            reasons.append("Short-term trend negative")

        # === RSI Analysis ===
        if conditions.spy_rsi > 75:
            add_score -= 1
            reasons.append(f"SPY RSI overbought ({conditions.spy_rsi:.0f})")
        elif conditions.spy_rsi < 25:
            # Contrarian - oversold can be opportunity
            add_score += 1
            reasons.append(f"SPY RSI oversold ({conditions.spy_rsi:.0f}) - potential opportunity")

        # === Determine Signal ===
        signal.reasons = reasons

        if add_score >= 3:
            signal.signal = TimingSignal.ADD_EXPOSURE
            signal.confidence = min(0.9, 0.5 + add_score * 0.1)
            signal.recommended_exposure = 1.0 + min(0.5, add_score * 0.1)
        elif add_score <= -3:
            signal.signal = TimingSignal.REDUCE_EXPOSURE
            signal.confidence = min(0.9, 0.5 + abs(add_score) * 0.1)
            signal.recommended_exposure = max(0.5, 1.0 - abs(add_score) * 0.1)
        else:
            signal.signal = TimingSignal.HOLD
            signal.confidence = 0.5 + abs(add_score) * 0.05
            signal.recommended_exposure = 1.0

        logger.info(f"Market timing: {signal.signal.value} (confidence={signal.confidence:.2f}, "
                   f"exposure={signal.recommended_exposure:.2f})")

        return signal

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Calculate RSI for a price series."""
        try:
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except Exception:
            return 50.0

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._last_update is None or self._last_signal is None:
            return False

        age = datetime.now() - self._last_update
        return age < timedelta(minutes=self._cache_ttl_minutes)

    def get_status(self) -> Dict[str, Any]:
        """Get timer status."""
        return {
            "cache_valid": self._is_cache_valid(),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "last_signal": self._last_signal.signal.value if self._last_signal else None,
            "last_confidence": self._last_signal.confidence if self._last_signal else None,
            "conditions": self._last_conditions.to_dict() if self._last_conditions else None,
        }
