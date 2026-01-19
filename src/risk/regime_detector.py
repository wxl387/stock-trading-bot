"""
Market regime detection for adaptive trading strategies.
Detects bull, bear, choppy, and volatile market conditions.
"""
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "bull"           # Uptrending market
    BEAR = "bear"           # Downtrending market
    CHOPPY = "choppy"       # Sideways, low trend
    VOLATILE = "volatile"   # High volatility (VIX spike)


@dataclass
class RegimeParameters:
    """Trading parameters for each regime."""
    regime: MarketRegime
    stop_loss_pct: float           # Stop loss percentage
    position_size_multiplier: float  # Position sizing multiplier
    min_confidence: float          # Minimum ML confidence to trade
    trailing_stop_enabled: bool    # Whether to use trailing stops
    description: str


# Default regime parameters
DEFAULT_REGIME_PARAMS: Dict[MarketRegime, RegimeParameters] = {
    MarketRegime.BULL: RegimeParameters(
        regime=MarketRegime.BULL,
        stop_loss_pct=0.05,          # 5% stop loss
        position_size_multiplier=1.0,  # Normal size
        min_confidence=0.55,          # Lower confidence threshold
        trailing_stop_enabled=True,
        description="Uptrending market - normal stops, trailing enabled"
    ),
    MarketRegime.BEAR: RegimeParameters(
        regime=MarketRegime.BEAR,
        stop_loss_pct=0.03,          # 3% tighter stop
        position_size_multiplier=0.7,  # Reduced size
        min_confidence=0.60,          # Higher confidence required
        trailing_stop_enabled=True,
        description="Downtrending market - tight stops, reduced sizing"
    ),
    MarketRegime.CHOPPY: RegimeParameters(
        regime=MarketRegime.CHOPPY,
        stop_loss_pct=0.02,          # 2% very tight stop
        position_size_multiplier=0.5,  # Half size
        min_confidence=0.70,          # High confidence required
        trailing_stop_enabled=False,   # No trailing in choppy
        description="Sideways market - very tight stops, minimal trading"
    ),
    MarketRegime.VOLATILE: RegimeParameters(
        regime=MarketRegime.VOLATILE,
        stop_loss_pct=0.08,          # 8% wider stop (ATR-based recommended)
        position_size_multiplier=0.5,  # Half size
        min_confidence=0.65,          # Higher confidence
        trailing_stop_enabled=True,
        description="High volatility - wide stops, reduced sizing"
    )
}


class RegimeDetector:
    """
    Detects current market regime based on technical indicators.

    Detection logic:
    1. VIX > 30: VOLATILE (overrides other signals)
    2. 50 SMA > 200 SMA and ADX > 25: BULL
    3. 50 SMA < 200 SMA and ADX > 25: BEAR
    4. ADX < 20: CHOPPY
    5. Default to previous regime or BULL
    """

    def __init__(
        self,
        vix_volatile_threshold: float = 30.0,
        adx_trend_threshold: float = 25.0,
        adx_choppy_threshold: float = 20.0,
        cache_ttl_hours: int = 4
    ):
        """
        Initialize regime detector.

        Args:
            vix_volatile_threshold: VIX level for volatile regime.
            adx_trend_threshold: ADX level for trending market.
            adx_choppy_threshold: ADX level below which market is choppy.
            cache_ttl_hours: Hours to cache regime detection.
        """
        self.vix_volatile_threshold = vix_volatile_threshold
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_choppy_threshold = adx_choppy_threshold
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        self._cached_regime: Optional[MarketRegime] = None
        self._cache_time: Optional[datetime] = None
        self._regime_params = DEFAULT_REGIME_PARAMS.copy()

        logger.info(f"RegimeDetector initialized: VIX>{vix_volatile_threshold}=volatile, "
                   f"ADX>{adx_trend_threshold}=trending, ADX<{adx_choppy_threshold}=choppy")

    def detect_regime(
        self,
        market_data: Optional[pd.DataFrame] = None,
        vix: Optional[float] = None,
        use_cache: bool = True
    ) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            market_data: DataFrame with OHLCV for a broad index (SPY/^GSPC).
            vix: Current VIX value (fetched if not provided).
            use_cache: Whether to use cached regime.

        Returns:
            Current MarketRegime.
        """
        # Check cache
        if use_cache and self._cached_regime is not None and self._cache_time is not None:
            if datetime.now() - self._cache_time < self.cache_ttl:
                return self._cached_regime

        # Fetch VIX if not provided
        if vix is None:
            vix = self._get_current_vix()

        # Check for volatile regime first (VIX override)
        if vix is not None and vix > self.vix_volatile_threshold:
            regime = MarketRegime.VOLATILE
            logger.info(f"Regime detected: VOLATILE (VIX={vix:.1f})")
            self._update_cache(regime)
            return regime

        # Need market data for trend detection
        if market_data is None:
            market_data = self._fetch_market_data()

        if market_data is None or len(market_data) < 200:
            # Not enough data, return cached or default
            return self._cached_regime or MarketRegime.BULL

        # Calculate indicators
        sma_50 = market_data['close'].rolling(window=50).mean().iloc[-1]
        sma_200 = market_data['close'].rolling(window=200).mean().iloc[-1]
        adx = self._calculate_adx(market_data)

        # Determine regime
        regime = self._classify_regime(sma_50, sma_200, adx, vix)

        logger.info(f"Regime detected: {regime.value.upper()} "
                   f"(SMA50={sma_50:.2f}, SMA200={sma_200:.2f}, ADX={adx:.1f}, VIX={vix or 'N/A'})")

        self._update_cache(regime)
        return regime

    def _classify_regime(
        self,
        sma_50: float,
        sma_200: float,
        adx: float,
        vix: Optional[float]
    ) -> MarketRegime:
        """Classify regime based on indicators."""
        # Choppy market (low ADX)
        if adx < self.adx_choppy_threshold:
            return MarketRegime.CHOPPY

        # Trending market (high ADX)
        if adx >= self.adx_trend_threshold:
            if sma_50 > sma_200:
                return MarketRegime.BULL
            else:
                return MarketRegime.BEAR

        # Middle ground - use SMA for direction
        if sma_50 > sma_200:
            return MarketRegime.BULL
        else:
            return MarketRegime.BEAR

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index (ADX)."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Smoothed values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0.0

    def _get_current_vix(self) -> Optional[float]:
        """Get current VIX value."""
        try:
            from src.data.macro_fetcher import get_macro_fetcher
            fetcher = get_macro_fetcher()

            vix_df = fetcher.fetch_indicator(
                "vix",
                start_date=datetime.now() - timedelta(days=7),
                end_date=datetime.now()
            )

            if not vix_df.empty:
                return float(vix_df.iloc[-1].iloc[0])

        except Exception as e:
            logger.warning(f"Failed to fetch VIX for regime detection: {e}")

        return None

    def _fetch_market_data(self) -> Optional[pd.DataFrame]:
        """Fetch market data for SPY or similar broad index."""
        try:
            from src.data.data_fetcher import DataFetcher
            fetcher = DataFetcher()

            # Use SPY as market proxy
            df = fetcher.fetch_stock_data(
                symbol="SPY",
                start_date=datetime.now() - timedelta(days=365),
                end_date=datetime.now()
            )

            return df

        except Exception as e:
            logger.warning(f"Failed to fetch market data for regime detection: {e}")
            return None

    def _update_cache(self, regime: MarketRegime) -> None:
        """Update cached regime."""
        self._cached_regime = regime
        self._cache_time = datetime.now()

    def get_regime_parameters(self, regime: Optional[MarketRegime] = None) -> RegimeParameters:
        """
        Get trading parameters for a regime.

        Args:
            regime: MarketRegime (uses current if not provided).

        Returns:
            RegimeParameters for the regime.
        """
        if regime is None:
            regime = self._cached_regime or self.detect_regime()

        return self._regime_params[regime]

    def set_regime_parameters(
        self,
        regime: MarketRegime,
        params: RegimeParameters
    ) -> None:
        """
        Override parameters for a specific regime.

        Args:
            regime: MarketRegime to configure.
            params: New RegimeParameters.
        """
        self._regime_params[regime] = params
        logger.info(f"Updated parameters for {regime.value}: {params.description}")

    def get_all_regime_parameters(self) -> Dict[MarketRegime, RegimeParameters]:
        """Get all regime parameters."""
        return self._regime_params.copy()

    def get_status(self) -> Dict:
        """Get current detector status."""
        return {
            "current_regime": self._cached_regime.value if self._cached_regime else None,
            "cache_time": self._cache_time.isoformat() if self._cache_time else None,
            "parameters": {
                r.value: {
                    "stop_loss_pct": p.stop_loss_pct,
                    "size_multiplier": p.position_size_multiplier,
                    "min_confidence": p.min_confidence,
                    "trailing_enabled": p.trailing_stop_enabled
                }
                for r, p in self._regime_params.items()
            }
        }


# Singleton instance
_regime_detector: Optional[RegimeDetector] = None


def get_regime_detector() -> RegimeDetector:
    """Get singleton regime detector instance."""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = RegimeDetector()
    return _regime_detector
