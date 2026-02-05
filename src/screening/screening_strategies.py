"""
Screening Strategies

Predefined strategies for stock screening with different weight allocations.
Each strategy emphasizes different factors based on investment style.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class StrategyWeights:
    """Weight allocation for scoring components."""
    fundamental: float = 0.25
    technical: float = 0.25
    momentum: float = 0.20
    sentiment: float = 0.15
    quality: float = 0.15

    def validate(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = self.fundamental + self.technical + self.momentum + self.sentiment + self.quality
        return abs(total - 1.0) < 0.001

    def to_dict(self) -> Dict[str, float]:
        return {
            "fundamental": self.fundamental,
            "technical": self.technical,
            "momentum": self.momentum,
            "sentiment": self.sentiment,
            "quality": self.quality,
        }


@dataclass
class FundamentalCriteria:
    """Criteria for fundamental scoring."""
    # Valuation preferences (lower percentile = better value)
    pe_preference: str = "moderate"  # low, moderate, high, any
    peg_max: float = 2.0

    # Growth requirements
    min_earnings_growth: float = 0.0
    min_revenue_growth: float = 0.0

    # Profitability requirements
    min_roe: float = 0.0
    min_profit_margin: float = 0.0

    # Health requirements
    max_debt_to_equity: float = 2.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pe_preference": self.pe_preference,
            "peg_max": self.peg_max,
            "min_earnings_growth": self.min_earnings_growth,
            "min_revenue_growth": self.min_revenue_growth,
            "min_roe": self.min_roe,
            "min_profit_margin": self.min_profit_margin,
            "max_debt_to_equity": self.max_debt_to_equity,
        }


@dataclass
class TechnicalCriteria:
    """Criteria for technical scoring."""
    # RSI preferences
    rsi_oversold: float = 30.0  # Consider bullish below this
    rsi_overbought: float = 70.0  # Consider bearish above this

    # Trend requirements
    require_above_sma20: bool = False
    require_above_sma50: bool = False
    require_above_sma200: bool = False

    # MACD
    require_macd_positive: bool = False
    require_macd_crossover: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "require_above_sma20": self.require_above_sma20,
            "require_above_sma50": self.require_above_sma50,
            "require_above_sma200": self.require_above_sma200,
            "require_macd_positive": self.require_macd_positive,
            "require_macd_crossover": self.require_macd_crossover,
        }


@dataclass
class MomentumCriteria:
    """Criteria for momentum scoring."""
    # Return requirements
    min_return_1m: float = -0.50  # -50% min (filter out extreme losers)
    min_return_3m: float = -0.50
    min_return_6m: float = -0.50

    # Relative strength
    compare_to_market: bool = True  # Compare to SPY
    min_relative_strength: float = -0.10  # Allow 10% underperformance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_return_1m": self.min_return_1m,
            "min_return_3m": self.min_return_3m,
            "min_return_6m": self.min_return_6m,
            "compare_to_market": self.compare_to_market,
            "min_relative_strength": self.min_relative_strength,
        }


class ScreeningStrategy(ABC):
    """Base class for screening strategies."""

    def __init__(self):
        self.name: str = "base"
        self.description: str = ""
        self.weights: StrategyWeights = StrategyWeights()
        self.fundamental: FundamentalCriteria = FundamentalCriteria()
        self.technical: TechnicalCriteria = TechnicalCriteria()
        self.momentum: MomentumCriteria = MomentumCriteria()

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return strategy description."""
        pass

    def get_weights(self) -> StrategyWeights:
        """Return weight allocation."""
        return self.weights

    def get_fundamental_criteria(self) -> FundamentalCriteria:
        """Return fundamental criteria."""
        return self.fundamental

    def get_technical_criteria(self) -> TechnicalCriteria:
        """Return technical criteria."""
        return self.technical

    def get_momentum_criteria(self) -> MomentumCriteria:
        """Return momentum criteria."""
        return self.momentum

    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary."""
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "weights": self.weights.to_dict(),
            "fundamental": self.fundamental.to_dict(),
            "technical": self.technical.to_dict(),
            "momentum": self.momentum.to_dict(),
        }


class GrowthStrategy(ScreeningStrategy):
    """
    Growth-focused strategy.

    Emphasizes earnings growth, revenue growth, and momentum.
    Accepts higher valuations for faster-growing companies.
    """

    def __init__(self):
        super().__init__()
        self.name = "growth"
        self.description = "Focus on high-growth companies with strong momentum"

        # Higher weight on growth and momentum
        self.weights = StrategyWeights(
            fundamental=0.30,
            technical=0.15,
            momentum=0.30,
            sentiment=0.10,
            quality=0.15,
        )

        # Growth-focused fundamentals
        self.fundamental = FundamentalCriteria(
            pe_preference="any",  # Accept high P/E for growth
            peg_max=3.0,  # Higher PEG acceptable
            min_earnings_growth=0.15,  # 15% min earnings growth
            min_revenue_growth=0.10,  # 10% min revenue growth
            min_roe=0.10,  # 10% ROE minimum
            min_profit_margin=0.0,
            max_debt_to_equity=2.0,
        )

        # Technical focus on trend
        self.technical = TechnicalCriteria(
            rsi_oversold=30.0,
            rsi_overbought=80.0,  # Allow higher RSI
            require_above_sma50=True,  # Must be trending up
            require_macd_positive=True,
        )

        # Momentum emphasis
        self.momentum = MomentumCriteria(
            min_return_3m=0.0,  # Positive 3-month return
            min_return_6m=0.0,  # Positive 6-month return
            compare_to_market=True,
            min_relative_strength=0.0,  # Must beat market
        )

    def get_name(self) -> str:
        return "growth"

    def get_description(self) -> str:
        return "Focus on high-growth companies with strong momentum"


class ValueStrategy(ScreeningStrategy):
    """
    Value-focused strategy.

    Emphasizes low valuations, strong balance sheets, and dividends.
    Looks for undervalued stocks with margin of safety.
    """

    def __init__(self):
        super().__init__()
        self.name = "value"
        self.description = "Focus on undervalued stocks with strong fundamentals"

        # Higher weight on fundamentals and quality
        self.weights = StrategyWeights(
            fundamental=0.40,
            technical=0.15,
            momentum=0.10,
            sentiment=0.10,
            quality=0.25,
        )

        # Value-focused fundamentals
        self.fundamental = FundamentalCriteria(
            pe_preference="low",  # Prefer low P/E
            peg_max=1.5,  # Low PEG ratio
            min_earnings_growth=-0.05,  # Tolerate slight decline
            min_revenue_growth=-0.05,
            min_roe=0.08,  # 8% ROE minimum
            min_profit_margin=0.05,  # 5% profit margin
            max_debt_to_equity=1.0,  # Lower debt
        )

        # Technical less important
        self.technical = TechnicalCriteria(
            rsi_oversold=25.0,  # Look for oversold
            rsi_overbought=70.0,
            require_above_sma200=True,  # Long-term uptrend
        )

        # Momentum less important
        self.momentum = MomentumCriteria(
            min_return_1m=-0.20,  # Allow recent weakness
            min_return_3m=-0.20,
            min_return_6m=-0.10,
            compare_to_market=False,  # Don't require market beating
        )

    def get_name(self) -> str:
        return "value"

    def get_description(self) -> str:
        return "Focus on undervalued stocks with strong fundamentals"


class MomentumStrategy(ScreeningStrategy):
    """
    Momentum-focused strategy.

    Emphasizes price momentum and relative strength.
    Follows trends and market leaders.
    """

    def __init__(self):
        super().__init__()
        self.name = "momentum"
        self.description = "Focus on stocks with strong price momentum"

        # Heavy weight on momentum and technical
        self.weights = StrategyWeights(
            fundamental=0.15,
            technical=0.30,
            momentum=0.35,
            sentiment=0.10,
            quality=0.10,
        )

        # Basic fundamental requirements
        self.fundamental = FundamentalCriteria(
            pe_preference="any",
            peg_max=5.0,  # Very lenient
            min_earnings_growth=0.0,
            min_revenue_growth=0.0,
            min_roe=0.0,
            max_debt_to_equity=3.0,
        )

        # Strong technical requirements
        self.technical = TechnicalCriteria(
            rsi_oversold=40.0,  # Avoid oversold
            rsi_overbought=85.0,  # Allow overbought (momentum!)
            require_above_sma20=True,
            require_above_sma50=True,
            require_macd_positive=True,
            require_macd_crossover=True,
        )

        # Strong momentum requirements
        self.momentum = MomentumCriteria(
            min_return_1m=0.02,  # 2% min 1-month return
            min_return_3m=0.05,  # 5% min 3-month return
            min_return_6m=0.10,  # 10% min 6-month return
            compare_to_market=True,
            min_relative_strength=0.05,  # Beat market by 5%
        )

    def get_name(self) -> str:
        return "momentum"

    def get_description(self) -> str:
        return "Focus on stocks with strong price momentum"


class QualityStrategy(ScreeningStrategy):
    """
    Quality-focused strategy.

    Emphasizes profitability, consistency, and low volatility.
    Looks for stable, well-managed companies.
    """

    def __init__(self):
        super().__init__()
        self.name = "quality"
        self.description = "Focus on high-quality, profitable companies"

        # Heavy weight on quality and fundamentals
        self.weights = StrategyWeights(
            fundamental=0.25,
            technical=0.15,
            momentum=0.15,
            sentiment=0.10,
            quality=0.35,
        )

        # High-quality fundamental requirements
        self.fundamental = FundamentalCriteria(
            pe_preference="moderate",
            peg_max=2.5,
            min_earnings_growth=0.05,
            min_revenue_growth=0.03,
            min_roe=0.15,  # High ROE
            min_profit_margin=0.10,  # Good margins
            max_debt_to_equity=0.8,  # Low debt
        )

        # Moderate technical
        self.technical = TechnicalCriteria(
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            require_above_sma50=True,
        )

        # Stable momentum
        self.momentum = MomentumCriteria(
            min_return_1m=-0.10,
            min_return_3m=-0.05,
            min_return_6m=0.0,
            compare_to_market=False,
        )

    def get_name(self) -> str:
        return "quality"

    def get_description(self) -> str:
        return "Focus on high-quality, profitable companies"


class BalancedStrategy(ScreeningStrategy):
    """
    Balanced strategy.

    Equal emphasis on all factors for diversified stock selection.
    Default strategy with moderate requirements.
    """

    def __init__(self):
        super().__init__()
        self.name = "balanced"
        self.description = "Balanced approach across all factors"

        # Equal weights
        self.weights = StrategyWeights(
            fundamental=0.25,
            technical=0.25,
            momentum=0.20,
            sentiment=0.15,
            quality=0.15,
        )

        # Moderate requirements
        self.fundamental = FundamentalCriteria(
            pe_preference="moderate",
            peg_max=2.0,
            min_earnings_growth=0.0,
            min_revenue_growth=0.0,
            min_roe=0.05,
            min_profit_margin=0.0,
            max_debt_to_equity=2.0,
        )

        self.technical = TechnicalCriteria(
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            require_above_sma50=True,
        )

        self.momentum = MomentumCriteria(
            min_return_1m=-0.15,
            min_return_3m=-0.10,
            min_return_6m=-0.05,
            compare_to_market=True,
            min_relative_strength=-0.05,
        )

    def get_name(self) -> str:
        return "balanced"

    def get_description(self) -> str:
        return "Balanced approach across all factors"


# Strategy registry
STRATEGIES: Dict[str, type] = {
    "growth": GrowthStrategy,
    "value": ValueStrategy,
    "momentum": MomentumStrategy,
    "quality": QualityStrategy,
    "balanced": BalancedStrategy,
}


def get_strategy(name: str) -> ScreeningStrategy:
    """
    Get a strategy by name.

    Args:
        name: Strategy name (growth, value, momentum, quality, balanced)

    Returns:
        Strategy instance

    Raises:
        ValueError: If strategy name is not recognized
    """
    name = name.lower()
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}")

    return STRATEGIES[name]()


def list_strategies() -> Dict[str, str]:
    """
    List available strategies with descriptions.

    Returns:
        Dictionary mapping strategy name to description
    """
    return {name: cls().get_description() for name, cls in STRATEGIES.items()}
