"""
LLM Client Module

Claude API client wrapper for intelligent agent analysis.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Module-level singleton
_llm_client: Optional["LLMClient"] = None
_lock = threading.Lock()


def get_llm_client(config: Optional[Dict[str, Any]] = None) -> "LLMClient":
    """
    Get or create the singleton LLMClient instance.

    Args:
        config: Configuration dictionary with LLM settings

    Returns:
        LLMClient singleton instance
    """
    global _llm_client
    with _lock:
        if _llm_client is None:
            _llm_client = LLMClient(config or {})
        return _llm_client


class LLMClient:
    """
    Claude API client wrapper for agent intelligence.

    Features:
    - Simple interface for text generation
    - Rate limiting and retry logic
    - System prompt customization per agent
    - Graceful degradation if API unavailable
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM client.

        Args:
            config: Configuration dictionary with optional keys:
                - llm_model: Model to use (default: claude-sonnet-4-20250514)
                - use_llm: Whether to enable LLM features (default: True)
        """
        self.config = config
        self.model = config.get("llm_model", self.DEFAULT_MODEL)
        self.enabled = config.get("use_llm", True)

        self._client = None
        self._last_request_time = 0
        self._min_request_interval = 0.5  # Rate limit: 2 requests/second

        if self.enabled:
            self._init_client()

    def _init_client(self) -> None:
        """Initialize the Anthropic client."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            logger.warning(
                "ANTHROPIC_API_KEY not set. LLM features will be disabled."
            )
            self.enabled = False
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"LLM client initialized with model: {self.model}")
        except ImportError:
            logger.warning(
                "anthropic package not installed. LLM features will be disabled. "
                "Install with: pip install anthropic"
            )
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.enabled = False

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt to send
            system_prompt: Optional system prompt for context
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)

        Returns:
            Generated text response, or None if failed
        """
        if not self.enabled or not self._client:
            logger.debug("LLM client not enabled, returning None")
            return None

        self._rate_limit()

        for attempt in range(self.MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": prompt}]

                kwargs = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "temperature": temperature,
                }

                if system_prompt:
                    kwargs["system"] = system_prompt

                response = self._client.messages.create(**kwargs)

                if response.content:
                    return response.content[0].text

                return None

            except Exception as e:
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}"
                )
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))

        logger.error("LLM request failed after all retries")
        return None

    def analyze_performance(
        self,
        metrics: Dict[str, Any],
        thresholds: Dict[str, float],
    ) -> Optional[str]:
        """
        Analyze trading performance metrics and provide insights.

        Args:
            metrics: Dictionary of performance metrics
            thresholds: Dictionary of threshold values for alerts

        Returns:
            Analysis text with insights and recommendations
        """
        system_prompt = """You are a quantitative trading analyst AI assistant.
Your role is to analyze trading system performance metrics and provide actionable insights.
Be concise, data-driven, and specific in your recommendations.
Format your response in clear sections with markdown."""

        prompt = f"""Analyze the following trading system performance metrics:

## Current Metrics
{self._format_metrics(metrics)}

## Alert Thresholds
{self._format_metrics(thresholds)}

Please provide:
1. A brief assessment of overall system health
2. Any concerning patterns or issues you identify
3. Specific, actionable recommendations for improvement
4. Priority level (low/medium/high) for each recommendation

Keep your analysis focused and actionable."""

        return self.generate(prompt, system_prompt=system_prompt)

    def evaluate_suggestion(
        self,
        suggestion: str,
        current_state: Dict[str, Any],
        available_actions: List[str],
    ) -> Optional[str]:
        """
        Evaluate a suggestion and decide on appropriate action.

        Args:
            suggestion: The suggestion from the Stock Analyst
            current_state: Current system state
            available_actions: List of actions the Developer can take

        Returns:
            Decision text with chosen action and reasoning
        """
        system_prompt = """You are a trading system developer AI assistant.
Your role is to evaluate suggestions from the Stock Analyst and decide on appropriate actions.
Be cautious with changes, prefer minimal interventions, and always explain your reasoning.
Format your response in clear sections with markdown."""

        prompt = f"""Evaluate this suggestion from the Stock Analyst:

## Suggestion
{suggestion}

## Current System State
{self._format_metrics(current_state)}

## Available Actions
{chr(10).join(f"- {action}" for action in available_actions)}

Please provide:
1. Your assessment of the suggestion's validity
2. The action you recommend (must be from the available actions list, or "no action")
3. Your reasoning for this decision
4. Any parameters or details for the action
5. Potential risks and how to mitigate them

Be conservative - only recommend changes if clearly beneficial."""

        return self.generate(prompt, system_prompt=system_prompt)

    def generate_daily_report(
        self,
        daily_metrics: Dict[str, Any],
        trades: List[Dict[str, Any]],
        positions: List[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Generate a comprehensive daily performance report.

        Args:
            daily_metrics: Dictionary of daily performance metrics
            trades: List of trades executed today
            positions: Current portfolio positions

        Returns:
            Formatted daily report text
        """
        system_prompt = """You are a trading analyst generating end-of-day reports.
Provide clear, concise summaries that highlight key performance indicators,
notable trades, and areas of concern. Use a professional tone."""

        trades_summary = "No trades today" if not trades else f"{len(trades)} trades executed"
        positions_summary = "No positions" if not positions else f"{len(positions)} open positions"

        prompt = f"""Generate a daily performance report for the trading system:

## Daily Metrics
{self._format_metrics(daily_metrics)}

## Trading Activity
{trades_summary}

## Portfolio
{positions_summary}

Please provide:
1. Executive summary (2-3 sentences)
2. Key performance highlights
3. Areas of concern (if any)
4. Recommendations for tomorrow

Keep the report concise and actionable."""

        return self.generate(prompt, system_prompt=system_prompt)

    def screen_stocks(
        self,
        candidates: List[Dict[str, Any]],
        portfolio: List[str],
        market_regime: str,
        strategy: str = "balanced"
    ) -> Optional[str]:
        """
        AI-enhanced stock screening and ranking.

        Args:
            candidates: List of candidate stocks with scores and metrics
            portfolio: Current portfolio symbols
            market_regime: Current market regime (bull, bear, choppy, volatile)
            strategy: Screening strategy (growth, value, momentum, quality, balanced)

        Returns:
            LLM analysis with rankings and recommendations
        """
        system_prompt = """You are a quantitative portfolio analyst AI assistant.
Your role is to analyze stock candidates and provide investment recommendations.
Consider both quantitative metrics and qualitative factors.
Be data-driven and specific in your rankings.
Format your response in clear sections with markdown."""

        # Format candidates
        candidates_str = self._format_candidates(candidates[:20])  # Top 20

        prompt = f"""Analyze and rank these stock candidates for portfolio addition:

## Current Market Regime
{market_regime.upper()} - Adjust your recommendations accordingly.

## Screening Strategy
{strategy.upper()} - Weight your analysis toward this investment style.

## Current Portfolio
{', '.join(portfolio) if portfolio else 'Empty portfolio'}

## Candidates (sorted by quantitative score)
{candidates_str}

Please provide:
1. **Top 5 Recommendations** - Your top picks with brief reasoning (2-3 sentences each)
2. **Stocks to Avoid** - Any candidates that should NOT be added and why
3. **Sector Considerations** - Comment on sector diversification
4. **Risk Assessment** - Overall risk level of your recommendations (low/medium/high)
5. **Entry Timing** - Whether now is a good time to add these positions

Keep your analysis concise and actionable. Focus on the most important factors."""

        return self.generate(prompt, system_prompt=system_prompt, max_tokens=1500)

    def evaluate_exit_candidates(
        self,
        holdings: List[Dict[str, Any]],
        market_outlook: str,
        benchmark_return: float = 0.0
    ) -> Optional[str]:
        """
        Evaluate holdings for potential exit.

        Args:
            holdings: List of current holdings with performance metrics
            market_outlook: Current market outlook description
            benchmark_return: Benchmark (e.g., SPY) return for comparison

        Returns:
            LLM analysis with exit recommendations
        """
        system_prompt = """You are a portfolio risk management AI assistant.
Your role is to evaluate current holdings and recommend exits when appropriate.
Be conservative - only recommend exits when clearly justified.
Consider both individual stock factors and portfolio-level impact.
Format your response in clear sections with markdown."""

        # Format holdings
        holdings_str = self._format_holdings(holdings)

        prompt = f"""Evaluate these holdings for potential exit:

## Market Outlook
{market_outlook}

## Benchmark Return (3-month)
{benchmark_return:.1%}

## Current Holdings
{holdings_str}

Please provide:
1. **Exit Recommendations** - Holdings that should be sold with clear reasoning
2. **Watch List** - Holdings to monitor closely but not exit yet
3. **Keep Positions** - Holdings that should be maintained
4. **Portfolio Adjustment** - Any rebalancing suggestions
5. **Risk Factors** - Key risks in the current portfolio

For each exit recommendation, specify:
- The urgency (immediate, within 1 week, when opportunity arises)
- Whether to exit fully or partially
- The primary reason (loss threshold, underperformance, fundamentals, etc.)

Be conservative in exit recommendations - avoid selling too quickly."""

        return self.generate(prompt, system_prompt=system_prompt, max_tokens=1500)

    def analyze_market_timing(
        self,
        conditions: Dict[str, Any],
        portfolio_exposure: float = 1.0
    ) -> Optional[str]:
        """
        Analyze market conditions for timing decisions.

        Args:
            conditions: Market condition data (VIX, SPY levels, etc.)
            portfolio_exposure: Current portfolio exposure level (0-1)

        Returns:
            LLM analysis with timing recommendations
        """
        system_prompt = """You are a market timing analyst AI assistant.
Your role is to analyze market conditions and recommend exposure adjustments.
Be measured in your recommendations - avoid extreme changes.
Consider both technical and fundamental factors.
Format your response in clear sections with markdown."""

        conditions_str = self._format_metrics(conditions)

        prompt = f"""Analyze current market conditions for timing decisions:

## Market Conditions
{conditions_str}

## Current Portfolio Exposure
{portfolio_exposure:.0%} invested

Please provide:
1. **Overall Market Assessment** - Current market state in 2-3 sentences
2. **Timing Signal** - Add exposure, reduce exposure, or hold
3. **Recommended Exposure Level** - What % should be invested (e.g., 80%)
4. **Key Factors** - 3-5 most important factors in your decision
5. **Time Horizon** - How long this recommendation is expected to be valid
6. **Risks** - What could invalidate this recommendation

Be measured - avoid recommending dramatic changes unless clearly warranted."""

        return self.generate(prompt, system_prompt=system_prompt, max_tokens=1000)

    def _format_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """Format candidate stocks for prompt."""
        lines = []
        for i, c in enumerate(candidates, 1):
            symbol = c.get("symbol", "N/A")
            score = c.get("score", c.get("total_score", 0))
            sector = c.get("sector", "N/A")
            pe = c.get("pe_ratio")
            growth = c.get("earnings_growth")
            rsi = c.get("rsi")
            ret_3m = c.get("return_3m")

            pe_str = f"P/E={pe:.1f}" if pe else "P/E=N/A"
            growth_str = f"Growth={growth:.1%}" if growth else "Growth=N/A"
            rsi_str = f"RSI={rsi:.0f}" if rsi else "RSI=N/A"
            ret_str = f"3M={ret_3m:.1%}" if ret_3m else "3M=N/A"

            lines.append(f"{i}. **{symbol}** (Score: {score:.1f})")
            lines.append(f"   - Sector: {sector}")
            lines.append(f"   - {pe_str}, {growth_str}, {rsi_str}, {ret_str}")

        return "\n".join(lines)

    def _format_holdings(self, holdings: List[Dict[str, Any]]) -> str:
        """Format holdings for prompt."""
        lines = []
        for h in holdings:
            symbol = h.get("symbol", "N/A")
            return_pct = h.get("total_return", h.get("return", 0))
            days_held = h.get("days_held", 0)
            sector = h.get("sector", "N/A")

            return_str = f"+{return_pct:.1%}" if return_pct >= 0 else f"{return_pct:.1%}"
            lines.append(f"- **{symbol}**: {return_str} ({days_held} days) - {sector}")

        return "\n".join(lines) if lines else "No holdings"

    def _format_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format metrics dictionary for prompt inclusion."""
        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            elif isinstance(value, dict):
                lines.append(f"- {key}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def is_available(self) -> bool:
        """Check if the LLM client is available and enabled."""
        return self.enabled and self._client is not None

    def get_status(self) -> Dict[str, Any]:
        """Get LLM client status."""
        return {
            "enabled": self.enabled,
            "available": self.is_available(),
            "model": self.model,
        }
