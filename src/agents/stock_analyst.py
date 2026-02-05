"""
Stock Analyst Agent

AI-powered agent that monitors market performance, detects issues, and suggests improvements.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_agent import (
    AgentMessage,
    AgentRole,
    BaseAgent,
    MessagePriority,
    MessageType,
)

logger = logging.getLogger(__name__)


class StockAnalystAgent(BaseAgent):
    """
    Stock Analyst Agent - monitors trading system performance.

    Responsibilities:
    - Regular health checks (Sharpe, drawdown, win rate)
    - Model degradation analysis
    - Daily portfolio reviews
    - Alert generation for concerning metrics

    Alert Thresholds:
    - Sharpe < 0.5 -> Warning
    - Drawdown > 10% -> High priority
    - Win rate < 45% -> Warning
    - Accuracy drop > 3% -> Degradation alert
    """

    def __init__(
        self,
        config: Dict[str, Any],
        message_queue,
        notifier=None,
        llm_client=None,
    ):
        """
        Initialize Stock Analyst agent.

        Args:
            config: Agent configuration
            message_queue: Shared message queue
            notifier: Optional Discord notifier
            llm_client: Optional LLM client for intelligent analysis
        """
        super().__init__(
            role=AgentRole.STOCK_ANALYST,
            config=config,
            message_queue=message_queue,
            notifier=notifier,
            llm_client=llm_client,
        )

        # Load thresholds from config
        analyst_config = config.get("stock_analyst", {})
        self.sharpe_warning = analyst_config.get("sharpe_warning", 0.5)
        self.drawdown_warning = analyst_config.get("drawdown_warning", 0.10)
        self.win_rate_warning = analyst_config.get("win_rate_warning", 0.45)
        self.accuracy_drop_threshold = analyst_config.get("accuracy_drop_threshold", 0.03)

        # Track last analysis times
        self._last_health_check: Optional[datetime] = None
        self._last_degradation_check: Optional[datetime] = None
        self._last_daily_review: Optional[datetime] = None
        self._last_stock_screening: Optional[datetime] = None
        self._last_portfolio_review: Optional[datetime] = None
        self._last_market_analysis: Optional[datetime] = None

        # Cache for data aggregator and degradation monitor
        self._data_aggregator = None
        self._degradation_monitor = None
        self._stock_screener = None
        self._market_timer = None
        self._symbol_manager = None

        logger.info(f"Stock Analyst agent initialized with thresholds: "
                   f"sharpe={self.sharpe_warning}, drawdown={self.drawdown_warning}, "
                   f"win_rate={self.win_rate_warning}")

    @property
    def data_aggregator(self):
        """Lazy load data aggregator."""
        if self._data_aggregator is None:
            try:
                from src.analytics.data_aggregator import DataAggregator
                self._data_aggregator = DataAggregator()
            except ImportError as e:
                logger.error(f"Failed to import DataAggregator: {e}")
        return self._data_aggregator

    @property
    def degradation_monitor(self):
        """Lazy load degradation monitor."""
        if self._degradation_monitor is None:
            try:
                from src.ml.degradation_monitor import DegradationMonitor
                from config.settings import Settings
                config = Settings.load_trading_config()
                deg_config = config.get("retraining", {}).get("degradation_detection", {})

                self._degradation_monitor = DegradationMonitor(
                    enabled=deg_config.get("enabled", True),
                    accuracy_drop_threshold=deg_config.get("accuracy_drop_threshold", 0.05),
                    confidence_collapse_threshold=deg_config.get("confidence_collapse_threshold", 0.55),
                    min_win_rate=deg_config.get("min_win_rate", 0.40),
                )
            except ImportError as e:
                logger.error(f"Failed to import DegradationMonitor: {e}")
        return self._degradation_monitor

    @property
    def stock_screener(self):
        """Lazy load stock screener."""
        if self._stock_screener is None:
            try:
                from src.screening.stock_screener import get_stock_screener
                from config.settings import Settings
                config = Settings.load_trading_config()
                self._stock_screener = get_stock_screener(config)
            except ImportError as e:
                logger.error(f"Failed to import StockScreener: {e}")
        return self._stock_screener

    @property
    def market_timer(self):
        """Lazy load market timer."""
        if self._market_timer is None:
            try:
                from src.risk.market_timing import get_market_timer
                from config.settings import Settings
                config = Settings.load_trading_config()
                self._market_timer = get_market_timer(config)
            except ImportError as e:
                logger.error(f"Failed to import MarketTimer: {e}")
        return self._market_timer

    @property
    def symbol_manager(self):
        """Lazy load symbol manager."""
        if self._symbol_manager is None:
            try:
                from src.core.symbol_manager import get_symbol_manager
                from config.settings import Settings
                config = Settings.load_trading_config()
                self._symbol_manager = get_symbol_manager(config)
            except ImportError as e:
                logger.error(f"Failed to import SymbolManager: {e}")
        return self._symbol_manager

    def analyze(self) -> List[AgentMessage]:
        """
        Perform analysis and generate observations/suggestions.

        This is called periodically by the orchestrator. The actual
        analysis type depends on the schedule that triggered it.

        Returns:
            List of messages to send to the Developer agent
        """
        # Default behavior: run health check if it's been a while
        messages = []

        # Check if we should run any analysis
        now = datetime.now()

        # Only run if explicitly called by orchestrator or if enough time has passed
        # The orchestrator will call specific methods for scheduled tasks

        return messages

    def run_health_check(self) -> List[AgentMessage]:
        """
        Run performance health check.

        Checks Sharpe ratio, drawdown, win rate, and other key metrics.

        Returns:
            List of messages for detected issues
        """
        self._last_health_check = datetime.now()
        messages = []

        try:
            metrics = self._gather_performance_metrics()
            if not metrics:
                logger.warning("No metrics available for health check")
                return messages

            issues = self._detect_issues(metrics)

            if issues:
                # Create alert message
                content = self._format_health_check_report(metrics, issues)
                priority = self._determine_priority(issues)

                # Use LLM to enhance analysis if available
                if self.llm_client and self.llm_client.is_available():
                    llm_analysis = self.llm_client.analyze_performance(
                        metrics,
                        {
                            "sharpe_warning": self.sharpe_warning,
                            "drawdown_warning": self.drawdown_warning,
                            "win_rate_warning": self.win_rate_warning,
                        }
                    )
                    if llm_analysis:
                        content += f"\n\n### AI Analysis\n{llm_analysis}"

                message = self.create_message(
                    recipient=AgentRole.DEVELOPER,
                    message_type=MessageType.OBSERVATION,
                    subject=f"Performance Alert: {len(issues)} issue(s) detected",
                    content=content,
                    priority=priority,
                    context={"metrics": metrics, "issues": issues},
                    requires_response=priority.value >= MessagePriority.HIGH.value,
                )
                messages.append(message)

                logger.info(f"Health check found {len(issues)} issues")
            else:
                logger.info("Health check: All metrics within acceptable ranges")

        except Exception as e:
            logger.error(f"Error during health check: {e}")

        return messages

    def run_degradation_check(self) -> List[AgentMessage]:
        """
        Run model degradation analysis.

        Checks if production models are degrading and suggests retraining.

        Returns:
            List of messages for degradation alerts
        """
        self._last_degradation_check = datetime.now()
        messages = []

        if not self.degradation_monitor:
            logger.warning("Degradation monitor not available")
            return messages

        try:
            reports = self.degradation_monitor.check_all_models()

            degraded_models = [r for r in reports if r.is_degraded]

            if degraded_models:
                content = self._format_degradation_report(degraded_models)

                # Use LLM to provide insights if available
                if self.llm_client and self.llm_client.is_available():
                    model_info = {
                        r.model_type: {
                            "metrics": r.metrics,
                            "baseline": r.baseline_metrics,
                            "reasons": r.degradation_reasons,
                        }
                        for r in degraded_models
                    }
                    llm_analysis = self.llm_client.analyze_performance(
                        {"degraded_models": model_info},
                        {"accuracy_drop_threshold": self.accuracy_drop_threshold}
                    )
                    if llm_analysis:
                        content += f"\n\n### AI Analysis\n{llm_analysis}"

                message = self.create_message(
                    recipient=AgentRole.DEVELOPER,
                    message_type=MessageType.SUGGESTION,
                    subject=f"Model Degradation: {len(degraded_models)} model(s) affected",
                    content=content,
                    priority=MessagePriority.HIGH,
                    context={
                        "degraded_models": [r.to_dict() for r in degraded_models],
                        "recommendations": [r.recommendation for r in degraded_models],
                    },
                    requires_response=True,
                )
                messages.append(message)

                logger.warning(f"Degradation check found {len(degraded_models)} degraded models")
            else:
                logger.info("Degradation check: All models healthy")

        except Exception as e:
            logger.error(f"Error during degradation check: {e}")

        return messages

    def run_daily_review(self) -> List[AgentMessage]:
        """
        Run comprehensive daily portfolio review.

        Generates end-of-day summary with insights and recommendations.

        Returns:
            List of messages with daily review
        """
        self._last_daily_review = datetime.now()
        messages = []

        try:
            # Get today's data
            today = datetime.now().date()
            start_of_day = datetime.combine(today, datetime.min.time())

            if not self.data_aggregator:
                logger.warning("Data aggregator not available")
                return messages

            # Gather daily data
            analytics_data = self.data_aggregator.prepare_analytics_data(
                start_date=start_of_day - timedelta(days=30),
                end_date=datetime.now()
            )

            trades = analytics_data.get("trades", pd.DataFrame())
            today_trades = trades[
                trades["timestamp"].dt.date == today
            ] if not trades.empty and "timestamp" in trades.columns else pd.DataFrame()

            metrics = analytics_data.get("portfolio_metrics", {})
            positions = analytics_data.get("positions", {})

            # Format daily report
            content = self._format_daily_report(
                metrics,
                today_trades,
                positions,
                analytics_data.get("returns", pd.Series())
            )

            # Use LLM to generate insights if available
            if self.llm_client and self.llm_client.is_available():
                llm_report = self.llm_client.generate_daily_report(
                    metrics,
                    today_trades.to_dict("records") if not today_trades.empty else [],
                    [{"symbol": k, **v} for k, v in positions.items()],
                )
                if llm_report:
                    content += f"\n\n### AI Insights\n{llm_report}"

            message = self.create_message(
                recipient=AgentRole.DEVELOPER,
                message_type=MessageType.STATUS_UPDATE,
                subject=f"Daily Review - {today.strftime('%Y-%m-%d')}",
                content=content,
                priority=MessagePriority.NORMAL,
                context={
                    "metrics": metrics,
                    "trades_count": len(today_trades),
                    "positions_count": len(positions),
                },
                requires_response=False,
            )
            messages.append(message)

            logger.info("Daily review completed")

        except Exception as e:
            logger.error(f"Error during daily review: {e}")

        return messages

    def run_stock_screening(self) -> List[AgentMessage]:
        """
        Run weekly stock screening to find new candidates.

        Screens stocks from the universe, ranks them, and sends recommendations
        to the Developer agent for potential addition.

        Returns:
            List of messages with screening recommendations
        """
        self._last_stock_screening = datetime.now()
        messages = []

        if not self.stock_screener:
            logger.warning("Stock screener not available")
            return messages

        try:
            # Get current portfolio symbols
            current_portfolio = []
            if self.symbol_manager:
                current_portfolio = self.symbol_manager.get_active_symbols()

            # Get top candidates excluding current portfolio
            from config.settings import Settings
            config = Settings.load_trading_config()
            strategy = config.get("dynamic_symbols", {}).get("screening", {}).get("strategy", "balanced")

            # Get stock universe
            try:
                from src.data.stock_universe import get_stock_universe
                universe = get_stock_universe()
                all_symbols = universe.get_universe()
            except Exception as e:
                logger.error(f"Failed to get universe: {e}")
                return messages

            # Score all stocks
            scores = self.stock_screener.score_stocks(all_symbols)

            # Filter out current portfolio
            portfolio_set = set(current_portfolio)
            candidates = [s for s in scores if s.symbol not in portfolio_set][:30]

            # Get market regime
            market_regime = "neutral"
            try:
                from src.risk.regime_detector import get_regime_detector
                detector = get_regime_detector()
                regime = detector.detect_regime()
                market_regime = regime.value if hasattr(regime, 'value') else str(regime)
            except Exception:
                pass

            # Format screening report
            content = self._format_screening_report(candidates[:20], current_portfolio, strategy)

            # Use LLM for enhanced analysis if available
            if self.llm_client and self.llm_client.is_available():
                llm_analysis = self.llm_client.screen_stocks(
                    candidates=[s.to_dict() for s in candidates[:20]],
                    portfolio=current_portfolio,
                    market_regime=market_regime,
                    strategy=strategy
                )
                if llm_analysis:
                    content += f"\n\n### AI Analysis\n{llm_analysis}"

            # Create message with top candidates as recommendations
            top_candidates = candidates[:10]
            recommendations = [
                {
                    "symbol": c.symbol,
                    "score": c.total_score,
                    "sector": c.sector,
                    "action": "add_symbol",
                    "reason": f"Score {c.total_score:.1f}, {c.sector}",
                }
                for c in top_candidates
            ]

            message = self.create_message(
                recipient=AgentRole.DEVELOPER,
                message_type=MessageType.SUGGESTION,
                subject=f"Weekly Stock Screening - {len(recommendations)} Candidates",
                content=content,
                priority=MessagePriority.NORMAL,
                context={
                    "screening_date": datetime.now().isoformat(),
                    "strategy": strategy,
                    "market_regime": market_regime,
                    "candidates_screened": len(all_symbols),
                    "recommendations": recommendations,
                },
                requires_response=True,
            )
            messages.append(message)

            logger.info(f"Stock screening complete: {len(recommendations)} recommendations")

        except Exception as e:
            logger.error(f"Error during stock screening: {e}")

        return messages

    def run_portfolio_review(self) -> List[AgentMessage]:
        """
        Run daily portfolio review to identify underperformers.

        Analyzes current holdings for potential exits based on performance,
        holding period, and other criteria.

        Returns:
            List of messages with exit recommendations
        """
        self._last_portfolio_review = datetime.now()
        messages = []

        if not self.symbol_manager:
            logger.warning("Symbol manager not available")
            return messages

        try:
            # Get benchmark return (SPY)
            benchmark_return = 0.0
            try:
                from src.data.data_fetcher import DataFetcher
                fetcher = DataFetcher()
                spy_df = fetcher.fetch_historical("SPY", period="3mo")
                if len(spy_df) >= 63:
                    benchmark_return = spy_df["close"].iloc[-1] / spy_df["close"].iloc[-63] - 1
            except Exception:
                pass

            # Review symbols
            review_result = self.symbol_manager.review_symbols(benchmark_return)

            if not review_result.recommendations:
                logger.info("Portfolio review: No action needed")
                return messages

            # Prepare holdings data for LLM
            holdings = []
            for symbol in self.symbol_manager.get_active_symbols():
                entry = self.symbol_manager.get_symbol_entry(symbol)
                if entry:
                    holdings.append({
                        "symbol": symbol,
                        "total_return": entry.total_return,
                        "days_held": entry.days_held,
                        "sector": entry.sector,
                        "entry_price": entry.entry_price,
                        "current_price": entry.current_price,
                    })

            # Format review report
            content = self._format_portfolio_review_report(
                review_result, holdings, benchmark_return
            )

            # Use LLM for enhanced analysis if available
            if self.llm_client and self.llm_client.is_available():
                # Get market outlook
                market_outlook = "Neutral market conditions"
                if self.market_timer:
                    signal = self.market_timer.get_timing_signal()
                    market_outlook = f"Market timing: {signal.signal.value}, conditions: {signal.conditions.overall_condition}"

                llm_analysis = self.llm_client.evaluate_exit_candidates(
                    holdings=holdings,
                    market_outlook=market_outlook,
                    benchmark_return=benchmark_return
                )
                if llm_analysis:
                    content += f"\n\n### AI Analysis\n{llm_analysis}"

            # Extract exit recommendations
            exit_recommendations = [
                r for r in review_result.recommendations
                if r.get("action") in ("remove", "consider_removal")
            ]

            message = self.create_message(
                recipient=AgentRole.DEVELOPER,
                message_type=MessageType.SUGGESTION if exit_recommendations else MessageType.STATUS_UPDATE,
                subject=f"Portfolio Review - {len(exit_recommendations)} Exit Candidates",
                content=content,
                priority=MessagePriority.HIGH if any(r.get("action") == "remove" for r in exit_recommendations) else MessagePriority.NORMAL,
                context={
                    "review_date": datetime.now().isoformat(),
                    "benchmark_return": benchmark_return,
                    "underperformers": review_result.underperformers,
                    "recommendations": review_result.recommendations,
                    "sector_exposure": review_result.sector_exposure,
                },
                requires_response=bool(exit_recommendations),
            )
            messages.append(message)

            logger.info(f"Portfolio review complete: {len(exit_recommendations)} exit candidates")

        except Exception as e:
            logger.error(f"Error during portfolio review: {e}")

        return messages

    def run_market_analysis(self) -> List[AgentMessage]:
        """
        Run market timing analysis.

        Analyzes market conditions and sends alerts if significant changes detected
        or if exposure adjustments are recommended.

        Returns:
            List of messages with market timing analysis
        """
        self._last_market_analysis = datetime.now()
        messages = []

        if not self.market_timer:
            logger.warning("Market timer not available")
            return messages

        try:
            # Get timing signal
            signal = self.market_timer.get_timing_signal()

            # Only send message if signal is not HOLD or confidence is high
            if signal.signal.value == "hold" and signal.confidence < 0.7:
                logger.info("Market analysis: Hold signal with moderate confidence, no alert needed")
                return messages

            # Format market analysis report
            content = self._format_market_analysis_report(signal)

            # Use LLM for enhanced analysis if available
            if self.llm_client and self.llm_client.is_available():
                llm_analysis = self.llm_client.analyze_market_timing(
                    conditions=signal.conditions.to_dict(),
                    portfolio_exposure=signal.recommended_exposure
                )
                if llm_analysis:
                    content += f"\n\n### AI Analysis\n{llm_analysis}"

            # Determine priority
            priority = MessagePriority.NORMAL
            if signal.signal.value == "reduce_exposure" and signal.confidence > 0.7:
                priority = MessagePriority.HIGH
            elif signal.conditions.vix_level > 35:
                priority = MessagePriority.HIGH

            message = self.create_message(
                recipient=AgentRole.DEVELOPER,
                message_type=MessageType.OBSERVATION,
                subject=f"Market Timing: {signal.signal.value.replace('_', ' ').title()}",
                content=content,
                priority=priority,
                context={
                    "signal": signal.signal.value,
                    "confidence": signal.confidence,
                    "recommended_exposure": signal.recommended_exposure,
                    "conditions": signal.conditions.to_dict(),
                    "reasons": signal.reasons,
                },
                requires_response=signal.signal.value != "hold",
            )
            messages.append(message)

            logger.info(f"Market analysis complete: {signal.signal.value} ({signal.confidence:.2f})")

        except Exception as e:
            logger.error(f"Error during market analysis: {e}")

        return messages

    def _format_screening_report(
        self,
        candidates: List,
        current_portfolio: List[str],
        strategy: str
    ) -> str:
        """Format stock screening report."""
        lines = [
            "## Weekly Stock Screening Report",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Strategy:** {strategy.title()}",
            f"**Current Portfolio:** {len(current_portfolio)} symbols",
            "",
            "### Top Candidates",
        ]

        for i, c in enumerate(candidates[:10], 1):
            score_breakdown = c.breakdown.to_dict() if hasattr(c, 'breakdown') else {}
            lines.append(f"\n**{i}. {c.symbol}** - Score: {c.total_score:.1f}")
            lines.append(f"   - Sector: {c.sector}")
            if score_breakdown:
                lines.append(f"   - Breakdown: Fund={score_breakdown.get('fundamental', 0):.0f}, "
                           f"Tech={score_breakdown.get('technical', 0):.0f}, "
                           f"Mom={score_breakdown.get('momentum', 0):.0f}")
            if c.pe_ratio:
                lines.append(f"   - P/E: {c.pe_ratio:.1f}")
            if c.earnings_growth:
                lines.append(f"   - Earnings Growth: {c.earnings_growth:.1%}")

        lines.extend([
            "",
            "### Sector Distribution (Top 10)",
        ])

        # Count sectors
        sector_counts = {}
        for c in candidates[:10]:
            if c.sector:
                sector_counts[c.sector] = sector_counts.get(c.sector, 0) + 1

        for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {sector}: {count}")

        return "\n".join(lines)

    def _format_portfolio_review_report(
        self,
        review_result,
        holdings: List[Dict],
        benchmark_return: float
    ) -> str:
        """Format portfolio review report."""
        lines = [
            "## Portfolio Review Report",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Benchmark (SPY) 3M Return:** {benchmark_return:.1%}",
            "",
            "### Holdings Summary",
            f"- Total Holdings: {len(holdings)}",
            f"- Underperformers: {len(review_result.underperformers)}",
            "",
        ]

        if review_result.recommendations:
            lines.append("### Exit Recommendations")
            for rec in review_result.recommendations:
                symbol = rec.get("symbol", "N/A")
                action = rec.get("action", "unknown")
                reason = rec.get("reason", "")
                ret = rec.get("return", 0)
                days = rec.get("days_held", 0)

                action_emoji = ":x:" if action == "remove" else ":warning:"
                lines.append(f"\n{action_emoji} **{symbol}** - {action.replace('_', ' ').title()}")
                lines.append(f"   - Return: {ret:.1%}")
                lines.append(f"   - Days Held: {days}")
                lines.append(f"   - Reason: {reason}")

        if review_result.sector_exposure:
            lines.extend([
                "",
                "### Sector Exposure",
            ])
            for sector, exposure in sorted(review_result.sector_exposure.items(), key=lambda x: -x[1]):
                lines.append(f"- {sector}: {exposure:.0%}")

        return "\n".join(lines)

    def _format_market_analysis_report(self, signal) -> str:
        """Format market timing analysis report."""
        conditions = signal.conditions

        lines = [
            "## Market Timing Analysis",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"### Signal: **{signal.signal.value.replace('_', ' ').upper()}**",
            f"- Confidence: {signal.confidence:.0%}",
            f"- Recommended Exposure: {signal.recommended_exposure:.0%}",
            "",
            "### Market Conditions",
            f"- Overall: {conditions.overall_condition.title()}",
            f"- VIX Level: {conditions.vix_level:.1f} (percentile: {conditions.vix_percentile:.0f}%)",
            f"- VIX Trend: {conditions.vix_trend}",
            "",
            "### SPY Technical",
            f"- Price: ${conditions.spy_price:.2f}",
            f"- vs 20 MA: {conditions.spy_vs_sma20:+.1f}%",
            f"- vs 50 MA: {conditions.spy_vs_sma50:+.1f}%",
            f"- vs 200 MA: {conditions.spy_vs_sma200:+.1f}%",
            f"- RSI: {conditions.spy_rsi:.0f}",
            "",
            "### Reasons",
        ]

        for reason in signal.reasons:
            lines.append(f"- {reason}")

        return "\n".join(lines)

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming messages from other agents.

        Args:
            message: Incoming message

        Returns:
            Optional response message
        """
        logger.info(f"Processing message: {message.subject}")

        # Handle queries from developer
        if message.message_type == MessageType.QUERY:
            return self._handle_query(message)

        # Acknowledge actions taken
        if message.message_type == MessageType.ACTION:
            return self._acknowledge_action(message)

        # Handle status updates
        if message.message_type == MessageType.STATUS_UPDATE:
            logger.info(f"Received status update: {message.subject}")
            return None

        return None

    def _handle_query(self, message: AgentMessage) -> AgentMessage:
        """Handle a query from the developer agent."""
        query_type = message.context.get("query_type", "general")

        if query_type == "current_metrics":
            metrics = self._gather_performance_metrics()
            content = self._format_metrics_response(metrics)
        elif query_type == "model_status":
            if self.degradation_monitor:
                reports = self.degradation_monitor.check_all_models()
                content = self._format_model_status(reports)
            else:
                content = "Degradation monitor not available"
        else:
            content = f"Query type '{query_type}' not recognized"

        return self.create_message(
            recipient=AgentRole.DEVELOPER,
            message_type=MessageType.RESPONSE,
            subject=f"Re: {message.subject}",
            content=content,
            priority=MessagePriority.NORMAL,
            parent_message_id=message.id,
        )

    def _acknowledge_action(self, message: AgentMessage) -> AgentMessage:
        """Acknowledge an action taken by the developer."""
        action = message.context.get("action", "unknown")

        return self.create_message(
            recipient=AgentRole.DEVELOPER,
            message_type=MessageType.ACKNOWLEDGMENT,
            subject=f"Acknowledged: {action}",
            content=f"Received confirmation of action: {action}. Will continue monitoring.",
            priority=MessagePriority.LOW,
            parent_message_id=message.id,
        )

    def _gather_performance_metrics(self) -> Dict[str, Any]:
        """Gather current performance metrics."""
        if not self.data_aggregator:
            return {}

        try:
            # Get recent data (last 30 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            analytics = self.data_aggregator.prepare_analytics_data(
                start_date=start_date,
                end_date=end_date
            )

            returns = analytics.get("returns", pd.Series())
            trades = analytics.get("trades", pd.DataFrame())
            portfolio_metrics = analytics.get("portfolio_metrics", {})

            # Calculate metrics
            metrics = {}

            # Sharpe ratio (annualized)
            if len(returns) > 0 and returns.std() != 0:
                metrics["sharpe_ratio"] = (returns.mean() / returns.std()) * np.sqrt(252)
            else:
                metrics["sharpe_ratio"] = 0.0

            # Max drawdown
            if len(returns) > 0:
                cumulative = (1 + returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                metrics["max_drawdown"] = abs(drawdown.min())
            else:
                metrics["max_drawdown"] = 0.0

            # Win rate
            if not trades.empty and "pnl" in trades.columns:
                winning_trades = trades[trades["pnl"] > 0]
                metrics["win_rate"] = len(winning_trades) / len(trades) if len(trades) > 0 else 0.0
            else:
                # Fallback to portfolio metrics
                metrics["win_rate"] = portfolio_metrics.get("win_rate", 0.0)

            # Total return
            if len(returns) > 0:
                metrics["total_return"] = (1 + returns).prod() - 1
            else:
                metrics["total_return"] = 0.0

            # Daily P&L
            if len(returns) > 0:
                metrics["daily_return"] = returns.iloc[-1] if len(returns) > 0 else 0.0
            else:
                metrics["daily_return"] = 0.0

            # Portfolio value
            metrics["portfolio_value"] = portfolio_metrics.get("portfolio_value", 0.0)

            # Trading days
            metrics["trading_days"] = len(returns)

            return metrics

        except Exception as e:
            logger.error(f"Error gathering metrics: {e}")
            return {}

    def _detect_issues(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect issues in the metrics."""
        issues = []

        # Check Sharpe ratio
        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe < self.sharpe_warning:
            issues.append({
                "type": "low_sharpe",
                "metric": "sharpe_ratio",
                "value": sharpe,
                "threshold": self.sharpe_warning,
                "severity": "warning",
            })

        # Check drawdown
        drawdown = metrics.get("max_drawdown", 0)
        if drawdown > self.drawdown_warning:
            issues.append({
                "type": "high_drawdown",
                "metric": "max_drawdown",
                "value": drawdown,
                "threshold": self.drawdown_warning,
                "severity": "high" if drawdown > self.drawdown_warning * 1.5 else "warning",
            })

        # Check win rate
        win_rate = metrics.get("win_rate", 0)
        if win_rate < self.win_rate_warning:
            issues.append({
                "type": "low_win_rate",
                "metric": "win_rate",
                "value": win_rate,
                "threshold": self.win_rate_warning,
                "severity": "warning",
            })

        return issues

    def _determine_priority(self, issues: List[Dict[str, Any]]) -> MessagePriority:
        """Determine message priority based on issues."""
        severities = [i.get("severity", "warning") for i in issues]

        if "urgent" in severities:
            return MessagePriority.URGENT
        elif "high" in severities:
            return MessagePriority.HIGH
        elif severities:
            return MessagePriority.NORMAL
        else:
            return MessagePriority.LOW

    def _format_health_check_report(
        self,
        metrics: Dict[str, Any],
        issues: List[Dict[str, Any]]
    ) -> str:
        """Format health check report."""
        lines = [
            "## Performance Analysis Report",
            f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "### Current Metrics:",
            f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"- Max Drawdown: {metrics.get('max_drawdown', 0):.1%}",
            f"- Win Rate: {metrics.get('win_rate', 0):.1%}",
            f"- Total Return (30d): {metrics.get('total_return', 0):.2%}",
            "",
            "### Issues Detected:",
        ]

        for issue in issues:
            severity_emoji = ":warning:" if issue["severity"] == "warning" else ":rotating_light:"
            lines.append(
                f"- {severity_emoji} **{issue['type'].replace('_', ' ').title()}**: "
                f"{issue['value']:.2f} (threshold: {issue['threshold']:.2f})"
            )

        lines.extend([
            "",
            "### Suggested Actions:",
        ])

        # Add suggestions based on issues
        for issue in issues:
            if issue["type"] == "low_sharpe":
                lines.append("1. Consider retraining models with recent volatile market data")
            elif issue["type"] == "high_drawdown":
                lines.append("2. Review risk parameters and consider reducing position sizes")
            elif issue["type"] == "low_win_rate":
                lines.append("3. Increase confidence threshold to reduce noise trades")

        return "\n".join(lines)

    def _format_degradation_report(self, reports: List) -> str:
        """Format degradation report."""
        lines = [
            "## Model Degradation Report",
            f"**Check Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        for report in reports:
            lines.extend([
                f"### {report.model_type.upper()}",
                f"- **Status:** Degraded",
                f"- **Recommendation:** {report.recommendation}",
                "",
                "**Current Metrics:**",
            ])

            for key, value in report.metrics.items():
                lines.append(f"- {key}: {value:.4f}")

            lines.extend([
                "",
                "**Degradation Reasons:**",
            ])

            for reason in report.degradation_reasons:
                lines.append(f"- {reason}")

            lines.append("")

        return "\n".join(lines)

    def _format_daily_report(
        self,
        metrics: Dict[str, Any],
        trades: pd.DataFrame,
        positions: Dict[str, Any],
        returns: pd.Series
    ) -> str:
        """Format daily portfolio report."""
        lines = [
            "## Daily Portfolio Review",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "### Portfolio Summary",
            f"- Portfolio Value: ${metrics.get('portfolio_value', 0):,.2f}",
            f"- Daily P&L: {returns.iloc[-1]:.2%}" if len(returns) > 0 else "- Daily P&L: N/A",
            "",
            "### Trading Activity",
            f"- Trades Today: {len(trades)}",
        ]

        if not trades.empty:
            buys = trades[trades.get("side", "").str.upper() == "BUY"] if "side" in trades.columns else pd.DataFrame()
            sells = trades[trades.get("side", "").str.upper() == "SELL"] if "side" in trades.columns else pd.DataFrame()
            lines.extend([
                f"- Buy Orders: {len(buys)}",
                f"- Sell Orders: {len(sells)}",
            ])

        lines.extend([
            "",
            f"### Open Positions: {len(positions)}",
        ])

        for symbol, pos in list(positions.items())[:5]:  # Top 5
            pnl = pos.get("unrealized_pnl", 0)
            pnl_str = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
            lines.append(f"- {symbol}: {pnl_str}")

        if len(positions) > 5:
            lines.append(f"- ... and {len(positions) - 5} more")

        return "\n".join(lines)

    def _format_metrics_response(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for response message."""
        lines = ["## Current Performance Metrics", ""]
        for key, value in metrics.items():
            if isinstance(value, float):
                if "ratio" in key or "return" in key:
                    lines.append(f"- {key}: {value:.4f}")
                elif "rate" in key or "drawdown" in key:
                    lines.append(f"- {key}: {value:.1%}")
                else:
                    lines.append(f"- {key}: {value:,.2f}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _format_model_status(self, reports: List) -> str:
        """Format model status response."""
        lines = ["## Model Health Status", ""]
        for report in reports:
            status = ":x: Degraded" if report.is_degraded else ":white_check_mark: Healthy"
            lines.append(f"### {report.model_type.upper()}: {status}")
            if report.is_degraded:
                lines.append(f"- Recommendation: {report.recommendation}")
                for reason in report.degradation_reasons:
                    lines.append(f"- {reason}")
            lines.append("")
        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        """Get agent status including last check times."""
        status = super().get_status()
        status.update({
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "last_degradation_check": self._last_degradation_check.isoformat() if self._last_degradation_check else None,
            "last_daily_review": self._last_daily_review.isoformat() if self._last_daily_review else None,
            "last_stock_screening": self._last_stock_screening.isoformat() if self._last_stock_screening else None,
            "last_portfolio_review": self._last_portfolio_review.isoformat() if self._last_portfolio_review else None,
            "last_market_analysis": self._last_market_analysis.isoformat() if self._last_market_analysis else None,
            "thresholds": {
                "sharpe_warning": self.sharpe_warning,
                "drawdown_warning": self.drawdown_warning,
                "win_rate_warning": self.win_rate_warning,
            },
        })
        return status
