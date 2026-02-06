"""
Portfolio Strategist Agent

Portfolio management agent that handles stock selection, allocation,
rebalancing recommendations, and performance attribution.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .base_agent import (
    AgentMessage,
    AgentRole,
    BaseAgent,
    MessagePriority,
    MessageType,
)

logger = logging.getLogger(__name__)


class PortfolioStrategistAgent(BaseAgent):
    """
    Portfolio Strategist Agent - manages portfolio composition and strategy.

    Responsibilities:
    - Stock screening and selection
    - Portfolio composition optimization
    - Target weight calculation
    - Rebalancing recommendations
    - Performance attribution
    - Underperformer identification

    Schedule:
    - Performance review: Every 4 hours
    - Rebalancing check: Daily (10 AM)
    - Stock screening: Weekly (Sunday 6 PM)
    - Full portfolio review: Weekly (Monday 9 AM)

    Outputs To:
    - Operations: Add/remove symbol requests, allocation changes
    - Risk Guardian: Proposed changes for risk assessment
    - Market Intelligence: Request research on candidates
    """

    def __init__(
        self,
        config: Dict[str, Any],
        message_queue,
        notifier=None,
        llm_client=None,
    ):
        """
        Initialize Portfolio Strategist agent.

        Args:
            config: Agent configuration
            message_queue: Shared message queue
            notifier: Optional Discord notifier
            llm_client: Optional LLM client for intelligent analysis
        """
        super().__init__(
            role=AgentRole.PORTFOLIO_STRATEGIST,
            config=config,
            message_queue=message_queue,
            notifier=notifier,
            llm_client=llm_client,
        )

        # Load configuration
        ps_config = config.get("portfolio_strategist", {})
        self.performance_review_hours = ps_config.get("performance_review_hours", 4)
        self.rebalancing_check_time = ps_config.get("rebalancing_check_time", "10:00")
        self.stock_screening_day = ps_config.get("stock_screening_day", "sunday")
        self.stock_screening_time = ps_config.get("stock_screening_time", "18:00")
        self.portfolio_review_day = ps_config.get("portfolio_review_day", "monday")
        self.portfolio_review_time = ps_config.get("portfolio_review_time", "09:00")
        self.min_score_to_add = ps_config.get("min_score_to_add", 70)
        self.underperformance_threshold = ps_config.get("underperformance_threshold", -0.10)

        # Track last analysis times
        self._last_performance_review: Optional[datetime] = None
        self._last_rebalancing_check: Optional[datetime] = None
        self._last_stock_screening: Optional[datetime] = None
        self._last_portfolio_review: Optional[datetime] = None

        # Risk constraints (updated by Risk Guardian)
        self._risk_constraints: Dict[str, Any] = {
            "max_new_position_size": 0.10,
            "allow_new_trades": True,
            "reduce_exposure": False,
        }

        # Lazy-loaded components
        self._stock_screener = None
        self._portfolio_optimizer = None
        self._correlation_analyzer = None
        self._symbol_manager = None
        self._data_aggregator = None

        logger.info(
            f"Portfolio Strategist agent initialized: "
            f"performance_review={self.performance_review_hours}h, "
            f"min_score={self.min_score_to_add}"
        )

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
    def portfolio_optimizer(self):
        """Lazy load portfolio optimizer."""
        if self._portfolio_optimizer is None:
            try:
                from src.portfolio.optimizer import get_portfolio_optimizer
                from config.settings import Settings
                config = Settings.load_trading_config()
                self._portfolio_optimizer = get_portfolio_optimizer(config)
            except ImportError as e:
                logger.debug(f"PortfolioOptimizer not available: {e}")
        return self._portfolio_optimizer

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

    def analyze(self) -> List[AgentMessage]:
        """
        Perform analysis and generate observations/suggestions.

        Returns:
            List of messages to send to other agents
        """
        return []

    def run_performance_review(self) -> List[AgentMessage]:
        """
        Review portfolio performance and identify issues.

        Returns:
            List of messages with performance analysis
        """
        self._last_performance_review = datetime.now()
        messages = []

        try:
            # Gather performance data
            performance_data = self._gather_performance_data()

            if not performance_data:
                logger.info("No performance data available")
                return messages

            # Analyze performance attribution
            attribution = self._calculate_attribution(performance_data)

            # Identify underperformers
            underperformers = self._identify_underperformers(performance_data)

            # Format report
            content = self._format_performance_report(performance_data, attribution, underperformers)

            # Use LLM for enhanced analysis
            if self.llm_client and self.llm_client.is_available():
                llm_analysis = self.llm_client.analyze_underperformers(
                    performance_data.get("holdings", []),
                    performance_data.get("benchmark_return", 0)
                )
                if llm_analysis:
                    content += f"\n\n### AI Analysis\n{llm_analysis}"

            # Determine if action needed
            if underperformers:
                # Send to Operations for potential action
                messages.append(self.create_message(
                    recipient=AgentRole.OPERATIONS,
                    message_type=MessageType.SUGGESTION,
                    subject=f"Performance Review: {len(underperformers)} underperformers",
                    content=content,
                    priority=MessagePriority.NORMAL,
                    context={
                        "underperformers": underperformers,
                        "performance_data": performance_data,
                        "recommendations": self._generate_exit_recommendations(underperformers),
                    },
                    requires_response=True,
                ))

                # Also notify Risk Guardian about performance issues
                messages.append(self.create_message(
                    recipient=AgentRole.RISK_GUARDIAN,
                    message_type=MessageType.OBSERVATION,
                    subject=f"Performance Alert: {len(underperformers)} positions underperforming",
                    content=self._format_underperformer_alert(underperformers),
                    priority=MessagePriority.NORMAL,
                    context={"underperformers": underperformers},
                ))
            else:
                # Just log as status update
                messages.append(self.create_message(
                    recipient=AgentRole.OPERATIONS,
                    message_type=MessageType.STATUS_UPDATE,
                    subject="Performance Review: All positions healthy",
                    content=content,
                    priority=MessagePriority.LOW,
                    context={"performance_data": performance_data},
                ))

            logger.info(f"Performance review complete: {len(underperformers)} underperformers")

        except Exception as e:
            logger.error(f"Error during performance review: {e}")

        return messages

    def run_rebalancing_check(self) -> List[AgentMessage]:
        """
        Check if portfolio needs rebalancing.

        Returns:
            List of messages with rebalancing recommendations
        """
        self._last_rebalancing_check = datetime.now()
        messages = []

        try:
            # Get current weights
            current_weights = self._get_current_weights()

            if not current_weights:
                logger.info("No positions to check for rebalancing")
                return messages

            # Get target weights
            target_weights = self._get_target_weights()

            if not target_weights:
                logger.info("No target weights defined")
                return messages

            # Calculate drift
            drift = self._calculate_drift(current_weights, target_weights)

            # Check if rebalancing needed
            rebalancing_needed, trades = self._check_rebalancing_threshold(
                current_weights, target_weights, drift
            )

            if rebalancing_needed:
                content = self._format_rebalancing_report(
                    current_weights, target_weights, drift, trades
                )

                # Use LLM for analysis
                if self.llm_client and self.llm_client.is_available():
                    llm_analysis = self.llm_client.recommend_rebalancing(
                        current_weights, target_weights, {"estimated_cost": sum(t.get("value", 0) * 0.001 for t in trades)}
                    )
                    if llm_analysis:
                        content += f"\n\n### AI Analysis\n{llm_analysis}"

                # Check with Risk Guardian first
                messages.append(self.create_message(
                    recipient=AgentRole.RISK_GUARDIAN,
                    message_type=MessageType.QUERY,
                    subject="Rebalancing Risk Check",
                    content=f"Proposing {len(trades)} trades for rebalancing. Please assess risk.",
                    priority=MessagePriority.NORMAL,
                    context={
                        "query_type": "risk_assessment",
                        "proposed_trades": trades,
                    },
                    requires_response=True,
                ))

                # Send rebalancing recommendation to Operations
                messages.append(self.create_message(
                    recipient=AgentRole.OPERATIONS,
                    message_type=MessageType.SUGGESTION,
                    subject=f"Rebalancing Required: {len(trades)} trades",
                    content=content,
                    priority=MessagePriority.NORMAL,
                    context={
                        "trades": trades,
                        "current_weights": current_weights,
                        "target_weights": target_weights,
                        "drift": drift,
                    },
                    requires_response=True,
                ))

            logger.info(f"Rebalancing check complete: {'needed' if rebalancing_needed else 'not needed'}")

        except Exception as e:
            logger.error(f"Error during rebalancing check: {e}")

        return messages

    def run_stock_screening(self) -> List[AgentMessage]:
        """
        Run weekly stock screening to find new candidates.

        Returns:
            List of messages with screening recommendations
        """
        self._last_stock_screening = datetime.now()
        messages = []

        if not self.stock_screener:
            logger.warning("Stock screener not available")
            return messages

        try:
            # Check risk constraints
            if not self._risk_constraints.get("allow_new_trades", True):
                logger.info("New trades not allowed due to risk constraints")
                return messages

            # Get current portfolio
            current_portfolio = []
            if self.symbol_manager:
                current_portfolio = self.symbol_manager.get_active_symbols()

            # Get stock universe
            try:
                from src.data.stock_universe import get_stock_universe
                universe = get_stock_universe()
                all_symbols = universe.get_universe()
            except Exception as e:
                logger.error(f"Failed to get universe: {e}")
                return messages

            # Get screening strategy
            from config.settings import Settings
            config = Settings.load_trading_config()
            strategy = config.get("dynamic_symbols", {}).get("screening", {}).get("strategy", "balanced")

            # Score all stocks
            scores = self.stock_screener.score_stocks(all_symbols)

            # Filter out current portfolio
            portfolio_set = set(current_portfolio)
            candidates = [s for s in scores if s.symbol not in portfolio_set]

            # Filter by minimum score
            qualified_candidates = [c for c in candidates if c.total_score >= self.min_score_to_add][:20]

            # Get market regime
            market_regime = "neutral"
            try:
                from src.risk.regime_detector import get_regime_detector
                detector = get_regime_detector()
                regime = detector.detect_regime()
                market_regime = regime.value if hasattr(regime, 'value') else str(regime)
            except Exception:
                pass

            # Format report
            content = self._format_screening_report(
                qualified_candidates, current_portfolio, strategy, market_regime
            )

            # Use LLM for enhanced analysis
            if self.llm_client and self.llm_client.is_available():
                llm_analysis = self.llm_client.evaluate_candidates(
                    [c.to_dict() for c in qualified_candidates[:10]],
                    current_portfolio,
                    self._risk_constraints
                )
                if llm_analysis:
                    content += f"\n\n### AI Analysis\n{llm_analysis}"

            # Create recommendations
            recommendations = [
                {
                    "symbol": c.symbol,
                    "score": c.total_score,
                    "sector": c.sector,
                    "action": "add_symbol",
                    "reason": f"Score {c.total_score:.1f}, {c.sector}",
                }
                for c in qualified_candidates[:5]
            ]

            # Request Market Intelligence to research top candidates
            if qualified_candidates:
                messages.append(self.create_message(
                    recipient=AgentRole.MARKET_INTELLIGENCE,
                    message_type=MessageType.QUERY,
                    subject=f"Research Request: {len(qualified_candidates[:5])} candidates",
                    content=f"Please provide news/sentiment analysis for:\n"
                            f"{', '.join(c.symbol for c in qualified_candidates[:5])}",
                    priority=MessagePriority.NORMAL,
                    context={
                        "query_type": "candidate_research",
                        "symbols": [c.symbol for c in qualified_candidates[:5]],
                    },
                ))

            # Send recommendations to Operations
            messages.append(self.create_message(
                recipient=AgentRole.OPERATIONS,
                message_type=MessageType.SUGGESTION,
                subject=f"Weekly Screening: {len(recommendations)} candidates",
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
            ))

            logger.info(f"Stock screening complete: {len(recommendations)} recommendations")

        except Exception as e:
            logger.error(f"Error during stock screening: {e}")

        return messages

    def run_portfolio_review(self) -> List[AgentMessage]:
        """
        Run comprehensive weekly portfolio review.

        Returns:
            List of messages with portfolio analysis
        """
        self._last_portfolio_review = datetime.now()
        messages = []

        try:
            # Get benchmark return (SPY 3-month)
            benchmark_return = self._get_benchmark_return()

            # Get portfolio holdings with performance
            holdings = self._get_holdings_with_performance()

            if not holdings:
                logger.info("No holdings for portfolio review")
                return messages

            # Analyze sector exposure
            sector_exposure = self._calculate_sector_exposure(holdings)

            # Identify candidates for exit
            exit_candidates = self._identify_exit_candidates(holdings, benchmark_return)

            # Calculate portfolio statistics
            portfolio_stats = self._calculate_portfolio_stats(holdings)

            # Format report
            content = self._format_portfolio_review_report(
                holdings, sector_exposure, exit_candidates, portfolio_stats, benchmark_return
            )

            # Use LLM for analysis
            if self.llm_client and self.llm_client.is_available():
                llm_analysis = self.llm_client.evaluate_exit_candidates(
                    holdings,
                    f"Benchmark (SPY) 3M return: {benchmark_return:.1%}",
                    benchmark_return
                )
                if llm_analysis:
                    content += f"\n\n### AI Analysis\n{llm_analysis}"

            # Generate recommendations
            recommendations = self._generate_exit_recommendations(exit_candidates)

            # Send to Operations
            priority = MessagePriority.HIGH if exit_candidates else MessagePriority.NORMAL
            messages.append(self.create_message(
                recipient=AgentRole.OPERATIONS,
                message_type=MessageType.SUGGESTION if exit_candidates else MessageType.STATUS_UPDATE,
                subject=f"Weekly Portfolio Review: {len(exit_candidates)} exit candidates",
                content=content,
                priority=priority,
                context={
                    "review_date": datetime.now().isoformat(),
                    "benchmark_return": benchmark_return,
                    "holdings_count": len(holdings),
                    "exit_candidates": exit_candidates,
                    "recommendations": recommendations,
                    "sector_exposure": sector_exposure,
                },
                requires_response=bool(exit_candidates),
            ))

            # Notify Risk Guardian about sector concentration
            max_sector_exposure = max(sector_exposure.values()) if sector_exposure else 0
            if max_sector_exposure > 0.25:
                messages.append(self.create_message(
                    recipient=AgentRole.RISK_GUARDIAN,
                    message_type=MessageType.OBSERVATION,
                    subject=f"Sector Concentration Alert: {max_sector_exposure:.0%}",
                    content=f"Sector concentration exceeds 25%:\n\n"
                            + "\n".join(f"- {s}: {e:.0%}" for s, e in sorted(sector_exposure.items(), key=lambda x: -x[1])),
                    priority=MessagePriority.NORMAL,
                    context={"sector_exposure": sector_exposure},
                ))

            logger.info(f"Portfolio review complete: {len(exit_candidates)} exit candidates")

        except Exception as e:
            logger.error(f"Error during portfolio review: {e}")

        return messages

    def _gather_performance_data(self) -> Dict[str, Any]:
        """Gather portfolio performance data."""
        data = {}

        try:
            if self.data_aggregator:
                analytics = self.data_aggregator.prepare_analytics_data(
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )

                data["returns"] = analytics.get("returns", pd.Series())
                data["trades"] = analytics.get("trades", pd.DataFrame())
                data["portfolio_metrics"] = analytics.get("portfolio_metrics", {})
                data["positions"] = analytics.get("positions", {})

            # Get holdings with returns
            data["holdings"] = self._get_holdings_with_performance()

            # Get benchmark return
            data["benchmark_return"] = self._get_benchmark_return()

        except Exception as e:
            logger.error(f"Error gathering performance data: {e}")

        return data

    def _calculate_attribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance attribution."""
        attribution = {
            "by_symbol": {},
            "by_sector": {},
            "total_return": 0,
        }

        holdings = data.get("holdings", [])
        if not holdings:
            return attribution

        total_value = sum(h.get("market_value", 0) for h in holdings)

        for holding in holdings:
            symbol = holding.get("symbol", "")
            sector = holding.get("sector", "Unknown")
            weight = holding.get("market_value", 0) / total_value if total_value > 0 else 0
            ret = holding.get("total_return", 0)
            contribution = weight * ret

            attribution["by_symbol"][symbol] = {
                "weight": weight,
                "return": ret,
                "contribution": contribution,
            }

            if sector not in attribution["by_sector"]:
                attribution["by_sector"][sector] = {"weight": 0, "contribution": 0}

            attribution["by_sector"][sector]["weight"] += weight
            attribution["by_sector"][sector]["contribution"] += contribution

        attribution["total_return"] = sum(a["contribution"] for a in attribution["by_symbol"].values())

        return attribution

    def _identify_underperformers(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify underperforming holdings."""
        underperformers = []
        benchmark_return = data.get("benchmark_return", 0)

        for holding in data.get("holdings", []):
            ret = holding.get("total_return", 0)

            # Check against threshold
            if ret < self.underperformance_threshold:
                underperformers.append({
                    "symbol": holding.get("symbol"),
                    "return": ret,
                    "days_held": holding.get("days_held", 0),
                    "sector": holding.get("sector"),
                    "reason": "loss_threshold",
                })
            # Check against benchmark
            elif ret < benchmark_return - 0.05:  # 5% underperformance vs benchmark
                underperformers.append({
                    "symbol": holding.get("symbol"),
                    "return": ret,
                    "days_held": holding.get("days_held", 0),
                    "sector": holding.get("sector"),
                    "reason": "benchmark_underperformance",
                    "vs_benchmark": ret - benchmark_return,
                })

        return underperformers

    def _get_holdings_with_performance(self) -> List[Dict[str, Any]]:
        """Get holdings with performance metrics."""
        holdings = []

        try:
            if self.symbol_manager:
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
                            "market_value": getattr(entry, 'market_value', 0),
                        })
        except Exception as e:
            logger.debug(f"Error getting holdings: {e}")

        return holdings

    def _get_benchmark_return(self, period_days: int = 63) -> float:
        """Get benchmark return (SPY)."""
        try:
            import yfinance as yf
            spy = yf.Ticker("SPY")
            hist = spy.history(period=f"{period_days + 10}d")

            if len(hist) >= period_days:
                return hist["Close"].iloc[-1] / hist["Close"].iloc[-period_days] - 1
        except Exception as e:
            logger.debug(f"Error getting benchmark return: {e}")

        return 0.0

    def _get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        weights = {}

        try:
            holdings = self._get_holdings_with_performance()
            total_value = sum(h.get("market_value", 0) for h in holdings)

            if total_value > 0:
                for h in holdings:
                    weights[h["symbol"]] = h.get("market_value", 0) / total_value
        except Exception as e:
            logger.debug(f"Error getting current weights: {e}")

        return weights

    def _get_target_weights(self) -> Dict[str, float]:
        """Get target portfolio weights."""
        try:
            if self.portfolio_optimizer:
                return self.portfolio_optimizer.get_target_weights()
        except Exception as e:
            logger.debug(f"Error getting target weights: {e}")

        # Fallback: equal weight
        current = self._get_current_weights()
        if current:
            equal_weight = 1.0 / len(current)
            return {symbol: equal_weight for symbol in current}

        return {}

    def _calculate_drift(
        self,
        current: Dict[str, float],
        target: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate weight drift from target."""
        drift = {}

        all_symbols = set(current.keys()) | set(target.keys())
        for symbol in all_symbols:
            curr_weight = current.get(symbol, 0)
            tgt_weight = target.get(symbol, 0)
            drift[symbol] = curr_weight - tgt_weight

        return drift

    def _check_rebalancing_threshold(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        drift: Dict[str, float]
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check if rebalancing is needed and generate trades."""
        # Get threshold from config
        from config.settings import Settings
        config = Settings.load_trading_config()
        threshold = config.get("portfolio_optimization", {}).get("rebalancing", {}).get("drift_threshold", 0.08)

        # Check if any symbol exceeds threshold
        needs_rebalancing = any(abs(d) > threshold for d in drift.values())

        trades = []
        if needs_rebalancing:
            holdings = self._get_holdings_with_performance()
            portfolio_value = sum(h.get("market_value", 0) for h in holdings) or 100000

            for symbol, d in drift.items():
                if abs(d) > 0.01:  # Minimum 1% to trade
                    trade_value = abs(d) * portfolio_value
                    trades.append({
                        "symbol": symbol,
                        "action": "sell" if d > 0 else "buy",
                        "weight_change": -d,
                        "value": trade_value,
                    })

        return needs_rebalancing, trades

    def _calculate_sector_exposure(self, holdings: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate sector exposure."""
        exposure = {}
        total_value = sum(h.get("market_value", 0) for h in holdings)

        if total_value > 0:
            for h in holdings:
                sector = h.get("sector", "Unknown")
                weight = h.get("market_value", 0) / total_value
                exposure[sector] = exposure.get(sector, 0) + weight

        return exposure

    def _calculate_portfolio_stats(self, holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate portfolio statistics."""
        stats = {
            "total_positions": len(holdings),
            "avg_days_held": 0,
            "avg_return": 0,
            "positive_positions": 0,
            "negative_positions": 0,
        }

        if not holdings:
            return stats

        days_held = [h.get("days_held", 0) for h in holdings]
        returns = [h.get("total_return", 0) for h in holdings]

        stats["avg_days_held"] = sum(days_held) / len(days_held)
        stats["avg_return"] = sum(returns) / len(returns)
        stats["positive_positions"] = sum(1 for r in returns if r > 0)
        stats["negative_positions"] = sum(1 for r in returns if r < 0)

        return stats

    def _identify_exit_candidates(
        self,
        holdings: List[Dict[str, Any]],
        benchmark_return: float
    ) -> List[Dict[str, Any]]:
        """Identify candidates for exit."""
        candidates = []

        # Get exit criteria from config
        from config.settings import Settings
        config = Settings.load_trading_config()
        exit_config = config.get("dynamic_symbols", {}).get("exit", {})

        loss_threshold = exit_config.get("loss_threshold", -0.15)
        underperf_threshold = exit_config.get("underperformance_threshold", -0.10)
        max_holding_days = exit_config.get("max_holding_days", 90)

        for h in holdings:
            ret = h.get("total_return", 0)
            days = h.get("days_held", 0)
            symbol = h.get("symbol", "")

            # Check loss threshold
            if ret < loss_threshold:
                candidates.append({
                    "symbol": symbol,
                    "return": ret,
                    "days_held": days,
                    "action": "remove",
                    "reason": f"Loss exceeds {loss_threshold:.0%} threshold",
                })
            # Check underperformance vs benchmark
            elif ret < underperf_threshold and ret < benchmark_return - 0.05:
                candidates.append({
                    "symbol": symbol,
                    "return": ret,
                    "days_held": days,
                    "action": "consider_removal",
                    "reason": f"Underperforming benchmark by {(ret - benchmark_return):.1%}",
                })
            # Check holding period
            elif days > max_holding_days and ret < 0:
                candidates.append({
                    "symbol": symbol,
                    "return": ret,
                    "days_held": days,
                    "action": "consider_removal",
                    "reason": f"Held {days} days with negative return",
                })

        return candidates

    def _generate_exit_recommendations(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate exit recommendations."""
        return [
            {
                "symbol": c["symbol"],
                "action": c["action"],
                "reason": c["reason"],
                "return": c.get("return", 0),
                "days_held": c.get("days_held", 0),
            }
            for c in candidates
        ]

    def _format_performance_report(
        self,
        data: Dict[str, Any],
        attribution: Dict[str, Any],
        underperformers: List[Dict[str, Any]]
    ) -> str:
        """Format performance report."""
        lines = [
            "## Performance Review",
            f"**Review Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Benchmark (SPY) 3M Return:** {data.get('benchmark_return', 0):.1%}",
            "",
            "### Portfolio Summary",
            f"- Total Return (30d): {attribution.get('total_return', 0):.2%}",
            f"- Holdings: {len(data.get('holdings', []))}",
            "",
        ]

        # Top contributors
        if attribution.get("by_symbol"):
            sorted_contrib = sorted(
                attribution["by_symbol"].items(),
                key=lambda x: x[1]["contribution"],
                reverse=True
            )

            lines.append("### Top Contributors")
            for symbol, attr in sorted_contrib[:3]:
                lines.append(f"- **{symbol}**: {attr['contribution']:+.2%} (return: {attr['return']:+.1%})")

            lines.extend(["", "### Bottom Contributors"])
            for symbol, attr in sorted_contrib[-3:]:
                lines.append(f"- **{symbol}**: {attr['contribution']:+.2%} (return: {attr['return']:+.1%})")

        # Underperformers
        if underperformers:
            lines.extend(["", "### Underperformers"])
            for up in underperformers:
                lines.append(f"- **{up['symbol']}**: {up['return']:+.1%} ({up['days_held']} days) - {up['reason']}")

        return "\n".join(lines)

    def _format_underperformer_alert(self, underperformers: List[Dict[str, Any]]) -> str:
        """Format underperformer alert."""
        lines = [
            "## Underperformer Alert",
            f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        for up in underperformers:
            lines.append(f"- **{up['symbol']}**: {up['return']:+.1%} ({up.get('reason', 'underperforming')})")

        return "\n".join(lines)

    def _format_rebalancing_report(
        self,
        current: Dict[str, float],
        target: Dict[str, float],
        drift: Dict[str, float],
        trades: List[Dict[str, Any]]
    ) -> str:
        """Format rebalancing report."""
        lines = [
            "## Rebalancing Analysis",
            f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "### Weight Drift Summary",
        ]

        # Sort by absolute drift
        sorted_drift = sorted(drift.items(), key=lambda x: abs(x[1]), reverse=True)
        for symbol, d in sorted_drift[:10]:
            curr = current.get(symbol, 0)
            tgt = target.get(symbol, 0)
            lines.append(f"- {symbol}: {curr:.1%} -> {tgt:.1%} (drift: {d:+.1%})")

        if trades:
            lines.extend(["", "### Proposed Trades"])
            for trade in trades:
                lines.append(f"- **{trade['action'].upper()}** {trade['symbol']}: {trade['weight_change']:+.1%} (${trade['value']:,.0f})")

        return "\n".join(lines)

    def _format_screening_report(
        self,
        candidates: List,
        current_portfolio: List[str],
        strategy: str,
        market_regime: str
    ) -> str:
        """Format stock screening report."""
        lines = [
            "## Weekly Stock Screening Report",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Strategy:** {strategy.title()}",
            f"**Market Regime:** {market_regime.title()}",
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

        # Sector distribution
        sector_counts = {}
        for c in candidates[:10]:
            if c.sector:
                sector_counts[c.sector] = sector_counts.get(c.sector, 0) + 1

        if sector_counts:
            lines.extend(["", "### Sector Distribution (Top 10)"])
            for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
                lines.append(f"- {sector}: {count}")

        return "\n".join(lines)

    def _format_portfolio_review_report(
        self,
        holdings: List[Dict[str, Any]],
        sector_exposure: Dict[str, float],
        exit_candidates: List[Dict[str, Any]],
        stats: Dict[str, Any],
        benchmark_return: float
    ) -> str:
        """Format portfolio review report."""
        lines = [
            "## Weekly Portfolio Review",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Benchmark (SPY) 3M Return:** {benchmark_return:.1%}",
            "",
            "### Portfolio Statistics",
            f"- Total Positions: {stats['total_positions']}",
            f"- Positive/Negative: {stats['positive_positions']}/{stats['negative_positions']}",
            f"- Average Return: {stats['avg_return']:.1%}",
            f"- Average Days Held: {stats['avg_days_held']:.0f}",
            "",
        ]

        # Exit candidates
        if exit_candidates:
            lines.append("### Exit Candidates")
            for ec in exit_candidates:
                emoji = ":x:" if ec["action"] == "remove" else ":warning:"
                lines.append(f"{emoji} **{ec['symbol']}**: {ec['return']:.1%} ({ec['days_held']} days)")
                lines.append(f"   Reason: {ec['reason']}")

        # Sector exposure
        if sector_exposure:
            lines.extend(["", "### Sector Exposure"])
            for sector, exp in sorted(sector_exposure.items(), key=lambda x: -x[1]):
                warning = " :warning:" if exp > 0.25 else ""
                lines.append(f"- {sector}: {exp:.0%}{warning}")

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

        # Handle risk constraints from Risk Guardian
        if message.sender == AgentRole.RISK_GUARDIAN:
            return self._handle_risk_update(message)

        # Handle market intelligence updates
        if message.sender == AgentRole.MARKET_INTELLIGENCE:
            return self._handle_market_update(message)

        # Handle queries
        if message.message_type == MessageType.QUERY:
            return self._handle_query(message)

        # Handle action confirmations
        if message.message_type == MessageType.ACTION:
            return self._handle_action_confirmation(message)

        return None

    def _handle_risk_update(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle risk update from Risk Guardian."""
        # Update risk constraints
        constraints = message.context.get("risk_constraints")
        if constraints:
            self._risk_constraints.update(constraints)
            logger.info(f"Updated risk constraints: {constraints}")

        return self.create_message(
            recipient=AgentRole.RISK_GUARDIAN,
            message_type=MessageType.ACKNOWLEDGMENT,
            subject="Risk Constraints Acknowledged",
            content=f"Risk constraints updated:\n"
                    f"- Allow new trades: {self._risk_constraints.get('allow_new_trades')}\n"
                    f"- Max position size: {self._risk_constraints.get('max_new_position_size'):.1%}",
            priority=MessagePriority.LOW,
            parent_message_id=message.id,
        )

    def _handle_market_update(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle market update from Market Intelligence."""
        # Process sector updates or earnings alerts
        sector_data = message.context.get("sector_data")
        earnings_news = message.context.get("earnings_news")

        if sector_data or earnings_news:
            # Could trigger early rebalancing check or screening
            logger.info("Received market update - may influence next analysis")

        return None

    def _handle_query(self, message: AgentMessage) -> AgentMessage:
        """Handle query from other agents."""
        query_type = message.context.get("query_type", "general")

        if query_type == "current_allocation":
            weights = self._get_current_weights()
            content = "## Current Portfolio Allocation\n\n"
            for symbol, weight in sorted(weights.items(), key=lambda x: -x[1]):
                content += f"- {symbol}: {weight:.1%}\n"
        elif query_type == "recommendations":
            content = "No pending recommendations"
        else:
            content = f"Unknown query type: {query_type}"

        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            subject=f"Re: {message.subject}",
            content=content,
            priority=MessagePriority.NORMAL,
            parent_message_id=message.id,
        )

    def _handle_action_confirmation(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle action confirmation from Operations."""
        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.ACKNOWLEDGMENT,
            subject=f"Acknowledged: {message.context.get('action', 'action')}",
            content="Action confirmation received.",
            priority=MessagePriority.LOW,
            parent_message_id=message.id,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get agent status including last analysis times."""
        status = super().get_status()
        status.update({
            "last_performance_review": self._last_performance_review.isoformat() if self._last_performance_review else None,
            "last_rebalancing_check": self._last_rebalancing_check.isoformat() if self._last_rebalancing_check else None,
            "last_stock_screening": self._last_stock_screening.isoformat() if self._last_stock_screening else None,
            "last_portfolio_review": self._last_portfolio_review.isoformat() if self._last_portfolio_review else None,
            "risk_constraints": self._risk_constraints,
            "config": {
                "performance_review_hours": self.performance_review_hours,
                "min_score_to_add": self.min_score_to_add,
                "underperformance_threshold": self.underperformance_threshold,
            },
        })
        return status
