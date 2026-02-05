"""
Risk Guardian Agent

Risk monitoring agent that protects the portfolio through risk assessment,
drawdown monitoring, correlation analysis, and emergency actions.
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


class RiskGuardianAgent(BaseAgent):
    """
    Risk Guardian Agent - monitors and protects portfolio risk.

    Responsibilities:
    - Real-time portfolio risk monitoring
    - Position-level and portfolio-level risk assessment
    - Drawdown protection and alerts
    - Correlation breakdown detection
    - Emergency position reduction
    - Circuit breaker enforcement

    Schedule:
    - Risk check: Every 30 minutes
    - Drawdown monitor: Every 15 minutes
    - Correlation check: Every 4 hours
    - Daily risk report: Daily (4 PM)

    Alert Thresholds:
    - Drawdown: Warning 5%, Critical 10%
    - Single position: Warning 12%, Critical 15%
    - Sector exposure: Warning 25%, Critical 35%
    - Daily loss: Warning 3%, Critical 5%

    Outputs To:
    - Operations: Risk-based config adjustments, emergency stops
    - Portfolio Strategist: Risk constraints for allocation
    - Market Intelligence: Request increased monitoring
    """

    def __init__(
        self,
        config: Dict[str, Any],
        message_queue,
        notifier=None,
        llm_client=None,
    ):
        """
        Initialize Risk Guardian agent.

        Args:
            config: Agent configuration
            message_queue: Shared message queue
            notifier: Optional Discord notifier
            llm_client: Optional LLM client for intelligent analysis
        """
        super().__init__(
            role=AgentRole.RISK_GUARDIAN,
            config=config,
            message_queue=message_queue,
            notifier=notifier,
            llm_client=llm_client,
        )

        # Load configuration
        rg_config = config.get("risk_guardian", {})
        self.risk_check_minutes = rg_config.get("risk_check_minutes", 30)
        self.drawdown_monitor_minutes = rg_config.get("drawdown_monitor_minutes", 15)
        self.correlation_check_hours = rg_config.get("correlation_check_hours", 4)
        self.daily_report_time = rg_config.get("daily_report_time", "16:00")

        # Load thresholds
        thresholds = rg_config.get("thresholds", {})
        self.drawdown_warning = thresholds.get("drawdown_warning", 0.05)
        self.drawdown_critical = thresholds.get("drawdown_critical", 0.10)
        self.position_warning = thresholds.get("position_warning", 0.12)
        self.position_critical = thresholds.get("position_critical", 0.15)
        self.sector_warning = thresholds.get("sector_warning", 0.25)
        self.sector_critical = thresholds.get("sector_critical", 0.35)
        self.daily_loss_warning = thresholds.get("daily_loss_warning", 0.03)
        self.daily_loss_critical = thresholds.get("daily_loss_critical", 0.05)

        # Emergency action settings
        emergency_config = rg_config.get("emergency_actions", {})
        self.emergency_enabled = emergency_config.get("enabled", True)
        self.auto_reduce_on_critical = emergency_config.get("auto_reduce_on_critical", True)
        self.reduce_percentage = emergency_config.get("reduce_percentage", 0.25)

        # Track last analysis times
        self._last_risk_check: Optional[datetime] = None
        self._last_drawdown_monitor: Optional[datetime] = None
        self._last_correlation_check: Optional[datetime] = None
        self._last_daily_report: Optional[datetime] = None

        # Track risk state
        self._peak_portfolio_value: Optional[float] = None
        self._current_drawdown: float = 0.0
        self._daily_start_value: Optional[float] = None
        self._daily_loss: float = 0.0
        self._trading_halted: bool = False
        self._last_correlation_matrix: Optional[pd.DataFrame] = None

        # Lazy-loaded components
        self._data_aggregator = None
        self._portfolio_manager = None
        self._symbol_manager = None

        logger.info(
            f"Risk Guardian agent initialized: "
            f"risk_check={self.risk_check_minutes}min, "
            f"drawdown_critical={self.drawdown_critical:.0%}"
        )

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
    def portfolio_manager(self):
        """Lazy load portfolio manager."""
        if self._portfolio_manager is None:
            try:
                from src.portfolio.portfolio_manager import get_portfolio_manager
                from config.settings import Settings
                config = Settings.load_trading_config()
                self._portfolio_manager = get_portfolio_manager(config)
            except ImportError as e:
                logger.debug(f"PortfolioManager not available: {e}")
        return self._portfolio_manager

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

        Returns:
            List of messages to send to other agents
        """
        return []

    def run_risk_check(self) -> List[AgentMessage]:
        """
        Run comprehensive portfolio risk check.

        Returns:
            List of messages with risk alerts
        """
        self._last_risk_check = datetime.now()
        messages = []

        try:
            risk_metrics = self._calculate_risk_metrics()

            if not risk_metrics:
                logger.warning("No risk metrics available")
                return messages

            # Analyze risk levels
            risk_alerts = self._analyze_risk_levels(risk_metrics)

            if risk_alerts:
                content = self._format_risk_check_report(risk_metrics, risk_alerts)

                # Use LLM for enhanced analysis
                if self.llm_client and self.llm_client.is_available():
                    llm_analysis = self.llm_client.assess_portfolio_risk(
                        risk_metrics,
                        self._get_positions_summary()
                    )
                    if llm_analysis:
                        content += f"\n\n### AI Risk Assessment\n{llm_analysis}"

                # Determine priority and whether to take emergency action
                priority = self._determine_alert_priority(risk_alerts)
                critical_alerts = [a for a in risk_alerts if a.get("severity") == "critical"]

                # Send risk warning to Operations
                messages.append(self.create_message(
                    recipient=AgentRole.OPERATIONS,
                    message_type=MessageType.SUGGESTION if critical_alerts else MessageType.OBSERVATION,
                    subject=f"Risk Alert: {len(risk_alerts)} issue(s) detected",
                    content=content,
                    priority=priority,
                    context={
                        "risk_metrics": risk_metrics,
                        "alerts": risk_alerts,
                        "emergency_action_needed": bool(critical_alerts),
                    },
                    requires_response=bool(critical_alerts),
                ))

                # Notify Portfolio Strategist about risk constraints
                messages.append(self.create_message(
                    recipient=AgentRole.PORTFOLIO_STRATEGIST,
                    message_type=MessageType.OBSERVATION,
                    subject=f"Risk Constraints Update",
                    content=self._format_risk_constraints(risk_metrics, risk_alerts),
                    priority=MessagePriority.HIGH if critical_alerts else MessagePriority.NORMAL,
                    context={
                        "risk_constraints": self._get_risk_constraints(risk_alerts),
                    },
                ))

                # Trigger emergency action if needed
                if critical_alerts and self.emergency_enabled and self.auto_reduce_on_critical:
                    emergency_msg = self.trigger_emergency_action(
                        reason=f"Critical risk level: {critical_alerts[0].get('type', 'unknown')}"
                    )
                    messages.append(emergency_msg)

            logger.info(f"Risk check complete: {len(risk_alerts)} alerts")

        except Exception as e:
            logger.error(f"Error during risk check: {e}")

        return messages

    def run_drawdown_monitor(self) -> List[AgentMessage]:
        """
        Monitor portfolio drawdown in real-time.

        Returns:
            List of messages if drawdown thresholds breached
        """
        self._last_drawdown_monitor = datetime.now()
        messages = []

        try:
            # Get current portfolio value
            portfolio_value = self._get_portfolio_value()

            if portfolio_value is None:
                return messages

            # Update peak value
            if self._peak_portfolio_value is None or portfolio_value > self._peak_portfolio_value:
                self._peak_portfolio_value = portfolio_value

            # Calculate drawdown
            if self._peak_portfolio_value > 0:
                self._current_drawdown = (self._peak_portfolio_value - portfolio_value) / self._peak_portfolio_value
            else:
                self._current_drawdown = 0.0

            # Update daily loss tracking
            today = datetime.now().date()
            if self._daily_start_value is None:
                self._daily_start_value = portfolio_value

            if self._daily_start_value > 0:
                self._daily_loss = (self._daily_start_value - portfolio_value) / self._daily_start_value
            else:
                self._daily_loss = 0.0

            # Check thresholds
            if self._current_drawdown >= self.drawdown_critical:
                messages.append(self.create_message(
                    recipient=AgentRole.OPERATIONS,
                    message_type=MessageType.SUGGESTION,
                    subject=f":rotating_light: CRITICAL Drawdown: {self._current_drawdown:.1%}",
                    content=self._format_drawdown_alert("critical", self._current_drawdown),
                    priority=MessagePriority.URGENT,
                    context={
                        "drawdown": self._current_drawdown,
                        "peak_value": self._peak_portfolio_value,
                        "current_value": portfolio_value,
                        "severity": "critical",
                    },
                    requires_response=True,
                ))
            elif self._current_drawdown >= self.drawdown_warning:
                messages.append(self.create_message(
                    recipient=AgentRole.OPERATIONS,
                    message_type=MessageType.OBSERVATION,
                    subject=f":warning: Drawdown Warning: {self._current_drawdown:.1%}",
                    content=self._format_drawdown_alert("warning", self._current_drawdown),
                    priority=MessagePriority.HIGH,
                    context={
                        "drawdown": self._current_drawdown,
                        "severity": "warning",
                    },
                ))

            # Check daily loss
            if self._daily_loss >= self.daily_loss_critical:
                messages.append(self.create_message(
                    recipient=AgentRole.OPERATIONS,
                    message_type=MessageType.SUGGESTION,
                    subject=f":rotating_light: CRITICAL Daily Loss: {self._daily_loss:.1%}",
                    content=f"Daily loss has exceeded critical threshold.\n\n"
                            f"- Daily Loss: {self._daily_loss:.1%}\n"
                            f"- Threshold: {self.daily_loss_critical:.1%}\n\n"
                            f"**Recommend halting trading for the day.**",
                    priority=MessagePriority.URGENT,
                    context={
                        "daily_loss": self._daily_loss,
                        "severity": "critical",
                        "recommended_action": "halt_trading",
                    },
                    requires_response=True,
                ))

            logger.debug(f"Drawdown monitor: {self._current_drawdown:.2%}, daily loss: {self._daily_loss:.2%}")

        except Exception as e:
            logger.error(f"Error during drawdown monitor: {e}")

        return messages

    def run_correlation_check(self) -> List[AgentMessage]:
        """
        Check for correlation breakdown between portfolio assets.

        Returns:
            List of messages with correlation alerts
        """
        self._last_correlation_check = datetime.now()
        messages = []

        try:
            # Get portfolio symbols
            symbols = []
            if self.symbol_manager:
                symbols = self.symbol_manager.get_active_symbols()

            if len(symbols) < 2:
                logger.info("Not enough symbols for correlation analysis")
                return messages

            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(symbols)

            if correlation_matrix is None or correlation_matrix.empty:
                return messages

            # Detect significant correlation changes
            changes = self._detect_correlation_changes(correlation_matrix)

            if changes:
                content = self._format_correlation_report(correlation_matrix, changes)

                # Use LLM for analysis
                if self.llm_client and self.llm_client.is_available():
                    llm_analysis = self.llm_client.analyze_correlation_breakdown(
                        correlation_matrix.to_dict(),
                        changes
                    )
                    if llm_analysis:
                        content += f"\n\n### AI Analysis\n{llm_analysis}"

                messages.append(self.create_message(
                    recipient=AgentRole.PORTFOLIO_STRATEGIST,
                    message_type=MessageType.OBSERVATION,
                    subject=f"Correlation Alert: {len(changes)} significant changes",
                    content=content,
                    priority=MessagePriority.NORMAL,
                    context={
                        "correlation_changes": changes,
                    },
                ))

            # Update stored correlation matrix
            self._last_correlation_matrix = correlation_matrix

            logger.info(f"Correlation check complete: {len(changes)} changes detected")

        except Exception as e:
            logger.error(f"Error during correlation check: {e}")

        return messages

    def run_daily_risk_report(self) -> List[AgentMessage]:
        """
        Generate comprehensive daily risk report.

        Returns:
            List of messages with daily risk summary
        """
        self._last_daily_report = datetime.now()
        messages = []

        try:
            # Gather comprehensive risk data
            risk_metrics = self._calculate_risk_metrics()
            positions = self._get_positions_summary()
            sector_exposure = self._calculate_sector_exposure()

            content = self._format_daily_risk_report(
                risk_metrics, positions, sector_exposure
            )

            # Use LLM for analysis
            if self.llm_client and self.llm_client.is_available():
                llm_analysis = self.llm_client.assess_portfolio_risk(
                    risk_metrics, positions
                )
                if llm_analysis:
                    content += f"\n\n### AI Risk Summary\n{llm_analysis}"

            # Send to Operations for logging
            messages.append(self.create_message(
                recipient=AgentRole.OPERATIONS,
                message_type=MessageType.STATUS_UPDATE,
                subject=f"Daily Risk Report - {datetime.now().strftime('%Y-%m-%d')}",
                content=content,
                priority=MessagePriority.NORMAL,
                context={
                    "risk_metrics": risk_metrics,
                    "positions_count": len(positions),
                    "sector_exposure": sector_exposure,
                },
            ))

            # Reset daily tracking
            self._daily_start_value = self._get_portfolio_value()
            self._daily_loss = 0.0

            logger.info("Daily risk report generated")

        except Exception as e:
            logger.error(f"Error generating daily risk report: {e}")

        return messages

    def trigger_emergency_action(self, reason: str) -> AgentMessage:
        """
        Trigger emergency risk reduction action.

        Args:
            reason: Reason for emergency action

        Returns:
            Message to Operations with emergency action request
        """
        logger.warning(f"EMERGENCY ACTION TRIGGERED: {reason}")

        self._trading_halted = True

        return self.create_message(
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.ACTION,
            subject=":rotating_light: EMERGENCY: Risk Reduction Required",
            content=f"## Emergency Risk Action Required\n\n"
                    f"**Reason:** {reason}\n\n"
                    f"**Recommended Actions:**\n"
                    f"1. Reduce all positions by {self.reduce_percentage:.0%}\n"
                    f"2. Halt new trades until risk levels normalize\n"
                    f"3. Close highest-risk positions first\n\n"
                    f"**Current State:**\n"
                    f"- Drawdown: {self._current_drawdown:.1%}\n"
                    f"- Daily Loss: {self._daily_loss:.1%}\n"
                    f"- Trading Halted: Yes",
            priority=MessagePriority.URGENT,
            context={
                "action": "emergency_reduce",
                "reduce_percentage": self.reduce_percentage,
                "reason": reason,
                "halt_trading": True,
            },
            requires_response=True,
        )

    def _calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics."""
        metrics = {}

        try:
            # Get returns data
            if self.data_aggregator:
                analytics = self.data_aggregator.prepare_analytics_data(
                    start_date=datetime.now() - timedelta(days=30),
                    end_date=datetime.now()
                )
                returns = analytics.get("returns", pd.Series())

                if len(returns) > 0:
                    # Calculate VaR (Value at Risk) - 95% confidence
                    metrics["var_95"] = np.percentile(returns, 5)

                    # Calculate CVaR (Expected Shortfall)
                    var_cutoff = metrics["var_95"]
                    metrics["cvar_95"] = returns[returns <= var_cutoff].mean() if len(returns[returns <= var_cutoff]) > 0 else var_cutoff

                    # Volatility (annualized)
                    metrics["volatility"] = returns.std() * np.sqrt(252)

                    # Sharpe ratio
                    if returns.std() > 0:
                        metrics["sharpe_ratio"] = (returns.mean() / returns.std()) * np.sqrt(252)
                    else:
                        metrics["sharpe_ratio"] = 0.0

            # Current drawdown and daily loss
            metrics["current_drawdown"] = self._current_drawdown
            metrics["daily_loss"] = self._daily_loss

            # Position metrics
            positions = self._get_positions_summary()
            if positions:
                position_sizes = [p.get("weight", 0) for p in positions]
                metrics["max_position_size"] = max(position_sizes) if position_sizes else 0
                metrics["avg_position_size"] = sum(position_sizes) / len(position_sizes) if position_sizes else 0
                metrics["position_count"] = len(positions)

            # Sector exposure
            sector_exposure = self._calculate_sector_exposure()
            if sector_exposure:
                metrics["max_sector_exposure"] = max(sector_exposure.values())
                metrics["sector_count"] = len(sector_exposure)

            # Portfolio beta (vs SPY)
            metrics["portfolio_beta"] = self._calculate_portfolio_beta()

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")

        return metrics

    def _analyze_risk_levels(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze risk metrics and generate alerts."""
        alerts = []

        # Check drawdown
        drawdown = metrics.get("current_drawdown", 0)
        if drawdown >= self.drawdown_critical:
            alerts.append({
                "type": "drawdown",
                "severity": "critical",
                "value": drawdown,
                "threshold": self.drawdown_critical,
                "message": f"Portfolio drawdown ({drawdown:.1%}) exceeds critical threshold ({self.drawdown_critical:.1%})",
            })
        elif drawdown >= self.drawdown_warning:
            alerts.append({
                "type": "drawdown",
                "severity": "warning",
                "value": drawdown,
                "threshold": self.drawdown_warning,
                "message": f"Portfolio drawdown ({drawdown:.1%}) exceeds warning threshold ({self.drawdown_warning:.1%})",
            })

        # Check daily loss
        daily_loss = metrics.get("daily_loss", 0)
        if daily_loss >= self.daily_loss_critical:
            alerts.append({
                "type": "daily_loss",
                "severity": "critical",
                "value": daily_loss,
                "threshold": self.daily_loss_critical,
                "message": f"Daily loss ({daily_loss:.1%}) exceeds critical threshold ({self.daily_loss_critical:.1%})",
            })
        elif daily_loss >= self.daily_loss_warning:
            alerts.append({
                "type": "daily_loss",
                "severity": "warning",
                "value": daily_loss,
                "threshold": self.daily_loss_warning,
                "message": f"Daily loss ({daily_loss:.1%}) exceeds warning threshold ({self.daily_loss_warning:.1%})",
            })

        # Check position concentration
        max_position = metrics.get("max_position_size", 0)
        if max_position >= self.position_critical:
            alerts.append({
                "type": "position_concentration",
                "severity": "critical",
                "value": max_position,
                "threshold": self.position_critical,
                "message": f"Max position ({max_position:.1%}) exceeds critical threshold ({self.position_critical:.1%})",
            })
        elif max_position >= self.position_warning:
            alerts.append({
                "type": "position_concentration",
                "severity": "warning",
                "value": max_position,
                "threshold": self.position_warning,
                "message": f"Max position ({max_position:.1%}) exceeds warning threshold ({self.position_warning:.1%})",
            })

        # Check sector exposure
        max_sector = metrics.get("max_sector_exposure", 0)
        if max_sector >= self.sector_critical:
            alerts.append({
                "type": "sector_concentration",
                "severity": "critical",
                "value": max_sector,
                "threshold": self.sector_critical,
                "message": f"Max sector exposure ({max_sector:.1%}) exceeds critical threshold ({self.sector_critical:.1%})",
            })
        elif max_sector >= self.sector_warning:
            alerts.append({
                "type": "sector_concentration",
                "severity": "warning",
                "value": max_sector,
                "threshold": self.sector_warning,
                "message": f"Max sector exposure ({max_sector:.1%}) exceeds warning threshold ({self.sector_warning:.1%})",
            })

        return alerts

    def _determine_alert_priority(self, alerts: List[Dict[str, Any]]) -> MessagePriority:
        """Determine message priority based on alerts."""
        severities = [a.get("severity", "warning") for a in alerts]

        if "critical" in severities:
            return MessagePriority.URGENT
        elif "warning" in severities:
            return MessagePriority.HIGH
        return MessagePriority.NORMAL

    def _get_portfolio_value(self) -> Optional[float]:
        """Get current portfolio value."""
        try:
            if self.portfolio_manager:
                return self.portfolio_manager.get_total_value()

            # Fallback to data aggregator
            if self.data_aggregator:
                analytics = self.data_aggregator.prepare_analytics_data(
                    start_date=datetime.now() - timedelta(days=1),
                    end_date=datetime.now()
                )
                return analytics.get("portfolio_metrics", {}).get("portfolio_value")
        except Exception as e:
            logger.debug(f"Error getting portfolio value: {e}")

        return None

    def _get_positions_summary(self) -> List[Dict[str, Any]]:
        """Get summary of current positions."""
        positions = []

        try:
            if self.data_aggregator:
                analytics = self.data_aggregator.prepare_analytics_data(
                    start_date=datetime.now() - timedelta(days=1),
                    end_date=datetime.now()
                )
                pos_data = analytics.get("positions", {})

                portfolio_value = self._get_portfolio_value() or 1

                for symbol, data in pos_data.items():
                    market_value = data.get("market_value", 0)
                    positions.append({
                        "symbol": symbol,
                        "market_value": market_value,
                        "weight": market_value / portfolio_value if portfolio_value > 0 else 0,
                        "unrealized_pnl": data.get("unrealized_pnl", 0),
                        "side": data.get("side", "long"),
                    })
        except Exception as e:
            logger.debug(f"Error getting positions: {e}")

        return positions

    def _calculate_sector_exposure(self) -> Dict[str, float]:
        """Calculate sector exposure."""
        exposure = {}

        try:
            if self.symbol_manager:
                symbols = self.symbol_manager.get_active_symbols()
                positions = self._get_positions_summary()

                # Map positions to sectors
                for pos in positions:
                    entry = self.symbol_manager.get_symbol_entry(pos["symbol"])
                    if entry:
                        sector = entry.sector or "Unknown"
                        exposure[sector] = exposure.get(sector, 0) + pos.get("weight", 0)
        except Exception as e:
            logger.debug(f"Error calculating sector exposure: {e}")

        return exposure

    def _calculate_portfolio_beta(self) -> float:
        """Calculate portfolio beta vs SPY."""
        try:
            import yfinance as yf

            # Get SPY returns
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="3mo")
            if spy_hist.empty:
                return 1.0

            spy_returns = spy_hist["Close"].pct_change().dropna()

            # Get portfolio returns
            if self.data_aggregator:
                analytics = self.data_aggregator.prepare_analytics_data(
                    start_date=datetime.now() - timedelta(days=90),
                    end_date=datetime.now()
                )
                portfolio_returns = analytics.get("returns", pd.Series())

                if len(portfolio_returns) > 10 and len(spy_returns) > 10:
                    # Align dates
                    common_dates = portfolio_returns.index.intersection(spy_returns.index)
                    if len(common_dates) > 10:
                        p_ret = portfolio_returns.loc[common_dates]
                        s_ret = spy_returns.loc[common_dates]

                        covariance = np.cov(p_ret, s_ret)[0, 1]
                        variance = np.var(s_ret)

                        if variance > 0:
                            return covariance / variance

        except Exception as e:
            logger.debug(f"Error calculating portfolio beta: {e}")

        return 1.0

    def _calculate_correlation_matrix(self, symbols: List[str]) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for symbols."""
        try:
            import yfinance as yf

            # Get price data
            data = {}
            for symbol in symbols[:20]:  # Limit to 20 symbols
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="3mo")
                    if not hist.empty:
                        data[symbol] = hist["Close"].pct_change().dropna()
                except Exception:
                    pass

            if len(data) < 2:
                return None

            # Create DataFrame and calculate correlation
            df = pd.DataFrame(data)
            return df.corr()

        except Exception as e:
            logger.debug(f"Error calculating correlation matrix: {e}")
            return None

    def _detect_correlation_changes(self, current_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect significant correlation changes."""
        changes = []

        if self._last_correlation_matrix is None:
            return changes

        try:
            # Compare matrices
            for col in current_matrix.columns:
                for row in current_matrix.index:
                    if col >= row:  # Skip diagonal and duplicates
                        continue

                    if col in self._last_correlation_matrix.columns and row in self._last_correlation_matrix.index:
                        old_corr = self._last_correlation_matrix.loc[row, col]
                        new_corr = current_matrix.loc[row, col]
                        change = abs(new_corr - old_corr)

                        if change > 0.2:  # Significant change threshold
                            changes.append({
                                "pair": f"{row}/{col}",
                                "old_correlation": old_corr,
                                "new_correlation": new_corr,
                                "change": change,
                                "direction": "increased" if new_corr > old_corr else "decreased",
                            })
        except Exception as e:
            logger.debug(f"Error detecting correlation changes: {e}")

        return changes

    def _get_risk_constraints(self, alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate risk constraints based on current alerts."""
        constraints = {
            "max_new_position_size": self.position_warning * 0.8,  # Reduce by 20%
            "allow_new_trades": True,
            "reduce_exposure": False,
        }

        for alert in alerts:
            if alert.get("severity") == "critical":
                constraints["allow_new_trades"] = False
                constraints["reduce_exposure"] = True
                constraints["max_new_position_size"] = 0

        return constraints

    def _format_risk_check_report(
        self,
        metrics: Dict[str, Any],
        alerts: List[Dict[str, Any]]
    ) -> str:
        """Format risk check report."""
        lines = [
            "## Risk Check Report",
            f"**Check Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "### Risk Metrics",
            f"- Current Drawdown: {metrics.get('current_drawdown', 0):.1%}",
            f"- Daily Loss: {metrics.get('daily_loss', 0):.1%}",
            f"- VaR (95%): {metrics.get('var_95', 0):.2%}",
            f"- Volatility: {metrics.get('volatility', 0):.1%}",
            f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}",
            f"- Portfolio Beta: {metrics.get('portfolio_beta', 1):.2f}",
            "",
            f"- Max Position: {metrics.get('max_position_size', 0):.1%}",
            f"- Max Sector: {metrics.get('max_sector_exposure', 0):.1%}",
            "",
            "### Alerts",
        ]

        for alert in alerts:
            emoji = ":rotating_light:" if alert["severity"] == "critical" else ":warning:"
            lines.append(f"{emoji} **{alert['type'].replace('_', ' ').title()}**: {alert['message']}")

        return "\n".join(lines)

    def _format_risk_constraints(
        self,
        metrics: Dict[str, Any],
        alerts: List[Dict[str, Any]]
    ) -> str:
        """Format risk constraints message."""
        constraints = self._get_risk_constraints(alerts)

        lines = [
            "## Risk Constraints Update",
            f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "### Current Constraints",
            f"- Max New Position Size: {constraints['max_new_position_size']:.1%}",
            f"- Allow New Trades: {'Yes' if constraints['allow_new_trades'] else 'No'}",
            f"- Reduce Exposure: {'Yes' if constraints['reduce_exposure'] else 'No'}",
            "",
            "### Risk State",
            f"- Drawdown: {metrics.get('current_drawdown', 0):.1%}",
            f"- Daily Loss: {metrics.get('daily_loss', 0):.1%}",
        ]

        return "\n".join(lines)

    def _format_drawdown_alert(self, severity: str, drawdown: float) -> str:
        """Format drawdown alert message."""
        lines = [
            f"## Drawdown Alert: {severity.upper()}",
            f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Current Drawdown:** {drawdown:.1%}",
            f"**Peak Value:** ${self._peak_portfolio_value:,.2f}" if self._peak_portfolio_value else "",
            "",
        ]

        if severity == "critical":
            lines.extend([
                "### Recommended Actions",
                "1. Consider reducing position sizes",
                "2. Review highest-loss positions for potential exit",
                "3. Tighten stop losses on remaining positions",
            ])

        return "\n".join(lines)

    def _format_correlation_report(
        self,
        matrix: pd.DataFrame,
        changes: List[Dict[str, Any]]
    ) -> str:
        """Format correlation report."""
        lines = [
            "## Correlation Analysis Report",
            f"**Check Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "### Significant Correlation Changes",
        ]

        for change in changes:
            direction = ":chart_with_upwards_trend:" if change["direction"] == "increased" else ":chart_with_downwards_trend:"
            lines.append(
                f"{direction} **{change['pair']}**: "
                f"{change['old_correlation']:.2f} -> {change['new_correlation']:.2f} "
                f"(change: {change['change']:.2f})"
            )

        # Add high correlations warning
        high_corr_pairs = []
        for col in matrix.columns:
            for row in matrix.index:
                if col >= row:
                    continue
                corr = matrix.loc[row, col]
                if abs(corr) > 0.8:
                    high_corr_pairs.append((row, col, corr))

        if high_corr_pairs:
            lines.extend(["", "### High Correlation Pairs (>0.8)"])
            for row, col, corr in high_corr_pairs:
                lines.append(f"- {row}/{col}: {corr:.2f}")

        return "\n".join(lines)

    def _format_daily_risk_report(
        self,
        metrics: Dict[str, Any],
        positions: List[Dict[str, Any]],
        sector_exposure: Dict[str, float]
    ) -> str:
        """Format daily risk report."""
        lines = [
            "## Daily Risk Report",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "### Portfolio Risk Summary",
            f"- Positions: {metrics.get('position_count', 0)}",
            f"- Current Drawdown: {metrics.get('current_drawdown', 0):.1%}",
            f"- Daily Loss: {metrics.get('daily_loss', 0):.1%}",
            f"- VaR (95%): {metrics.get('var_95', 0):.2%}",
            f"- CVaR (95%): {metrics.get('cvar_95', 0):.2%}",
            f"- Volatility (ann.): {metrics.get('volatility', 0):.1%}",
            f"- Portfolio Beta: {metrics.get('portfolio_beta', 1):.2f}",
            "",
            "### Position Concentration",
            f"- Max Position: {metrics.get('max_position_size', 0):.1%}",
            f"- Avg Position: {metrics.get('avg_position_size', 0):.1%}",
            "",
            "### Sector Exposure",
        ]

        if sector_exposure:
            for sector, exp in sorted(sector_exposure.items(), key=lambda x: -x[1]):
                lines.append(f"- {sector}: {exp:.1%}")

        lines.extend([
            "",
            "### Thresholds",
            f"- Drawdown Warning/Critical: {self.drawdown_warning:.0%}/{self.drawdown_critical:.0%}",
            f"- Position Warning/Critical: {self.position_warning:.0%}/{self.position_critical:.0%}",
            f"- Sector Warning/Critical: {self.sector_warning:.0%}/{self.sector_critical:.0%}",
        ])

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

        # Handle market alerts from Market Intelligence
        if message.sender == AgentRole.MARKET_INTELLIGENCE:
            return self._handle_market_alert(message)

        # Handle risk check requests
        if message.message_type == MessageType.QUERY:
            return self._handle_query(message)

        # Handle action confirmations
        if message.message_type == MessageType.ACTION:
            return self._handle_action_confirmation(message)

        return None

    def _handle_market_alert(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle market alert from Market Intelligence."""
        vix_alert = message.context.get("vix_alert")

        if vix_alert and vix_alert.get("current", 0) > self.alert_on_vix_spike:
            # Trigger additional risk monitoring
            return self.create_message(
                recipient=AgentRole.OPERATIONS,
                message_type=MessageType.OBSERVATION,
                subject="Elevated VIX - Increased Risk Monitoring",
                content=f"VIX has spiked to {vix_alert['current']:.1f}. "
                        f"Increasing risk monitoring frequency.",
                priority=MessagePriority.HIGH,
                context={"vix_level": vix_alert["current"]},
            )

        return None

    def _handle_query(self, message: AgentMessage) -> AgentMessage:
        """Handle query from other agents."""
        query_type = message.context.get("query_type", "general")

        if query_type == "risk_status":
            metrics = self._calculate_risk_metrics()
            content = self._format_risk_check_report(metrics, [])
        elif query_type == "can_trade":
            constraints = self._get_risk_constraints([])
            can_trade = constraints.get("allow_new_trades", True) and not self._trading_halted
            content = f"Trading allowed: {'Yes' if can_trade else 'No'}"
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
        action = message.context.get("action")

        if action == "resume_trading":
            self._trading_halted = False
            logger.info("Trading resumed after emergency halt")

        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.ACKNOWLEDGMENT,
            subject=f"Acknowledged: {action}",
            content=f"Action '{action}' acknowledged.",
            priority=MessagePriority.LOW,
            parent_message_id=message.id,
        )

    @property
    def alert_on_vix_spike(self) -> float:
        """VIX spike threshold."""
        return self.config.get("market_intelligence", {}).get("alert_on_vix_spike", 25)

    def get_status(self) -> Dict[str, Any]:
        """Get agent status including risk state."""
        status = super().get_status()
        status.update({
            "last_risk_check": self._last_risk_check.isoformat() if self._last_risk_check else None,
            "last_drawdown_monitor": self._last_drawdown_monitor.isoformat() if self._last_drawdown_monitor else None,
            "last_correlation_check": self._last_correlation_check.isoformat() if self._last_correlation_check else None,
            "last_daily_report": self._last_daily_report.isoformat() if self._last_daily_report else None,
            "risk_state": {
                "current_drawdown": self._current_drawdown,
                "daily_loss": self._daily_loss,
                "trading_halted": self._trading_halted,
                "peak_portfolio_value": self._peak_portfolio_value,
            },
            "thresholds": {
                "drawdown_warning": self.drawdown_warning,
                "drawdown_critical": self.drawdown_critical,
                "position_warning": self.position_warning,
                "position_critical": self.position_critical,
                "sector_warning": self.sector_warning,
                "sector_critical": self.sector_critical,
            },
        })
        return status
