"""
Agent Orchestrator

APScheduler-based coordinator for the 4-agent trading system.
Manages agent schedules, message passing, and lifecycle.
"""

import logging
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .base_agent import AgentRole
from .message_queue import MessageQueue, get_message_queue
from .llm_client import LLMClient, get_llm_client
from .agent_notifier import AgentNotifier, get_agent_notifier
from .market_intelligence import MarketIntelligenceAgent
from .risk_guardian import RiskGuardianAgent
from .portfolio_strategist import PortfolioStrategistAgent
from .operations_agent import OperationsAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Coordinates the 4-agent trading system using APScheduler.

    Agent System Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Agent Orchestrator                          │
    │                  (APScheduler coordination)                      │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
    │     Market       │ │    Portfolio     │ │   Operations     │
    │  Intelligence    │ │   Strategist     │ │                  │
    │                  │ │                  │ │ - Config changes │
    │ - News/Events    │ │ - Stock screen   │ │ - Retraining     │
    │ - Macro data     │ │ - Allocation     │ │ - Execution QA   │
    │ - Sector trends  │ │ - Rebalancing    │ │ - System health  │
    └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
             │                    │                    │
             │                    ▼                    │
             │           ┌──────────────────┐          │
             └──────────▶│   Risk Guardian  │◀─────────┘
                         │                  │
                         │ - Portfolio risk │
                         │ - Drawdown alert │
                         │ - Position limits│
                         │ - EMERGENCY STOP │
                         └──────────────────┘

    Schedules:
    - Market Intelligence:
      - News scan: Every 1 hour
      - Earnings check: Every 4 hours
      - Macro analysis: Every 6 hours
      - Sector analysis: Daily (6 AM)

    - Risk Guardian:
      - Risk check: Every 30 minutes
      - Drawdown monitor: Every 15 minutes
      - Correlation check: Every 4 hours
      - Daily risk report: Daily (4 PM)

    - Portfolio Strategist:
      - Performance review: Every 4 hours
      - Rebalancing check: Daily (10 AM)
      - Stock screening: Weekly (Sunday 6 PM)
      - Portfolio review: Weekly (Monday 9 AM)

    - Operations:
      - Process messages: Every 15 minutes
      - Execution quality check: Every 2 hours
      - System health check: Every 4 hours
      - Degradation check: Every 12 hours
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the orchestrator.

        Args:
            config: Full trading configuration dictionary
        """
        self.config = config
        self.agents_config = config.get("agents", {})

        # Check if agents are enabled
        self.enabled = self.agents_config.get("enabled", False)
        if not self.enabled:
            logger.info("Agent orchestrator disabled in config")
            return

        # Initialize shared infrastructure
        self.message_queue = get_message_queue()

        # Initialize notifier if Discord is enabled
        self.notifier = None
        if self.agents_config.get("notifications", {}).get("discord_enabled", True):
            self.notifier = get_agent_notifier(self.agents_config.get("notifications", {}))

        # Initialize LLM client if enabled
        self.llm_client = None
        if self.agents_config.get("use_llm", True):
            self.llm_client = get_llm_client(self.agents_config)

        # Initialize agents
        self._init_agents()

        # Initialize scheduler
        self.scheduler = BackgroundScheduler()
        self._is_running = False
        self._lock = threading.Lock()

        logger.info("Agent orchestrator initialized with 4-agent system")

    def _init_agents(self) -> None:
        """Initialize agent instances."""
        # Market Intelligence Agent
        self.market_intelligence = MarketIntelligenceAgent(
            config=self.agents_config,
            message_queue=self.message_queue,
            notifier=self.notifier,
            llm_client=self.llm_client,
        )

        # Risk Guardian Agent
        self.risk_guardian = RiskGuardianAgent(
            config=self.agents_config,
            message_queue=self.message_queue,
            notifier=self.notifier,
            llm_client=self.llm_client,
        )

        # Portfolio Strategist Agent
        self.portfolio_strategist = PortfolioStrategistAgent(
            config=self.agents_config,
            message_queue=self.message_queue,
            notifier=self.notifier,
            llm_client=self.llm_client,
        )

        # Operations Agent
        self.operations = OperationsAgent(
            config=self.agents_config,
            message_queue=self.message_queue,
            notifier=self.notifier,
            llm_client=self.llm_client,
        )

        self.agents = {
            AgentRole.MARKET_INTELLIGENCE: self.market_intelligence,
            AgentRole.RISK_GUARDIAN: self.risk_guardian,
            AgentRole.PORTFOLIO_STRATEGIST: self.portfolio_strategist,
            AgentRole.OPERATIONS: self.operations,
        }

    def start(self) -> None:
        """Start the orchestrator and all scheduled jobs."""
        if not self.enabled:
            logger.info("Agent orchestrator is disabled")
            return

        if self._is_running:
            logger.warning("Orchestrator already running")
            return

        with self._lock:
            # Get configuration for each agent
            mi_config = self.agents_config.get("market_intelligence", {})
            rg_config = self.agents_config.get("risk_guardian", {})
            ps_config = self.agents_config.get("portfolio_strategist", {})
            ops_config = self.agents_config.get("operations", {})

            # ============================================================
            # Market Intelligence Agent Schedule
            # ============================================================

            # News scan: Every N hours (default 1)
            news_scan_hours = mi_config.get("news_scan_hours", 1)
            self.scheduler.add_job(
                self._run_news_scan,
                trigger=IntervalTrigger(hours=news_scan_hours),
                id="mi_news_scan",
                name="Market Intelligence News Scan",
                replace_existing=True,
            )

            # Earnings check: Every N hours (default 4)
            earnings_check_hours = mi_config.get("earnings_check_hours", 4)
            self.scheduler.add_job(
                self._run_earnings_check,
                trigger=IntervalTrigger(hours=earnings_check_hours),
                id="mi_earnings_check",
                name="Market Intelligence Earnings Check",
                replace_existing=True,
            )

            # Macro analysis: Every N hours (default 6)
            macro_analysis_hours = mi_config.get("macro_analysis_hours", 6)
            self.scheduler.add_job(
                self._run_macro_analysis,
                trigger=IntervalTrigger(hours=macro_analysis_hours),
                id="mi_macro_analysis",
                name="Market Intelligence Macro Analysis",
                replace_existing=True,
            )

            # Sector analysis: Daily at specified time (default 6 AM)
            sector_analysis_time = mi_config.get("sector_analysis_time", "06:00")
            hour, minute = map(int, sector_analysis_time.split(":"))
            self.scheduler.add_job(
                self._run_sector_analysis,
                trigger=CronTrigger(hour=hour, minute=minute),
                id="mi_sector_analysis",
                name="Market Intelligence Sector Analysis",
                replace_existing=True,
            )

            # ============================================================
            # Risk Guardian Agent Schedule
            # ============================================================

            # Risk check: Every N minutes (default 30)
            risk_check_minutes = rg_config.get("risk_check_minutes", 30)
            self.scheduler.add_job(
                self._run_risk_check,
                trigger=IntervalTrigger(minutes=risk_check_minutes),
                id="rg_risk_check",
                name="Risk Guardian Risk Check",
                replace_existing=True,
            )

            # Drawdown monitor: Every N minutes (default 15)
            drawdown_monitor_minutes = rg_config.get("drawdown_monitor_minutes", 15)
            self.scheduler.add_job(
                self._run_drawdown_monitor,
                trigger=IntervalTrigger(minutes=drawdown_monitor_minutes),
                id="rg_drawdown_monitor",
                name="Risk Guardian Drawdown Monitor",
                replace_existing=True,
            )

            # Correlation check: Every N hours (default 4)
            correlation_check_hours = rg_config.get("correlation_check_hours", 4)
            self.scheduler.add_job(
                self._run_correlation_check,
                trigger=IntervalTrigger(hours=correlation_check_hours),
                id="rg_correlation_check",
                name="Risk Guardian Correlation Check",
                replace_existing=True,
            )

            # Daily risk report: Daily at specified time (default 4 PM)
            daily_report_time = rg_config.get("daily_report_time", "16:00")
            hour, minute = map(int, daily_report_time.split(":"))
            self.scheduler.add_job(
                self._run_daily_risk_report,
                trigger=CronTrigger(hour=hour, minute=minute),
                id="rg_daily_report",
                name="Risk Guardian Daily Report",
                replace_existing=True,
            )

            # ============================================================
            # Portfolio Strategist Agent Schedule
            # ============================================================

            # Performance review: Every N hours (default 4)
            performance_review_hours = ps_config.get("performance_review_hours", 4)
            self.scheduler.add_job(
                self._run_performance_review,
                trigger=IntervalTrigger(hours=performance_review_hours),
                id="ps_performance_review",
                name="Portfolio Strategist Performance Review",
                replace_existing=True,
            )

            # Rebalancing check: Daily at specified time (default 10 AM)
            rebalancing_check_time = ps_config.get("rebalancing_check_time", "10:00")
            hour, minute = map(int, rebalancing_check_time.split(":"))
            self.scheduler.add_job(
                self._run_rebalancing_check,
                trigger=CronTrigger(hour=hour, minute=minute),
                id="ps_rebalancing_check",
                name="Portfolio Strategist Rebalancing Check",
                replace_existing=True,
            )

            # Day map for weekly scheduling
            day_map = {
                "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                "friday": 4, "saturday": 5, "sunday": 6
            }

            # Stock screening: Daily or Weekly based on config
            screening_frequency = ps_config.get("stock_screening_frequency", "weekly")
            if screening_frequency == "daily":
                # Daily screening at interval
                screening_hours = ps_config.get("stock_screening_hours", 12)
                self.scheduler.add_job(
                    self._run_stock_screening,
                    trigger=IntervalTrigger(hours=screening_hours),
                    id="ps_stock_screening",
                    name="Portfolio Strategist Stock Screening",
                    replace_existing=True,
                )
            else:
                # Weekly at specified day and time
                screening_day = ps_config.get("stock_screening_day", "sunday")
                screening_time = ps_config.get("stock_screening_time", "18:00")
                screening_hour, screening_minute = map(int, screening_time.split(":"))
                screening_dow = day_map.get(screening_day.lower(), 6)
                self.scheduler.add_job(
                    self._run_stock_screening,
                    trigger=CronTrigger(
                        day_of_week=screening_dow,
                        hour=screening_hour,
                        minute=screening_minute
                    ),
                    id="ps_stock_screening",
                    name="Portfolio Strategist Stock Screening",
                    replace_existing=True,
                )

            # Portfolio review: Weekly at specified day and time
            review_day = ps_config.get("portfolio_review_day", "monday")
            review_time = ps_config.get("portfolio_review_time", "09:00")
            review_hour, review_minute = map(int, review_time.split(":"))
            review_dow = day_map.get(review_day.lower(), 0)
            self.scheduler.add_job(
                self._run_portfolio_review,
                trigger=CronTrigger(
                    day_of_week=review_dow,
                    hour=review_hour,
                    minute=review_minute
                ),
                id="ps_portfolio_review",
                name="Portfolio Strategist Portfolio Review",
                replace_existing=True,
            )

            # ============================================================
            # Operations Agent Schedule
            # ============================================================

            # Process messages: Every N minutes (default 15)
            process_messages_minutes = ops_config.get("process_messages_minutes", 15)
            self.scheduler.add_job(
                self._run_operations_cycle,
                trigger=IntervalTrigger(minutes=process_messages_minutes),
                id="ops_process_messages",
                name="Operations Process Messages",
                replace_existing=True,
            )

            # Execution quality check: Every N hours (default 2)
            execution_quality_hours = ops_config.get("execution_quality_hours", 2)
            self.scheduler.add_job(
                self._run_execution_quality_check,
                trigger=IntervalTrigger(hours=execution_quality_hours),
                id="ops_execution_quality",
                name="Operations Execution Quality Check",
                replace_existing=True,
            )

            # System health check: Every N hours (default 4)
            system_health_hours = ops_config.get("system_health_hours", 4)
            self.scheduler.add_job(
                self._run_system_health_check,
                trigger=IntervalTrigger(hours=system_health_hours),
                id="ops_system_health",
                name="Operations System Health Check",
                replace_existing=True,
            )

            # Degradation check: Every N hours (default 12)
            degradation_check_hours = ops_config.get("degradation_check_hours", 12)
            self.scheduler.add_job(
                self._run_degradation_check,
                trigger=IntervalTrigger(hours=degradation_check_hours),
                id="ops_degradation_check",
                name="Operations Degradation Check",
                replace_existing=True,
            )

            # Start scheduler
            self.scheduler.start()
            self._is_running = True

            # Notify startup
            if self.notifier:
                self.notifier.notify_agent_startup([
                    "Market Intelligence",
                    "Risk Guardian",
                    "Portfolio Strategist",
                    "Operations",
                ])

            logger.info("Agent orchestrator started with 4-agent system")
            self._log_schedule()

    def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        if not self._is_running:
            return

        with self._lock:
            # Notify shutdown
            if self.notifier:
                self.notifier.notify_agent_shutdown("Orchestrator stopping")

            # Shutdown scheduler
            self.scheduler.shutdown(wait=True)
            self._is_running = False

            # Close message queue connections
            self.message_queue.close()

            logger.info("Agent orchestrator stopped")

    # ============================================================
    # Market Intelligence Agent Methods
    # ============================================================

    def _run_news_scan(self) -> None:
        """Run Market Intelligence news scan."""
        logger.info("Running scheduled news scan")
        try:
            messages = self.market_intelligence.run_news_scan()
            for msg in messages:
                self.market_intelligence.send_message(msg)
            logger.info(f"News scan complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"News scan failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.MARKET_INTELLIGENCE,
                    "News scan failed",
                    str(e)
                )

    def _run_earnings_check(self) -> None:
        """Run Market Intelligence earnings check."""
        logger.info("Running scheduled earnings check")
        try:
            messages = self.market_intelligence.run_earnings_check()
            for msg in messages:
                self.market_intelligence.send_message(msg)
            logger.info(f"Earnings check complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Earnings check failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.MARKET_INTELLIGENCE,
                    "Earnings check failed",
                    str(e)
                )

    def _run_macro_analysis(self) -> None:
        """Run Market Intelligence macro analysis."""
        logger.info("Running scheduled macro analysis")
        try:
            messages = self.market_intelligence.run_macro_analysis()
            for msg in messages:
                self.market_intelligence.send_message(msg)
            logger.info(f"Macro analysis complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.MARKET_INTELLIGENCE,
                    "Macro analysis failed",
                    str(e)
                )

    def _run_sector_analysis(self) -> None:
        """Run Market Intelligence sector analysis."""
        logger.info("Running scheduled sector analysis")
        try:
            messages = self.market_intelligence.run_sector_analysis()
            for msg in messages:
                self.market_intelligence.send_message(msg)
            logger.info(f"Sector analysis complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Sector analysis failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.MARKET_INTELLIGENCE,
                    "Sector analysis failed",
                    str(e)
                )

    # ============================================================
    # Risk Guardian Agent Methods
    # ============================================================

    def _run_risk_check(self) -> None:
        """Run Risk Guardian risk check."""
        logger.info("Running scheduled risk check")
        try:
            messages = self.risk_guardian.run_risk_check()
            for msg in messages:
                self.risk_guardian.send_message(msg)
            logger.info(f"Risk check complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.RISK_GUARDIAN,
                    "Risk check failed",
                    str(e)
                )

    def _run_drawdown_monitor(self) -> None:
        """Run Risk Guardian drawdown monitor."""
        logger.debug("Running scheduled drawdown monitor")
        try:
            messages = self.risk_guardian.run_drawdown_monitor()
            for msg in messages:
                self.risk_guardian.send_message(msg)
            if messages:
                logger.info(f"Drawdown monitor: {len(messages)} alerts generated")
        except Exception as e:
            logger.error(f"Drawdown monitor failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.RISK_GUARDIAN,
                    "Drawdown monitor failed",
                    str(e)
                )

    def _run_correlation_check(self) -> None:
        """Run Risk Guardian correlation check."""
        logger.info("Running scheduled correlation check")
        try:
            messages = self.risk_guardian.run_correlation_check()
            for msg in messages:
                self.risk_guardian.send_message(msg)
            logger.info(f"Correlation check complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Correlation check failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.RISK_GUARDIAN,
                    "Correlation check failed",
                    str(e)
                )

    def _run_daily_risk_report(self) -> None:
        """Run Risk Guardian daily risk report."""
        logger.info("Running scheduled daily risk report")
        try:
            messages = self.risk_guardian.run_daily_risk_report()
            for msg in messages:
                self.risk_guardian.send_message(msg)
            logger.info(f"Daily risk report complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Daily risk report failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.RISK_GUARDIAN,
                    "Daily risk report failed",
                    str(e)
                )

    # ============================================================
    # Portfolio Strategist Agent Methods
    # ============================================================

    def _run_performance_review(self) -> None:
        """Run Portfolio Strategist performance review."""
        logger.info("Running scheduled performance review")
        try:
            messages = self.portfolio_strategist.run_performance_review()
            for msg in messages:
                self.portfolio_strategist.send_message(msg)
            logger.info(f"Performance review complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Performance review failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.PORTFOLIO_STRATEGIST,
                    "Performance review failed",
                    str(e)
                )

    def _run_rebalancing_check(self) -> None:
        """Run Portfolio Strategist rebalancing check."""
        logger.info("Running scheduled rebalancing check")
        try:
            messages = self.portfolio_strategist.run_rebalancing_check()
            for msg in messages:
                self.portfolio_strategist.send_message(msg)
            logger.info(f"Rebalancing check complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Rebalancing check failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.PORTFOLIO_STRATEGIST,
                    "Rebalancing check failed",
                    str(e)
                )

    def _run_stock_screening(self) -> None:
        """Run Portfolio Strategist stock screening."""
        logger.info("Running scheduled stock screening")
        try:
            messages = self.portfolio_strategist.run_stock_screening()
            for msg in messages:
                self.portfolio_strategist.send_message(msg)
            logger.info(f"Stock screening complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Stock screening failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.PORTFOLIO_STRATEGIST,
                    "Stock screening failed",
                    str(e)
                )

    def _run_portfolio_review(self) -> None:
        """Run Portfolio Strategist portfolio review."""
        logger.info("Running scheduled portfolio review")
        try:
            messages = self.portfolio_strategist.run_portfolio_review()
            for msg in messages:
                self.portfolio_strategist.send_message(msg)
            logger.info(f"Portfolio review complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Portfolio review failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.PORTFOLIO_STRATEGIST,
                    "Portfolio review failed",
                    str(e)
                )

    # ============================================================
    # Operations Agent Methods
    # ============================================================

    def _run_operations_cycle(self) -> None:
        """Run Operations agent message processing cycle."""
        logger.info("Running operations message processing")
        try:
            result = self.operations.run_cycle()
            logger.info(
                f"Operations cycle complete: "
                f"processed={result['messages_processed']}, "
                f"sent={result['messages_sent']}"
            )

            for error in result.get("errors", []):
                logger.error(f"Operations cycle error: {error}")

        except Exception as e:
            logger.error(f"Operations cycle failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.OPERATIONS,
                    "Message processing failed",
                    str(e)
                )

    def _run_execution_quality_check(self) -> None:
        """Run Operations execution quality check."""
        logger.info("Running scheduled execution quality check")
        try:
            messages = self.operations.run_execution_quality_check()
            for msg in messages:
                self.operations.send_message(msg)
            logger.info(f"Execution quality check complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Execution quality check failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.OPERATIONS,
                    "Execution quality check failed",
                    str(e)
                )

    def _run_system_health_check(self) -> None:
        """Run Operations system health check."""
        logger.info("Running scheduled system health check")
        try:
            messages = self.operations.run_system_health_check()
            for msg in messages:
                self.operations.send_message(msg)
            logger.info(f"System health check complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.OPERATIONS,
                    "System health check failed",
                    str(e)
                )

    def _run_degradation_check(self) -> None:
        """Run Operations model degradation check."""
        logger.info("Running scheduled degradation check")
        try:
            messages = self.operations.run_degradation_check()
            for msg in messages:
                self.operations.send_message(msg)
            logger.info(f"Degradation check complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Degradation check failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.OPERATIONS,
                    "Degradation check failed",
                    str(e)
                )

    # ============================================================
    # Manual Trigger Methods
    # ============================================================

    def trigger_news_scan(self) -> None:
        """Manually trigger news scan."""
        self._run_news_scan()

    def trigger_earnings_check(self) -> None:
        """Manually trigger earnings check."""
        self._run_earnings_check()

    def trigger_macro_analysis(self) -> None:
        """Manually trigger macro analysis."""
        self._run_macro_analysis()

    def trigger_sector_analysis(self) -> None:
        """Manually trigger sector analysis."""
        self._run_sector_analysis()

    def trigger_risk_check(self) -> None:
        """Manually trigger risk check."""
        self._run_risk_check()

    def trigger_drawdown_monitor(self) -> None:
        """Manually trigger drawdown monitor."""
        self._run_drawdown_monitor()

    def trigger_correlation_check(self) -> None:
        """Manually trigger correlation check."""
        self._run_correlation_check()

    def trigger_daily_risk_report(self) -> None:
        """Manually trigger daily risk report."""
        self._run_daily_risk_report()

    def trigger_performance_review(self) -> None:
        """Manually trigger performance review."""
        self._run_performance_review()

    def trigger_rebalancing_check(self) -> None:
        """Manually trigger rebalancing check."""
        self._run_rebalancing_check()

    def trigger_stock_screening(self) -> None:
        """Manually trigger stock screening."""
        self._run_stock_screening()

    def trigger_portfolio_review(self) -> None:
        """Manually trigger portfolio review."""
        self._run_portfolio_review()

    def trigger_operations_cycle(self) -> None:
        """Manually trigger operations message processing."""
        self._run_operations_cycle()

    def trigger_execution_quality_check(self) -> None:
        """Manually trigger execution quality check."""
        self._run_execution_quality_check()

    def trigger_system_health_check(self) -> None:
        """Manually trigger system health check."""
        self._run_system_health_check()

    def trigger_degradation_check(self) -> None:
        """Manually trigger degradation check."""
        self._run_degradation_check()

    # ============================================================
    # Status and Utility Methods
    # ============================================================

    def _log_schedule(self) -> None:
        """Log the scheduled jobs."""
        logger.info("Scheduled jobs:")
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            logger.info(f"  - {job.name}: next run at {next_run}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get orchestrator status.

        Returns:
            Dictionary with orchestrator and agent status
        """
        status = {
            "enabled": self.enabled,
            "running": self._is_running,
            "message_queue": self.message_queue.get_stats() if self.enabled else {},
            "llm_available": self.llm_client.is_available() if self.llm_client else False,
            "notifier_enabled": self.notifier.enabled if self.notifier else False,
        }

        if self.enabled and self._is_running:
            status["agents"] = {
                role.value: agent.get_status()
                for role, agent in self.agents.items()
            }

            status["scheduled_jobs"] = [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                }
                for job in self.scheduler.get_jobs()
            ]

        return status

    def get_conversation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent conversation history between agents.

        Args:
            limit: Maximum number of messages to return

        Returns:
            List of message dictionaries
        """
        if not self.enabled:
            return []

        # Get conversations between all agent pairs
        all_messages = []

        # Key agent pairs for conversation history
        pairs = [
            (AgentRole.MARKET_INTELLIGENCE, AgentRole.RISK_GUARDIAN),
            (AgentRole.MARKET_INTELLIGENCE, AgentRole.PORTFOLIO_STRATEGIST),
            (AgentRole.RISK_GUARDIAN, AgentRole.OPERATIONS),
            (AgentRole.RISK_GUARDIAN, AgentRole.PORTFOLIO_STRATEGIST),
            (AgentRole.PORTFOLIO_STRATEGIST, AgentRole.OPERATIONS),
        ]

        for agent1, agent2 in pairs:
            messages = self.message_queue.get_conversation(
                agent1, agent2, limit=limit // len(pairs)
            )
            all_messages.extend([msg.to_dict() for msg in messages])

        # Sort by timestamp and limit
        all_messages.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return all_messages[:limit]

    def cleanup_old_messages(self, days: int = 30) -> int:
        """
        Clean up old messages from the queue.

        Args:
            days: Delete messages older than this many days

        Returns:
            Number of deleted messages
        """
        if not self.enabled:
            return 0

        return self.message_queue.delete_old_messages(days)


# Singleton instance
_orchestrator: Optional[AgentOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_orchestrator(config: Optional[Dict[str, Any]] = None) -> AgentOrchestrator:
    """
    Get or create the singleton AgentOrchestrator instance.

    Args:
        config: Configuration dictionary (required for first call)

    Returns:
        AgentOrchestrator singleton instance
    """
    global _orchestrator
    with _orchestrator_lock:
        if _orchestrator is None:
            if config is None:
                from config.settings import Settings
                config = Settings.load_trading_config()
            _orchestrator = AgentOrchestrator(config)
        return _orchestrator
