"""
Agent Orchestrator

APScheduler-based coordinator for the multi-agent system.
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
from .stock_analyst import StockAnalystAgent
from .developer_agent import DeveloperAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Coordinates the multi-agent system using APScheduler.

    Schedules:
    - Stock Analyst:
      - Health check: Every 4 hours
      - Degradation check: Every 12 hours
      - Daily review: 4:30 PM daily
      - Stock screening: Weekly (Sunday 6PM)
      - Portfolio review: Daily (after market close)
      - Market analysis: Every 4 hours

    - Developer:
      - Process messages: Every 30 minutes
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

        logger.info("Agent orchestrator initialized")

    def _init_agents(self) -> None:
        """Initialize agent instances."""
        # Stock Analyst Agent
        self.stock_analyst = StockAnalystAgent(
            config=self.agents_config,
            message_queue=self.message_queue,
            notifier=self.notifier,
            llm_client=self.llm_client,
        )

        # Developer Agent
        self.developer = DeveloperAgent(
            config=self.agents_config,
            message_queue=self.message_queue,
            notifier=self.notifier,
            llm_client=self.llm_client,
        )

        self.agents = {
            AgentRole.STOCK_ANALYST: self.stock_analyst,
            AgentRole.DEVELOPER: self.developer,
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
            # Get schedule configuration
            analyst_config = self.agents_config.get("stock_analyst", {})
            developer_config = self.agents_config.get("developer", {})

            # Stock Analyst: Health check every N hours
            health_check_hours = analyst_config.get("health_check_hours", 4)
            self.scheduler.add_job(
                self._run_health_check,
                trigger=IntervalTrigger(hours=health_check_hours),
                id="analyst_health_check",
                name="Stock Analyst Health Check",
                replace_existing=True,
            )

            # Stock Analyst: Degradation check every N hours
            degradation_check_hours = analyst_config.get("degradation_check_hours", 12)
            self.scheduler.add_job(
                self._run_degradation_check,
                trigger=IntervalTrigger(hours=degradation_check_hours),
                id="analyst_degradation_check",
                name="Stock Analyst Degradation Check",
                replace_existing=True,
            )

            # Stock Analyst: Daily review at specified time
            daily_review_time = analyst_config.get("daily_review_time", "16:30")
            hour, minute = map(int, daily_review_time.split(":"))
            self.scheduler.add_job(
                self._run_daily_review,
                trigger=CronTrigger(hour=hour, minute=minute),
                id="analyst_daily_review",
                name="Stock Analyst Daily Review",
                replace_existing=True,
            )

            # Developer: Process messages every 30 minutes
            process_interval_minutes = developer_config.get("process_interval_minutes", 30)
            self.scheduler.add_job(
                self._run_developer_cycle,
                trigger=IntervalTrigger(minutes=process_interval_minutes),
                id="developer_process_messages",
                name="Developer Process Messages",
                replace_existing=True,
            )

            # Stock Analyst: Weekly stock screening (if dynamic symbols enabled)
            dynamic_symbols_config = self.config.get("dynamic_symbols", {})
            if dynamic_symbols_config.get("enabled", False):
                screening_config = dynamic_symbols_config.get("screening", {})
                screening_day = screening_config.get("refresh_day", "sunday")
                screening_time = screening_config.get("refresh_time", "18:00")
                screening_hour, screening_minute = map(int, screening_time.split(":"))

                # Map day names to cron day of week
                day_map = {
                    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                    "friday": 4, "saturday": 5, "sunday": 6
                }
                screening_dow = day_map.get(screening_day.lower(), 6)

                self.scheduler.add_job(
                    self._run_stock_screening,
                    trigger=CronTrigger(
                        day_of_week=screening_dow,
                        hour=screening_hour,
                        minute=screening_minute
                    ),
                    id="analyst_stock_screening",
                    name="Stock Analyst Weekly Screening",
                    replace_existing=True,
                )

                # Stock Analyst: Daily portfolio review (after market close)
                portfolio_review_hours = analyst_config.get("portfolio_review_hours", 24)
                self.scheduler.add_job(
                    self._run_portfolio_review,
                    trigger=IntervalTrigger(hours=portfolio_review_hours),
                    id="analyst_portfolio_review",
                    name="Stock Analyst Portfolio Review",
                    replace_existing=True,
                )

                # Stock Analyst: Market analysis every N hours
                market_analysis_hours = analyst_config.get("market_analysis_hours", 4)
                self.scheduler.add_job(
                    self._run_market_analysis,
                    trigger=IntervalTrigger(hours=market_analysis_hours),
                    id="analyst_market_analysis",
                    name="Stock Analyst Market Analysis",
                    replace_existing=True,
                )

                logger.info("Dynamic symbols enabled - added screening, review, and market analysis schedules")

            # Start scheduler
            self.scheduler.start()
            self._is_running = True

            # Notify startup
            if self.notifier:
                self.notifier.notify_agent_startup([
                    "Stock Analyst",
                    "Developer",
                ])

            logger.info("Agent orchestrator started")
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

    def _run_health_check(self) -> None:
        """Run Stock Analyst health check."""
        logger.info("Running scheduled health check")
        try:
            messages = self.stock_analyst.run_health_check()
            for msg in messages:
                self.stock_analyst.send_message(msg)
            logger.info(f"Health check complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.STOCK_ANALYST,
                    "Health check failed",
                    str(e)
                )

    def _run_degradation_check(self) -> None:
        """Run Stock Analyst degradation check."""
        logger.info("Running scheduled degradation check")
        try:
            messages = self.stock_analyst.run_degradation_check()
            for msg in messages:
                self.stock_analyst.send_message(msg)
            logger.info(f"Degradation check complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Degradation check failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.STOCK_ANALYST,
                    "Degradation check failed",
                    str(e)
                )

    def _run_daily_review(self) -> None:
        """Run Stock Analyst daily review."""
        logger.info("Running scheduled daily review")
        try:
            messages = self.stock_analyst.run_daily_review()
            for msg in messages:
                self.stock_analyst.send_message(msg)
            logger.info(f"Daily review complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Daily review failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.STOCK_ANALYST,
                    "Daily review failed",
                    str(e)
                )

    def _run_developer_cycle(self) -> None:
        """Run Developer agent message processing cycle."""
        logger.info("Running developer message processing")
        try:
            result = self.developer.run_cycle()
            logger.info(
                f"Developer cycle complete: "
                f"processed={result['messages_processed']}, "
                f"sent={result['messages_sent']}"
            )

            # Log any errors
            for error in result.get("errors", []):
                logger.error(f"Developer cycle error: {error}")

        except Exception as e:
            logger.error(f"Developer cycle failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.DEVELOPER,
                    "Message processing failed",
                    str(e)
                )

    def _run_stock_screening(self) -> None:
        """Run Stock Analyst weekly stock screening."""
        logger.info("Running scheduled stock screening")
        try:
            messages = self.stock_analyst.run_stock_screening()
            for msg in messages:
                self.stock_analyst.send_message(msg)
            logger.info(f"Stock screening complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Stock screening failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.STOCK_ANALYST,
                    "Stock screening failed",
                    str(e)
                )

    def _run_portfolio_review(self) -> None:
        """Run Stock Analyst portfolio review."""
        logger.info("Running scheduled portfolio review")
        try:
            messages = self.stock_analyst.run_portfolio_review()
            for msg in messages:
                self.stock_analyst.send_message(msg)
            logger.info(f"Portfolio review complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Portfolio review failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.STOCK_ANALYST,
                    "Portfolio review failed",
                    str(e)
                )

    def _run_market_analysis(self) -> None:
        """Run Stock Analyst market timing analysis."""
        logger.info("Running scheduled market analysis")
        try:
            messages = self.stock_analyst.run_market_analysis()
            for msg in messages:
                self.stock_analyst.send_message(msg)
            logger.info(f"Market analysis complete: {len(messages)} messages generated")
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            if self.notifier:
                self.notifier.notify_agent_error(
                    AgentRole.STOCK_ANALYST,
                    "Market analysis failed",
                    str(e)
                )

    def trigger_health_check(self) -> None:
        """Manually trigger a health check."""
        self._run_health_check()

    def trigger_degradation_check(self) -> None:
        """Manually trigger a degradation check."""
        self._run_degradation_check()

    def trigger_daily_review(self) -> None:
        """Manually trigger a daily review."""
        self._run_daily_review()

    def trigger_developer_cycle(self) -> None:
        """Manually trigger developer message processing."""
        self._run_developer_cycle()

    def trigger_stock_screening(self) -> None:
        """Manually trigger stock screening."""
        self._run_stock_screening()

    def trigger_portfolio_review(self) -> None:
        """Manually trigger portfolio review."""
        self._run_portfolio_review()

    def trigger_market_analysis(self) -> None:
        """Manually trigger market analysis."""
        self._run_market_analysis()

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

        messages = self.message_queue.get_conversation(
            AgentRole.STOCK_ANALYST,
            AgentRole.DEVELOPER,
            limit=limit,
        )

        return [msg.to_dict() for msg in messages]

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
