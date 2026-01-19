"""
Scheduled model retraining using APScheduler.
Runs alongside the trading bot to automatically retrain models on a schedule.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import Settings, MODELS_DIR
from src.ml.retraining import RetrainingPipeline

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependency
def _get_notifier():
    """Get notifier instance (lazy load to avoid circular import)."""
    try:
        from src.notifications.notifier_manager import get_notifier
        return get_notifier()
    except Exception:
        return None


class ScheduledRetrainer:
    """
    Manages scheduled model retraining.

    Integrates with APScheduler to run retraining on a configurable schedule.
    Logs results and auto-deploys models that show improvement.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        schedule: str = "weekly",
        day_of_week: str = "sun",
        hour: int = 2,
        auto_deploy: bool = True,
        min_improvement: float = 0.01,
        enabled: bool = True
    ):
        """
        Initialize the scheduled retrainer.

        Args:
            symbols: List of symbols to train on (default: from config).
            schedule: Schedule type: "weekly", "daily", or "monthly".
            day_of_week: Day for weekly schedule (mon, tue, wed, thu, fri, sat, sun).
            hour: Hour to run (0-23).
            auto_deploy: Whether to auto-deploy improved models.
            min_improvement: Minimum accuracy improvement to deploy (default: 1%).
            enabled: Whether scheduled retraining is enabled.
        """
        self.enabled = enabled
        self.schedule = schedule
        self.day_of_week = day_of_week
        self.hour = hour
        self.auto_deploy = auto_deploy
        self.min_improvement = min_improvement

        # Load config for symbols if not provided
        config = Settings.load_trading_config()
        self.symbols = symbols or config.get("trading", {}).get("symbols", [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"
        ])

        # Initialize pipeline
        self.pipeline = RetrainingPipeline(
            symbols=self.symbols,
            min_improvement_threshold=min_improvement
        )

        # Initialize scheduler
        self.scheduler = BackgroundScheduler()
        self._is_running = False
        self._last_retrain: Optional[datetime] = None
        self._last_result: Optional[Dict] = None

        # Retraining log file
        self.log_file = MODELS_DIR / "retraining_log.json"

    def start(self) -> None:
        """Start the scheduler."""
        if not self.enabled:
            logger.info("Scheduled retraining is disabled")
            return

        if self._is_running:
            logger.warning("Scheduler already running")
            return

        # Create cron trigger based on schedule type
        trigger = self._create_trigger()

        # Add the retraining job
        self.scheduler.add_job(
            self.run_retrain,
            trigger=trigger,
            id="model_retraining",
            name="Scheduled Model Retraining",
            replace_existing=True
        )

        self.scheduler.start()
        self._is_running = True

        next_run = self.scheduler.get_job("model_retraining").next_run_time
        logger.info(f"Scheduled retraining started. Next run: {next_run}")

    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self._is_running:
            self.scheduler.shutdown(wait=True)
            self._is_running = False
            logger.info("Scheduled retraining stopped")

    def _create_trigger(self) -> CronTrigger:
        """Create cron trigger based on schedule type."""
        if self.schedule == "daily":
            return CronTrigger(hour=self.hour)
        elif self.schedule == "weekly":
            return CronTrigger(day_of_week=self.day_of_week, hour=self.hour)
        elif self.schedule == "monthly":
            return CronTrigger(day=1, hour=self.hour)
        else:
            # Default to weekly
            return CronTrigger(day_of_week="sun", hour=2)

    def run_retrain(self) -> Dict:
        """
        Execute the retraining pipeline.

        Returns:
            Dict with retraining results.
        """
        logger.info("=" * 60)
        logger.info("SCHEDULED RETRAINING STARTED")
        logger.info(f"Time: {datetime.now().isoformat()}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info("=" * 60)

        # Send start notification
        notifier = _get_notifier()
        if notifier:
            notifier.notify_retraining(status="started")

        start_time = datetime.now()
        result = {
            "started_at": start_time.isoformat(),
            "symbols": self.symbols,
            "models": {},
            "deployments": {},
            "errors": []
        }

        try:
            # Retrain all models
            retrain_results = self.pipeline.retrain_all(
                use_cache=False,  # Always fetch fresh data
                models_to_train=["xgboost", "lstm", "cnn"]
            )

            # Process results for each model
            production_names = {
                "xgboost": "trading_model",
                "lstm": "lstm_trading_model",
                "cnn": "cnn_trading_model"
            }

            for model_type, model_result in retrain_results.items():
                if "error" in model_result:
                    result["errors"].append({
                        "model": model_type,
                        "error": model_result["error"]
                    })
                    logger.error(f"Failed to retrain {model_type}: {model_result['error']}")
                    continue

                version = model_result["version"]
                metrics = model_result["metrics"]

                result["models"][model_type] = {
                    "version": version,
                    "accuracy": metrics.get("accuracy", 0),
                    "f1": metrics.get("f1", 0)
                }

                logger.info(f"{model_type.upper()}: accuracy={metrics.get('accuracy', 0):.4f}")

                # Auto-deploy if enabled
                if self.auto_deploy:
                    deployed = self.pipeline.compare_and_deploy(
                        model_type=model_type,
                        new_version=version,
                        production_name=production_names[model_type]
                    )
                    result["deployments"][model_type] = deployed

                    if deployed:
                        logger.info(f"DEPLOYED: {model_type} -> {version}")
                    else:
                        logger.info(f"KEPT CURRENT: {model_type} (no improvement)")

        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            result["errors"].append({"pipeline": str(e)})

        # Finalize result
        end_time = datetime.now()
        result["completed_at"] = end_time.isoformat()
        result["duration_seconds"] = (end_time - start_time).total_seconds()

        self._last_retrain = end_time
        self._last_result = result

        # Log to file
        self._log_result(result)

        logger.info("=" * 60)
        logger.info(f"SCHEDULED RETRAINING COMPLETED")
        logger.info(f"Duration: {result['duration_seconds']:.1f} seconds")
        logger.info(f"Deployments: {sum(result['deployments'].values())} / {len(result['models'])}")
        logger.info("=" * 60)

        # Send completion notification
        if notifier:
            notifier.notify_retraining(
                status="completed",
                models=result.get("models"),
                deployments=result.get("deployments"),
                duration_seconds=result.get("duration_seconds")
            )

        return result

    def _log_result(self, result: Dict) -> None:
        """Append result to retraining log file."""
        import json

        try:
            # Load existing log
            if self.log_file.exists():
                with open(self.log_file, "r") as f:
                    log = json.load(f)
            else:
                log = {"history": []}

            # Append new result
            log["history"].append(result)

            # Keep last 100 entries
            log["history"] = log["history"][-100:]

            # Save
            with open(self.log_file, "w") as f:
                json.dump(log, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to log retraining result: {e}")

    def get_status(self) -> Dict:
        """
        Get current scheduler status.

        Returns:
            Dict with status info.
        """
        status = {
            "enabled": self.enabled,
            "is_running": self._is_running,
            "schedule": self.schedule,
            "day_of_week": self.day_of_week if self.schedule == "weekly" else None,
            "hour": self.hour,
            "auto_deploy": self.auto_deploy,
            "min_improvement": self.min_improvement,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "next_run": None
        }

        if self._is_running:
            job = self.scheduler.get_job("model_retraining")
            if job and job.next_run_time:
                status["next_run"] = job.next_run_time.isoformat()

        return status

    def get_last_result(self) -> Optional[Dict]:
        """Get the result of the last retraining run."""
        return self._last_result

    def get_retraining_history(self, limit: int = 10) -> List[Dict]:
        """
        Get retraining history from log file.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of retraining results (most recent first).
        """
        import json

        if not self.log_file.exists():
            return []

        try:
            with open(self.log_file, "r") as f:
                log = json.load(f)
            return log.get("history", [])[-limit:][::-1]
        except Exception:
            return []

    def trigger_retrain_now(self) -> Dict:
        """
        Manually trigger retraining immediately.

        Returns:
            Retraining result.
        """
        logger.info("Manual retraining triggered")
        return self.run_retrain()


def get_scheduled_retrainer(config: Optional[Dict] = None) -> ScheduledRetrainer:
    """
    Factory function to create ScheduledRetrainer from config.

    Args:
        config: Optional config dict. If None, loads from trading_config.yaml.

    Returns:
        Configured ScheduledRetrainer instance.
    """
    if config is None:
        config = Settings.load_trading_config()

    retraining_config = config.get("retraining", {})
    trading_config = config.get("trading", {})

    return ScheduledRetrainer(
        symbols=trading_config.get("symbols"),
        schedule=retraining_config.get("schedule", "weekly"),
        day_of_week=retraining_config.get("day_of_week", "sun"),
        hour=retraining_config.get("hour", 2),
        auto_deploy=retraining_config.get("auto_deploy", True),
        min_improvement=retraining_config.get("min_improvement", 0.01),
        enabled=retraining_config.get("enabled", True)
    )
