"""
Scheduled model retraining using APScheduler.
Runs alongside the trading bot to automatically retrain models on a schedule.
Supports degradation detection, walk-forward validation, and auto-rollback (Phase 18).
"""
import logging
import os
import tempfile
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

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

    Phase 18 additions:
    - Degradation detection: periodic checks on production model health
    - Walk-forward validation: multi-metric validation before deployment
    - Auto-rollback: grace period monitoring with automatic rollback
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        schedule: str = "weekly",
        day_of_week: str = "sun",
        hour: int = 2,
        auto_deploy: bool = True,
        min_improvement: float = 0.01,
        enabled: bool = True,
        # Phase 18 parameters
        degradation_check_enabled: bool = False,
        degradation_check_interval_hours: int = 12,
        auto_rollback_enabled: bool = False,
        rollback_grace_period_days: int = 5,
        use_walk_forward_validation: bool = False,
        walk_forward_config: Optional[Dict] = None
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
            degradation_check_enabled: Whether to run periodic degradation checks.
            degradation_check_interval_hours: Hours between degradation checks.
            auto_rollback_enabled: Whether to auto-rollback bad deployments.
            rollback_grace_period_days: Days to monitor after deployment.
            use_walk_forward_validation: Whether to validate before deployment.
            walk_forward_config: Config dict for walk-forward validation parameters.
        """
        self.enabled = enabled
        self.schedule = schedule
        self.day_of_week = day_of_week
        self.hour = hour
        self.auto_deploy = auto_deploy
        self.min_improvement = min_improvement

        # Phase 18 settings
        self.degradation_check_enabled = degradation_check_enabled
        self.degradation_check_interval_hours = degradation_check_interval_hours
        self.auto_rollback_enabled = auto_rollback_enabled
        self.rollback_grace_period_days = rollback_grace_period_days
        self.use_walk_forward_validation = use_walk_forward_validation
        self.walk_forward_config = walk_forward_config or {}

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
        self._is_retraining = False
        self._retrain_lock = threading.Lock()
        self._last_retrain: Optional[datetime] = None
        self._last_result: Optional[Dict] = None

        # Retraining log file
        self.log_file = MODELS_DIR / "retraining_log.json"

        # Phase 18: Initialize degradation monitor and rollback manager
        self._degradation_monitor = None
        self._rollback_manager = None

        if self.degradation_check_enabled:
            from src.ml.degradation_monitor import get_degradation_monitor
            self._degradation_monitor = get_degradation_monitor(config)

        if self.auto_rollback_enabled:
            from src.ml.auto_rollback import AutoRollbackManager
            self._rollback_manager = AutoRollbackManager(
                grace_period_days=self.rollback_grace_period_days,
                enabled=True
            )

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

        # Phase 18: Add degradation check job
        if self.degradation_check_enabled and self._degradation_monitor:
            self.scheduler.add_job(
                self.run_degradation_check,
                trigger=IntervalTrigger(hours=self.degradation_check_interval_hours),
                id="degradation_check",
                name="Model Degradation Check",
                replace_existing=True
            )
            logger.info(
                f"Degradation check enabled (every {self.degradation_check_interval_hours}h)"
            )

        # Phase 18: Add grace period check job (daily)
        if self.auto_rollback_enabled and self._rollback_manager:
            self.scheduler.add_job(
                self.run_grace_period_check,
                trigger=CronTrigger(hour=(self.hour + 1) % 24),
                id="grace_period_check",
                name="Grace Period Check",
                replace_existing=True
            )
            logger.info("Auto-rollback grace period check enabled (daily)")

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

    def run_retrain(self, trigger_reason: str = "scheduled") -> Dict:
        """
        Execute the retraining pipeline.

        Args:
            trigger_reason: Why retraining was triggered ("scheduled", "degradation", "manual").

        Returns:
            Dict with retraining results.
        """
        # Prevent concurrent retraining
        if not self._retrain_lock.acquire(blocking=False):
            logger.warning("Retraining already in progress, skipping")
            return {"skipped": True, "reason": "already_running"}

        try:
            self._is_retraining = True
            return self._execute_retrain(trigger_reason)
        finally:
            self._is_retraining = False
            self._retrain_lock.release()

    def _execute_retrain(self, trigger_reason: str) -> Dict:
        """Internal retraining execution."""
        logger.info("=" * 60)
        logger.info(f"SCHEDULED RETRAINING STARTED (reason: {trigger_reason})")
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
            "trigger_reason": trigger_reason,
            "symbols": self.symbols,
            "models": {},
            "deployments": {},
            "walk_forward_validations": {},
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
                    if self.use_walk_forward_validation:
                        # Phase 18: Enhanced deployment with walk-forward validation
                        deployed = self.pipeline.compare_and_deploy_enhanced(
                            model_type=model_type,
                            new_version=version,
                            production_name=production_names[model_type],
                            new_model=model_result.get("model"),
                            X=model_result.get("X"),
                            y=model_result.get("y"),
                            use_walk_forward=True,
                            walk_forward_config=self.walk_forward_config,
                            auto_rollback_manager=self._rollback_manager
                        )
                    else:
                        # Original deployment logic
                        deployed = self.pipeline.compare_and_deploy(
                            model_type=model_type,
                            new_version=version,
                            production_name=production_names[model_type]
                        )
                        # Register with rollback manager if enabled
                        if deployed and self._rollback_manager:
                            self._rollback_manager.register_deployment(
                                model_type, version
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
        logger.info("SCHEDULED RETRAINING COMPLETED")
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

    def run_degradation_check(self) -> Dict:
        """
        Run degradation checks on all production models.
        If degradation detected, trigger proactive retraining or rollback.

        Returns:
            Dict with check results per model.
        """
        if not self._degradation_monitor:
            return {}

        logger.info("Running scheduled degradation check")
        results = {}

        try:
            reports = self._degradation_monitor.check_all_models()

            for model_type, report in reports.items():
                results[model_type] = {
                    "is_degraded": report.is_degraded,
                    "recommendation": report.recommendation,
                    "reasons": report.degradation_reasons
                }

                if report.is_degraded:
                    self._handle_degradation(report)

        except Exception as e:
            logger.error(f"Degradation check failed: {e}")
            results["error"] = str(e)

        return results

    def _handle_degradation(self, report) -> None:
        """Handle a degradation report - trigger retraining or rollback."""
        logger.warning(
            f"Degradation detected for {report.model_type}: "
            f"{', '.join(report.degradation_reasons)}"
        )

        if report.recommendation == "rollback" and self._rollback_manager:
            # Model is in grace period - rollback
            reason = f"Degradation during grace period: {'; '.join(report.degradation_reasons)}"
            event = self._rollback_manager.rollback(report.model_type, reason=reason)
            if event:
                logger.info(
                    f"Auto-rollback executed: {report.model_type} "
                    f"restored to {event.restored_version}"
                )
                notifier = _get_notifier()
                if notifier:
                    notifier.notify_retraining(
                        status="rollback",
                        models={report.model_type: {"rolled_back": True}}
                    )
        elif report.recommendation == "retrain":
            # Trigger proactive retraining
            logger.info(f"Triggering proactive retraining for {report.model_type}")
            if not self._is_retraining:
                self.run_retrain(trigger_reason="degradation")

    def run_grace_period_check(self) -> None:
        """Check all models in grace periods and clear expired ones."""
        if not self._rollback_manager:
            return

        logger.debug("Running grace period check")
        self._rollback_manager.check_all_grace_periods()

    def _log_result(self, result: Dict) -> None:
        """Append result to retraining log file atomically."""
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

            # Save atomically using tempfile + os.replace
            log_dir = self.log_file.parent
            log_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=log_dir, suffix=".json.tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(log, f, indent=2)
                os.replace(tmp_path, self.log_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise

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
            "is_retraining": self._is_retraining,
            "schedule": self.schedule,
            "day_of_week": self.day_of_week if self.schedule == "weekly" else None,
            "hour": self.hour,
            "auto_deploy": self.auto_deploy,
            "min_improvement": self.min_improvement,
            "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
            "next_run": None,
            # Phase 18 status
            "degradation_check_enabled": self.degradation_check_enabled,
            "auto_rollback_enabled": self.auto_rollback_enabled,
            "walk_forward_validation": self.use_walk_forward_validation
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
        return self.run_retrain(trigger_reason="manual")


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

    # Phase 18 config sections
    deg_config = retraining_config.get("degradation_detection", {})
    rollback_config = retraining_config.get("auto_rollback", {})
    wf_config = retraining_config.get("walk_forward_validation", {})

    return ScheduledRetrainer(
        symbols=trading_config.get("symbols"),
        schedule=retraining_config.get("schedule", "weekly"),
        day_of_week=retraining_config.get("day_of_week", "sun"),
        hour=retraining_config.get("hour", 2),
        auto_deploy=retraining_config.get("auto_deploy", True),
        min_improvement=retraining_config.get("min_improvement", 0.01),
        enabled=retraining_config.get("enabled", True),
        # Phase 18
        degradation_check_enabled=deg_config.get("enabled", False),
        degradation_check_interval_hours=deg_config.get("check_interval_hours", 12),
        auto_rollback_enabled=rollback_config.get("enabled", False),
        rollback_grace_period_days=rollback_config.get("grace_period_days", 5),
        use_walk_forward_validation=wf_config.get("enabled", False),
        walk_forward_config=wf_config
    )
