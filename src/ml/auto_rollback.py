"""
Auto-rollback manager for model deployments.
Monitors newly deployed models during a grace period and automatically
rolls back to the previous version if degradation is detected.
"""
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


@dataclass
class RollbackEvent:
    """Record of a rollback event."""
    model_type: str
    rolled_back_version: str
    restored_version: str
    reason: str
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


class AutoRollbackManager:
    """
    Manages model deployment grace periods and automatic rollback.

    Flow:
    1. When a new model is deployed, record previous_production in registry
    2. During grace period (N days), degradation checks can trigger rollback
    3. If degradation detected during grace period, rollback to previous version
    4. After grace period passes, clear the grace period markers
    """

    def __init__(
        self,
        grace_period_days: int = 5,
        enabled: bool = True
    ):
        self.grace_period_days = grace_period_days
        self.enabled = enabled
        self.registry_path = MODELS_DIR / "versions" / "registry.json"
        self.versions_dir = MODELS_DIR / "versions"

    def register_deployment(
        self,
        model_type: str,
        new_version: str,
        previous_version: Optional[str] = None
    ) -> None:
        """
        Register a new deployment for grace-period monitoring.

        Args:
            model_type: Model type (xgboost, lstm, cnn).
            new_version: Version string of newly deployed model.
            previous_version: Version string of the model being replaced.
        """
        if not self.enabled:
            return

        registry = self._load_registry()
        production = registry.get("production", {}).get(model_type, {})

        # Determine previous version
        if previous_version is None:
            previous_version = production.get("version")

        if previous_version is None or previous_version == new_version:
            logger.debug(f"No previous version to track for {model_type}")
            return

        # Set grace period timestamps
        now = datetime.now()
        grace_end = now + timedelta(days=self.grace_period_days)

        # Update production entry with grace period info
        if "production" not in registry:
            registry["production"] = {}
        if model_type not in registry["production"]:
            registry["production"][model_type] = {}

        registry["production"][model_type]["previous_production"] = previous_version
        registry["production"][model_type]["grace_period_start"] = now.isoformat()
        registry["production"][model_type]["grace_period_end"] = grace_end.isoformat()

        self._save_registry(registry)
        logger.info(
            f"Registered deployment for {model_type}: {new_version} "
            f"(grace period until {grace_end.strftime('%Y-%m-%d %H:%M')})"
        )

    def check_grace_period(self, model_type: str) -> Optional[RollbackEvent]:
        """
        Check if a model's grace period has expired cleanly.
        If expired without issues, clear the grace period markers.

        Returns:
            None (grace period management only, no rollback triggered here).
        """
        if not self.enabled:
            return None

        registry = self._load_registry()
        production = registry.get("production", {}).get(model_type, {})

        grace_end_str = production.get("grace_period_end")
        if not grace_end_str:
            return None

        grace_end = datetime.fromisoformat(grace_end_str)
        if datetime.now() >= grace_end:
            # Grace period expired - model passed
            self.clear_grace_period(model_type)
            logger.info(f"{model_type} passed grace period successfully")

        return None

    def check_all_grace_periods(self) -> None:
        """Check all models for expired grace periods and clear them."""
        if not self.enabled:
            return

        registry = self._load_registry()
        production = registry.get("production", {})

        for model_type in list(production.keys()):
            self.check_grace_period(model_type)

    def rollback(self, model_type: str, reason: str = "") -> Optional[RollbackEvent]:
        """
        Roll back a model to its previous production version.

        Args:
            model_type: Model type to rollback.
            reason: Reason for rollback.

        Returns:
            RollbackEvent if rollback succeeded, None otherwise.
        """
        registry = self._load_registry()
        production = registry.get("production", {}).get(model_type, {})

        previous_version = production.get("previous_production")
        current_version = production.get("version")

        if not previous_version:
            logger.warning(f"No previous version to rollback to for {model_type}")
            return None

        # Verify previous version exists
        prev_version_dir = self.versions_dir / previous_version
        if not prev_version_dir.exists():
            logger.error(f"Previous version directory not found: {prev_version_dir}")
            return None

        logger.info(
            f"Rolling back {model_type}: {current_version} -> {previous_version}"
        )

        # Restore model files
        try:
            self._restore_model_files(model_type, previous_version)
        except Exception as e:
            logger.error(f"Failed to restore model files: {e}")
            return None

        # Get previous version metrics from registry
        prev_metrics = registry.get("versions", {}).get(
            previous_version, {}
        ).get("metrics", {})

        # Update registry
        registry["production"][model_type] = {
            "version": previous_version,
            "deployed_at": datetime.now().isoformat(),
            "metrics": prev_metrics,
            "rolled_back_from": current_version
        }

        # Log rollback event
        event = RollbackEvent(
            model_type=model_type,
            rolled_back_version=current_version,
            restored_version=previous_version,
            reason=reason,
            timestamp=datetime.now().isoformat()
        )

        if "rollback_history" not in registry:
            registry["rollback_history"] = []
        registry["rollback_history"].append(event.to_dict())
        registry["rollback_history"] = registry["rollback_history"][-50:]

        self._save_registry(registry)

        logger.info(
            f"Rollback complete: {model_type} restored to {previous_version}"
        )
        return event

    def is_in_grace_period(self, model_type: str) -> bool:
        """Check if a model is currently in its deployment grace period."""
        registry = self._load_registry()
        production = registry.get("production", {}).get(model_type, {})

        grace_end_str = production.get("grace_period_end")
        if not grace_end_str:
            return False

        return datetime.now() < datetime.fromisoformat(grace_end_str)

    def clear_grace_period(self, model_type: str) -> None:
        """Clear grace period markers after it expires without issues."""
        registry = self._load_registry()
        production = registry.get("production", {}).get(model_type, {})

        for key in ["previous_production", "grace_period_start", "grace_period_end"]:
            production.pop(key, None)

        self._save_registry(registry)
        logger.debug(f"Cleared grace period for {model_type}")

    def get_rollback_history(self, limit: int = 20) -> List[Dict]:
        """Get history of rollback events."""
        registry = self._load_registry()
        history = registry.get("rollback_history", [])
        return history[-limit:][::-1]

    def _restore_model_files(self, model_type: str, version: str) -> None:
        """Copy versioned model files to production location."""
        version_dir = self.versions_dir / version
        production_names = {
            "xgboost": "trading_model",
            "lstm": "lstm_trading_model",
            "cnn": "cnn_trading_model"
        }
        production_name = production_names.get(model_type, model_type)

        if model_type == "xgboost":
            # Find the pkl file in version directory
            src = version_dir / "model.pkl"
            if not src.exists():
                # Try directory format
                model_dir = version_dir / "model"
                if model_dir.is_dir():
                    for f in model_dir.glob("*.pkl"):
                        src = f
                        break
            if src.exists():
                dst = MODELS_DIR / f"{production_name}.pkl"
                shutil.copy(src, dst)
            else:
                raise FileNotFoundError(f"No pkl file found in {version_dir}")
        else:
            # LSTM/CNN - copy .keras and metadata
            dst_dir = MODELS_DIR / production_name
            dst_dir.mkdir(exist_ok=True)

            src_model = version_dir / "model.keras"
            if src_model.exists():
                shutil.copy(src_model, dst_dir / "model.keras")

            src_meta = version_dir / "metadata.json"
            if src_meta.exists():
                shutil.copy(src_meta, dst_dir / "metadata.json")

            if not src_model.exists():
                raise FileNotFoundError(f"No model.keras found in {version_dir}")

    def _load_registry(self) -> Dict:
        """Load version registry."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {}

    def _save_registry(self, registry: Dict) -> None:
        """Save version registry atomically using tempfile + os.replace."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=self.registry_path.parent, suffix=".json.tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(registry, f, indent=2)
            os.replace(tmp_path, self.registry_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def get_auto_rollback_manager(config: Optional[Dict] = None) -> AutoRollbackManager:
    """Factory function to create AutoRollbackManager from config."""
    if config is None:
        config = Settings.load_trading_config()

    retraining_config = config.get("retraining", {})
    rollback_config = retraining_config.get("auto_rollback", {})

    return AutoRollbackManager(
        grace_period_days=rollback_config.get("grace_period_days", 5),
        enabled=rollback_config.get("enabled", False)
    )
