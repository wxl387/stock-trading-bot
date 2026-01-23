"""Tests for Phase 18: degradation detection, auto-rollback, and walk-forward validation."""
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.degradation_monitor import DegradationMonitor, DegradationReport
from src.ml.auto_rollback import AutoRollbackManager, RollbackEvent


class TestDegradationMonitor:
    """Tests for DegradationMonitor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = DegradationMonitor(
            symbols=["AAPL"],
            evaluation_window_days=30,
            accuracy_drop_threshold=0.05,
            confidence_collapse_threshold=0.55,
            sharpe_decline_threshold=-0.5,
            min_win_rate=0.40,
            enabled=True
        )

    def test_initialization(self):
        """Test DegradationMonitor initialization."""
        assert self.monitor.evaluation_window_days == 30
        assert self.monitor.accuracy_drop_threshold == 0.05
        assert self.monitor.confidence_collapse_threshold == 0.55
        assert self.monitor.enabled is True

    def test_confidence_collapse_detection(self):
        """Test that probabilities near 0.5 are detected as collapse."""
        # Probabilities clustered around 0.5 (low confidence)
        probs = np.random.normal(0.5, 0.02, size=100)
        probs = np.clip(probs, 0, 1)
        dist = self.monitor._analyze_confidence(probs)

        assert dist["mean"] < 0.55
        assert dist["pct_near_0_5"] > 0.5  # Most predictions near 0.5

    def test_no_collapse_with_confident_predictions(self):
        """Test that high-confidence predictions pass the check."""
        # Mix of high and low confidence predictions
        probs = np.concatenate([
            np.random.uniform(0.7, 0.95, 50),
            np.random.uniform(0.05, 0.3, 50)
        ])
        dist = self.monitor._analyze_confidence(probs)

        assert dist["pct_near_0_5"] < 0.2
        assert dist["pct_above_0_6"] > 0.3

    def test_accuracy_drop_detection(self):
        """Test detection when accuracy drops below threshold."""
        current_metrics = {"accuracy": 0.45, "win_rate": 0.50, "sharpe_ratio": 0.5}
        baseline_metrics = {"accuracy": 0.55}  # 10% drop

        is_degraded, reasons = self.monitor._check_degradation_signals(
            current_metrics, baseline_metrics, None
        )

        assert any("Accuracy dropped" in r for r in reasons)

    def test_no_accuracy_degradation_within_threshold(self):
        """Test that small accuracy drops don't trigger degradation."""
        current_metrics = {"accuracy": 0.52, "win_rate": 0.50, "sharpe_ratio": 0.5}
        baseline_metrics = {"accuracy": 0.54}  # Only 2% drop

        is_degraded, reasons = self.monitor._check_degradation_signals(
            current_metrics, baseline_metrics, None
        )

        assert not any("Accuracy dropped" in r for r in reasons)

    def test_sharpe_decline_detection(self):
        """Test Sharpe ratio degradation detection."""
        current_metrics = {
            "accuracy": 0.55,
            "win_rate": 0.50,
            "sharpe_ratio": -0.8  # Below threshold of -0.5
        }
        baseline_metrics = {"accuracy": 0.55}

        is_degraded, reasons = self.monitor._check_degradation_signals(
            current_metrics, baseline_metrics, None
        )

        assert any("Sharpe ratio degraded" in r for r in reasons)

    def test_win_rate_degradation(self):
        """Test win rate below minimum threshold."""
        current_metrics = {
            "accuracy": 0.55,
            "win_rate": 0.35,  # Below 0.40 threshold
            "sharpe_ratio": 0.5
        }
        baseline_metrics = {"accuracy": 0.55}

        is_degraded, reasons = self.monitor._check_degradation_signals(
            current_metrics, baseline_metrics, None
        )

        assert any("Win rate below minimum" in r for r in reasons)

    def test_multiple_signals_required_for_degradation(self):
        """Test that 2+ signals are required to flag as degraded."""
        # Only one signal (win rate)
        current_metrics = {
            "accuracy": 0.55,
            "win_rate": 0.35,
            "sharpe_ratio": 0.5
        }
        baseline_metrics = {"accuracy": 0.55}

        is_degraded, reasons = self.monitor._check_degradation_signals(
            current_metrics, baseline_metrics, None
        )

        assert len(reasons) == 1
        assert not is_degraded  # Only 1 signal, need 2+

    def test_multiple_signals_trigger_degradation(self):
        """Test that 2+ signals correctly flag degradation."""
        current_metrics = {
            "accuracy": 0.45,  # Drop from baseline
            "win_rate": 0.35,  # Below threshold
            "sharpe_ratio": -0.8  # Below threshold
        }
        baseline_metrics = {"accuracy": 0.55}

        is_degraded, reasons = self.monitor._check_degradation_signals(
            current_metrics, baseline_metrics, None
        )

        assert len(reasons) >= 2
        assert is_degraded

    def test_confidence_collapse_signal(self):
        """Test confidence collapse as a degradation signal."""
        current_metrics = {
            "accuracy": 0.45,  # One signal: accuracy drop
            "win_rate": 0.50,
            "sharpe_ratio": 0.5
        }
        baseline_metrics = {"accuracy": 0.55}
        confidence_dist = {
            "mean": 0.51,  # Near 0.5 = collapse
            "std": 0.02,
            "pct_near_0_5": 0.8,
            "pct_above_0_6": 0.05,
            "pct_below_0_4": 0.05
        }

        is_degraded, reasons = self.monitor._check_degradation_signals(
            current_metrics, baseline_metrics, confidence_dist
        )

        assert any("Confidence collapse" in r for r in reasons)
        assert is_degraded  # 2 signals: accuracy + confidence

    def test_report_to_dict(self):
        """Test DegradationReport serialization."""
        report = DegradationReport(
            model_type="xgboost",
            check_time="2026-01-23T12:00:00",
            is_degraded=True,
            metrics={"accuracy": 0.45},
            baseline_metrics={"accuracy": 0.55},
            degradation_reasons=["Accuracy dropped"],
            recommendation="retrain"
        )

        d = report.to_dict()
        assert d["model_type"] == "xgboost"
        assert d["is_degraded"] is True
        assert d["recommendation"] == "retrain"

    @patch('src.ml.degradation_monitor.DegradationMonitor._load_registry')
    def test_check_model_no_production(self, mock_registry):
        """Test check_model when no production model exists."""
        mock_registry.return_value = {"production": {}}

        report = self.monitor.check_model("xgboost")
        assert not report.is_degraded
        assert "No production model found" in report.degradation_reasons

    def test_check_all_disabled(self):
        """Test that disabled monitor returns empty results."""
        self.monitor.enabled = False
        results = self.monitor.check_all_models()
        assert results == {}


class TestAutoRollbackManager:
    """Tests for AutoRollbackManager."""

    def setup_method(self, tmp_path=None):
        """Set up test fixtures with temp directory."""
        self.manager = AutoRollbackManager(
            grace_period_days=5,
            enabled=True
        )

    def test_initialization(self):
        """Test AutoRollbackManager initialization."""
        assert self.manager.grace_period_days == 5
        assert self.manager.enabled is True

    @patch('src.ml.auto_rollback.AutoRollbackManager._load_registry')
    @patch('src.ml.auto_rollback.AutoRollbackManager._save_registry')
    def test_register_deployment(self, mock_save, mock_load):
        """Test grace period registration."""
        mock_load.return_value = {
            "production": {
                "xgboost": {"version": "v_old"}
            }
        }

        self.manager.register_deployment("xgboost", "v_new", "v_old")

        # Verify registry was saved with grace period info
        saved = mock_save.call_args[0][0]
        prod = saved["production"]["xgboost"]
        assert prod["previous_production"] == "v_old"
        assert "grace_period_start" in prod
        assert "grace_period_end" in prod

    @patch('src.ml.auto_rollback.AutoRollbackManager._load_registry')
    def test_is_in_grace_period_true(self, mock_load):
        """Test detecting active grace period."""
        future = (datetime.now() + timedelta(days=3)).isoformat()
        mock_load.return_value = {
            "production": {
                "xgboost": {"grace_period_end": future}
            }
        }

        assert self.manager.is_in_grace_period("xgboost") is True

    @patch('src.ml.auto_rollback.AutoRollbackManager._load_registry')
    def test_is_in_grace_period_expired(self, mock_load):
        """Test detecting expired grace period."""
        past = (datetime.now() - timedelta(days=1)).isoformat()
        mock_load.return_value = {
            "production": {
                "xgboost": {"grace_period_end": past}
            }
        }

        assert self.manager.is_in_grace_period("xgboost") is False

    @patch('src.ml.auto_rollback.AutoRollbackManager._load_registry')
    def test_is_in_grace_period_none(self, mock_load):
        """Test when no grace period is set."""
        mock_load.return_value = {
            "production": {
                "xgboost": {"version": "v1"}
            }
        }

        assert self.manager.is_in_grace_period("xgboost") is False

    @patch('src.ml.auto_rollback.AutoRollbackManager._restore_model_files')
    @patch('src.ml.auto_rollback.AutoRollbackManager._save_registry')
    @patch('src.ml.auto_rollback.AutoRollbackManager._load_registry')
    def test_rollback_success(self, mock_load, mock_save, mock_restore):
        """Test successful rollback."""
        mock_load.return_value = {
            "production": {
                "xgboost": {
                    "version": "v_new",
                    "previous_production": "v_old"
                }
            },
            "versions": {
                "v_old": {"metrics": {"accuracy": 0.55}}
            }
        }
        # Mock the versions directory to exist
        self.manager.versions_dir = Mock()
        mock_dir = Mock()
        mock_dir.exists.return_value = True
        self.manager.versions_dir.__truediv__ = Mock(return_value=mock_dir)

        event = self.manager.rollback("xgboost", reason="Test rollback")

        assert event is not None
        assert event.rolled_back_version == "v_new"
        assert event.restored_version == "v_old"
        assert event.reason == "Test rollback"
        mock_restore.assert_called_once_with("xgboost", "v_old")

    @patch('src.ml.auto_rollback.AutoRollbackManager._load_registry')
    def test_rollback_no_previous(self, mock_load):
        """Test rollback when no previous version exists."""
        mock_load.return_value = {
            "production": {
                "xgboost": {"version": "v1"}
            }
        }

        event = self.manager.rollback("xgboost")
        assert event is None

    @patch('src.ml.auto_rollback.AutoRollbackManager._save_registry')
    @patch('src.ml.auto_rollback.AutoRollbackManager._load_registry')
    def test_clear_grace_period(self, mock_load, mock_save):
        """Test clearing grace period markers."""
        mock_load.return_value = {
            "production": {
                "xgboost": {
                    "version": "v1",
                    "previous_production": "v_old",
                    "grace_period_start": "2026-01-20T00:00:00",
                    "grace_period_end": "2026-01-25T00:00:00"
                }
            }
        }

        self.manager.clear_grace_period("xgboost")

        saved = mock_save.call_args[0][0]
        prod = saved["production"]["xgboost"]
        assert "previous_production" not in prod
        assert "grace_period_start" not in prod
        assert "grace_period_end" not in prod

    @patch('src.ml.auto_rollback.AutoRollbackManager._load_registry')
    def test_get_rollback_history(self, mock_load):
        """Test retrieving rollback history."""
        mock_load.return_value = {
            "rollback_history": [
                {"model_type": "xgboost", "timestamp": "2026-01-20"},
                {"model_type": "lstm", "timestamp": "2026-01-21"}
            ]
        }

        history = self.manager.get_rollback_history(limit=5)
        assert len(history) == 2
        # Most recent first
        assert history[0]["model_type"] == "lstm"

    def test_disabled_manager(self):
        """Test that disabled manager does nothing."""
        manager = AutoRollbackManager(enabled=False)
        assert manager.is_in_grace_period("xgboost") is False


class TestWalkForwardValidation:
    """Tests for walk-forward pre-deployment validation in RetrainingPipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.ml.retraining import RetrainingPipeline
        self.pipeline = RetrainingPipeline(
            symbols=["AAPL"],
            min_improvement_threshold=0.01
        )

    def test_walk_forward_validation_pass(self):
        """Test that a good model passes walk-forward validation."""
        # Create mock model that returns appropriate-sized predictions
        def mock_predict_proba(X_input):
            n = len(X_input)
            probas = np.zeros((n, 2))
            probas[:, 1] = np.random.uniform(0.6, 0.9, n)
            probas[:, 0] = 1 - probas[:, 1]
            return probas

        mock_model = Mock()
        mock_model.predict_proba = Mock(side_effect=mock_predict_proba)

        X = pd.DataFrame(np.random.randn(500, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(np.random.randint(0, 2, 500))

        passed, metrics = self.pipeline.validate_with_walk_forward(
            model=mock_model,
            model_type="xgboost",
            X=X,
            y=y,
            n_windows=3,
            min_sharpe=-1.0,  # Lenient threshold
            min_profit_factor=0.0,
            max_drawdown=1.0
        )

        assert passed
        assert "sharpe_ratio" in metrics
        assert "profit_factor" in metrics
        assert "max_drawdown" in metrics
        assert metrics["n_windows"] > 0

    def test_walk_forward_validation_fail_sharpe(self):
        """Test that model fails when Sharpe is too low."""
        # Model predicts randomly (bad model)
        mock_model = Mock()
        probas = np.zeros((100, 2))
        probas[:, 1] = 0.51  # Always barely above threshold
        probas[:, 0] = 0.49
        mock_model.predict_proba = Mock(return_value=probas)

        X = pd.DataFrame(np.random.randn(500, 10), columns=[f"f{i}" for i in range(10)])
        y = pd.Series(np.random.randint(0, 2, 500))

        passed, metrics = self.pipeline.validate_with_walk_forward(
            model=mock_model,
            model_type="xgboost",
            X=X,
            y=y,
            n_windows=3,
            min_sharpe=2.0,  # Very strict threshold
            min_profit_factor=0.0,
            max_drawdown=1.0
        )

        assert not passed

    def test_walk_forward_insufficient_data(self):
        """Test handling of insufficient data."""
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))

        X = pd.DataFrame(np.random.randn(10, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 10))

        passed, metrics = self.pipeline.validate_with_walk_forward(
            model=mock_model,
            model_type="xgboost",
            X=X,
            y=y,
            n_windows=3
        )

        # Should fail gracefully with no valid windows
        assert not passed
        assert metrics == {}


try:
    import apscheduler
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False


@pytest.mark.skipif(not HAS_APSCHEDULER, reason="apscheduler not installed")
class TestScheduledRetrainerPhase18:
    """Tests for Phase 18 additions to ScheduledRetrainer."""

    def test_init_with_phase18_disabled(self):
        """Test initialization with Phase 18 features disabled."""
        from src.ml.scheduled_retrainer import ScheduledRetrainer

        with patch('config.settings.Settings.load_trading_config') as mock_config:
            mock_config.return_value = {"trading": {"symbols": ["AAPL"]}}
            retrainer = ScheduledRetrainer(
                symbols=["AAPL"],
                enabled=False,
                degradation_check_enabled=False,
                auto_rollback_enabled=False,
                use_walk_forward_validation=False
            )

        assert retrainer._degradation_monitor is None
        assert retrainer._rollback_manager is None
        assert retrainer.use_walk_forward_validation is False

    def test_retrain_lock_prevents_concurrent(self):
        """Test that concurrent retraining is prevented."""
        from src.ml.scheduled_retrainer import ScheduledRetrainer

        with patch('config.settings.Settings.load_trading_config') as mock_config:
            mock_config.return_value = {"trading": {"symbols": ["AAPL"]}}
            retrainer = ScheduledRetrainer(symbols=["AAPL"], enabled=True)

        # Simulate lock being held
        retrainer._retrain_lock.acquire()
        result = retrainer.run_retrain()
        retrainer._retrain_lock.release()

        assert result.get("skipped") is True
        assert result.get("reason") == "already_running"

    def test_factory_function_phase18(self):
        """Test factory function reads Phase 18 config."""
        mock_config_data = {
            "trading": {"symbols": ["AAPL"]},
            "retraining": {
                "enabled": True,
                "schedule": "daily",
                "hour": 3,
                "degradation_detection": {
                    "enabled": False,
                    "check_interval_hours": 6
                },
                "auto_rollback": {
                    "enabled": False,
                    "grace_period_days": 7
                },
                "walk_forward_validation": {
                    "enabled": False,
                    "n_windows": 4
                }
            }
        }

        from src.ml.scheduled_retrainer import get_scheduled_retrainer

        with patch('config.settings.Settings.load_trading_config', return_value=mock_config_data):
            retrainer = get_scheduled_retrainer(config=mock_config_data)

        assert retrainer.degradation_check_interval_hours == 6
        assert retrainer.rollback_grace_period_days == 7
        assert retrainer.walk_forward_config.get("n_windows") == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
