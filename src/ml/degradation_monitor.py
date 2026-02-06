"""
Walk-forward degradation detection for production models.
Monitors model performance on recent data and triggers retraining
when degradation exceeds configurable thresholds.
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR, Settings
from src.data.data_fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class DegradationReport:
    """Report from a degradation check."""
    model_type: str
    check_time: str
    is_degraded: bool
    metrics: Dict[str, float]
    baseline_metrics: Dict[str, float]
    degradation_reasons: List[str] = field(default_factory=list)
    confidence_distribution: Optional[Dict[str, float]] = None
    recommendation: str = "none"  # "none", "retrain", "rollback"

    def to_dict(self) -> Dict:
        return asdict(self)


class DegradationMonitor:
    """
    Monitors production model health using walk-forward evaluation on recent data.

    Detection signals:
    1. Accuracy drop below threshold relative to deployment baseline
    2. Prediction confidence collapse (probabilities clustering around 0.5)
    3. Sharpe ratio decline on recent walk-forward windows
    4. Win rate degradation below minimum threshold
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        evaluation_window_days: int = 63,
        n_eval_windows: int = 3,
        accuracy_drop_threshold: float = 0.05,
        confidence_collapse_threshold: float = 0.55,
        sharpe_decline_threshold: float = -0.5,
        min_win_rate: float = 0.40,
        prediction_horizon: int = 5,
        sequence_length: int = 20,
        enabled: bool = True
    ):
        self.enabled = enabled
        self.evaluation_window_days = evaluation_window_days
        self.n_eval_windows = n_eval_windows
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.confidence_collapse_threshold = confidence_collapse_threshold
        self.sharpe_decline_threshold = sharpe_decline_threshold
        self.min_win_rate = min_win_rate
        self.prediction_horizon = prediction_horizon
        self.sequence_length = sequence_length

        config = Settings.load_trading_config()
        self.symbols = symbols or config.get("trading", {}).get("symbols", [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"
        ])

        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.registry_path = MODELS_DIR / "versions" / "registry.json"
        self.monitoring_log_path = MODELS_DIR / "monitoring_log.json"

    def check_model(self, model_type: str) -> DegradationReport:
        """
        Run degradation check for a specific production model.

        Args:
            model_type: One of "xgboost", "lstm", "cnn".

        Returns:
            DegradationReport with findings.
        """
        check_time = datetime.now()
        logger.info(f"Running degradation check for {model_type}")

        # Get baseline metrics from registry
        registry = self._load_registry()
        production_info = registry.get("production", {}).get(model_type)
        if not production_info:
            logger.warning(f"No production model found for {model_type}")
            return DegradationReport(
                model_type=model_type,
                check_time=check_time.isoformat(),
                is_degraded=False,
                metrics={},
                baseline_metrics={},
                degradation_reasons=["No production model found"],
                recommendation="none"
            )

        baseline_metrics = production_info.get("metrics", {})

        try:
            # Load the production model
            model = self._load_production_model(model_type)

            # Evaluate on recent data
            eval_results = self._evaluate_on_recent_data(model, model_type)

            if eval_results is None:
                return DegradationReport(
                    model_type=model_type,
                    check_time=check_time.isoformat(),
                    is_degraded=False,
                    metrics={},
                    baseline_metrics=baseline_metrics,
                    degradation_reasons=["Insufficient data for evaluation"],
                    recommendation="none"
                )

            current_metrics = eval_results["metrics"]
            confidence_dist = eval_results.get("confidence_distribution")

            # Check for degradation signals
            is_degraded, reasons = self._check_degradation_signals(
                current_metrics, baseline_metrics, confidence_dist
            )

            # Determine recommendation
            if is_degraded:
                # Check if in grace period (rollback) or not (retrain)
                grace_info = production_info.get("grace_period_end")
                if grace_info and datetime.fromisoformat(grace_info) > check_time:
                    recommendation = "rollback"
                else:
                    recommendation = "retrain"
            else:
                recommendation = "none"

            report = DegradationReport(
                model_type=model_type,
                check_time=check_time.isoformat(),
                is_degraded=is_degraded,
                metrics=current_metrics,
                baseline_metrics=baseline_metrics,
                degradation_reasons=reasons,
                confidence_distribution=confidence_dist,
                recommendation=recommendation
            )

            self._log_check_result(report)
            return report

        except Exception as e:
            logger.error(f"Degradation check failed for {model_type}: {e}")
            return DegradationReport(
                model_type=model_type,
                check_time=check_time.isoformat(),
                is_degraded=False,
                metrics={},
                baseline_metrics=baseline_metrics,
                degradation_reasons=[f"Check failed: {str(e)}"],
                recommendation="none"
            )

    def check_all_models(self) -> Dict[str, DegradationReport]:
        """Check all production models for degradation."""
        if not self.enabled:
            return {}

        registry = self._load_registry()
        production_models = registry.get("production", {})

        reports = {}
        for model_type in production_models:
            reports[model_type] = self.check_model(model_type)

        return reports

    def _load_production_model(self, model_type: str):
        """Load the current production model for inference."""
        if model_type == "xgboost":
            from src.ml.models.xgboost_model import XGBoostModel
            model = XGBoostModel()
            model.load("trading_model")
            return model
        elif model_type == "lstm":
            from src.ml.models.lstm_model import LSTMModel
            model = LSTMModel(sequence_length=self.sequence_length)
            model.load("lstm_trading_model")
            return model
        elif model_type == "cnn":
            from src.ml.models.cnn_model import CNNModel
            model = CNNModel(sequence_length=self.sequence_length)
            model.load("cnn_trading_model")
            return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _evaluate_on_recent_data(
        self,
        model,
        model_type: str
    ) -> Optional[Dict]:
        """
        Evaluate model on recent walk-forward windows.

        Returns dict with metrics and confidence distribution, or None if
        insufficient data.
        """
        # Fetch recent data for all symbols
        all_predictions = []
        all_actuals = []
        all_probabilities = []
        all_returns = []

        for symbol in self.symbols:
            try:
                df = self.data_fetcher.fetch_historical(
                    symbol=symbol,
                    period=f"{self.evaluation_window_days + 60}d",
                    use_cache=True
                )

                if df.empty or len(df) < 50:
                    continue

                df = self.feature_engineer.add_all_features_extended(
                    df, symbol=symbol,
                    include_sentiment=False,
                    include_macro=False,
                    include_cross_asset=True,
                    include_interactions=False,
                    include_lagged=True,
                    use_cache=True
                )
                df = self.feature_engineer.create_labels(
                    df, horizon=self.prediction_horizon
                )
                df = df.dropna()

                if len(df) < 30:
                    continue

                # Use last evaluation_window_days as test data
                test_df = df.tail(min(self.evaluation_window_days, len(df)))
                feature_cols = self._get_feature_columns(test_df)
                X_test = test_df[feature_cols]
                y_test = test_df["label_binary"].values

                # Get predictions
                prob_up = None
                preds = None
                if model_type == "xgboost":
                    probas = model.predict_proba(X_test)
                    if probas is not None and len(probas) > 0:
                        prob_up = probas[:, 1]
                        preds = (prob_up > 0.5).astype(int)
                else:
                    # LSTM/CNN need sequences
                    from src.ml.sequence_utils import create_sequences
                    X_seq, y_seq = create_sequences(
                        X_test, pd.Series(y_test), self.sequence_length
                    )
                    if len(X_seq) < 5:
                        continue
                    probas = model.predict_proba(X_seq)
                    if probas is not None and len(probas) > 0:
                        prob_up = probas[:, 1]
                        preds = (prob_up > 0.5).astype(int)
                        y_test = y_seq  # Align with sequences

                if prob_up is None or preds is None:
                    logger.debug(f"No predictions for {symbol}, skipping")
                    continue

                all_predictions.extend(preds.tolist())
                all_actuals.extend(y_test.tolist())
                all_probabilities.extend(prob_up.tolist())

                # Calculate returns for Sharpe/win rate
                if "close" in test_df.columns:
                    returns = test_df["close"].pct_change().dropna()
                    # Only count returns where we had a long signal
                    signal_returns = []
                    for i, p in enumerate(prob_up):
                        if i < len(returns) and p > 0.5:
                            signal_returns.append(returns.iloc[i] if i < len(returns) else 0)
                    all_returns.extend(signal_returns)

            except Exception as e:
                logger.debug(f"Failed to evaluate {symbol}: {e}")
                continue

        if len(all_predictions) < 20:
            return None

        # Calculate metrics
        from sklearn.metrics import accuracy_score
        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)
        probabilities = np.array(all_probabilities)
        returns = np.array(all_returns) if all_returns else np.array([0.0])

        accuracy = accuracy_score(actuals, predictions)

        # Trading metrics
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
        profit_factor = (
            abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0
            else float('inf') if len(wins) > 0 else 0.0
        )

        # Sharpe ratio (annualized)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Confidence distribution
        confidence_dist = self._analyze_confidence(probabilities)

        metrics = {
            "accuracy": float(accuracy),
            "win_rate": float(win_rate),
            "sharpe_ratio": float(sharpe),
            "profit_factor": float(min(profit_factor, 10.0)),  # Cap at 10
            "n_predictions": len(predictions),
            "n_signals": int((probabilities > 0.5).sum())
        }

        return {
            "metrics": metrics,
            "confidence_distribution": confidence_dist
        }

    def _analyze_confidence(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Analyze prediction confidence distribution."""
        return {
            "mean": float(np.mean(probabilities)),
            "std": float(np.std(probabilities)),
            "pct_near_0_5": float(
                np.mean(np.abs(probabilities - 0.5) < 0.05)
            ),
            "pct_above_0_6": float(np.mean(probabilities > 0.6)),
            "pct_below_0_4": float(np.mean(probabilities < 0.4))
        }

    def _check_degradation_signals(
        self,
        current_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float],
        confidence_dist: Optional[Dict[str, float]]
    ) -> Tuple[bool, List[str]]:
        """Check all degradation signals and return (is_degraded, reasons)."""
        reasons = []

        # 1. Accuracy drop
        baseline_acc = baseline_metrics.get("accuracy", 0.5)
        current_acc = current_metrics.get("accuracy", 0.5)
        acc_drop = baseline_acc - current_acc
        if acc_drop > self.accuracy_drop_threshold:
            reasons.append(
                f"Accuracy dropped {acc_drop:.3f} "
                f"(baseline: {baseline_acc:.3f}, current: {current_acc:.3f})"
            )

        # 2. Confidence collapse
        if confidence_dist:
            avg_conf = max(
                confidence_dist.get("mean", 0.5),
                1 - confidence_dist.get("mean", 0.5)
            )
            if avg_conf < self.confidence_collapse_threshold:
                reasons.append(
                    f"Confidence collapse: avg={confidence_dist['mean']:.3f}, "
                    f"pct_near_0.5={confidence_dist['pct_near_0_5']:.1%}"
                )

        # 3. Sharpe ratio decline
        sharpe = current_metrics.get("sharpe_ratio", 0.0)
        if sharpe < self.sharpe_decline_threshold:
            reasons.append(
                f"Sharpe ratio degraded: {sharpe:.3f} "
                f"(threshold: {self.sharpe_decline_threshold})"
            )

        # 4. Win rate too low
        win_rate = current_metrics.get("win_rate", 0.5)
        if win_rate < self.min_win_rate:
            reasons.append(
                f"Win rate below minimum: {win_rate:.3f} "
                f"(threshold: {self.min_win_rate})"
            )

        is_degraded = len(reasons) >= 2  # Require 2+ signals to reduce false positives
        return is_degraded, reasons

    def _log_check_result(self, report: DegradationReport) -> None:
        """Persist check result to monitoring log."""
        try:
            if self.monitoring_log_path.exists():
                with open(self.monitoring_log_path, "r") as f:
                    log = json.load(f)
            else:
                log = {"checks": []}

            log["checks"].append(report.to_dict())
            log["checks"] = log["checks"][-200:]  # Keep last 200

            self.monitoring_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.monitoring_log_path, "w") as f:
                json.dump(log, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to log monitoring result: {e}")

    def get_monitoring_history(
        self, model_type: Optional[str] = None, limit: int = 30
    ) -> List[Dict]:
        """Get historical degradation check results."""
        if not self.monitoring_log_path.exists():
            return []

        try:
            with open(self.monitoring_log_path, "r") as f:
                log = json.load(f)

            checks = log.get("checks", [])
            if model_type:
                checks = [c for c in checks if c.get("model_type") == model_type]

            return checks[-limit:][::-1]
        except Exception as e:
            logger.warning(f"Failed to load monitoring history: {e}")
            return []

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns from DataFrame."""
        exclude = [
            "symbol", "date", "dividends", "stock_splits",
            "future_returns", "label_binary", "label_3class"
        ]
        return [c for c in df.columns if c not in exclude]

    def _load_registry(self) -> Dict:
        """Load version registry."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {}


def get_degradation_monitor(config: Optional[Dict] = None) -> DegradationMonitor:
    """Factory function to create DegradationMonitor from config."""
    if config is None:
        config = Settings.load_trading_config()

    retraining_config = config.get("retraining", {})
    deg_config = retraining_config.get("degradation_detection", {})
    trading_config = config.get("trading", {})
    ml_config = config.get("ml_model", {})

    return DegradationMonitor(
        symbols=trading_config.get("symbols"),
        evaluation_window_days=deg_config.get("evaluation_window_days", 63),
        n_eval_windows=deg_config.get("n_eval_windows", 3),
        accuracy_drop_threshold=deg_config.get("accuracy_drop_threshold", 0.05),
        confidence_collapse_threshold=deg_config.get("confidence_collapse_threshold", 0.55),
        sharpe_decline_threshold=deg_config.get("sharpe_decline_threshold", -0.5),
        min_win_rate=deg_config.get("min_win_rate", 0.40),
        prediction_horizon=ml_config.get("prediction_horizon", 5),
        sequence_length=ml_config.get("sequence_length", 20),
        enabled=deg_config.get("enabled", False)
    )
