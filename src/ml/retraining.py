"""
Automated model retraining pipeline with versioning.
"""
import logging
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from config.settings import MODELS_DIR, DATA_DIR
from src.data.data_fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelVersion:
    """Represents a model version with metadata."""

    def __init__(
        self,
        model_type: str,
        version: str,
        created_at: datetime,
        metrics: Dict[str, float],
        data_info: Dict
    ):
        self.model_type = model_type
        self.version = version
        self.created_at = created_at
        self.metrics = metrics
        self.data_info = data_info

    def to_dict(self) -> Dict:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "metrics": self.metrics,
            "data_info": self.data_info
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelVersion":
        return cls(
            model_type=data["model_type"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metrics=data["metrics"],
            data_info=data["data_info"]
        )


class RetrainingPipeline:
    """
    Automated retraining pipeline with versioning and comparison.

    Features:
    - On-demand or scheduled retraining
    - Automatic data refresh
    - Model versioning
    - Performance comparison before deployment
    """

    def __init__(
        self,
        symbols: List[str],
        prediction_horizon: int = 5,
        train_period_days: int = 252,
        test_ratio: float = 0.2,
        sequence_length: int = 20,
        min_improvement_threshold: float = 0.01  # 1% improvement required
    ):
        """
        Initialize retraining pipeline.

        Args:
            symbols: List of stock symbols to train on.
            prediction_horizon: Days ahead to predict.
            train_period_days: Days of historical data for training.
            test_ratio: Fraction of data for testing.
            sequence_length: Sequence length for LSTM/CNN.
            min_improvement_threshold: Minimum accuracy improvement to deploy.
        """
        self.symbols = symbols
        self.prediction_horizon = prediction_horizon
        self.train_period_days = train_period_days
        self.test_ratio = test_ratio
        self.sequence_length = sequence_length
        self.min_improvement_threshold = min_improvement_threshold

        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()

        # Version registry
        self.versions_dir = MODELS_DIR / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        self.registry_path = self.versions_dir / "registry.json"

    def refresh_data(self, use_cache: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fetch fresh training data for all symbols.

        Args:
            use_cache: Whether to use cached data.

        Returns:
            (X, y) tuple of features and labels.
        """
        logger.info(f"Refreshing data for {len(self.symbols)} symbols")

        all_features = []
        all_labels = []

        for symbol in self.symbols:
            try:
                df = self.data_fetcher.fetch_historical(
                    symbol=symbol,
                    period=f"{self.train_period_days + 60}d",
                    use_cache=use_cache
                )

                if df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue

                df = self.feature_engineer.add_all_features_extended(
                    df,
                    symbol=symbol,
                    include_sentiment=True,
                    include_macro=True,
                    include_cross_asset=True,
                    include_interactions=True,
                    include_lagged=True,
                    use_cache=use_cache
                )
                df = self.feature_engineer.create_labels(df, horizon=self.prediction_horizon)
                df = df.dropna()

                if len(df) < 50:
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                    continue

                feature_cols = self._get_feature_columns(df)
                all_features.append(df[feature_cols])
                all_labels.append(df["label_binary"])

                logger.debug(f"Processed {symbol}: {len(df)} samples")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid training data from any symbol")

        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)

        logger.info(f"Refreshed data: {len(X)} samples, {len(X.columns)} features")
        return X, y

    def retrain_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tune: bool = False,
        n_trials: int = 30
    ) -> Tuple[any, Dict[str, float]]:
        """
        Retrain XGBoost model.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            tune: Whether to tune hyperparameters.
            n_trials: Number of Optuna trials if tuning.

        Returns:
            (model, metrics) tuple.
        """
        from src.ml.models.xgboost_model import XGBoostModel

        logger.info("Retraining XGBoost model")

        split_idx = int(len(X) * (1 - self.test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = XGBoostModel()

        if tune:
            logger.info(f"Tuning XGBoost with {n_trials} trials")
            model.tune(X_train, y_train, n_trials=n_trials)

        model.train(X_train, y_train, eval_set=(X_test, y_test))

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)

        logger.info(f"XGBoost retrained: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        return model, metrics

    def retrain_lstm(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = 50,
        tune: bool = False,
        n_trials: int = 20
    ) -> Tuple[any, Dict[str, float]]:
        """
        Retrain LSTM model.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            epochs: Number of training epochs.
            tune: Whether to tune hyperparameters with Optuna.
            n_trials: Number of Optuna trials if tuning.

        Returns:
            (model, metrics) tuple.
        """
        from src.ml.models.lstm_model import LSTMModel
        from src.ml.sequence_utils import create_sequences, split_sequences_time_series

        logger.info("Retraining LSTM model")

        # Create sequences
        X_seq, y_seq = create_sequences(X, y, self.sequence_length)
        X_train, X_test, y_train, y_test = split_sequences_time_series(
            X_seq, y_seq, self.test_ratio
        )

        model = LSTMModel(
            sequence_length=self.sequence_length,
            n_features=X_seq.shape[2]
        )

        # Optuna tuning if requested
        if tune:
            logger.info(f"Tuning LSTM with {n_trials} trials")
            tune_result = model.tune(X_train, y_train, n_trials=n_trials)
            logger.info(f"LSTM tuning best accuracy: {tune_result['best_score']:.4f}")

        model.train(
            X_train, y_train,
            eval_set=(X_test, y_test),
            epochs=epochs
        )

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)

        logger.info(f"LSTM retrained: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        return model, metrics

    def retrain_cnn(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = 30,
        tune: bool = False,
        n_trials: int = 20
    ) -> Tuple[any, Dict[str, float]]:
        """
        Retrain CNN model.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            epochs: Number of training epochs.
            tune: Whether to tune hyperparameters with Optuna.
            n_trials: Number of Optuna trials if tuning.

        Returns:
            (model, metrics) tuple.
        """
        from src.ml.models.cnn_model import CNNModel
        from src.ml.sequence_utils import create_sequences, split_sequences_time_series

        logger.info("Retraining CNN model")

        # Create sequences
        X_seq, y_seq = create_sequences(X, y, self.sequence_length)
        X_train, X_test, y_train, y_test = split_sequences_time_series(
            X_seq, y_seq, self.test_ratio
        )

        model = CNNModel(
            sequence_length=self.sequence_length,
            n_features=X_seq.shape[2]
        )

        # Optuna tuning if requested
        if tune:
            logger.info(f"Tuning CNN with {n_trials} trials")
            tune_result = model.tune(X_train, y_train, n_trials=n_trials)
            logger.info(f"CNN tuning best accuracy: {tune_result['best_score']:.4f}")

        model.train(
            X_train, y_train,
            eval_set=(X_test, y_test),
            epochs=epochs
        )

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)

        logger.info(f"CNN retrained: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        return model, metrics

    def retrain_transformer(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        epochs: int = 50,
        tune: bool = False,
        n_trials: int = 15
    ) -> Tuple[any, Dict[str, float]]:
        """
        Retrain Transformer model.

        Args:
            X: Feature DataFrame.
            y: Target Series.
            epochs: Number of training epochs.
            tune: Whether to tune hyperparameters with Optuna.
            n_trials: Number of Optuna trials if tuning.

        Returns:
            (model, metrics) tuple.
        """
        from src.ml.models.transformer_model import TransformerModel
        from src.ml.sequence_utils import create_sequences, split_sequences_time_series

        logger.info("Retraining Transformer model")

        # Create sequences
        X_seq, y_seq = create_sequences(X, y, self.sequence_length)
        X_train, X_test, y_train, y_test = split_sequences_time_series(
            X_seq, y_seq, self.test_ratio
        )

        model = TransformerModel(
            sequence_length=self.sequence_length,
            n_features=X_seq.shape[2]
        )

        # Optuna tuning if requested
        if tune:
            logger.info(f"Tuning Transformer with {n_trials} trials")
            tune_result = model.tune(X_train, y_train, n_trials=n_trials)
            logger.info(f"Transformer tuning best accuracy: {tune_result['best_score']:.4f}")

        model.train(
            X_train, y_train,
            eval_set=(X_test, y_test),
            epochs=epochs
        )

        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred)

        logger.info(f"Transformer retrained: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        return model, metrics

    def retrain_all(
        self,
        use_cache: bool = False,
        tune_xgboost: bool = False,
        tune_lstm: bool = False,
        tune_cnn: bool = False,
        tune_transformer: bool = False,
        tune_all: bool = False,
        n_trials: int = 20,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Retrain all models with fresh data.

        Args:
            use_cache: Whether to use cached data.
            tune_xgboost: Whether to tune XGBoost hyperparameters.
            tune_lstm: Whether to tune LSTM hyperparameters.
            tune_cnn: Whether to tune CNN hyperparameters.
            tune_transformer: Whether to tune Transformer hyperparameters.
            tune_all: Tune all models (overrides individual flags).
            n_trials: Number of Optuna trials for tuning.
            models_to_train: List of models to train (default: all).

        Returns:
            Dict with model results and versions.
        """
        logger.info("Starting full retraining pipeline")

        if models_to_train is None:
            models_to_train = ["xgboost", "lstm", "cnn"]

        # Override individual tune flags if tune_all is set
        if tune_all:
            tune_xgboost = tune_lstm = tune_cnn = tune_transformer = True

        # Refresh data
        X, y = self.refresh_data(use_cache=use_cache)

        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Retrain each model
        for model_type in models_to_train:
            try:
                if model_type == "xgboost":
                    model, metrics = self.retrain_xgboost(X, y, tune=tune_xgboost, n_trials=n_trials)
                elif model_type == "lstm":
                    model, metrics = self.retrain_lstm(X, y, tune=tune_lstm, n_trials=n_trials)
                elif model_type == "cnn":
                    model, metrics = self.retrain_cnn(X, y, tune=tune_cnn, n_trials=n_trials)
                elif model_type == "transformer":
                    model, metrics = self.retrain_transformer(X, y, tune=tune_transformer, n_trials=n_trials)
                else:
                    logger.warning(f"Unknown model type: {model_type}")
                    continue

                # Version and save
                version = f"{model_type}_v{timestamp}"
                self._save_versioned_model(model, model_type, version, metrics, X)

                results[model_type] = {
                    "version": version,
                    "metrics": metrics,
                    "model": model
                }

            except Exception as e:
                logger.error(f"Failed to retrain {model_type}: {e}")
                results[model_type] = {"error": str(e)}

        return results

    def compare_and_deploy(
        self,
        model_type: str,
        new_version: str,
        production_name: str
    ) -> bool:
        """
        Compare new model with production and deploy if better.

        Args:
            model_type: Type of model (xgboost, lstm, cnn).
            new_version: Version string of new model.
            production_name: Name for production model.

        Returns:
            True if new model was deployed.
        """
        registry = self._load_registry()

        # Get current production metrics
        current_prod = registry.get("production", {}).get(model_type)
        new_metrics = registry.get("versions", {}).get(new_version, {}).get("metrics", {})

        if not new_metrics:
            logger.warning(f"No metrics found for version {new_version}")
            return False

        if current_prod is None:
            # No production model, deploy new one
            self._deploy_model(model_type, new_version, production_name)
            logger.info(f"Deployed {new_version} as new production model")
            return True

        current_metrics = current_prod.get("metrics", {})

        # Compare accuracy (or other metric)
        new_accuracy = new_metrics.get("accuracy", 0)
        current_accuracy = current_metrics.get("accuracy", 0)
        improvement = new_accuracy - current_accuracy

        if improvement >= self.min_improvement_threshold:
            self._deploy_model(model_type, new_version, production_name)
            logger.info(
                f"Deployed {new_version} (improvement: {improvement:.4f}, "
                f"new: {new_accuracy:.4f}, old: {current_accuracy:.4f})"
            )
            return True
        else:
            logger.info(
                f"Kept current model (new accuracy: {new_accuracy:.4f}, "
                f"improvement: {improvement:.4f} < threshold {self.min_improvement_threshold})"
            )
            return False

    def _save_versioned_model(
        self,
        model,
        model_type: str,
        version: str,
        metrics: Dict,
        X: pd.DataFrame
    ) -> Dict:
        """Save model with version info."""
        version_dir = self.versions_dir / version
        version_dir.mkdir(exist_ok=True)

        # Save model based on type
        if model_type == "xgboost":
            model.save(str(version_dir / "model"))
        else:
            # LSTM/CNN - save Keras model with .keras extension
            model.model.save(str(version_dir / "model.keras"))
            # Save full metadata including model-specific params
            import json
            metadata = {
                "model_type": model_type,
                "sequence_length": model.sequence_length,
                "n_features": model.n_features,
                "dropout_rate": model.dropout_rate,
                "learning_rate": model.learning_rate,
                "normalization_stats": {
                    "mean": model.normalization_stats["mean"].tolist(),
                    "std": model.normalization_stats["std"].tolist()
                } if model.normalization_stats else None
            }
            # Add model-specific parameters
            if model_type == "lstm":
                metadata["lstm_units"] = list(model.lstm_units)
            elif model_type == "cnn":
                metadata["filters"] = list(model.filters)
                metadata["kernel_size"] = model.kernel_size
            elif model_type == "transformer":
                metadata["embed_dim"] = model.embed_dim
                metadata["num_heads"] = model.num_heads
                metadata["ff_dim"] = model.ff_dim
                metadata["num_transformer_blocks"] = model.num_transformer_blocks
            with open(version_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        # Create version info
        version_info = ModelVersion(
            model_type=model_type,
            version=version,
            created_at=datetime.now(),
            metrics=metrics,
            data_info={
                "n_samples": len(X),
                "n_features": len(X.columns),
                "symbols": self.symbols
            }
        )

        # Update registry
        registry = self._load_registry()
        if "versions" not in registry:
            registry["versions"] = {}
        registry["versions"][version] = version_info.to_dict()
        self._save_registry(registry)

        logger.info(f"Saved versioned model: {version}")
        return version_info.to_dict()

    def _deploy_model(
        self,
        model_type: str,
        version: str,
        production_name: str
    ) -> None:
        """Deploy versioned model to production."""
        version_dir = self.versions_dir / version

        # Determine source path based on model type
        if model_type == "xgboost":
            src = version_dir / "model.pkl"
            dst = MODELS_DIR / f"{production_name}.pkl"
            if src.exists():
                shutil.copy(src, dst)
            else:
                # Try directory format
                src_dir = version_dir / "model"
                if src_dir.is_dir():
                    # Copy pickle from directory
                    for f in src_dir.glob("*.pkl"):
                        shutil.copy(f, MODELS_DIR / f"{production_name}.pkl")
                        break
        else:
            # LSTM/CNN - copy .keras file and metadata.json
            dst_dir = MODELS_DIR / production_name
            dst_dir.mkdir(exist_ok=True)

            # Copy model.keras file
            src_model = version_dir / "model.keras"
            if src_model.exists():
                shutil.copy(src_model, dst_dir / "model.keras")

            # Copy metadata.json
            src_meta = version_dir / "metadata.json"
            if src_meta.exists():
                shutil.copy(src_meta, dst_dir / "metadata.json")

        # Update registry
        registry = self._load_registry()
        if "production" not in registry:
            registry["production"] = {}
        registry["production"][model_type] = {
            "version": version,
            "deployed_at": datetime.now().isoformat(),
            "metrics": registry.get("versions", {}).get(version, {}).get("metrics", {})
        }
        self._save_registry(registry)

    def _load_registry(self) -> Dict:
        """Load version registry."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {}

    def _save_registry(self, registry: Dict) -> None:
        """Save version registry."""
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns from DataFrame."""
        exclude = [
            "symbol", "date", "dividends", "stock_splits",
            "future_returns", "label_binary", "label_3class"
        ]
        return [c for c in df.columns if c not in exclude]

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    def get_retraining_status(self) -> Dict:
        """Get current retraining status and history."""
        registry = self._load_registry()
        versions = registry.get("versions", {})

        return {
            "production_models": registry.get("production", {}),
            "version_count": len(versions),
            "recent_versions": sorted(versions.keys(), reverse=True)[:5]
        }

    def list_versions(self, model_type: Optional[str] = None) -> List[Dict]:
        """List all model versions."""
        registry = self._load_registry()
        versions = registry.get("versions", {})

        result = []
        for version, info in sorted(versions.items(), reverse=True):
            if model_type is None or info.get("model_type") == model_type:
                result.append({
                    "version": version,
                    "model_type": info.get("model_type"),
                    "created_at": info.get("created_at"),
                    "accuracy": info.get("metrics", {}).get("accuracy", 0)
                })

        return result
