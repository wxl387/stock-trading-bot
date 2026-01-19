"""
Model training pipeline.
"""
import logging
from typing import Optional, Dict, Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.data_fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer
from src.ml.models.xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Training pipeline for ML models.
    Handles data preparation, training, and evaluation.
    """

    def __init__(
        self,
        symbols: List[str],
        prediction_horizon: int = 5,
        train_period_days: int = 252,
        test_ratio: float = 0.2
    ):
        """
        Initialize ModelTrainer.

        Args:
            symbols: List of stock symbols to train on.
            prediction_horizon: Days ahead to predict.
            train_period_days: Days of historical data to use.
            test_ratio: Ratio of data for testing.
        """
        self.symbols = symbols
        self.prediction_horizon = prediction_horizon
        self.train_period_days = train_period_days
        self.test_ratio = test_ratio

        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.model: Optional[XGBoostModel] = None

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from multiple symbols.

        Returns:
            Tuple of (features DataFrame, labels Series).
        """
        logger.info(f"Preparing training data for {len(self.symbols)} symbols")

        all_features = []
        all_labels = []

        for symbol in self.symbols:
            try:
                # Fetch historical data
                df = self.data_fetcher.fetch_historical(
                    symbol=symbol,
                    period=f"{self.train_period_days + 60}d"  # Extra buffer
                )

                if df.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue

                # Add features (including extended features for better accuracy)
                df = self.feature_engineer.add_all_features_extended(
                    df,
                    symbol=symbol,
                    include_sentiment=True,
                    include_macro=True,
                    include_cross_asset=True,
                    include_interactions=True,
                    include_lagged=True,
                    use_cache=True
                )

                # Create labels
                df = self.feature_engineer.create_labels(
                    df,
                    horizon=self.prediction_horizon
                )

                # Drop rows with NaN
                df = df.dropna()

                if len(df) < 50:
                    logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue

                # Select feature columns
                feature_cols = self._get_feature_columns(df)

                all_features.append(df[feature_cols])
                all_labels.append(df["label_binary"])

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue

        if not all_features:
            raise ValueError("No valid training data")

        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)

        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} features")

        return X, y

    def train(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X: Optional pre-prepared features.
            y: Optional pre-prepared labels.

        Returns:
            Dictionary with training metrics.
        """
        # Prepare data if not provided
        if X is None or y is None:
            X, y = self.prepare_data()

        # Split data (maintaining time order for time series)
        split_idx = int(len(X) * (1 - self.test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")

        # Initialize and train model
        self.model = XGBoostModel()

        train_metrics = self.model.train(
            X_train, y_train,
            eval_set=(X_test, y_test)
        )

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        test_metrics = self.model._calculate_metrics(y_test, y_pred)

        logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1: {test_metrics['f1']:.4f}")

        return {
            "train": train_metrics,
            "test": test_metrics
        }

    def cross_validate(
        self,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation.

        Args:
            n_splits: Number of CV folds.

        Returns:
            Dictionary with CV metrics.
        """
        X, y = self.prepare_data()

        model = XGBoostModel()
        return model.cross_validate(X, y, n_splits=n_splits)

    def save_model(self, name: str = "trading_model") -> None:
        """Save trained model."""
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")

        self.model.save(name)

    def load_model(self, name: str = "trading_model") -> None:
        """Load trained model."""
        self.model = XGBoostModel()
        self.model.load(name)

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns from DataFrame."""
        exclude_cols = [
            "symbol", "date", "dividends", "stock_splits",
            "future_returns", "label_binary", "label_3class"
        ]
        return [col for col in df.columns if col not in exclude_cols]
