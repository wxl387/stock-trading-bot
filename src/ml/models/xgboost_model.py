"""
XGBoost model for stock price prediction.
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from config.settings import MODELS_DIR

logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost classifier for predicting stock price direction.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        gamma: float = 1.0,
        reg_alpha: float = 0.3,
        reg_lambda: float = 1.5,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            subsample: Subsample ratio of training data.
            colsample_bytree: Subsample ratio of columns.
            min_child_weight: Minimum sum of instance weight in a child.
            gamma: Minimum loss reduction for further partition.
            reg_alpha: L1 regularization on weights.
            reg_lambda: L2 regularization on weights.
            random_state: Random seed.
        """
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False
        }

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self.is_trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 10
    ) -> Dict[str, float]:
        """
        Train the XGBoost model.

        Args:
            X: Feature DataFrame.
            y: Target labels.
            eval_set: Optional validation set (X_val, y_val).
            early_stopping_rounds: Early stopping patience.

        Returns:
            Dictionary with training metrics.
        """
        logger.info(f"Training XGBoost model with {len(X)} samples")

        self.feature_names = list(X.columns)

        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)

        # Prepare evaluation set
        eval_sets = [(X, y)]
        if eval_set is not None:
            eval_sets.append(eval_set)

        # Train with early stopping (set via constructor for XGBoost 2.x+)
        if eval_set is not None:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        self.model.fit(
            X, y,
            eval_set=eval_sets,
            verbose=False
        )

        self.is_trained = True

        # Calculate feature importance
        self._calculate_feature_importance()

        # Calculate training metrics
        y_pred = self.model.predict(X)
        metrics = self._calculate_metrics(y, y_pred)

        logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of predicted labels.
        """
        self._ensure_trained()
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of prediction probabilities [prob_class_0, prob_class_1].
        """
        self._ensure_trained()
        # Ensure feature names match training data to avoid XGBoost mismatch errors
        if self.feature_names and not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        elif self.feature_names and isinstance(X, pd.DataFrame) and list(X.columns) != self.feature_names:
            X = X.reindex(columns=self.feature_names)
        return self.model.predict_proba(X)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform time series cross-validation.

        Args:
            X: Feature DataFrame.
            y: Target labels.
            n_splits: Number of CV splits.

        Returns:
            Dictionary with metrics for each fold.
        """
        logger.info(f"Running {n_splits}-fold time series cross-validation")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": []
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train fold
            fold_model = xgb.XGBClassifier(**self.params)
            fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Evaluate
            y_pred = fold_model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)

            for key in results:
                results[key].append(metrics[key])

            logger.debug(f"Fold {fold + 1}: Accuracy = {metrics['accuracy']:.4f}")

        # Log average metrics
        for key in results:
            avg = np.mean(results[key])
            std = np.std(results[key])
            logger.info(f"CV {key}: {avg:.4f} (+/- {std:.4f})")

        return results

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        n_splits: int = 3
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.

        Args:
            X: Feature DataFrame.
            y: Target labels.
            n_trials: Number of optimization trials.
            n_splits: Number of CV splits for evaluation.

        Returns:
            Dictionary with best parameters.
        """
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
                "random_state": 42,
                "objective": "binary:logistic",
                "eval_metric": "logloss"
            }

            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                y_pred = model.predict(X_val)
                scores.append(accuracy_score(y_val, y_pred))

            return np.mean(scores)

        # Create study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Get best parameters
        best_params = study.best_params
        best_params.update({
            "random_state": 42,
            "objective": "binary:logistic",
            "eval_metric": "logloss"
        })

        logger.info(f"Best CV accuracy: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Update model parameters
        self.params = best_params

        return {
            "best_params": best_params,
            "best_score": study.best_value,
            "n_trials": n_trials
        }

    def save(self, name: str = "xgboost_model") -> Path:
        """
        Save model to disk.

        Args:
            name: Model name.

        Returns:
            Path to saved model.
        """
        self._ensure_trained()

        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / f"{name}.pkl"

        model_data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def load(self, name: str = "xgboost_model") -> None:
        """
        Load model from disk.

        Args:
            name: Model name.
        """
        model_path = MODELS_DIR / f"{name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.params = model_data["params"]
        self.feature_names = model_data["feature_names"]
        self.feature_importance = model_data["feature_importance"]
        self.is_trained = True

        logger.info(f"Model loaded from {model_path}")

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            top_n: Number of top features to return.

        Returns:
            DataFrame with feature importance.
        """
        self._ensure_trained()

        if self.feature_importance is None:
            self._calculate_feature_importance()

        return self.feature_importance.head(top_n)

    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance."""
        importance = self.model.feature_importances_

        self.feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    def _ensure_trained(self) -> None:
        """Ensure model is trained before prediction."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
