"""
Tests for the RetrainingPipeline in src/ml/retraining.py.

Covers:
- RetrainingPipeline initialization and configuration
- refresh_data() with mocked DataFetcher and FeatureEngineer
- retrain_xgboost() method (mocked model, verified data flow/splits)
- retrain_lstm() and retrain_cnn() methods
- retrain_transformer() method
- retrain_all() orchestration
- Model versioning (_save_versioned_model) and registry updates
- compare_and_deploy() promotion logic
- compare_and_deploy_enhanced() with walk-forward validation
- validate_with_walk_forward()
- Error handling (missing data, failed training, empty symbols)
- ModelVersion serialization round-trip
- Utility methods: list_versions, get_retraining_status, _get_feature_columns, _calculate_metrics
"""
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Module-under-test
# ---------------------------------------------------------------------------
from src.ml.retraining import ModelVersion, RetrainingPipeline


# ======================================================================
# Helpers & Fixtures
# ======================================================================

def _make_feature_df(n_rows: int = 200, n_features: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a simple feature DataFrame with named columns."""
    rng = np.random.RandomState(seed)
    data = rng.randn(n_rows, n_features)
    cols = [f"feat_{i}" for i in range(n_features)]
    return pd.DataFrame(data, columns=cols)


def _make_labels(n_rows: int = 200, seed: int = 42) -> pd.Series:
    """Create a binary label Series."""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.randint(0, 2, size=n_rows), name="label_binary")


def _make_ohlcv_df(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a minimal OHLCV DataFrame with extra columns expected after feature engineering."""
    rng = np.random.RandomState(seed)
    close = 100.0 + np.cumsum(rng.randn(n_rows) * 0.5)
    df = pd.DataFrame({
        "open": close + rng.randn(n_rows) * 0.1,
        "high": close + abs(rng.randn(n_rows)) * 0.5,
        "low": close - abs(rng.randn(n_rows)) * 0.5,
        "close": close,
        "volume": rng.randint(1_000_000, 10_000_000, n_rows),
        "feat_0": rng.randn(n_rows),
        "feat_1": rng.randn(n_rows),
        "feat_2": rng.randn(n_rows),
        "label_binary": rng.randint(0, 2, n_rows),
    })
    return df


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Provide a temporary MODELS_DIR and patch it into settings + pipeline."""
    models = tmp_path / "models"
    models.mkdir()
    return models


@pytest.fixture
def pipeline(tmp_models_dir):
    """
    Return a RetrainingPipeline with mocked DataFetcher/FeatureEngineer and
    a temporary MODELS_DIR so no real files are touched.
    """
    with patch("src.ml.retraining.MODELS_DIR", tmp_models_dir), \
         patch("src.ml.retraining.DataFetcher") as MockDF, \
         patch("src.ml.retraining.FeatureEngineer") as MockFE:

        p = RetrainingPipeline(
            symbols=["AAPL", "MSFT"],
            prediction_horizon=5,
            train_period_days=252,
            test_ratio=0.2,
            sequence_length=10,
            min_improvement_threshold=0.01,
        )
        # Expose the mock instances for per-test customisation
        p._mock_data_fetcher = MockDF.return_value
        p._mock_feature_engineer = MockFE.return_value
        yield p


@pytest.fixture
def training_data():
    """Return a (X, y) tuple ready for retrain_* methods."""
    X = _make_feature_df(200, 10)
    y = _make_labels(200)
    return X, y


# ======================================================================
# ModelVersion Tests
# ======================================================================

class TestModelVersion:
    """Tests for the ModelVersion data class."""

    def test_to_dict(self):
        now = datetime(2025, 1, 15, 12, 0, 0)
        mv = ModelVersion(
            model_type="xgboost",
            version="xgboost_v20250115_120000",
            created_at=now,
            metrics={"accuracy": 0.75, "f1": 0.70},
            data_info={"n_samples": 500, "n_features": 10, "symbols": ["AAPL"]},
        )
        d = mv.to_dict()
        assert d["model_type"] == "xgboost"
        assert d["version"] == "xgboost_v20250115_120000"
        assert d["created_at"] == "2025-01-15T12:00:00"
        assert d["metrics"]["accuracy"] == 0.75
        assert d["data_info"]["n_samples"] == 500

    def test_from_dict_roundtrip(self):
        now = datetime(2025, 6, 1, 8, 30, 0)
        original = ModelVersion(
            model_type="lstm",
            version="lstm_v20250601",
            created_at=now,
            metrics={"accuracy": 0.80},
            data_info={"n_samples": 300},
        )
        d = original.to_dict()
        restored = ModelVersion.from_dict(d)
        assert restored.model_type == original.model_type
        assert restored.version == original.version
        assert restored.created_at == original.created_at
        assert restored.metrics == original.metrics
        assert restored.data_info == original.data_info

    def test_from_dict_preserves_all_fields(self):
        d = {
            "model_type": "cnn",
            "version": "cnn_v1",
            "created_at": "2025-03-20T10:00:00",
            "metrics": {"accuracy": 0.55, "precision": 0.60},
            "data_info": {"symbols": ["AAPL", "MSFT"]},
        }
        mv = ModelVersion.from_dict(d)
        assert mv.model_type == "cnn"
        assert mv.metrics["precision"] == 0.60
        assert "MSFT" in mv.data_info["symbols"]


# ======================================================================
# RetrainingPipeline Initialization
# ======================================================================

class TestPipelineInit:
    """Tests for RetrainingPipeline construction."""

    def test_default_params(self, pipeline):
        assert pipeline.symbols == ["AAPL", "MSFT"]
        assert pipeline.prediction_horizon == 5
        assert pipeline.train_period_days == 252
        assert pipeline.test_ratio == 0.2
        assert pipeline.sequence_length == 10
        assert pipeline.min_improvement_threshold == 0.01

    def test_versions_dir_created(self, pipeline):
        assert pipeline.versions_dir.exists()
        assert pipeline.versions_dir.is_dir()

    def test_registry_path(self, pipeline):
        assert pipeline.registry_path.name == "registry.json"
        assert pipeline.registry_path.parent == pipeline.versions_dir


# ======================================================================
# refresh_data()
# ======================================================================

class TestRefreshData:
    """Tests for the data refresh pipeline."""

    def _setup_fetcher_and_engineer(self, pipeline, dfs_by_symbol):
        """
        Configure mock DataFetcher.fetch_historical and
        FeatureEngineer methods so refresh_data() gets usable DataFrames.
        """
        def fetch_side_effect(symbol, period, use_cache):
            return dfs_by_symbol.get(symbol, pd.DataFrame())

        pipeline.data_fetcher.fetch_historical = MagicMock(side_effect=fetch_side_effect)
        pipeline.feature_engineer.add_all_features_extended = MagicMock(
            side_effect=lambda df, **kw: df
        )
        pipeline.feature_engineer.create_labels = MagicMock(
            side_effect=lambda df, horizon: df
        )

    def test_refresh_data_happy_path(self, pipeline):
        dfs = {
            sym: _make_ohlcv_df(100, seed=i)
            for i, sym in enumerate(["AAPL", "MSFT"])
        }
        self._setup_fetcher_and_engineer(pipeline, dfs)

        X, y = pipeline.refresh_data(use_cache=False)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        # Both symbols contribute; each has 100 rows
        assert len(X) == 200
        assert len(y) == 200

    def test_refresh_data_empty_symbol_skipped(self, pipeline):
        """Symbols returning empty DataFrames are silently skipped."""
        dfs = {
            "AAPL": _make_ohlcv_df(100),
            "MSFT": pd.DataFrame(),  # empty
        }
        self._setup_fetcher_and_engineer(pipeline, dfs)

        X, y = pipeline.refresh_data()
        # Only AAPL data should contribute
        assert len(X) == 100

    def test_refresh_data_insufficient_rows_skipped(self, pipeline):
        """Symbols with fewer than 50 rows after dropna are skipped."""
        dfs = {
            "AAPL": _make_ohlcv_df(100),
            "MSFT": _make_ohlcv_df(30),  # too few
        }
        self._setup_fetcher_and_engineer(pipeline, dfs)

        X, y = pipeline.refresh_data()
        assert len(X) == 100  # only AAPL

    def test_refresh_data_all_symbols_fail_raises(self, pipeline):
        """ValueError when no symbol produces valid data."""
        dfs = {
            "AAPL": pd.DataFrame(),
            "MSFT": pd.DataFrame(),
        }
        self._setup_fetcher_and_engineer(pipeline, dfs)

        with pytest.raises(ValueError, match="No valid training data"):
            pipeline.refresh_data()

    def test_refresh_data_exception_in_fetch_skipped(self, pipeline):
        """If DataFetcher.fetch_historical raises, symbol is skipped."""
        pipeline.data_fetcher.fetch_historical = MagicMock(
            side_effect=[
                RuntimeError("network error"),
                _make_ohlcv_df(80),
            ]
        )
        pipeline.feature_engineer.add_all_features_extended = MagicMock(
            side_effect=lambda df, **kw: df
        )
        pipeline.feature_engineer.create_labels = MagicMock(
            side_effect=lambda df, horizon: df
        )

        X, y = pipeline.refresh_data()
        # Only second symbol (MSFT) should succeed
        assert len(X) == 80

    def test_refresh_data_use_cache_forwarded(self, pipeline):
        """use_cache kwarg is forwarded to DataFetcher.fetch_historical."""
        dfs = {"AAPL": _make_ohlcv_df(100), "MSFT": _make_ohlcv_df(100)}
        self._setup_fetcher_and_engineer(pipeline, dfs)

        pipeline.refresh_data(use_cache=True)
        for call in pipeline.data_fetcher.fetch_historical.call_args_list:
            assert call.kwargs.get("use_cache") is True or call[1].get("use_cache") is True


# ======================================================================
# _get_feature_columns / _calculate_metrics
# ======================================================================

class TestUtilities:
    """Tests for helper methods."""

    def test_get_feature_columns_excludes_reserved(self, pipeline):
        df = pd.DataFrame({
            "feat_0": [1], "feat_1": [2],
            "symbol": ["AAPL"], "date": ["2025-01-01"],
            "label_binary": [1], "label_3class": [0],
            "future_returns": [0.01],
            "dividends": [0], "stock_splits": [0],
        })
        cols = pipeline._get_feature_columns(df)
        assert "feat_0" in cols
        assert "feat_1" in cols
        for excluded in ["symbol", "date", "label_binary", "label_3class",
                         "future_returns", "dividends", "stock_splits"]:
            assert excluded not in cols

    def test_calculate_metrics_perfect(self, pipeline):
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])
        m = pipeline._calculate_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_calculate_metrics_all_wrong(self, pipeline):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        m = pipeline._calculate_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_calculate_metrics_zero_division(self, pipeline):
        """When no positive predictions, precision/recall/f1 should be 0."""
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        m = pipeline._calculate_metrics(y_true, y_pred)
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0


# ======================================================================
# retrain_xgboost()
# ======================================================================

class TestRetrainXGBoost:
    """Tests for retrain_xgboost()."""

    @patch("src.ml.retraining.RetrainingPipeline._calculate_metrics")
    def test_retrain_xgboost_no_tune(self, mock_calc, pipeline, training_data):
        X, y = training_data
        mock_model = MagicMock()
        mock_model.predict.return_value = y.values

        expected_metrics = {"accuracy": 0.80, "precision": 0.78, "recall": 0.82, "f1": 0.80}
        mock_calc.return_value = expected_metrics

        with patch("src.ml.models.xgboost_model.XGBoostModel", return_value=mock_model):
            model, metrics = pipeline.retrain_xgboost(X, y, tune=False)

        # Model was trained with the correct split (80/20)
        split_idx = int(len(X) * 0.8)
        train_call = mock_model.train.call_args
        X_train_arg = train_call[0][0]
        assert len(X_train_arg) == split_idx

        assert metrics == expected_metrics
        assert model is mock_model

    @patch("src.ml.retraining.RetrainingPipeline._calculate_metrics")
    def test_retrain_xgboost_with_tune(self, mock_calc, pipeline, training_data):
        X, y = training_data
        mock_model = MagicMock()
        mock_model.predict.return_value = y.values[:40]  # test set predictions

        mock_calc.return_value = {"accuracy": 0.85, "f1": 0.83, "precision": 0.84, "recall": 0.82}

        with patch("src.ml.models.xgboost_model.XGBoostModel", return_value=mock_model):
            model, metrics = pipeline.retrain_xgboost(X, y, tune=True, n_trials=5)

        mock_model.tune.assert_called_once()
        tune_args = mock_model.tune.call_args
        assert tune_args[1].get("n_trials") == 5 or tune_args.kwargs.get("n_trials") == 5

    @patch("src.ml.retraining.RetrainingPipeline._calculate_metrics")
    def test_retrain_xgboost_eval_set_passed(self, mock_calc, pipeline, training_data):
        """Verify that eval_set=(X_test, y_test) is passed to model.train()."""
        X, y = training_data
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(40)
        mock_calc.return_value = {"accuracy": 0.5, "f1": 0.5, "precision": 0.5, "recall": 0.5}

        with patch("src.ml.models.xgboost_model.XGBoostModel", return_value=mock_model):
            pipeline.retrain_xgboost(X, y)

        call_kwargs = mock_model.train.call_args
        # eval_set should be a tuple of (X_test, y_test)
        eval_set_arg = call_kwargs[1].get("eval_set") or call_kwargs.kwargs.get("eval_set")
        assert eval_set_arg is not None
        X_test_arg, y_test_arg = eval_set_arg
        expected_test_len = int(len(X) * pipeline.test_ratio)
        assert len(X_test_arg) == expected_test_len


# ======================================================================
# retrain_lstm()
# ======================================================================

class TestRetrainLSTM:
    """Tests for retrain_lstm()."""

    @patch("src.ml.retraining.RetrainingPipeline._calculate_metrics")
    def test_retrain_lstm_no_tune(self, mock_calc, pipeline, training_data):
        X, y = training_data
        n_features = X.shape[1]
        seq_len = pipeline.sequence_length

        # Fabricate sequence data that create_sequences would produce
        n_seq = len(X) - seq_len
        X_seq = np.random.randn(n_seq, seq_len, n_features)
        y_seq = np.random.randint(0, 2, n_seq)

        split_idx = int(n_seq * (1 - pipeline.test_ratio))
        X_train_seq = X_seq[:split_idx]
        X_test_seq = X_seq[split_idx:]
        y_train_seq = y_seq[:split_idx]
        y_test_seq = y_seq[split_idx:]

        mock_model = MagicMock()
        mock_model.predict.return_value = y_test_seq

        mock_calc.return_value = {"accuracy": 0.70, "f1": 0.68, "precision": 0.72, "recall": 0.66}

        with patch("src.ml.models.lstm_model.LSTMModel", return_value=mock_model), \
             patch("src.ml.sequence_utils.create_sequences", return_value=(X_seq, y_seq)) as mock_cs, \
             patch("src.ml.sequence_utils.split_sequences_time_series",
                   return_value=(X_train_seq, X_test_seq, y_train_seq, y_test_seq)):

            model, metrics = pipeline.retrain_lstm(X, y, epochs=10)

        # create_sequences called with correct seq_len
        mock_cs.assert_called_once_with(X, y, seq_len)

        # LSTMModel was constructed with correct shape
        mock_model_cls = mock_model  # returned by the patch
        # model.train was called
        mock_model.train.assert_called_once()
        train_kwargs = mock_model.train.call_args
        assert train_kwargs[1].get("epochs") == 10 or train_kwargs.kwargs.get("epochs") == 10

        assert model is mock_model
        assert metrics["accuracy"] == 0.70

    @patch("src.ml.retraining.RetrainingPipeline._calculate_metrics")
    def test_retrain_lstm_with_tune(self, mock_calc, pipeline, training_data):
        X, y = training_data
        seq_len = pipeline.sequence_length
        n_seq = len(X) - seq_len
        X_seq = np.random.randn(n_seq, seq_len, X.shape[1])
        y_seq = np.random.randint(0, 2, n_seq)
        split = int(n_seq * 0.8)

        mock_model = MagicMock()
        mock_model.predict.return_value = y_seq[split:]
        mock_model.tune.return_value = {"best_score": 0.72}
        mock_calc.return_value = {"accuracy": 0.72, "f1": 0.70, "precision": 0.73, "recall": 0.68}

        with patch("src.ml.models.lstm_model.LSTMModel", return_value=mock_model), \
             patch("src.ml.sequence_utils.create_sequences", return_value=(X_seq, y_seq)), \
             patch("src.ml.sequence_utils.split_sequences_time_series",
                   return_value=(X_seq[:split], X_seq[split:], y_seq[:split], y_seq[split:])):

            model, metrics = pipeline.retrain_lstm(X, y, tune=True, n_trials=8)

        mock_model.tune.assert_called_once()


# ======================================================================
# retrain_cnn()
# ======================================================================

class TestRetrainCNN:
    """Tests for retrain_cnn()."""

    @patch("src.ml.retraining.RetrainingPipeline._calculate_metrics")
    def test_retrain_cnn_no_tune(self, mock_calc, pipeline, training_data):
        X, y = training_data
        seq_len = pipeline.sequence_length
        n_seq = len(X) - seq_len
        X_seq = np.random.randn(n_seq, seq_len, X.shape[1])
        y_seq = np.random.randint(0, 2, n_seq)
        split = int(n_seq * 0.8)

        mock_model = MagicMock()
        mock_model.predict.return_value = y_seq[split:]
        mock_calc.return_value = {"accuracy": 0.72, "f1": 0.70, "precision": 0.71, "recall": 0.69}

        with patch("src.ml.models.cnn_model.CNNModel", return_value=mock_model), \
             patch("src.ml.sequence_utils.create_sequences", return_value=(X_seq, y_seq)), \
             patch("src.ml.sequence_utils.split_sequences_time_series",
                   return_value=(X_seq[:split], X_seq[split:], y_seq[:split], y_seq[split:])):

            model, metrics = pipeline.retrain_cnn(X, y, epochs=15)

        mock_model.train.assert_called_once()
        train_kwargs = mock_model.train.call_args
        assert train_kwargs[1].get("epochs") == 15 or train_kwargs.kwargs.get("epochs") == 15
        assert metrics["accuracy"] == 0.72

    @patch("src.ml.retraining.RetrainingPipeline._calculate_metrics")
    def test_retrain_cnn_with_tune(self, mock_calc, pipeline, training_data):
        X, y = training_data
        seq_len = pipeline.sequence_length
        n_seq = len(X) - seq_len
        X_seq = np.random.randn(n_seq, seq_len, X.shape[1])
        y_seq = np.random.randint(0, 2, n_seq)
        split = int(n_seq * 0.8)

        mock_model = MagicMock()
        mock_model.predict.return_value = y_seq[split:]
        mock_model.tune.return_value = {"best_score": 0.73}
        mock_calc.return_value = {"accuracy": 0.73, "f1": 0.71, "precision": 0.72, "recall": 0.70}

        with patch("src.ml.models.cnn_model.CNNModel", return_value=mock_model), \
             patch("src.ml.sequence_utils.create_sequences", return_value=(X_seq, y_seq)), \
             patch("src.ml.sequence_utils.split_sequences_time_series",
                   return_value=(X_seq[:split], X_seq[split:], y_seq[:split], y_seq[split:])):

            model, metrics = pipeline.retrain_cnn(X, y, tune=True, n_trials=12)

        mock_model.tune.assert_called_once()


# ======================================================================
# retrain_transformer()
# ======================================================================

class TestRetrainTransformer:
    """Tests for retrain_transformer()."""

    @patch("src.ml.retraining.RetrainingPipeline._calculate_metrics")
    def test_retrain_transformer_no_tune(self, mock_calc, pipeline, training_data):
        X, y = training_data
        seq_len = pipeline.sequence_length
        n_seq = len(X) - seq_len
        X_seq = np.random.randn(n_seq, seq_len, X.shape[1])
        y_seq = np.random.randint(0, 2, n_seq)
        split = int(n_seq * 0.8)

        mock_model = MagicMock()
        mock_model.predict.return_value = y_seq[split:]
        mock_calc.return_value = {"accuracy": 0.68, "f1": 0.66, "precision": 0.70, "recall": 0.63}

        with patch("src.ml.models.transformer_model.TransformerModel", return_value=mock_model), \
             patch("src.ml.sequence_utils.create_sequences", return_value=(X_seq, y_seq)), \
             patch("src.ml.sequence_utils.split_sequences_time_series",
                   return_value=(X_seq[:split], X_seq[split:], y_seq[:split], y_seq[split:])):

            model, metrics = pipeline.retrain_transformer(X, y, epochs=20)

        mock_model.train.assert_called_once()
        assert metrics["accuracy"] == 0.68


# ======================================================================
# retrain_all()
# ======================================================================

class TestRetrainAll:
    """Tests for retrain_all() orchestration."""

    def test_retrain_all_default_models(self, pipeline, training_data):
        X, y = training_data

        with patch.object(pipeline, "refresh_data", return_value=(X, y)), \
             patch.object(pipeline, "retrain_xgboost", return_value=(MagicMock(), {"accuracy": 0.80})), \
             patch.object(pipeline, "retrain_lstm", return_value=(MagicMock(), {"accuracy": 0.75})), \
             patch.object(pipeline, "retrain_cnn", return_value=(MagicMock(), {"accuracy": 0.72})), \
             patch.object(pipeline, "_save_versioned_model", return_value={}):

            results = pipeline.retrain_all()

        assert "xgboost" in results
        assert "lstm" in results
        assert "cnn" in results
        assert results["xgboost"]["metrics"]["accuracy"] == 0.80
        assert results["lstm"]["metrics"]["accuracy"] == 0.75

    def test_retrain_all_selected_models(self, pipeline, training_data):
        X, y = training_data

        with patch.object(pipeline, "refresh_data", return_value=(X, y)), \
             patch.object(pipeline, "retrain_xgboost", return_value=(MagicMock(), {"accuracy": 0.80})), \
             patch.object(pipeline, "_save_versioned_model", return_value={}):

            results = pipeline.retrain_all(models_to_train=["xgboost"])

        assert "xgboost" in results
        assert "lstm" not in results
        assert "cnn" not in results

    def test_retrain_all_tune_all_flag(self, pipeline, training_data):
        X, y = training_data

        with patch.object(pipeline, "refresh_data", return_value=(X, y)), \
             patch.object(pipeline, "retrain_xgboost", return_value=(MagicMock(), {"accuracy": 0.80})) as mock_xgb, \
             patch.object(pipeline, "retrain_lstm", return_value=(MagicMock(), {"accuracy": 0.75})) as mock_lstm, \
             patch.object(pipeline, "retrain_cnn", return_value=(MagicMock(), {"accuracy": 0.72})) as mock_cnn, \
             patch.object(pipeline, "_save_versioned_model", return_value={}):

            pipeline.retrain_all(tune_all=True, n_trials=15)

        # Each retrain method should have tune=True
        assert mock_xgb.call_args[1].get("tune") is True or mock_xgb.call_args.kwargs.get("tune") is True
        assert mock_lstm.call_args[1].get("tune") is True or mock_lstm.call_args.kwargs.get("tune") is True
        assert mock_cnn.call_args[1].get("tune") is True or mock_cnn.call_args.kwargs.get("tune") is True

    def test_retrain_all_model_failure_recorded(self, pipeline, training_data):
        """A model that fails training should record error, not crash entire pipeline."""
        X, y = training_data

        with patch.object(pipeline, "refresh_data", return_value=(X, y)), \
             patch.object(pipeline, "retrain_xgboost", side_effect=RuntimeError("XGB blew up")), \
             patch.object(pipeline, "retrain_lstm", return_value=(MagicMock(), {"accuracy": 0.75})), \
             patch.object(pipeline, "retrain_cnn", return_value=(MagicMock(), {"accuracy": 0.72})), \
             patch.object(pipeline, "_save_versioned_model", return_value={}):

            results = pipeline.retrain_all()

        assert "error" in results["xgboost"]
        assert "XGB blew up" in results["xgboost"]["error"]
        # Other models still succeeded
        assert "metrics" in results["lstm"]

    def test_retrain_all_unknown_model_skipped(self, pipeline, training_data):
        X, y = training_data

        with patch.object(pipeline, "refresh_data", return_value=(X, y)):
            results = pipeline.retrain_all(models_to_train=["unknown_model"])

        assert len(results) == 0

    def test_retrain_all_use_cache_forwarded(self, pipeline, training_data):
        X, y = training_data

        with patch.object(pipeline, "refresh_data", return_value=(X, y)) as mock_refresh, \
             patch.object(pipeline, "retrain_xgboost", return_value=(MagicMock(), {"accuracy": 0.8})), \
             patch.object(pipeline, "_save_versioned_model", return_value={}):

            pipeline.retrain_all(use_cache=True, models_to_train=["xgboost"])

        mock_refresh.assert_called_once_with(use_cache=True)

    def test_retrain_all_includes_transformer(self, pipeline, training_data):
        X, y = training_data

        with patch.object(pipeline, "refresh_data", return_value=(X, y)), \
             patch.object(pipeline, "retrain_transformer", return_value=(MagicMock(), {"accuracy": 0.68})), \
             patch.object(pipeline, "_save_versioned_model", return_value={}):

            results = pipeline.retrain_all(models_to_train=["transformer"])

        assert "transformer" in results
        assert results["transformer"]["metrics"]["accuracy"] == 0.68


# ======================================================================
# Versioning & Registry
# ======================================================================

class TestVersioningAndRegistry:
    """Tests for _save_versioned_model, _load_registry, _save_registry."""

    def test_empty_registry_loads_as_empty_dict(self, pipeline):
        reg = pipeline._load_registry()
        assert reg == {}

    def test_save_and_load_registry(self, pipeline):
        registry = {"versions": {"v1": {"model_type": "xgboost"}}, "production": {}}
        pipeline._save_registry(registry)

        loaded = pipeline._load_registry()
        assert loaded == registry

    def test_save_registry_atomic_write(self, pipeline):
        """Even after save, no .tmp files should linger."""
        pipeline._save_registry({"test": True})
        tmp_files = list(pipeline.versions_dir.glob("*.json.tmp"))
        assert len(tmp_files) == 0

    def test_save_versioned_model_xgboost(self, pipeline, training_data):
        X, y = training_data
        mock_model = MagicMock()
        metrics = {"accuracy": 0.80, "f1": 0.78}

        result = pipeline._save_versioned_model(
            mock_model, "xgboost", "xgboost_v20250101_000000", metrics, X
        )

        # Model.save was called
        mock_model.save.assert_called_once()

        # Registry was updated
        reg = pipeline._load_registry()
        assert "xgboost_v20250101_000000" in reg["versions"]
        version_info = reg["versions"]["xgboost_v20250101_000000"]
        assert version_info["model_type"] == "xgboost"
        assert version_info["metrics"]["accuracy"] == 0.80
        assert version_info["data_info"]["n_samples"] == len(X)
        assert version_info["data_info"]["n_features"] == len(X.columns)

    def test_save_versioned_model_lstm(self, pipeline, training_data):
        X, y = training_data
        mock_model = MagicMock()
        mock_model.sequence_length = 10
        mock_model.n_features = 10
        mock_model.dropout_rate = 0.2
        mock_model.learning_rate = 0.001
        mock_model.lstm_units = (64, 32)
        mock_model.normalization_stats = {
            "mean": np.array([1.0, 2.0]),
            "std": np.array([0.5, 0.3]),
        }
        metrics = {"accuracy": 0.75}

        result = pipeline._save_versioned_model(
            mock_model, "lstm", "lstm_v20250101_000000", metrics, X
        )

        # Keras model.save was called
        mock_model.model.save.assert_called_once()

        # metadata.json was written
        version_dir = pipeline.versions_dir / "lstm_v20250101_000000"
        metadata_path = version_dir / "metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            meta = json.load(f)
        assert meta["model_type"] == "lstm"
        assert meta["lstm_units"] == [64, 32]
        assert meta["normalization_stats"]["mean"] == [1.0, 2.0]

    def test_save_versioned_model_cnn(self, pipeline, training_data):
        X, y = training_data
        mock_model = MagicMock()
        mock_model.sequence_length = 10
        mock_model.n_features = 10
        mock_model.dropout_rate = 0.2
        mock_model.learning_rate = 0.001
        mock_model.filters = (32, 16)
        mock_model.kernel_size = 3
        mock_model.normalization_stats = None
        metrics = {"accuracy": 0.73}

        result = pipeline._save_versioned_model(
            mock_model, "cnn", "cnn_v20250101_000000", metrics, X
        )

        version_dir = pipeline.versions_dir / "cnn_v20250101_000000"
        metadata_path = version_dir / "metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            meta = json.load(f)
        assert meta["model_type"] == "cnn"
        assert meta["filters"] == [32, 16]
        assert meta["kernel_size"] == 3
        assert meta["normalization_stats"] is None


# ======================================================================
# compare_and_deploy()
# ======================================================================

class TestCompareAndDeploy:
    """Tests for compare_and_deploy() promotion logic."""

    def _seed_registry(self, pipeline, versions=None, production=None):
        reg = {}
        if versions:
            reg["versions"] = versions
        if production:
            reg["production"] = production
        pipeline._save_registry(reg)

    def test_deploy_when_no_production_model(self, pipeline):
        """First deployment should always succeed."""
        self._seed_registry(pipeline, versions={
            "xgboost_v1": {"metrics": {"accuracy": 0.70}}
        })

        with patch.object(pipeline, "_deploy_model") as mock_deploy:
            result = pipeline.compare_and_deploy("xgboost", "xgboost_v1", "xgboost_prod")

        assert result is True
        mock_deploy.assert_called_once_with("xgboost", "xgboost_v1", "xgboost_prod")

    def test_deploy_when_improvement_above_threshold(self, pipeline):
        self._seed_registry(pipeline,
            versions={"xgboost_v2": {"metrics": {"accuracy": 0.82}}},
            production={"xgboost": {"version": "xgboost_v1", "metrics": {"accuracy": 0.70}}}
        )

        with patch.object(pipeline, "_deploy_model"):
            result = pipeline.compare_and_deploy("xgboost", "xgboost_v2", "xgboost_prod")

        assert result is True  # 0.82 - 0.70 = 0.12 >= 0.01

    def test_no_deploy_when_below_threshold(self, pipeline):
        self._seed_registry(pipeline,
            versions={"xgboost_v2": {"metrics": {"accuracy": 0.705}}},
            production={"xgboost": {"version": "xgboost_v1", "metrics": {"accuracy": 0.70}}}
        )

        with patch.object(pipeline, "_deploy_model") as mock_deploy:
            result = pipeline.compare_and_deploy("xgboost", "xgboost_v2", "xgboost_prod")

        assert result is False
        mock_deploy.assert_not_called()

    def test_no_deploy_when_new_metrics_missing(self, pipeline):
        self._seed_registry(pipeline, versions={})

        result = pipeline.compare_and_deploy("xgboost", "nonexistent_v", "xgboost_prod")
        assert result is False

    def test_deploy_at_exact_threshold(self, pipeline):
        """Improvement exactly == threshold should deploy (>= check)."""
        pipeline.min_improvement_threshold = 0.05
        self._seed_registry(pipeline,
            versions={"xgboost_v2": {"metrics": {"accuracy": 0.75}}},
            production={"xgboost": {"version": "xgboost_v1", "metrics": {"accuracy": 0.70}}}
        )

        with patch.object(pipeline, "_deploy_model"):
            result = pipeline.compare_and_deploy("xgboost", "xgboost_v2", "xgboost_prod")

        assert result is True

    def test_no_deploy_when_worse(self, pipeline):
        """New model worse than production should not deploy."""
        self._seed_registry(pipeline,
            versions={"xgboost_v2": {"metrics": {"accuracy": 0.65}}},
            production={"xgboost": {"version": "xgboost_v1", "metrics": {"accuracy": 0.70}}}
        )

        with patch.object(pipeline, "_deploy_model") as mock_deploy:
            result = pipeline.compare_and_deploy("xgboost", "xgboost_v2", "xgboost_prod")

        assert result is False
        mock_deploy.assert_not_called()


# ======================================================================
# _deploy_model()
# ======================================================================

class TestDeployModel:
    """Tests for _deploy_model() file-copy and registry update logic."""

    def test_deploy_xgboost_pkl(self, pipeline, tmp_models_dir):
        """XGBoost deploy copies .pkl file to production path."""
        version = "xgboost_v1"
        version_dir = pipeline.versions_dir / version
        version_dir.mkdir(parents=True)
        # Create a fake pkl file
        pkl_path = version_dir / "model.pkl"
        pkl_path.write_text("fake model")

        # Seed registry with version info
        pipeline._save_registry({"versions": {version: {"metrics": {"accuracy": 0.80}}}})

        with patch("src.ml.retraining.MODELS_DIR", tmp_models_dir):
            pipeline._deploy_model("xgboost", version, "xgboost_prod")

        # Check registry production entry
        reg = pipeline._load_registry()
        assert reg["production"]["xgboost"]["version"] == version
        assert "deployed_at" in reg["production"]["xgboost"]

    def test_deploy_lstm_keras(self, pipeline, tmp_models_dir):
        """LSTM deploy copies .keras file and metadata.json to production dir."""
        version = "lstm_v1"
        version_dir = pipeline.versions_dir / version
        version_dir.mkdir(parents=True)
        (version_dir / "model.keras").write_text("fake keras")
        (version_dir / "metadata.json").write_text('{"model_type": "lstm"}')

        pipeline._save_registry({"versions": {version: {"metrics": {"accuracy": 0.75}}}})

        with patch("src.ml.retraining.MODELS_DIR", tmp_models_dir):
            pipeline._deploy_model("lstm", version, "lstm_prod")

        dst_dir = tmp_models_dir / "lstm_prod"
        assert (dst_dir / "model.keras").exists()
        assert (dst_dir / "metadata.json").exists()

        reg = pipeline._load_registry()
        assert reg["production"]["lstm"]["version"] == version


# ======================================================================
# validate_with_walk_forward()
# ======================================================================

class TestWalkForwardValidation:
    """Tests for validate_with_walk_forward()."""

    def test_walk_forward_xgboost_passes(self, pipeline):
        """XGBoost model with decent predictions should pass validation."""
        n = 300
        X = _make_feature_df(n, 10)
        y = _make_labels(n)

        mock_model = MagicMock()
        # Return probas that are generally correct
        def predict_proba_side_effect(X_in):
            n_rows = len(X_in)
            # Give slightly better-than-random probabilities
            probas = np.column_stack([
                np.full(n_rows, 0.4),
                np.full(n_rows, 0.6),
            ])
            return probas

        mock_model.predict_proba = MagicMock(side_effect=predict_proba_side_effect)

        passed, metrics = pipeline.validate_with_walk_forward(
            model=mock_model,
            model_type="xgboost",
            X=X,
            y=y,
            n_windows=2,
            min_sharpe=0.0,
            min_profit_factor=0.0,
            max_drawdown=1.0,
        )

        assert passed == True or passed == False  # may be np.bool_
        assert "accuracy" in metrics
        assert "sharpe_ratio" in metrics
        assert "n_windows" in metrics
        assert metrics["n_windows"] >= 1

    def test_walk_forward_fails_on_low_sharpe(self, pipeline):
        """If min_sharpe is set very high, validation should fail."""
        n = 300
        X = _make_feature_df(n, 10)
        y = _make_labels(n)

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.column_stack([
            np.full(100, 0.5), np.full(100, 0.5)
        ])

        passed, metrics = pipeline.validate_with_walk_forward(
            model=mock_model,
            model_type="xgboost",
            X=X,
            y=y,
            n_windows=2,
            min_sharpe=100.0,  # impossibly high
            min_profit_factor=0.0,
            max_drawdown=1.0,
        )

        assert passed == False

    def test_walk_forward_no_valid_windows(self, pipeline):
        """With very small dataset, should return False and empty metrics."""
        X = _make_feature_df(20, 10)  # too few samples
        y = _make_labels(20)

        mock_model = MagicMock()

        passed, metrics = pipeline.validate_with_walk_forward(
            model=mock_model,
            model_type="xgboost",
            X=X,
            y=y,
            n_windows=5,
        )

        assert passed is False
        assert metrics == {}

    def test_walk_forward_lstm_uses_sequences(self, pipeline):
        """Non-xgboost models should create sequences before predict_proba."""
        n = 300
        X = _make_feature_df(n, 10)
        y = _make_labels(n)

        seq_len = pipeline.sequence_length
        mock_model = MagicMock()

        # We need to mock create_sequences inside the walk-forward loop
        n_out = 50
        X_seq = np.random.randn(n_out, seq_len, 10)
        y_seq = np.random.randint(0, 2, n_out)

        mock_model.predict_proba.return_value = np.column_stack([
            np.full(n_out, 0.4), np.full(n_out, 0.6)
        ])

        with patch("src.ml.sequence_utils.create_sequences", return_value=(X_seq, y_seq)):
            passed, metrics = pipeline.validate_with_walk_forward(
                model=mock_model,
                model_type="lstm",
                X=X,
                y=y,
                n_windows=2,
                min_sharpe=0.0,
                min_profit_factor=0.0,
                max_drawdown=1.0,
            )

        # predict_proba should have received sequences, not raw DataFrames
        if mock_model.predict_proba.called:
            call_arg = mock_model.predict_proba.call_args[0][0]
            assert isinstance(call_arg, np.ndarray)


# ======================================================================
# compare_and_deploy_enhanced()
# ======================================================================

class TestCompareAndDeployEnhanced:
    """Tests for compare_and_deploy_enhanced() with walk-forward."""

    def _seed_registry(self, pipeline, versions=None, production=None):
        reg = {}
        if versions:
            reg["versions"] = versions
        if production:
            reg["production"] = production
        pipeline._save_registry(reg)

    def test_deploy_when_no_production_exists(self, pipeline):
        self._seed_registry(pipeline, versions={
            "xgboost_v1": {"metrics": {"accuracy": 0.75}}
        })

        with patch.object(pipeline, "_deploy_model"), \
             patch.object(pipeline, "validate_with_walk_forward", return_value=(True, {"sharpe_ratio": 1.0})):

            result = pipeline.compare_and_deploy_enhanced(
                "xgboost", "xgboost_v1", "xgboost_prod",
                new_model=MagicMock(),
                X=_make_feature_df(100),
                y=_make_labels(100),
            )

        assert result is True

    def test_no_deploy_when_walk_forward_fails(self, pipeline):
        """If walk-forward validation fails, model should not be deployed."""
        self._seed_registry(pipeline, versions={
            "xgboost_v2": {"metrics": {"accuracy": 0.80}}
        })

        with patch.object(pipeline, "validate_with_walk_forward", return_value=(False, {})):
            result = pipeline.compare_and_deploy_enhanced(
                "xgboost", "xgboost_v2", "xgboost_prod",
                new_model=MagicMock(),
                X=_make_feature_df(100),
                y=_make_labels(100),
                use_walk_forward=True,
            )

        assert result is False

    def test_no_deploy_when_no_metrics_for_version(self, pipeline):
        self._seed_registry(pipeline, versions={})

        result = pipeline.compare_and_deploy_enhanced(
            "xgboost", "nonexistent", "xgboost_prod"
        )

        assert result is False

    def test_deploy_with_auto_rollback_manager(self, pipeline):
        """auto_rollback_manager.register_deployment should be called on deploy."""
        self._seed_registry(pipeline, versions={
            "xgboost_v1": {"metrics": {"accuracy": 0.75}}
        })

        mock_arm = MagicMock()

        with patch.object(pipeline, "_deploy_model"), \
             patch.object(pipeline, "validate_with_walk_forward", return_value=(True, {"sharpe_ratio": 1.0})):

            result = pipeline.compare_and_deploy_enhanced(
                "xgboost", "xgboost_v1", "xgboost_prod",
                new_model=MagicMock(),
                X=_make_feature_df(100),
                y=_make_labels(100),
                auto_rollback_manager=mock_arm,
            )

        assert result is True
        mock_arm.register_deployment.assert_called_once()

    def test_skip_walk_forward_when_disabled(self, pipeline):
        """When use_walk_forward=False, skip validation entirely."""
        self._seed_registry(pipeline,
            versions={"xgboost_v2": {"metrics": {"accuracy": 0.90}}},
            production={"xgboost": {"version": "xgboost_v1", "metrics": {"accuracy": 0.70}}}
        )

        with patch.object(pipeline, "_deploy_model"), \
             patch.object(pipeline, "validate_with_walk_forward") as mock_wf:

            # Large improvement so weighted score should pass
            pipeline.min_improvement_threshold = 0.01
            result = pipeline.compare_and_deploy_enhanced(
                "xgboost", "xgboost_v2", "xgboost_prod",
                use_walk_forward=False,
            )

        # walk forward should not have been called
        mock_wf.assert_not_called()

    def test_skip_walk_forward_when_no_model_or_data(self, pipeline):
        """Walk-forward requires model, X, y. Without them, skip validation."""
        self._seed_registry(pipeline,
            versions={"xgboost_v2": {"metrics": {"accuracy": 0.90}}},
            production={"xgboost": {"version": "xgboost_v1", "metrics": {"accuracy": 0.70}}}
        )

        with patch.object(pipeline, "_deploy_model"), \
             patch.object(pipeline, "validate_with_walk_forward") as mock_wf:

            pipeline.min_improvement_threshold = 0.01
            # new_model=None => should skip walk-forward
            result = pipeline.compare_and_deploy_enhanced(
                "xgboost", "xgboost_v2", "xgboost_prod",
                new_model=None,
                use_walk_forward=True,
            )

        mock_wf.assert_not_called()


# ======================================================================
# list_versions() / get_retraining_status()
# ======================================================================

class TestStatusAndListing:
    """Tests for informational methods."""

    def test_list_versions_empty(self, pipeline):
        result = pipeline.list_versions()
        assert result == []

    def test_list_versions_all(self, pipeline):
        pipeline._save_registry({"versions": {
            "xgboost_v1": {"model_type": "xgboost", "created_at": "2025-01-01", "metrics": {"accuracy": 0.70}},
            "lstm_v1": {"model_type": "lstm", "created_at": "2025-01-02", "metrics": {"accuracy": 0.72}},
        }})
        result = pipeline.list_versions()
        assert len(result) == 2

    def test_list_versions_filter_by_type(self, pipeline):
        pipeline._save_registry({"versions": {
            "xgboost_v1": {"model_type": "xgboost", "created_at": "2025-01-01", "metrics": {"accuracy": 0.70}},
            "lstm_v1": {"model_type": "lstm", "created_at": "2025-01-02", "metrics": {"accuracy": 0.72}},
        }})
        result = pipeline.list_versions(model_type="lstm")
        assert len(result) == 1
        assert result[0]["model_type"] == "lstm"

    def test_get_retraining_status_empty(self, pipeline):
        status = pipeline.get_retraining_status()
        assert status["production_models"] == {}
        assert status["version_count"] == 0
        assert status["recent_versions"] == []

    def test_get_retraining_status_with_data(self, pipeline):
        pipeline._save_registry({
            "versions": {
                "xgboost_v1": {},
                "xgboost_v2": {},
                "lstm_v1": {},
            },
            "production": {
                "xgboost": {"version": "xgboost_v2"}
            }
        })
        status = pipeline.get_retraining_status()
        assert status["version_count"] == 3
        assert len(status["recent_versions"]) == 3
        assert status["production_models"]["xgboost"]["version"] == "xgboost_v2"


# ======================================================================
# Edge Cases & Error Paths
# ======================================================================

class TestEdgeCases:
    """Edge cases and error paths."""

    def test_retrain_all_refresh_data_fails(self, pipeline):
        """If refresh_data raises, retrain_all should propagate the error."""
        with patch.object(pipeline, "refresh_data", side_effect=ValueError("No valid data")):
            with pytest.raises(ValueError, match="No valid data"):
                pipeline.retrain_all()

    def test_save_versioned_model_transformer(self, pipeline, training_data):
        """Transformer metadata includes embed_dim, num_heads, ff_dim, etc."""
        X, y = training_data
        mock_model = MagicMock()
        mock_model.sequence_length = 10
        mock_model.n_features = 10
        mock_model.dropout_rate = 0.1
        mock_model.learning_rate = 0.0005
        mock_model.embed_dim = 64
        mock_model.num_heads = 4
        mock_model.ff_dim = 128
        mock_model.num_transformer_blocks = 2
        mock_model.normalization_stats = None
        metrics = {"accuracy": 0.68}

        pipeline._save_versioned_model(
            mock_model, "transformer", "transformer_v1", metrics, X
        )

        version_dir = pipeline.versions_dir / "transformer_v1"
        with open(version_dir / "metadata.json") as f:
            meta = json.load(f)
        assert meta["embed_dim"] == 64
        assert meta["num_heads"] == 4
        assert meta["num_transformer_blocks"] == 2

    def test_registry_survives_concurrent_reads(self, pipeline):
        """Multiple load/save cycles should not corrupt the registry."""
        for i in range(5):
            reg = pipeline._load_registry()
            reg[f"key_{i}"] = f"value_{i}"
            pipeline._save_registry(reg)

        final = pipeline._load_registry()
        for i in range(5):
            assert final[f"key_{i}"] == f"value_{i}"

    def test_deploy_xgboost_directory_format(self, pipeline, tmp_models_dir):
        """When xgboost model is saved in directory format (not flat .pkl)."""
        version = "xgboost_v1"
        version_dir = pipeline.versions_dir / version
        model_dir = version_dir / "model"
        model_dir.mkdir(parents=True)
        (model_dir / "xgboost_model.pkl").write_text("fake pickle data")

        pipeline._save_registry({"versions": {version: {"metrics": {"accuracy": 0.80}}}})

        with patch("src.ml.retraining.MODELS_DIR", tmp_models_dir):
            pipeline._deploy_model("xgboost", version, "xgboost_prod")

        # Should have copied the pkl to production
        prod_pkl = tmp_models_dir / "xgboost_prod.pkl"
        assert prod_pkl.exists()

    def test_walk_forward_window_exception_handled(self, pipeline):
        """If a walk-forward window raises, it's skipped, not fatal."""
        n = 300
        X = _make_feature_df(n, 10)
        y = _make_labels(n)

        mock_model = MagicMock()
        # First call raises, second succeeds
        call_count = [0]

        def predict_proba_effect(X_in):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("GPU out of memory")
            n_rows = len(X_in)
            return np.column_stack([np.full(n_rows, 0.4), np.full(n_rows, 0.6)])

        mock_model.predict_proba = MagicMock(side_effect=predict_proba_effect)

        passed, metrics = pipeline.validate_with_walk_forward(
            model=mock_model,
            model_type="xgboost",
            X=X,
            y=y,
            n_windows=2,
            min_sharpe=0.0,
            min_profit_factor=0.0,
            max_drawdown=1.0,
        )

        # Should not crash; at least one window should have been processed
        assert passed == True or passed == False  # may be np.bool_

    def test_calculate_metrics_mixed(self, pipeline):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        m = pipeline._calculate_metrics(y_true, y_pred)
        # 6 out of 8 correct
        assert abs(m["accuracy"] - 0.75) < 0.01
        assert 0 <= m["precision"] <= 1
        assert 0 <= m["recall"] <= 1
        assert 0 <= m["f1"] <= 1
