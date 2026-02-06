"""
Comprehensive unit tests for MLStrategy.

Tests cover:
- Signal generation: mock the ML model predict_proba for controlled values
- Confidence threshold: signals above/below threshold
- Trade recommendations: buy/sell signal generation
- Portfolio value and position-aware sizing
- Ensemble mode: mock ensemble predict_single
- Edge cases: empty symbols, model not loaded, NaN predictions
- Backtest signals
- Hold signal generation
- Feature column detection
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timedelta

from src.strategy.ml_strategy import MLStrategy, SignalType, TradingSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n_days=100, base_price=100.0):
    """Create a realistic OHLCV DataFrame with the feature columns
    that MLStrategy's generate_signal expects after feature engineering."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="B")
    returns = np.random.randn(n_days) * 0.02
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_days)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n_days)) * 0.01)
    open_ = low + (high - low) * np.random.rand(n_days)
    volume = np.random.randint(1_000_000, 10_000_000, n_days)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    # Add the feature columns that generate_signal reads for the signal object
    df["rsi_14"] = 50.0 + np.random.randn(n_days) * 10
    df["macd"] = np.random.randn(n_days) * 0.5
    df["price_vs_sma20"] = np.random.randn(n_days) * 2
    # Extra features that _get_feature_columns would pick up
    df["sma_20"] = close
    df["sma_50"] = close * 0.98
    df["ema_12"] = close * 1.01
    df["bb_upper"] = close * 1.02
    df["bb_lower"] = close * 0.98
    return df


def _make_mock_model(prob_up=0.7, prob_down=0.3):
    """Create a mock XGBoost-like model that returns fixed probabilities."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[prob_down, prob_up]])
    model.feature_names = None  # use fallback feature selection
    return model


def _make_mock_ensemble(prediction=1, confidence=0.75):
    """Create a mock EnsembleModel that returns fixed prediction/confidence."""
    model = MagicMock()
    model.predict_single.return_value = (
        prediction,
        confidence,
        {"xgboost": 0.7, "lstm": 0.8, "cnn": 0.75},
    )
    model.feature_names = None
    return model


def _make_strategy(
    model=None,
    model_type="xgboost",
    confidence_threshold=0.6,
    min_confidence_sell=0.55,
):
    """Create an MLStrategy with all external dependencies mocked."""
    with patch("src.strategy.ml_strategy.DataFetcher") as MockDF, \
         patch("src.strategy.ml_strategy.FeatureEngineer") as MockFE:

        mock_fetcher = MockDF.return_value
        mock_fe = MockFE.return_value

        # DataFetcher: fetch_historical returns a raw DF
        raw_df = _make_ohlcv_df()
        mock_fetcher.fetch_historical.return_value = raw_df

        # FeatureEngineer: add_all_features_extended returns the same DF
        # (it already has the feature columns we added)
        mock_fe.add_all_features_extended.return_value = raw_df

        # DataFetcher.get_latest_price returns a sensible default
        mock_fetcher.get_latest_price.return_value = 150.0

        strategy = MLStrategy(
            model=model,
            model_type=model_type,
            confidence_threshold=confidence_threshold,
            min_confidence_sell=min_confidence_sell,
        )

    # Re-assign the mocks since __init__ already set them
    strategy.data_fetcher = mock_fetcher
    strategy.feature_engineer = mock_fe
    return strategy


@pytest.fixture
def mock_model():
    return _make_mock_model()


@pytest.fixture
def strategy(mock_model):
    return _make_strategy(model=mock_model)


@pytest.fixture
def mock_risk_manager():
    rm = MagicMock()
    rm.calculate_stop_loss.return_value = 142.50
    rm.calculate_position_size.return_value = 10
    return rm


# ===================================================================
# Signal Generation (single symbol)
# ===================================================================

class TestSignalGeneration:

    def test_generate_signal_returns_trading_signal(self, strategy):
        signal = strategy.generate_signal("AAPL")
        assert isinstance(signal, TradingSignal)
        assert signal.symbol == "AAPL"

    def test_buy_signal_when_prob_up_above_threshold(self):
        model = _make_mock_model(prob_up=0.75, prob_down=0.25)
        strategy = _make_strategy(model=model, confidence_threshold=0.6)
        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.BUY

    def test_sell_signal_when_prob_down_above_sell_threshold(self):
        # prob_down = 0.60 >= min_confidence_sell (0.55), prob_up = 0.40 < 0.6
        model = _make_mock_model(prob_up=0.40, prob_down=0.60)
        strategy = _make_strategy(
            model=model, confidence_threshold=0.6, min_confidence_sell=0.55,
        )
        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.SELL

    def test_hold_signal_when_below_both_thresholds(self):
        # prob_up=0.52, prob_down=0.48: neither is >= threshold
        model = _make_mock_model(prob_up=0.52, prob_down=0.48)
        strategy = _make_strategy(
            model=model, confidence_threshold=0.6, min_confidence_sell=0.55,
        )
        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.HOLD

    def test_confidence_value_is_max_of_probs(self):
        model = _make_mock_model(prob_up=0.7, prob_down=0.3)
        strategy = _make_strategy(model=model)
        signal = strategy.generate_signal("AAPL")
        assert signal.confidence == pytest.approx(0.7)

    def test_predicted_direction_up(self):
        model = _make_mock_model(prob_up=0.8, prob_down=0.2)
        strategy = _make_strategy(model=model)
        signal = strategy.generate_signal("AAPL")
        assert signal.predicted_direction == 1

    def test_predicted_direction_down(self):
        model = _make_mock_model(prob_up=0.3, prob_down=0.7)
        strategy = _make_strategy(model=model)
        signal = strategy.generate_signal("AAPL")
        assert signal.predicted_direction == 0

    def test_signal_has_feature_dict(self):
        model = _make_mock_model(prob_up=0.7, prob_down=0.3)
        strategy = _make_strategy(model=model)
        signal = strategy.generate_signal("AAPL")
        assert signal.features is not None
        assert "rsi_14" in signal.features
        assert "macd" in signal.features
        assert "price_vs_sma20" in signal.features


# ===================================================================
# Confidence Thresholds
# ===================================================================

class TestConfidenceThresholds:

    def test_exact_buy_threshold(self):
        model = _make_mock_model(prob_up=0.6, prob_down=0.4)
        strategy = _make_strategy(model=model, confidence_threshold=0.6)
        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.BUY

    def test_just_below_buy_threshold(self):
        model = _make_mock_model(prob_up=0.599, prob_down=0.401)
        strategy = _make_strategy(
            model=model, confidence_threshold=0.6, min_confidence_sell=0.55,
        )
        signal = strategy.generate_signal("AAPL")
        # prob_up < 0.6 so not BUY; prob_down=0.401 < 0.55 so not SELL => HOLD
        assert signal.signal == SignalType.HOLD

    def test_exact_sell_threshold(self):
        model = _make_mock_model(prob_up=0.45, prob_down=0.55)
        strategy = _make_strategy(
            model=model, confidence_threshold=0.6, min_confidence_sell=0.55,
        )
        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.SELL

    def test_just_below_sell_threshold(self):
        model = _make_mock_model(prob_up=0.451, prob_down=0.549)
        strategy = _make_strategy(
            model=model, confidence_threshold=0.6, min_confidence_sell=0.55,
        )
        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.HOLD

    def test_high_confidence_threshold(self):
        model = _make_mock_model(prob_up=0.85, prob_down=0.15)
        strategy = _make_strategy(model=model, confidence_threshold=0.9)
        signal = strategy.generate_signal("AAPL")
        # 0.85 < 0.9 threshold, not a BUY
        assert signal.signal != SignalType.BUY

    def test_low_confidence_threshold_triggers_buy(self):
        model = _make_mock_model(prob_up=0.51, prob_down=0.49)
        strategy = _make_strategy(model=model, confidence_threshold=0.5)
        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.BUY


# ===================================================================
# Trade Recommendations
# ===================================================================

class TestTradeRecommendations:

    def test_buy_recommendation_generated(self, mock_risk_manager):
        model = _make_mock_model(prob_up=0.8, prob_down=0.2)
        strategy = _make_strategy(model=model, confidence_threshold=0.6)

        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={},
            risk_manager=mock_risk_manager,
        )
        assert len(recs) == 1
        assert recs[0]["action"] == "BUY"
        assert recs[0]["symbol"] == "AAPL"
        assert recs[0]["shares"] == 10
        assert "stop_loss" in recs[0]

    def test_sell_recommendation_generated(self, mock_risk_manager):
        model = _make_mock_model(prob_up=0.35, prob_down=0.65)
        strategy = _make_strategy(
            model=model, confidence_threshold=0.6, min_confidence_sell=0.55,
        )

        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={"AAPL": 20},
            risk_manager=mock_risk_manager,
        )
        assert len(recs) == 1
        assert recs[0]["action"] == "SELL"
        assert recs[0]["symbol"] == "AAPL"
        assert recs[0]["shares"] == 20

    def test_hold_generates_no_recommendation(self, mock_risk_manager):
        model = _make_mock_model(prob_up=0.52, prob_down=0.48)
        strategy = _make_strategy(
            model=model, confidence_threshold=0.6, min_confidence_sell=0.55,
        )

        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={},
            risk_manager=mock_risk_manager,
        )
        assert len(recs) == 0

    def test_buy_signal_no_rec_if_already_holding(self, mock_risk_manager):
        """BUY signal ignored if we already hold the symbol (traditional mode)."""
        model = _make_mock_model(prob_up=0.8, prob_down=0.2)
        strategy = _make_strategy(model=model, confidence_threshold=0.6)

        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={"AAPL": 10},
            risk_manager=mock_risk_manager,
        )
        assert len(recs) == 0

    def test_sell_signal_no_rec_if_no_position(self, mock_risk_manager):
        """SELL signal ignored if we don't hold the symbol."""
        model = _make_mock_model(prob_up=0.35, prob_down=0.65)
        strategy = _make_strategy(
            model=model, confidence_threshold=0.6, min_confidence_sell=0.55,
        )

        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={},
            risk_manager=mock_risk_manager,
        )
        assert len(recs) == 0

    def test_recommendations_sorted_by_confidence(self, mock_risk_manager):
        """Multiple symbols: recommendations sorted highest confidence first."""
        model = MagicMock()
        model.feature_names = None

        # AAPL -> 0.70 confidence BUY, MSFT -> 0.85 confidence BUY
        call_count = {"n": 0}

        def predict_side_effect(X):
            call_count["n"] += 1
            if call_count["n"] % 2 == 1:
                return np.array([[0.3, 0.7]])  # AAPL first
            else:
                return np.array([[0.15, 0.85]])  # MSFT second
        model.predict_proba.side_effect = predict_side_effect

        strategy = _make_strategy(model=model, confidence_threshold=0.6)

        recs = strategy.get_trade_recommendations(
            symbols=["AAPL", "MSFT"],
            portfolio_value=100_000.0,
            current_positions={},
            risk_manager=mock_risk_manager,
        )
        assert len(recs) == 2
        assert recs[0]["confidence"] >= recs[1]["confidence"]

    def test_recommendations_with_zero_position_size(self, mock_risk_manager):
        """If risk manager returns 0 shares, no recommendation is made."""
        mock_risk_manager.calculate_position_size.return_value = 0
        model = _make_mock_model(prob_up=0.8, prob_down=0.2)
        strategy = _make_strategy(model=model, confidence_threshold=0.6)

        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={},
            risk_manager=mock_risk_manager,
        )
        assert len(recs) == 0

    def test_recommendation_includes_price(self, mock_risk_manager):
        model = _make_mock_model(prob_up=0.8, prob_down=0.2)
        strategy = _make_strategy(model=model)

        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={},
            risk_manager=mock_risk_manager,
        )
        assert recs[0]["price"] == pytest.approx(150.0)

    def test_recommendation_skips_invalid_price(self, mock_risk_manager):
        model = _make_mock_model(prob_up=0.8, prob_down=0.2)
        strategy = _make_strategy(model=model)
        strategy.data_fetcher.get_latest_price.return_value = 0.0

        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={},
            risk_manager=mock_risk_manager,
        )
        assert len(recs) == 0


# ===================================================================
# Portfolio-aware Sizing (target_weights mode)
# ===================================================================

class TestPortfolioWeights:

    def test_target_weight_buy(self, mock_risk_manager):
        model = _make_mock_model(prob_up=0.5, prob_down=0.5)
        strategy = _make_strategy(model=model)

        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={"AAPL": 0},
            risk_manager=mock_risk_manager,
            target_weights={"AAPL": 0.10},  # 10% = $10k / $150 = 66 shares
        )
        assert len(recs) == 1
        assert recs[0]["action"] == "BUY"
        assert recs[0]["shares"] == 66  # int(10000/150) = 66

    def test_target_weight_sell(self, mock_risk_manager):
        model = _make_mock_model(prob_up=0.5, prob_down=0.5)
        strategy = _make_strategy(model=model)

        # Currently hold 100, target=10% = 66 shares -> sell 34
        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={"AAPL": 100},
            risk_manager=mock_risk_manager,
            target_weights={"AAPL": 0.10},
        )
        assert len(recs) == 1
        assert recs[0]["action"] == "SELL"
        assert recs[0]["shares"] == 34  # 100 - 66

    def test_target_weight_no_trade_if_close(self, mock_risk_manager):
        model = _make_mock_model(prob_up=0.5, prob_down=0.5)
        strategy = _make_strategy(model=model)

        # Currently hold 66 shares, target is 66 -> difference = 0
        recs = strategy.get_trade_recommendations(
            symbols=["AAPL"],
            portfolio_value=100_000.0,
            current_positions={"AAPL": 66},
            risk_manager=mock_risk_manager,
            target_weights={"AAPL": 0.10},
        )
        assert len(recs) == 0


# ===================================================================
# Ensemble Mode
# ===================================================================

class TestEnsembleMode:

    def test_ensemble_buy_signal(self):
        from src.ml.models.ensemble_model import EnsembleModel

        model = _make_mock_ensemble(prediction=1, confidence=0.80)
        strategy = _make_strategy(model=model, model_type="ensemble")

        # Make isinstance(model, EnsembleModel) return True
        model.__class__ = EnsembleModel

        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.BUY
        assert signal.confidence == pytest.approx(0.80)

    def test_ensemble_sell_signal(self):
        from src.ml.models.ensemble_model import EnsembleModel

        # prediction=0 (down), confidence=0.70
        # For prediction=0: prob_up = 1-0.70 = 0.30, prob_down = 0.70
        model = _make_mock_ensemble(prediction=0, confidence=0.70)
        strategy = _make_strategy(
            model=model,
            model_type="ensemble",
            confidence_threshold=0.6,
            min_confidence_sell=0.55,
        )

        model.__class__ = EnsembleModel

        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.SELL

    def test_ensemble_hold_signal(self):
        from src.ml.models.ensemble_model import EnsembleModel

        # prediction=1 (up), confidence=0.52, so prob_up=0.52 < 0.6 threshold
        # prob_down=0.48 < 0.55 => HOLD
        model = _make_mock_ensemble(prediction=1, confidence=0.52)
        strategy = _make_strategy(
            model=model,
            model_type="ensemble",
            confidence_threshold=0.6,
            min_confidence_sell=0.55,
        )
        model.__class__ = EnsembleModel

        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.HOLD

    def test_ensemble_model_probs_logged(self):
        from src.ml.models.ensemble_model import EnsembleModel

        model = _make_mock_ensemble(prediction=1, confidence=0.80)
        strategy = _make_strategy(model=model, model_type="ensemble")
        model.__class__ = EnsembleModel

        # Just ensure it does not crash and returns valid signal
        signal = strategy.generate_signal("AAPL")
        assert isinstance(signal, TradingSignal)


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:

    def test_model_not_loaded_raises(self):
        strategy = _make_strategy(model=None)
        with pytest.raises(RuntimeError, match="No model loaded"):
            strategy.generate_signal("AAPL")

    def test_empty_symbols_list(self, mock_risk_manager):
        model = _make_mock_model()
        strategy = _make_strategy(model=model)

        signals = strategy.generate_signals([])
        assert signals == {}

        recs = strategy.get_trade_recommendations(
            symbols=[],
            portfolio_value=100_000.0,
            current_positions={},
            risk_manager=mock_risk_manager,
        )
        assert recs == []

    def test_no_data_returns_hold(self):
        model = _make_mock_model()
        strategy = _make_strategy(model=model)
        strategy.data_fetcher.fetch_historical.return_value = pd.DataFrame()

        signal = strategy.generate_signal("FAKE")
        assert signal.signal == SignalType.HOLD

    def test_insufficient_data_returns_hold(self):
        model = _make_mock_model()
        strategy = _make_strategy(model=model)

        # Return only 3 rows (less than the 5-row minimum)
        tiny_df = _make_ohlcv_df(n_days=3)
        strategy.data_fetcher.fetch_historical.return_value = tiny_df
        strategy.feature_engineer.add_all_features_extended.return_value = tiny_df

        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.HOLD

    def test_nan_predictions_return_hold(self):
        """If the model returns NaN probabilities, the exception path returns HOLD."""
        model = MagicMock()
        model.predict_proba.return_value = np.array([[np.nan, np.nan]])
        model.feature_names = None
        strategy = _make_strategy(model=model)

        signal = strategy.generate_signal("AAPL")
        # NaN comparisons (>= threshold) are False, so we get HOLD
        assert signal.signal == SignalType.HOLD

    def test_model_predict_raises_returns_hold(self):
        """If prediction raises, generate_signal returns HOLD."""
        model = MagicMock()
        model.predict_proba.side_effect = RuntimeError("Model crashed")
        model.feature_names = None
        strategy = _make_strategy(model=model)

        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.HOLD

    def test_data_fetcher_error_returns_hold(self):
        """If data fetching raises, generate_signal returns HOLD."""
        model = _make_mock_model()
        strategy = _make_strategy(model=model)
        strategy.data_fetcher.fetch_historical.side_effect = Exception("Network error")

        signal = strategy.generate_signal("AAPL")
        assert signal.signal == SignalType.HOLD


# ===================================================================
# Generate Signals (multiple symbols)
# ===================================================================

class TestGenerateSignals:

    def test_generate_signals_multiple(self):
        model = _make_mock_model(prob_up=0.8, prob_down=0.2)
        strategy = _make_strategy(model=model)

        signals = strategy.generate_signals(["AAPL", "MSFT", "GOOGL"])
        assert len(signals) == 3
        assert all(isinstance(s, TradingSignal) for s in signals.values())
        assert set(signals.keys()) == {"AAPL", "MSFT", "GOOGL"}

    def test_generate_signals_error_in_one_symbol(self):
        """If one symbol fails, others still return signals."""
        model = MagicMock()
        model.feature_names = None
        call_count = {"n": 0}

        def predict_side_effect(X):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("model error for symbol 2")
            return np.array([[0.3, 0.7]])

        model.predict_proba.side_effect = predict_side_effect
        strategy = _make_strategy(model=model)

        signals = strategy.generate_signals(["AAPL", "MSFT", "GOOGL"])
        assert len(signals) == 3
        # MSFT should be HOLD due to error
        assert signals["MSFT"].signal == SignalType.HOLD


# ===================================================================
# Hold Signal
# ===================================================================

class TestHoldSignal:

    def test_hold_signal_properties(self):
        model = _make_mock_model()
        strategy = _make_strategy(model=model)
        hold = strategy._hold_signal("AAPL", reason="test")
        assert hold.symbol == "AAPL"
        assert hold.signal == SignalType.HOLD
        assert hold.confidence == pytest.approx(0.5)
        assert hold.predicted_direction == 0

    def test_hold_signal_default_reason(self):
        model = _make_mock_model()
        strategy = _make_strategy(model=model)
        hold = strategy._hold_signal("AAPL")
        assert hold.signal == SignalType.HOLD


# ===================================================================
# Feature Column Detection
# ===================================================================

class TestFeatureColumns:

    def test_feature_columns_excludes_metadata(self):
        model = _make_mock_model()
        strategy = _make_strategy(model=model)

        df = _make_ohlcv_df()
        df["symbol"] = "AAPL"
        df["date"] = df.index
        df["label_binary"] = 1

        cols = strategy._get_feature_columns(df)
        assert "symbol" not in cols
        assert "date" not in cols
        assert "label_binary" not in cols

    def test_feature_columns_uses_model_feature_names_if_available(self):
        model = _make_mock_model()
        model.feature_names = ["rsi_14", "macd", "sma_20"]
        strategy = _make_strategy(model=model)

        df = _make_ohlcv_df()
        cols = strategy._get_feature_columns(df)
        assert cols == ["rsi_14", "macd", "sma_20"]

    def test_feature_columns_handles_missing_model_features(self):
        model = _make_mock_model()
        model.feature_names = ["rsi_14", "nonexistent_feature"]
        strategy = _make_strategy(model=model)

        df = _make_ohlcv_df()
        cols = strategy._get_feature_columns(df)
        # Should return only features present in df
        assert "rsi_14" in cols
        assert "nonexistent_feature" not in cols


# ===================================================================
# Backtest Signals
# ===================================================================

class TestBacktestSignals:

    def test_backtest_returns_results_dict(self):
        model = _make_mock_model()
        # predict_proba needs to return one row per data row
        n_rows = 90  # after dropna we'll have fewer rows
        model.predict_proba.return_value = np.column_stack([
            np.full(n_rows, 0.3),
            np.full(n_rows, 0.7),
        ])
        model.feature_names = None
        strategy = _make_strategy(model=model)

        df = _make_ohlcv_df(n_days=n_rows)
        result = strategy.backtest_signals(df, initial_capital=100_000)

        assert "initial_capital" in result
        assert "final_value" in result
        assert "total_return" in result
        assert "total_trades" in result
        assert "trades" in result
        assert result["initial_capital"] == pytest.approx(100_000)

    def test_backtest_no_model_raises(self):
        strategy = _make_strategy(model=None)
        df = _make_ohlcv_df()
        with pytest.raises(RuntimeError, match="No model loaded"):
            strategy.backtest_signals(df)

    def test_backtest_buy_then_sell(self):
        """Create a sequence that forces exactly one buy then one sell."""
        n = 10
        model = MagicMock()
        model.feature_names = None

        # Row 0: high prob_up -> BUY; rows 1-8: hold; row 9: low prob_up -> SELL
        probs = np.column_stack([
            np.full(n, 0.5),
            np.full(n, 0.5),
        ])
        probs[0] = [0.2, 0.8]  # BUY trigger
        probs[-1] = [0.8, 0.2]  # SELL trigger (prob_up=0.2 < min_confidence_sell=0.55)

        model.predict_proba.return_value = probs
        strategy = _make_strategy(
            model=model, confidence_threshold=0.6, min_confidence_sell=0.55,
        )

        df = _make_ohlcv_df(n_days=n)
        result = strategy.backtest_signals(df, initial_capital=100_000)

        # Should have exactly one buy and one sell
        buys = [t for t in result["trades"] if t["type"] == "BUY"]
        sells = [t for t in result["trades"] if t["type"] == "SELL"]
        assert len(buys) == 1
        assert len(sells) == 1


# ===================================================================
# Load Model / Load Ensemble
# ===================================================================

class TestLoadModel:

    def test_load_model_sets_xgboost(self):
        strategy = _make_strategy(model=None)

        with patch("src.strategy.ml_strategy.XGBoostModel") as MockXGB:
            mock_instance = MockXGB.return_value
            strategy.load_model("test_model")

        assert strategy.model is mock_instance
        assert strategy.model_type == "xgboost"
        mock_instance.load.assert_called_once_with("test_model")

    def test_load_ensemble_sets_ensemble(self):
        strategy = _make_strategy(model=None)

        with patch("src.strategy.ml_strategy.EnsembleModel") as MockEns:
            mock_instance = MockEns.return_value
            mock_instance.load_models.return_value = ["xgboost", "lstm"]
            loaded = strategy.load_ensemble()

        assert strategy.model is mock_instance
        assert strategy.model_type == "ensemble"
        assert "xgboost" in loaded
