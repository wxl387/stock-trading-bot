"""
Machine learning based trading strategy.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Tuple, Union
import numpy as np

# Import TF BEFORE pandas to avoid model.predict() deadlock
# (pandas 2.3+ / TF 2.20+ import order bug on macOS)
from src.ml.device_config import configure_tensorflow_device
configure_tensorflow_device()

import pandas as pd

from src.data.data_fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer
from src.ml.models.xgboost_model import XGBoostModel
from src.ml.models.ensemble_model import EnsembleModel, VotingMethod
from src.risk.risk_manager import RiskManager, StopLossType

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal type."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    symbol: str
    signal: SignalType
    confidence: float
    predicted_direction: int  # 1 for up, 0 for down
    features: Optional[Dict] = None
    timestamp: Optional[pd.Timestamp] = None


class MLStrategy:
    """
    Machine learning based trading strategy.
    Generates trading signals using trained ML models.
    Supports single models (XGBoost) or ensemble (XGBoost + LSTM + CNN).
    """

    def __init__(
        self,
        model: Optional[Union[XGBoostModel, EnsembleModel]] = None,
        model_type: str = "xgboost",  # "xgboost" or "ensemble"
        confidence_threshold: float = 0.6,
        min_confidence_sell: float = 0.55,
        sequence_length: int = 20
    ):
        """
        Initialize ML Strategy.

        Args:
            model: Trained model (XGBoost or Ensemble).
            model_type: Type of model: "xgboost" or "ensemble".
            confidence_threshold: Minimum confidence to generate BUY signal.
            min_confidence_sell: Minimum confidence to generate SELL signal.
            sequence_length: Sequence length for LSTM/CNN models.
        """
        self.model = model
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.min_confidence_sell = min_confidence_sell
        self.sequence_length = sequence_length

        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()

    def load_model(self, model_name: str = "trading_model") -> None:
        """
        Load a trained XGBoost model.

        Args:
            model_name: Name of the saved model.
        """
        self.model = XGBoostModel()
        self.model.load(model_name)
        self.model_type = "xgboost"
        logger.info(f"Loaded XGBoost model: {model_name}")

    def load_ensemble(
        self,
        voting_method: VotingMethod = VotingMethod.SOFT,
        model_weights: Optional[Dict[str, float]] = None,
        xgboost_name: str = "trading_model",
        lstm_name: str = "lstm_trading_model",
        cnn_name: str = "cnn_trading_model"
    ) -> List[str]:
        """
        Load ensemble model with multiple sub-models.

        Args:
            voting_method: How to combine predictions (SOFT, HARD, WEIGHTED).
            model_weights: Custom weights for each model type.
            xgboost_name: Name of saved XGBoost model.
            lstm_name: Name of saved LSTM model.
            cnn_name: Name of saved CNN model.

        Returns:
            List of successfully loaded model names.
        """
        self.model = EnsembleModel(
            voting_method=voting_method,
            model_weights=model_weights,
            sequence_length=self.sequence_length
        )
        loaded = self.model.load_models(
            xgboost_name=xgboost_name,
            lstm_name=lstm_name,
            cnn_name=cnn_name
        )
        self.model_type = "ensemble"
        logger.info(f"Loaded ensemble with models: {loaded}")
        return loaded

    def generate_signal(self, symbol: str) -> TradingSignal:
        """
        Generate trading signal for a single symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            TradingSignal with recommendation.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() or load_ensemble() first.")

        try:
            # Fetch recent data (need 1y for SMA_200 feature)
            df = self.data_fetcher.fetch_historical(
                symbol=symbol,
                period="1y",
                use_cache=False
            )

            if df.empty:
                logger.warning(f"No data for {symbol}")
                return self._hold_signal(symbol, reason="No data")

            # Add features (must match training flags in src/ml/training.py)
            df = self.feature_engineer.add_all_features_extended(
                df,
                symbol=symbol,
                include_sentiment=False,
                include_macro=False,
                include_cross_asset=True,
                include_interactions=False,
                include_lagged=True,
                use_cache=True
            )
            df = df.dropna()

            if len(df) < 5:
                return self._hold_signal(symbol, reason="Insufficient data")

            # Get latest features
            latest = df.iloc[-1:]
            feature_cols = self._get_feature_columns(df)

            # Get prediction based on model type
            if self.model_type == "ensemble" and isinstance(self.model, EnsembleModel):
                # Ensemble: use predict_single with feature history for LSTM/CNN
                feature_history = df[feature_cols].iloc[-self.sequence_length:]

                prediction, confidence, model_probs = self.model.predict_single(
                    X_flat=latest[feature_cols],
                    feature_history=feature_history
                )

                # Use ensemble's weighted prediction (confidence = max(prob, 1-prob))
                prob_up = confidence if prediction == 1 else (1 - confidence)
                prob_down = 1 - prob_up

                # Log individual model predictions
                if model_probs:
                    probs_str = ", ".join(f"{m}={p:.2%}" for m, p in model_probs.items())
                    logger.debug(f"{symbol} model probs: {probs_str}")

            else:
                # XGBoost: use direct predict_proba
                X = latest[feature_cols]
                proba = self.model.predict_proba(X)[0]

                # proba[0] = prob down, proba[1] = prob up
                prob_up = proba[1]
                prob_down = proba[0]
                confidence = max(prob_up, prob_down)

            # Generate signal based on confidence
            if prob_up >= self.confidence_threshold:
                signal_type = SignalType.BUY
            elif prob_down >= self.min_confidence_sell:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD

            signal = TradingSignal(
                symbol=symbol,
                signal=signal_type,
                confidence=confidence,
                predicted_direction=1 if prob_up > prob_down else 0,
                features={
                    "rsi_14": latest["rsi_14"].values[0],
                    "macd": latest["macd"].values[0],
                    "price_vs_sma20": latest["price_vs_sma20"].values[0]
                } if "rsi_14" in latest.columns else None,
                timestamp=latest.index[0] if isinstance(latest.index, pd.DatetimeIndex) else None
            )

            model_info = f"({self.model_type})" if self.model_type == "ensemble" else ""
            logger.info(
                f"Signal for {symbol}{model_info}: {signal_type.value} "
                f"(confidence: {confidence:.2%}, prob_up: {prob_up:.2%})"
            )

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return self._hold_signal(symbol, reason=str(e))

    def generate_signals(self, symbols: List[str]) -> Dict[str, TradingSignal]:
        """
        Generate trading signals for multiple symbols.

        Args:
            symbols: List of stock ticker symbols.

        Returns:
            Dictionary mapping symbol to TradingSignal.
        """
        signals = {}

        for symbol in symbols:
            try:
                signals[symbol] = self.generate_signal(symbol)
            except Exception as e:
                logger.error(f"Failed to generate signal for {symbol}: {e}")
                signals[symbol] = self._hold_signal(symbol, reason=str(e))

        return signals

    def get_trade_recommendations(
        self,
        symbols: List[str],
        portfolio_value: float,
        current_positions: Dict[str, int],
        risk_manager: RiskManager,
        target_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        Get actionable trade recommendations.

        Args:
            symbols: List of symbols to analyze.
            portfolio_value: Total portfolio value.
            current_positions: Dict of symbol -> shares held.
            risk_manager: RiskManager instance.
            target_weights: Optional target portfolio weights from optimizer.

        Returns:
            List of trade recommendation dictionaries.
        """
        signals = self.generate_signals(symbols)
        recommendations = []

        # Portfolio-aware position sizing if target weights provided
        use_portfolio_weights = target_weights is not None and len(target_weights) > 0

        for symbol, signal in signals.items():
            current_shares = current_positions.get(symbol, 0)
            price = self.data_fetcher.get_latest_price(symbol)

            if use_portfolio_weights and symbol in target_weights:
                # Portfolio optimization mode: use target weights, but respect ML signals
                if not (price > 0 and np.isfinite(price)):
                    logger.warning(f"Skipping {symbol}: invalid price {price}")
                    continue
                target_weight = target_weights[symbol]
                target_value = portfolio_value * target_weight
                target_shares = int(target_value / price)
                shares_diff = target_shares - current_shares

                # ML SELL signal overrides optimizer: sell entire position
                if signal.signal == SignalType.SELL and current_shares > 0:
                    recommendations.append({
                        "action": "SELL",
                        "symbol": symbol,
                        "shares": current_shares,
                        "price": price,
                        "confidence": signal.confidence,
                        "reason": f"ML sell signal overrides target {target_weight:.1%} (confidence: {signal.confidence:.1%})"
                    })
                    continue

                # Only rebalance if difference is significant (>5% of target or >$100)
                min_trade_shares = max(1, int(target_shares * 0.05))
                min_trade_value = 100 / price if price > 0 else 1

                if abs(shares_diff) >= max(min_trade_shares, min_trade_value):
                    if shares_diff > 0:
                        # Need to buy more shares
                        stop_loss_price = risk_manager.calculate_stop_loss(price)
                        recommendations.append({
                            "action": "BUY",
                            "symbol": symbol,
                            "shares": shares_diff,
                            "price": price,
                            "stop_loss": stop_loss_price,
                            "confidence": signal.confidence,
                            "reason": f"Portfolio rebalance to {target_weight:.1%} (signal: {signal.confidence:.1%})"
                        })
                    else:
                        # Need to sell shares
                        recommendations.append({
                            "action": "SELL",
                            "symbol": symbol,
                            "shares": abs(shares_diff),
                            "price": price,
                            "confidence": signal.confidence,
                            "reason": f"Portfolio rebalance to {target_weight:.1%} (signal: {signal.confidence:.1%})"
                        })

            else:
                # Traditional signal-based trading (original logic)
                if price <= 0:
                    logger.warning(f"Skipping {symbol}: invalid price {price}")
                    continue

                if signal.signal == SignalType.BUY and current_shares == 0:
                    # New buy opportunity
                    # Calculate position size
                    stop_loss_price = risk_manager.calculate_stop_loss(price)
                    shares = risk_manager.calculate_position_size(
                        portfolio_value=portfolio_value,
                        entry_price=price,
                        stop_loss_price=stop_loss_price
                    )

                    if shares > 0:
                        recommendations.append({
                            "action": "BUY",
                            "symbol": symbol,
                            "shares": shares,
                            "price": price,
                            "stop_loss": stop_loss_price,
                            "confidence": signal.confidence,
                            "reason": f"ML signal: {signal.confidence:.1%} confidence"
                        })

                elif signal.signal == SignalType.SELL and current_shares > 0:
                    # Sell existing position
                    recommendations.append({
                        "action": "SELL",
                        "symbol": symbol,
                        "shares": current_shares,
                        "price": price,
                        "confidence": signal.confidence,
                        "reason": f"ML sell signal: {signal.confidence:.1%} confidence"
                    })

        # Sort by confidence (highest first)
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)

        return recommendations

    def backtest_signals(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000
    ) -> Dict:
        """
        Backtest strategy on historical data.

        Args:
            df: DataFrame with OHLCV and features.
            initial_capital: Starting capital.

        Returns:
            Dictionary with backtest results.
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")

        feature_cols = self._get_feature_columns(df)
        df = df.dropna(subset=feature_cols)

        # Generate predictions for all rows
        X = df[feature_cols]
        predictions = self.model.predict_proba(X)

        # Simulate trading
        capital = initial_capital
        position = 0
        trades = []

        for i in range(len(df)):
            prob_up = predictions[i][1]
            current_price = df["close"].iloc[i]

            if prob_up >= self.confidence_threshold and position == 0:
                # Buy
                shares = int(capital * 0.95 / current_price)
                if shares > 0:
                    position = shares
                    capital -= shares * current_price
                    trades.append({
                        "type": "BUY",
                        "price": current_price,
                        "shares": shares,
                        "date": df.index[i]
                    })

            elif prob_up < self.min_confidence_sell and position > 0:
                # Sell
                capital += position * current_price
                trades.append({
                    "type": "SELL",
                    "price": current_price,
                    "shares": position,
                    "date": df.index[i]
                })
                position = 0

        # Final value
        final_value = capital + position * df["close"].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital

        return {
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "total_trades": len(trades),
            "trades": trades
        }

    def _hold_signal(self, symbol: str, reason: str = "") -> TradingSignal:
        """Create a HOLD signal."""
        return TradingSignal(
            symbol=symbol,
            signal=SignalType.HOLD,
            confidence=0.5,
            predicted_direction=0
        )

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns from DataFrame."""
        # Use model's feature names if available for exact match
        if self.model is not None and hasattr(self.model, 'feature_names') and self.model.feature_names:
            missing = [f for f in self.model.feature_names if f not in df.columns]
            if missing:
                logger.warning(f"Missing {len(missing)} model features: {missing[:5]}...")
                return [f for f in self.model.feature_names if f in df.columns]
            return self.model.feature_names

        # Fallback: exclude non-feature columns
        exclude_cols = [
            "symbol", "date", "dividends", "stock_splits",
            "future_returns", "label_binary", "label_3class"
        ]
        return [col for col in df.columns if col not in exclude_cols]
