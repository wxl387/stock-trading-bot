"""
Walk-forward hyperparameter optimization using Optuna.

Evaluates hyperparameters across multiple walk-forward windows to find
parameters that generalize to unseen market conditions.
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from config.settings import MODELS_DIR
from src.data.data_fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class WalkForwardOptimizer:
    """
    Walk-forward hyperparameter optimization.

    Uses Optuna to search for XGBoost hyperparameters that maximize
    out-of-sample performance across multiple walk-forward windows.
    """

    def __init__(
        self,
        symbols: List[str],
        train_period: int = 252,
        test_period: int = 63,
        step: int = 126,
        n_windows: int = 4,
        n_trials: int = 50,
        optimize_metric: str = "sharpe_ratio",
        optimize_trading_params: bool = True,
        optimize_features: bool = False,
        confidence_threshold: float = 0.6,
        max_positions: int = 5,
        initial_capital: float = 100000,
        commission: float = 0.0,
        slippage: float = 0.001,
        use_regime: bool = False,
        regime_mode: str = "adjust",
        enable_stop_loss: bool = False,
        enable_trailing_stop: bool = False,
        enable_kelly: bool = False,
        enable_circuit_breaker: bool = False,
        trailing_atr_multiplier: float = 2.0,
        circuit_breaker_threshold: float = 0.15,
        circuit_breaker_recovery: float = 0.05,
    ):
        self.symbols = symbols
        self.train_period = train_period
        self.test_period = test_period
        self.step = step
        self.n_windows = n_windows
        self.n_trials = n_trials
        self.optimize_metric = optimize_metric
        self.optimize_trading_params = optimize_trading_params
        self.optimize_features = optimize_features
        self.confidence_threshold = confidence_threshold
        self.max_positions = max_positions
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.use_regime = use_regime
        self.regime_mode = regime_mode
        self.enable_stop_loss = enable_stop_loss
        self.enable_trailing_stop = enable_trailing_stop
        self.enable_kelly = enable_kelly
        self.enable_circuit_breaker = enable_circuit_breaker
        self.trailing_atr_multiplier = trailing_atr_multiplier
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_recovery = circuit_breaker_recovery

        # Cached data (loaded once)
        self.all_data: Dict[str, pd.DataFrame] = {}
        self.common_dates: List = []
        self.windows: List = []
        self._data_loaded = False
        self._regime_series: Optional[pd.Series] = None

        # Results
        self.best_params: Optional[Dict] = None
        self.study: Optional[optuna.Study] = None

    def optimize(self, n_jobs: int = 1) -> Dict[str, Any]:
        """
        Run walk-forward hyperparameter optimization.

        Args:
            n_jobs: Number of parallel Optuna trials.

        Returns:
            Dictionary with best parameters and optimization stats.
        """
        # Load and cache data
        if not self._data_loaded:
            self._prepare_data()

        if not self.windows:
            raise ValueError("Not enough data for walk-forward optimization")

        print(f"\nStarting optimization: {self.n_trials} trials, "
              f"{len(self.windows)} windows, metric={self.optimize_metric}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Window config: train={self.train_period}d, test={self.test_period}d, step={self.step}d")
        print("-" * 60)

        # Create Optuna study
        self.study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=1
            ),
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            callbacks=[self._trial_callback]
        )

        # Extract best params
        best_trial = self.study.best_trial
        self.best_params = self._extract_params(best_trial)

        print(f"\n{'=' * 60}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Best score ({self.optimize_metric}): {best_trial.value:.4f}")
        print(f"Best trial: #{best_trial.number}")
        print(f"Model params: {self.best_params['model_params']}")
        if self.optimize_trading_params:
            print(f"Trading params: {self.best_params['trading_params']}")
        if self.optimize_features:
            print(f"Feature flags: {self.best_params['feature_flags']}")

        stats = {
            "best_score": best_trial.value,
            "best_trial": best_trial.number,
            "total_trials": len(self.study.trials),
            "pruned_trials": len([t for t in self.study.trials
                                  if t.state == optuna.trial.TrialState.PRUNED]),
        }

        return {
            "best_score": best_trial.value,
            "best_model_params": self.best_params["model_params"],
            "best_trading_params": self.best_params["trading_params"],
            "best_feature_flags": self.best_params.get("feature_flags"),
            "study_stats": stats,
        }

    def _trial_callback(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Print progress after each trial."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if trial.number % 5 == 0 or trial.number == self.n_trials - 1:
                print(f"  Trial {trial.number:3d}: score={trial.value:.4f} "
                      f"(best={study.best_value:.4f} @ trial #{study.best_trial.number})")

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: evaluate params across walk-forward windows."""
        # Build search spaces
        model_params = self._build_xgboost_space(trial)
        trading_params = (
            self._build_trading_space(trial)
            if self.optimize_trading_params
            else {"confidence_threshold": self.confidence_threshold,
                  "max_positions": self.max_positions}
        )
        feature_flags = (
            self._build_feature_space(trial)
            if self.optimize_features
            else None
        )

        # Evaluate across walk-forward windows
        window_metrics = self._evaluate_params(model_params, trading_params, feature_flags)

        if not window_metrics or all(m == 0 for m in window_metrics):
            return float("-inf")

        # Score: mean - 0.5 * std (penalize variance)
        avg_metric = np.mean(window_metrics)
        std_metric = np.std(window_metrics) if len(window_metrics) > 1 else 0

        score = avg_metric - 0.5 * std_metric

        # Report intermediate values for pruning
        for i, m in enumerate(window_metrics):
            trial.report(m, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return score

    def _build_xgboost_space(self, trial: optuna.Trial) -> Dict:
        """XGBoost hyperparameter search space."""
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        }

    def _build_trading_space(self, trial: optuna.Trial) -> Dict:
        """Trading parameter search space."""
        return {
            "confidence_threshold": trial.suggest_float("confidence_threshold", 0.5, 0.8),
            "max_positions": trial.suggest_int("max_positions", 3, 10),
        }

    def _build_feature_space(self, trial: optuna.Trial) -> Dict:
        """Feature group toggle search space."""
        return {
            "include_macro": trial.suggest_categorical("include_macro", [True, False]),
            "include_cross_asset": trial.suggest_categorical("include_cross_asset", [True, False]),
            "include_interactions": trial.suggest_categorical("include_interactions", [True, False]),
            "include_lagged": trial.suggest_categorical("include_lagged", [True, False]),
        }

    def _prepare_data(self):
        """Fetch and feature-engineer data for all symbols (cached)."""
        print("Loading and preparing data...")
        data_fetcher = DataFetcher()
        feature_engineer = FeatureEngineer()

        for symbol in self.symbols:
            try:
                df = data_fetcher.fetch_historical(symbol, period="5y")
                if df.empty:
                    continue
                df = feature_engineer.add_all_features_extended(
                    df, symbol=symbol,
                    include_sentiment=False,
                    include_macro=True,
                    include_cross_asset=True,
                    include_interactions=True,
                    include_lagged=True,
                    use_cache=True
                )
                df = df.ffill().fillna(0)
                df = df[df['close'] > 0]
                if len(df) > self.train_period + self.test_period:
                    self.all_data[symbol] = df
                    print(f"  {symbol}: {len(df)} samples")
            except Exception as e:
                print(f"  {symbol}: failed - {e}")

        if not self.all_data:
            raise ValueError("No valid data loaded")

        # Find common dates
        common_dates_set = None
        for df in self.all_data.values():
            dates = set(df.index)
            common_dates_set = dates if common_dates_set is None else common_dates_set.intersection(dates)
        self.common_dates = sorted(common_dates_set)
        total_days = len(self.common_dates)

        # Pre-compute walk-forward windows
        self.windows = []
        for window_start in range(0, total_days - self.train_period - self.test_period + 1, self.step):
            train_end = window_start + self.train_period
            test_end = min(train_end + self.test_period, total_days)

            train_dates = self.common_dates[window_start:train_end]
            test_dates = self.common_dates[train_end:test_end]

            if len(test_dates) > 0:
                self.windows.append((train_dates, test_dates))

            if len(self.windows) >= self.n_windows:
                break

        # Pre-compute regime series if enabled
        if self.use_regime:
            from src.risk.regime_detector import compute_regime_series
            try:
                spy_data = data_fetcher.fetch_historical("SPY", period="5y")
                if not spy_data.empty and len(spy_data) >= 200:
                    if spy_data.index.tz is not None:
                        spy_data.index = spy_data.index.tz_localize(None)
                    self._regime_series = compute_regime_series(spy_data)
                    print(f"  Regime series: {len(self._regime_series)} days")
            except Exception as e:
                print(f"  Regime detection failed: {e}")

        self._data_loaded = True
        print(f"Data ready: {total_days} common days, {len(self.windows)} windows")

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns excluding metadata/label columns."""
        exclude = [
            "symbol", "date", "dividends", "stock_splits",
            "future_returns", "label_binary", "label_3class"
        ]
        return [c for c in df.columns if c not in exclude]

    def _prepare_data_with_features(self, symbol: str, feature_flags: Optional[Dict]) -> pd.DataFrame:
        """Get data for a symbol, optionally re-engineering features based on flags."""
        if feature_flags is None:
            return self.all_data[symbol]

        # Re-engineer features with specific flags
        data_fetcher = DataFetcher()
        feature_engineer = FeatureEngineer()

        df = data_fetcher.fetch_historical(symbol, period="5y")
        if df.empty:
            return pd.DataFrame()

        df = feature_engineer.add_all_features_extended(
            df, symbol=symbol,
            include_sentiment=False,
            include_macro=feature_flags.get("include_macro", True),
            include_cross_asset=feature_flags.get("include_cross_asset", True),
            include_interactions=feature_flags.get("include_interactions", True),
            include_lagged=feature_flags.get("include_lagged", True),
            use_cache=True
        )
        df = df.ffill().fillna(0)
        df = df[df['close'] > 0]
        return df

    def _evaluate_params(
        self,
        model_params: Dict,
        trading_params: Dict,
        feature_flags: Optional[Dict]
    ) -> List[float]:
        """
        Run condensed walk-forward with given hyperparameters.
        Returns list of per-window metric values.
        """
        from src.ml.models.xgboost_model import XGBoostModel

        window_metrics = []
        capital = self.initial_capital

        for window_idx, (train_dates, test_dates) in enumerate(self.windows):
            # Prepare training data
            X_train_list, y_train_list = [], []
            feature_cols = None

            for symbol in self.all_data:
                if feature_flags is not None:
                    df = self._prepare_data_with_features(symbol, feature_flags)
                else:
                    df = self.all_data[symbol]

                if df.empty:
                    continue

                train_df = df.loc[df.index.isin(train_dates)]
                if len(train_df) < 50:
                    continue

                if feature_cols is None:
                    feature_cols = self._get_feature_columns(train_df)

                X_sym = train_df[feature_cols].copy()
                y_sym = (train_df['close'].shift(-5) / train_df['close'] - 1 > 0).astype(int)
                X_sym = X_sym.iloc[:-5]
                y_sym = y_sym.iloc[:-5]
                X_train_list.append(X_sym.reset_index(drop=True))
                y_train_list.append(y_sym.reset_index(drop=True))

            if not X_train_list:
                window_metrics.append(0.0)
                continue

            X_train = pd.concat(X_train_list, ignore_index=True)
            y_train = pd.concat(y_train_list, ignore_index=True)

            # Train XGBoost with trial params
            try:
                model = XGBoostModel(**model_params)
                model.train(X_train, y_train)
            except Exception:
                window_metrics.append(0.0)
                continue

            # Generate predictions on test window
            test_signals = []
            test_prices = {}
            test_highs = {}
            test_lows = {}
            test_atr = {}
            confidence_threshold = trading_params["confidence_threshold"]
            max_positions = trading_params["max_positions"]

            for symbol in self.all_data:
                if feature_flags is not None:
                    df = self._prepare_data_with_features(symbol, feature_flags)
                else:
                    df = self.all_data[symbol]

                if df.empty:
                    continue

                test_df = df.loc[df.index.isin(test_dates)]
                if len(test_df) < 1:
                    continue

                try:
                    X_test = test_df[feature_cols]
                    probas = model.predict_proba(X_test)
                    prob_up = probas[:, 1] if probas.ndim > 1 else probas

                    if len(prob_up) != len(test_df):
                        logger.warning(
                            f"Prediction length mismatch for {symbol}: "
                            f"{len(prob_up)} predictions vs {len(test_df)} test samples"
                        )
                    for i, (date, price) in enumerate(test_df['close'].items()):
                        if i < len(prob_up):
                            test_signals.append({
                                "date": date,
                                "symbol": symbol,
                                "price": price,
                                "prob_up": prob_up[i]
                            })
                    test_prices[symbol] = test_df['close']
                    # Risk management data
                    if 'high' in test_df.columns:
                        test_highs[symbol] = test_df['high']
                    if 'low' in test_df.columns:
                        test_lows[symbol] = test_df['low']
                    if 'atr_14' in test_df.columns:
                        test_atr[symbol] = test_df['atr_14']
                except Exception:
                    continue

            if not test_signals:
                window_metrics.append(0.0)
                continue

            # Simulate portfolio
            signals_df = pd.DataFrame(test_signals)
            window_result = self._simulate_window(
                signals_df, test_prices, confidence_threshold, max_positions, capital,
                regime_series=self._regime_series, regime_mode=self.regime_mode,
                highs=test_highs, lows=test_lows, atr=test_atr,
                enable_stop_loss=self.enable_stop_loss,
                enable_trailing_stop=self.enable_trailing_stop,
                enable_kelly=self.enable_kelly,
                enable_circuit_breaker=self.enable_circuit_breaker,
                trailing_atr_multiplier=self.trailing_atr_multiplier,
                circuit_breaker_threshold=self.circuit_breaker_threshold,
                circuit_breaker_recovery=self.circuit_breaker_recovery,
            )

            # Extract metric
            metric_value = self._extract_metric(window_result, capital)
            window_metrics.append(metric_value)

            # Carry capital forward
            capital = window_result["final_value"]

        return window_metrics

    def _lookup_regime(self, regime_series, date):
        """Look up regime for a given date."""
        from src.risk.regime_detector import MarketRegime
        lookup_date = date
        if hasattr(lookup_date, 'tz') and lookup_date.tz is not None:
            lookup_date = lookup_date.tz_localize(None)
        if lookup_date in regime_series.index:
            return regime_series.loc[lookup_date]
        valid_dates = regime_series.index[regime_series.index <= lookup_date]
        if len(valid_dates) > 0:
            return regime_series.loc[valid_dates[-1]]
        return MarketRegime.BULL

    def _simulate_window(
        self,
        signals_df: pd.DataFrame,
        prices: Dict[str, pd.Series],
        confidence_threshold: float,
        max_positions: int,
        starting_capital: float,
        regime_series: Optional[pd.Series] = None,
        regime_mode: str = "adjust",
        highs: Optional[Dict[str, pd.Series]] = None,
        lows: Optional[Dict[str, pd.Series]] = None,
        atr: Optional[Dict[str, pd.Series]] = None,
        enable_stop_loss: bool = False,
        enable_trailing_stop: bool = False,
        enable_kelly: bool = False,
        enable_circuit_breaker: bool = False,
        trailing_atr_multiplier: float = 2.0,
        circuit_breaker_threshold: float = 0.15,
        circuit_breaker_recovery: float = 0.05,
    ) -> Dict:
        """Simulate portfolio trading for a single window. Returns dict with metrics."""
        from src.risk.regime_detector import MarketRegime, DEFAULT_REGIME_PARAMS

        capital = starting_capital
        positions = {}
        trades = []  # list of dicts with 'return' and 'pnl'
        equity_history = []

        # Risk management state
        trailing_stops: Dict[str, float] = {}
        peak_equity = starting_capital
        circuit_breaker_active = False
        completed_trades: list = []

        for date in sorted(signals_df["date"].unique()):
            day_signals = signals_df[signals_df["date"] == date]

            # Regime-adaptive parameters
            day_confidence = confidence_threshold
            day_size_mult = 1.0
            day_stop_loss_pct = 0.05
            day_trailing_enabled = True

            if regime_series is not None:
                regime = self._lookup_regime(regime_series, date)
                regime_params = DEFAULT_REGIME_PARAMS[regime]
                if regime_mode == "filter":
                    if regime in (MarketRegime.BEAR, MarketRegime.CHOPPY):
                        day_confidence = 0.99
                    else:
                        day_confidence = max(confidence_threshold, regime_params.min_confidence)
                else:
                    day_confidence = max(confidence_threshold, regime_params.min_confidence)
                day_size_mult = regime_params.position_size_multiplier
                day_stop_loss_pct = regime_params.stop_loss_pct
                day_trailing_enabled = regime_params.trailing_stop_enabled

            # Current equity
            equity = capital
            for sym, pos in positions.items():
                if sym in prices and date in prices[sym].index:
                    equity += pos["shares"] * prices[sym].loc[date]
            equity_history.append(equity)

            # Circuit breaker
            if enable_circuit_breaker:
                if equity > peak_equity:
                    peak_equity = equity
                current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                if current_dd >= circuit_breaker_threshold:
                    circuit_breaker_active = True
                elif circuit_breaker_active and current_dd <= circuit_breaker_recovery:
                    circuit_breaker_active = False

            # Stop-loss exits
            if enable_stop_loss or enable_trailing_stop:
                for sym in list(positions.keys()):
                    pos = positions[sym]

                    day_low = lows[sym].loc[date] if (lows and sym in lows and date in lows[sym].index) else None
                    day_high = highs[sym].loc[date] if (highs and sym in highs and date in highs[sym].index) else None
                    day_atr = atr[sym].loc[date] if (atr and sym in atr and date in atr[sym].index) else None

                    if day_low is None:
                        sym_signal = day_signals[day_signals["symbol"] == sym]
                        day_low = sym_signal.iloc[0]["price"] if not sym_signal.empty else None
                    if day_low is None:
                        continue

                    fixed_stop = None
                    if enable_stop_loss:
                        fixed_stop = pos["entry_price"] * (1 - day_stop_loss_pct)

                    trail_stop = None
                    if enable_trailing_stop and day_trailing_enabled and day_atr is not None and day_atr > 0:
                        trail_distance = day_atr * trailing_atr_multiplier
                        if sym not in trailing_stops:
                            trailing_stops[sym] = pos["entry_price"] - trail_distance
                        if day_high is not None:
                            trailing_stops[sym] = max(trailing_stops[sym], day_high - trail_distance)
                        trail_stop = trailing_stops[sym]

                    effective_stop = None
                    if fixed_stop is not None and trail_stop is not None:
                        effective_stop = max(fixed_stop, trail_stop)
                    elif fixed_stop is not None:
                        effective_stop = fixed_stop
                    elif trail_stop is not None:
                        effective_stop = trail_stop

                    if effective_stop is not None and day_low <= effective_stop:
                        pos = positions.pop(sym)
                        if sym in trailing_stops:
                            del trailing_stops[sym]
                        exit_price = min(effective_stop, day_low) if day_low < effective_stop else effective_stop
                        proceeds = pos["shares"] * exit_price * (1 - self.commission - self.slippage)
                        capital += proceeds
                        entry_cost = pos["shares"] * pos["entry_price"] * (1 + self.commission + self.slippage)
                        trade_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                        pnl = proceeds - entry_cost
                        trade_record = {"return": trade_ret, "pnl": pnl}
                        trades.append(trade_record)
                        completed_trades.append(trade_record)

            # Signal-based exits
            for sym in list(positions.keys()):
                sym_signal = day_signals[day_signals["symbol"] == sym]
                if not sym_signal.empty:
                    prob = sym_signal.iloc[0]["prob_up"]
                    price = sym_signal.iloc[0]["price"]
                    if prob < (1 - day_confidence):
                        pos = positions.pop(sym)
                        if sym in trailing_stops:
                            del trailing_stops[sym]
                        proceeds = pos["shares"] * price * (1 - self.commission - self.slippage)
                        capital += proceeds
                        entry_cost = pos["shares"] * pos["entry_price"] * (1 + self.commission + self.slippage)
                        trade_ret = (price - pos["entry_price"]) / pos["entry_price"]
                        pnl = proceeds - entry_cost
                        trade_record = {"return": trade_ret, "pnl": pnl}
                        trades.append(trade_record)
                        completed_trades.append(trade_record)

            # Entries (blocked if circuit breaker active)
            if len(positions) < max_positions and not circuit_breaker_active:
                candidates = day_signals[day_signals["prob_up"] > day_confidence]
                candidates = candidates[~candidates["symbol"].isin(positions.keys())]
                candidates = candidates.sort_values("prob_up", ascending=False)

                for _, row in candidates.iterrows():
                    if len(positions) >= max_positions:
                        break

                    # Position sizing: Kelly or default
                    if enable_kelly and len(completed_trades) >= 10:
                        recent = completed_trades[-20:]
                        wins = [t for t in recent if t["return"] > 0]
                        losses = [t for t in recent if t["return"] <= 0]
                        win_rate_k = len(wins) / len(recent)
                        avg_win = np.mean([t["return"] for t in wins]) if wins else 0.0
                        avg_loss = abs(np.mean([t["return"] for t in losses])) if losses else 1.0
                        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
                        kelly_f = win_rate_k - (1 - win_rate_k) / win_loss_ratio if win_loss_ratio > 0 else -1.0
                        half_kelly = max(0.05, min(0.25, kelly_f / 2.0))
                        position_value = equity * half_kelly * day_size_mult
                    else:
                        base_value = capital / (max_positions - len(positions) + 1) * 0.95
                        position_value = base_value * day_size_mult

                    shares = int(position_value / row["price"])
                    if shares > 0:
                        cost = shares * row["price"] * (1 + self.commission + self.slippage)
                        if cost <= capital:
                            capital -= cost
                            positions[row["symbol"]] = {
                                "shares": shares,
                                "entry_price": row["price"],
                            }
                            if enable_trailing_stop and atr and row["symbol"] in atr:
                                if date in atr[row["symbol"]].index:
                                    entry_atr = atr[row["symbol"]].loc[date]
                                    if entry_atr > 0:
                                        trailing_stops[row["symbol"]] = row["price"] - (entry_atr * trailing_atr_multiplier)

        # Close remaining positions
        for sym, pos in positions.items():
            if sym in prices:
                last_price = prices[sym].iloc[-1]
                proceeds = pos["shares"] * last_price * (1 - self.commission - self.slippage)
                capital += proceeds
                entry_cost = pos["shares"] * pos["entry_price"] * (1 + self.commission + self.slippage)
                trade_ret = (last_price - pos["entry_price"]) / pos["entry_price"]
                pnl = proceeds - entry_cost
                trade_record = {"return": trade_ret, "pnl": pnl}
                trades.append(trade_record)
                completed_trades.append(trade_record)

        # Compute metrics
        equity_series = pd.Series(equity_history) if equity_history else pd.Series([starting_capital])
        daily_returns = equity_series.pct_change().dropna()

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino
        if len(daily_returns) > 1:
            downside = daily_returns[daily_returns < 0]
            if len(downside) > 0:
                downside_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252)
                ann_return = daily_returns.mean() * 252
                sortino = ann_return / downside_dev if downside_dev > 0 else 0.0
            else:
                sortino = 0.0
        else:
            sortino = 0.0

        total_return = (capital - starting_capital) / starting_capital

        # Win rate and profit factor
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(trades) if trades else 0
        total_wins = sum(t["pnl"] for t in wins) if wins else 0
        total_losses = abs(sum(t["pnl"] for t in losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else (float('inf') if total_wins > 0 else 0)

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "total_return": total_return,
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "final_value": capital,
            "total_trades": len(trades),
        }

    def _extract_metric(self, result: Dict, starting_capital: float) -> float:
        """Extract the target metric from a window result."""
        return result.get(self.optimize_metric, 0.0)

    def _extract_params(self, trial: optuna.trial.FrozenTrial) -> Dict:
        """Extract model/trading/feature params from a completed trial."""
        model_params = {
            "n_estimators": trial.params["n_estimators"],
            "max_depth": trial.params["max_depth"],
            "learning_rate": trial.params["learning_rate"],
            "subsample": trial.params["subsample"],
            "colsample_bytree": trial.params["colsample_bytree"],
            "min_child_weight": trial.params["min_child_weight"],
            "gamma": trial.params["gamma"],
            "reg_alpha": trial.params["reg_alpha"],
            "reg_lambda": trial.params["reg_lambda"],
        }

        if self.optimize_trading_params:
            trading_params = {
                "confidence_threshold": trial.params["confidence_threshold"],
                "max_positions": trial.params["max_positions"],
            }
        else:
            trading_params = {
                "confidence_threshold": self.confidence_threshold,
                "max_positions": self.max_positions,
            }

        feature_flags = None
        if self.optimize_features:
            feature_flags = {
                "include_macro": trial.params["include_macro"],
                "include_cross_asset": trial.params["include_cross_asset"],
                "include_interactions": trial.params["include_interactions"],
                "include_lagged": trial.params["include_lagged"],
            }

        return {
            "model_params": model_params,
            "trading_params": trading_params,
            "feature_flags": feature_flags,
        }

    def save_results(self, path: str):
        """Save optimization results to JSON."""
        if self.best_params is None or self.study is None:
            raise ValueError("No optimization results to save. Run optimize() first.")

        best_trial = self.study.best_trial

        # Collect window details
        window_details = []
        for i, (train_dates, test_dates) in enumerate(self.windows):
            window_details.append({
                "window": i + 1,
                "train_start": str(train_dates[0].date()),
                "train_end": str(train_dates[-1].date()),
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
            })

        output = {
            "optimization_date": datetime.now().isoformat(),
            "model_type": "xgboost",
            "optimize_metric": self.optimize_metric,
            "n_trials": self.n_trials,
            "n_windows": len(self.windows),
            "symbols": self.symbols,
            "best_score": best_trial.value,
            "best_model_params": self.best_params["model_params"],
            "best_trading_params": self.best_params["trading_params"],
            "best_feature_flags": self.best_params.get("feature_flags"),
            "window_details": window_details,
            "study_stats": {
                "best_trial": best_trial.number,
                "total_trials": len(self.study.trials),
                "pruned_trials": len([t for t in self.study.trials
                                      if t.state == optuna.trial.TrialState.PRUNED]),
            }
        }

        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to: {path}")

    @staticmethod
    def load_results(path: str) -> Dict:
        """Load optimization results from JSON."""
        with open(path, 'r') as f:
            return json.load(f)
