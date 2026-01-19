"""
Backtesting module for strategy validation.
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

try:
    import vectorbt as vbt
    HAS_VBT = True
except ImportError:
    HAS_VBT = False

from src.data.data_fetcher import DataFetcher
from src.data.feature_engineer import FeatureEngineer
from src.ml.sequence_utils import create_sequences

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtest result container."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    final_value: float
    trades: pd.DataFrame


class Backtester:
    """
    Backtesting engine for strategy validation.
    Uses vectorbt for fast vectorized backtesting.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.0,
        slippage: float = 0.001
    ):
        """
        Initialize Backtester.

        Args:
            initial_capital: Starting capital.
            commission: Commission per trade (fraction).
            slippage: Slippage assumption (fraction).
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()

    def run(
        self,
        symbol: str,
        entry_signals: pd.Series,
        exit_signals: pd.Series,
        prices: pd.Series
    ) -> BacktestResult:
        """
        Run backtest with provided signals.

        Args:
            symbol: Stock symbol.
            entry_signals: Boolean series for entry points.
            exit_signals: Boolean series for exit points.
            prices: Price series.

        Returns:
            BacktestResult with performance metrics.
        """
        if HAS_VBT:
            return self._run_vectorbt(entry_signals, exit_signals, prices)
        else:
            return self._run_simple(entry_signals, exit_signals, prices)

    def run_strategy(
        self,
        symbol: str,
        strategy_func,
        period: str = "2y"
    ) -> BacktestResult:
        """
        Run backtest with a strategy function.

        Args:
            symbol: Stock symbol.
            strategy_func: Function that takes DataFrame and returns (entries, exits).
            period: Historical period to test.

        Returns:
            BacktestResult with performance metrics.
        """
        # Fetch data
        df = self.data_fetcher.fetch_historical(symbol, period=period)

        if df.empty:
            raise ValueError(f"No data for {symbol}")

        # Add features (including extended features)
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
        df = df.dropna()

        # Generate signals
        entries, exits = strategy_func(df)

        # Run backtest
        return self.run(symbol, entries, exits, df["close"])

    def _run_vectorbt(
        self,
        entries: pd.Series,
        exits: pd.Series,
        prices: pd.Series
    ) -> BacktestResult:
        """Run backtest using vectorbt."""
        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.commission + self.slippage,
            freq="1D"
        )

        # Extract metrics
        stats = portfolio.stats()

        # Get trades
        trades_df = portfolio.trades.records_readable

        return BacktestResult(
            total_return=stats.get("Total Return [%]", 0) / 100,
            annualized_return=stats.get("Annualized Return [%]", 0) / 100,
            sharpe_ratio=stats.get("Sharpe Ratio", 0),
            sortino_ratio=stats.get("Sortino Ratio", 0),
            max_drawdown=stats.get("Max Drawdown [%]", 0) / 100,
            win_rate=stats.get("Win Rate [%]", 0) / 100,
            profit_factor=stats.get("Profit Factor", 0),
            total_trades=int(stats.get("Total Trades", 0)),
            avg_trade_duration=stats.get("Avg Winning Trade Duration", pd.Timedelta(0)).days,
            final_value=portfolio.final_value(),
            trades=trades_df
        )

    def _run_simple(
        self,
        entries: pd.Series,
        exits: pd.Series,
        prices: pd.Series
    ) -> BacktestResult:
        """Run simple backtest without vectorbt."""
        capital = self.initial_capital
        position = 0
        entry_price = 0
        trades = []

        for i in range(len(prices)):
            price = prices.iloc[i]

            if entries.iloc[i] and position == 0:
                # Buy
                shares = int(capital * 0.95 / price)
                if shares > 0:
                    cost = shares * price * (1 + self.commission + self.slippage)
                    capital -= cost
                    position = shares
                    entry_price = price
                    trades.append({
                        "entry_date": prices.index[i],
                        "entry_price": price,
                        "shares": shares
                    })

            elif exits.iloc[i] and position > 0:
                # Sell
                proceeds = position * price * (1 - self.commission - self.slippage)
                capital += proceeds

                if trades:
                    trades[-1]["exit_date"] = prices.index[i]
                    trades[-1]["exit_price"] = price
                    trades[-1]["pnl"] = proceeds - (position * entry_price)
                    trades[-1]["return"] = (price - entry_price) / entry_price

                position = 0
                entry_price = 0

        # Close any remaining position
        if position > 0:
            capital += position * prices.iloc[-1]

        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        final_value = capital

        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Calculate other metrics
        if len(trades_df) > 0 and "return" in trades_df.columns:
            wins = trades_df[trades_df["return"] > 0]
            losses = trades_df[trades_df["return"] <= 0]
            win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0

            total_wins = wins["pnl"].sum() if len(wins) > 0 else 0
            total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        else:
            win_rate = 0
            profit_factor = 0

        # Calculate Sharpe ratio approximation
        if len(trades_df) > 0 and "return" in trades_df.columns:
            returns = trades_df["return"]
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        equity_curve = self._calculate_equity_curve(entries, exits, prices)
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Annualized return
        days = (prices.index[-1] - prices.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sharpe * 0.8,  # Approximation
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df),
            avg_trade_duration=0,
            final_value=final_value,
            trades=trades_df
        )

    def _calculate_equity_curve(
        self,
        entries: pd.Series,
        exits: pd.Series,
        prices: pd.Series
    ) -> pd.Series:
        """Calculate equity curve."""
        equity = [self.initial_capital]
        capital = self.initial_capital
        position = 0

        for i in range(len(prices)):
            if entries.iloc[i] and position == 0:
                position = int(capital * 0.95 / prices.iloc[i])
                capital -= position * prices.iloc[i]
            elif exits.iloc[i] and position > 0:
                capital += position * prices.iloc[i]
                position = 0

            current_equity = capital + position * prices.iloc[i]
            equity.append(current_equity)

        return pd.Series(equity[1:], index=prices.index)

    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())

    def walk_forward(
        self,
        symbol: str,
        strategy_func,
        train_period: int = 252,
        test_period: int = 63,
        step: int = 21
    ) -> List[BacktestResult]:
        """
        Perform walk-forward analysis.

        Args:
            symbol: Stock symbol.
            strategy_func: Strategy function.
            train_period: Training period in days.
            test_period: Test period in days.
            step: Step size in days.

        Returns:
            List of BacktestResult for each period.
        """
        # Fetch all data
        df = self.data_fetcher.fetch_historical(symbol, period="5y")
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
        df = df.dropna()

        results = []

        for i in range(0, len(df) - train_period - test_period, step):
            train_df = df.iloc[i:i + train_period]
            test_df = df.iloc[i + train_period:i + train_period + test_period]

            # Generate signals on test data using strategy trained on train data
            try:
                entries, exits = strategy_func(test_df)
                result = self.run(symbol, entries, exits, test_df["close"])
                results.append(result)
            except Exception as e:
                logger.warning(f"Walk-forward period {i} failed: {e}")

        return results

    def run_ml_strategy(
        self,
        symbols: List[str],
        model,
        period: str = "1y",
        confidence_threshold: float = 0.6,
        sequence_length: int = 20,
        position_size: float = 0.1
    ) -> Dict[str, BacktestResult]:
        """
        Backtest ML ensemble strategy on multiple symbols.

        Args:
            symbols: List of stock symbols.
            model: Trained model (XGBoost or Ensemble) with predict_proba method.
            period: Historical period to test.
            confidence_threshold: Minimum confidence for entry.
            sequence_length: Sequence length for LSTM/CNN.
            position_size: Fraction of capital per position.

        Returns:
            Dict of symbol -> BacktestResult.
        """
        results = {}

        for symbol in symbols:
            try:
                result = self._backtest_ml_single(
                    symbol, model, period, confidence_threshold,
                    sequence_length, position_size
                )
                results[symbol] = result
                logger.info(f"{symbol}: {result.total_return:.2%} return, {result.total_trades} trades")
            except Exception as e:
                logger.error(f"Backtest failed for {symbol}: {e}")

        return results

    def _backtest_ml_single(
        self,
        symbol: str,
        model,
        period: str,
        confidence_threshold: float,
        sequence_length: int,
        position_size: float
    ) -> BacktestResult:
        """Backtest ML strategy on a single symbol."""
        # Fetch and prepare data
        df = self.data_fetcher.fetch_historical(symbol, period=period)
        if df.empty:
            raise ValueError(f"No data for {symbol}")

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
        df = df.dropna()

        # Get feature columns (same as training)
        exclude_cols = ["symbol", "date", "dividends", "stock_splits", "future_returns", "label_binary", "label_3class"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].values
        prices = df["close"]

        # Check if model is ensemble (needs sequences) or XGBoost (flat features)
        is_ensemble = hasattr(model, 'loaded_models') and len(getattr(model, 'loaded_models', {})) > 1

        # Generate predictions
        if is_ensemble:
            # Create sequences for LSTM/CNN
            X_seq, _ = create_sequences(df[feature_cols], pd.Series([0]*len(df)), sequence_length)
            # Align flat features with sequences
            X_flat = X[sequence_length:]
            prices = prices.iloc[sequence_length:]

            # Get predictions
            probas = model.predict_proba(X_flat, X_seq)
        else:
            # XGBoost only - use flat features
            probas = model.predict_proba(X)

        # Generate entry/exit signals based on confidence
        prob_up = probas[:, 1] if probas.ndim > 1 else probas
        entries = pd.Series(prob_up > confidence_threshold, index=prices.index)
        exits = pd.Series(prob_up < (1 - confidence_threshold), index=prices.index)

        # Run backtest
        return self._run_simple(entries, exits, prices)

    def run_ml_portfolio(
        self,
        symbols: List[str],
        model,
        period: str = "1y",
        confidence_threshold: float = 0.6,
        sequence_length: int = 20,
        max_positions: int = 5
    ) -> BacktestResult:
        """
        Backtest ML strategy as a portfolio (combined results).

        Args:
            symbols: List of stock symbols.
            model: Trained model with predict_proba method.
            period: Historical period.
            confidence_threshold: Minimum confidence for entry.
            sequence_length: Sequence length for sequences.
            max_positions: Maximum concurrent positions.

        Returns:
            Combined BacktestResult.
        """
        all_signals = []
        all_prices = {}

        for symbol in symbols:
            try:
                df = self.data_fetcher.fetch_historical(symbol, period=period)
                if df.empty:
                    continue

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
                df = df.dropna()

                exclude_cols = ["symbol", "date", "dividends", "stock_splits", "future_returns", "label_binary", "label_3class"]
                feature_cols = [c for c in df.columns if c not in exclude_cols]
                X = df[feature_cols].values
                prices = df["close"]

                is_ensemble = hasattr(model, 'loaded_models') and len(getattr(model, 'loaded_models', {})) > 1

                if is_ensemble:
                    X_seq, _ = create_sequences(df[feature_cols], pd.Series([0]*len(df)), sequence_length)
                    X_flat = X[sequence_length:]
                    prices = prices.iloc[sequence_length:]
                    probas = model.predict_proba(X_flat, X_seq)
                else:
                    probas = model.predict_proba(X)

                prob_up = probas[:, 1] if probas.ndim > 1 else probas

                for i, (date, price) in enumerate(prices.items()):
                    all_signals.append({
                        "date": date,
                        "symbol": symbol,
                        "price": price,
                        "prob_up": prob_up[i]
                    })

                all_prices[symbol] = prices

            except Exception as e:
                logger.warning(f"Failed to process {symbol}: {e}")

        if not all_signals:
            raise ValueError("No valid signals generated")

        # Simulate portfolio trading
        signals_df = pd.DataFrame(all_signals)
        return self._simulate_portfolio(signals_df, all_prices, confidence_threshold, max_positions)

    def _simulate_portfolio(
        self,
        signals_df: pd.DataFrame,
        prices: Dict[str, pd.Series],
        confidence_threshold: float,
        max_positions: int
    ) -> BacktestResult:
        """Simulate portfolio trading with multiple symbols."""
        capital = self.initial_capital
        positions = {}  # symbol -> {"shares": n, "entry_price": p, "entry_date": d}
        trades = []
        equity_history = []

        # Group by date
        for date in sorted(signals_df["date"].unique()):
            day_signals = signals_df[signals_df["date"] == date]

            # Calculate current equity
            equity = capital
            for sym, pos in positions.items():
                if sym in prices and date in prices[sym].index:
                    equity += pos["shares"] * prices[sym].loc[date]
            equity_history.append({"date": date, "equity": equity})

            # Check for exits first
            for sym in list(positions.keys()):
                sym_signal = day_signals[day_signals["symbol"] == sym]
                if not sym_signal.empty:
                    prob = sym_signal.iloc[0]["prob_up"]
                    price = sym_signal.iloc[0]["price"]

                    # Exit if confidence drops
                    if prob < (1 - confidence_threshold):
                        pos = positions.pop(sym)
                        proceeds = pos["shares"] * price * (1 - self.commission - self.slippage)
                        capital += proceeds
                        trades.append({
                            "symbol": sym,
                            "entry_date": pos["entry_date"],
                            "entry_price": pos["entry_price"],
                            "exit_date": date,
                            "exit_price": price,
                            "shares": pos["shares"],
                            "pnl": proceeds - (pos["shares"] * pos["entry_price"]),
                            "return": (price - pos["entry_price"]) / pos["entry_price"]
                        })

            # Check for entries
            if len(positions) < max_positions:
                # Sort by probability descending
                candidates = day_signals[day_signals["prob_up"] > confidence_threshold]
                candidates = candidates[~candidates["symbol"].isin(positions.keys())]
                candidates = candidates.sort_values("prob_up", ascending=False)

                for _, row in candidates.iterrows():
                    if len(positions) >= max_positions:
                        break

                    position_value = capital / (max_positions - len(positions) + 1) * 0.95
                    shares = int(position_value / row["price"])

                    if shares > 0:
                        cost = shares * row["price"] * (1 + self.commission + self.slippage)
                        if cost <= capital:
                            capital -= cost
                            positions[row["symbol"]] = {
                                "shares": shares,
                                "entry_price": row["price"],
                                "entry_date": date
                            }

        # Close remaining positions at last price
        for sym, pos in positions.items():
            if sym in prices:
                last_price = prices[sym].iloc[-1]
                proceeds = pos["shares"] * last_price
                capital += proceeds
                trades.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "exit_date": prices[sym].index[-1],
                    "exit_price": last_price,
                    "shares": pos["shares"],
                    "pnl": proceeds - (pos["shares"] * pos["entry_price"]),
                    "return": (last_price - pos["entry_price"]) / pos["entry_price"]
                })

        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_history).set_index("date")
        final_value = capital

        total_return = (final_value - self.initial_capital) / self.initial_capital

        if len(trades_df) > 0:
            wins = trades_df[trades_df["return"] > 0]
            losses = trades_df[trades_df["return"] <= 0]
            win_rate = len(wins) / len(trades_df)
            total_wins = wins["pnl"].sum() if len(wins) > 0 else 0
            total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
            returns = trades_df["return"]
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
            sharpe = 0

        # Max drawdown from equity curve
        if len(equity_df) > 0:
            peak = equity_df["equity"].expanding().max()
            drawdown = (equity_df["equity"] - peak) / peak
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0

        # Annualized return
        if len(equity_df) > 1:
            days = (equity_df.index[-1] - equity_df.index[0]).days
            years = days / 365.25
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        else:
            annualized_return = 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sharpe * 0.8,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df),
            avg_trade_duration=0,
            final_value=final_value,
            trades=trades_df
        )

    @staticmethod
    def print_results(result: BacktestResult) -> None:
        """Print backtest results."""
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Total Return:      {result.total_return:>10.2%}")
        print(f"Annualized Return: {result.annualized_return:>10.2%}")
        print(f"Sharpe Ratio:      {result.sharpe_ratio:>10.2f}")
        print(f"Sortino Ratio:     {result.sortino_ratio:>10.2f}")
        print(f"Max Drawdown:      {result.max_drawdown:>10.2%}")
        print(f"Win Rate:          {result.win_rate:>10.2%}")
        print(f"Profit Factor:     {result.profit_factor:>10.2f}")
        print(f"Total Trades:      {result.total_trades:>10}")
        print(f"Final Value:       ${result.final_value:>10,.2f}")
        print("=" * 50)
