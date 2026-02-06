"""
Backtesting module for strategy validation.
"""
import logging
from dataclasses import dataclass, field
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
    equity_curve: Optional[pd.Series] = None
    benchmark_return: Optional[float] = None
    spy_return: Optional[float] = None
    daily_returns: Optional[pd.Series] = None


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

        # Store benchmark data for plotting after run
        self._last_bh_equity = None
        self._last_spy_equity = None

    def _calculate_sortino(self, daily_returns: pd.Series, risk_free_rate: float = 0.05) -> float:
        """Calculate Sortino ratio using downside deviation."""
        if len(daily_returns) < 2 or daily_returns.std() == 0:
            return 0.0
        daily_rf = risk_free_rate / 252
        excess_returns = daily_returns - daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
        downside_dev = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)
        if downside_dev == 0:
            return 0.0
        ann_return = daily_returns.mean() * 252
        return (ann_return - risk_free_rate) / downside_dev

    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns excluding metadata/label columns."""
        exclude = [
            "symbol", "date", "dividends", "stock_splits",
            "future_returns", "label_binary", "label_3class"
        ]
        return [c for c in df.columns if c not in exclude]

    def run_simple_backtest(
        self,
        prices: pd.Series,
        signals: pd.Series
    ) -> BacktestResult:
        """
        Run backtest with numeric signals.

        Args:
            prices: Price series with datetime index.
            signals: Numeric series: 1=buy, -1=sell, 0=hold.

        Returns:
            BacktestResult with performance metrics.
        """
        if prices.empty or signals.empty:
            raise ValueError("Prices and signals must be non-empty")

        entries = (signals > 0)
        exits = (signals < 0)
        return self._run_simple(entries, exits, prices)

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
            include_sentiment=False,
            include_macro=False,
            include_cross_asset=True,
            include_interactions=False,
            include_lagged=True,
            use_cache=True
        )
        df = df.ffill().fillna(0)
        df = df[df['close'] > 0]

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
                    trades[-1]["pnl"] = proceeds - (position * entry_price * (1 + self.commission + self.slippage))
                    trades[-1]["return"] = (price - entry_price) / entry_price

                position = 0
                entry_price = 0

        # Close any remaining position
        if position > 0:
            last_price = prices.iloc[-1]
            liquidation_proceeds = position * last_price * (1 - self.commission - self.slippage)
            capital += liquidation_proceeds
            if trades and "exit_date" not in trades[-1]:
                trades[-1]["exit_date"] = prices.index[-1]
                trades[-1]["exit_price"] = last_price
                trades[-1]["pnl"] = liquidation_proceeds - (position * entry_price * (1 + self.commission + self.slippage))
                trades[-1]["return"] = (last_price - entry_price) / entry_price

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

        # Calculate equity curve and daily returns
        equity_curve = self._calculate_equity_curve(entries, exits, prices)
        daily_returns = equity_curve.pct_change().dropna()

        # Sharpe ratio from daily returns
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio from daily returns
        sortino = self._calculate_sortino(daily_returns)

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(equity_curve)

        # Annualized return
        days = (prices.index[-1] - prices.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df),
            avg_trade_duration=0,
            final_value=final_value,
            trades=trades_df,
            equity_curve=equity_curve,
            daily_returns=daily_returns
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
                capital -= position * prices.iloc[i] * (1 + self.commission + self.slippage)
            elif exits.iloc[i] and position > 0:
                capital += position * prices.iloc[i] * (1 - self.commission - self.slippage)
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
            include_sentiment=False,
            include_macro=False,
            include_cross_asset=True,
            include_interactions=False,
            include_lagged=True,
            use_cache=True
        )
        df = df.ffill().fillna(0)
        df = df[df['close'] > 0]

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
            include_sentiment=False,
            include_macro=False,
            include_cross_asset=True,
            include_interactions=False,
            include_lagged=True,
            use_cache=True
        )
        df = df.ffill().fillna(0)
        df = df[df['close'] > 0]

        # Get feature columns (same as training)
        exclude_cols = ["symbol", "date", "dividends", "stock_splits", "future_returns", "label_binary", "label_3class"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].values
        prices = df["close"]

        # Check if model is ensemble (needs sequences) or XGBoost (flat features)
        is_ensemble = hasattr(model, 'active_models') and len(getattr(model, 'active_models', [])) > 1

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
                    include_sentiment=False,
                    include_macro=False,
                    include_cross_asset=True,
                    include_interactions=False,
                    include_lagged=True,
                    use_cache=True
                )
                # Fill NaN from optional features (macro/sentiment may be unavailable)
                df = df.ffill().fillna(0)
                df = df[df['close'] > 0]

                exclude_cols = ["symbol", "date", "dividends", "stock_splits", "future_returns", "label_binary", "label_3class"]
                feature_cols = [c for c in df.columns if c not in exclude_cols]
                X = df[feature_cols].values
                prices = df["close"]

                is_ensemble = hasattr(model, 'active_models') and len(getattr(model, 'active_models', [])) > 1

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
        result = self._simulate_portfolio(signals_df, all_prices, confidence_threshold, max_positions)

        # Add benchmark comparison
        if result.equity_curve is not None and len(result.equity_curve) > 0:
            bh_return, spy_return, bh_equity, spy_equity = self._calculate_benchmarks(
                result.equity_curve, symbols, period
            )
            result.benchmark_return = bh_return
            result.spy_return = spy_return
            self._last_bh_equity = bh_equity
            self._last_spy_equity = spy_equity

        return result

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
                            "pnl": proceeds - (pos["shares"] * pos["entry_price"] * (1 + self.commission + self.slippage)),
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
                proceeds = pos["shares"] * last_price * (1 - self.commission - self.slippage)
                capital += proceeds
                entry_cost = pos["shares"] * pos["entry_price"] * (1 + self.commission + self.slippage)
                trades.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "exit_date": prices[sym].index[-1],
                    "exit_price": last_price,
                    "shares": pos["shares"],
                    "pnl": proceeds - entry_cost,
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
        else:
            win_rate = 0
            profit_factor = 0

        # Equity curve and daily returns
        if len(equity_df) > 0:
            equity_series = equity_df["equity"]
            daily_returns = equity_series.pct_change().dropna()

            # Sharpe from daily equity returns
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0.0

            sortino = self._calculate_sortino(daily_returns)

            # Max drawdown
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = abs(drawdown.min())
        else:
            equity_series = pd.Series(dtype=float)
            daily_returns = pd.Series(dtype=float)
            sharpe = 0.0
            sortino = 0.0
            max_drawdown = 0.0

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
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df),
            avg_trade_duration=0,
            final_value=final_value,
            trades=trades_df,
            equity_curve=equity_series,
            daily_returns=daily_returns
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

        if result.benchmark_return is not None:
            print(f"\nBENCHMARK COMPARISON")
            print("-" * 50)
            print(f"Strategy Return:   {result.total_return:>10.2%}")
            print(f"Buy & Hold Return: {result.benchmark_return:>10.2%}")
            if result.spy_return is not None:
                print(f"SPY Return:        {result.spy_return:>10.2%}")
            alpha_bh = result.total_return - result.benchmark_return
            print(f"Alpha vs B&H:      {alpha_bh:>+10.2%}")
            if result.spy_return is not None:
                alpha_spy = result.total_return - result.spy_return
                print(f"Alpha vs SPY:      {alpha_spy:>+10.2%}")

    def _calculate_benchmarks(
        self,
        equity_curve: pd.Series,
        symbols: List[str],
        period: str
    ) -> Tuple[float, float, pd.Series, pd.Series]:
        """
        Calculate benchmark returns for comparison.

        Returns:
            (buy_hold_return, spy_return, buy_hold_equity, spy_equity)
        """
        # Ensure equity curve index is tz-naive for comparison
        eq_index = equity_curve.index
        if eq_index.tz is not None:
            eq_index = eq_index.tz_localize(None)

        # Equal-weight buy-and-hold of same symbols
        buy_hold_values = []
        for symbol in symbols:
            try:
                df = self.data_fetcher.fetch_historical(symbol, period=period)
                if not df.empty:
                    prices = df['close'].copy()
                    if prices.index.tz is not None:
                        prices.index = prices.index.tz_localize(None)
                    prices = prices.reindex(eq_index, method='ffill').dropna()
                    if len(prices) > 0:
                        normalized = prices / prices.iloc[0]
                        buy_hold_values.append(normalized)
            except Exception:
                pass

        if buy_hold_values:
            equal_weight = pd.concat(buy_hold_values, axis=1).mean(axis=1)
            buy_hold_equity = equal_weight * self.initial_capital
            buy_hold_return = float(equal_weight.iloc[-1] - 1.0)
        else:
            buy_hold_equity = pd.Series(self.initial_capital, index=eq_index)
            buy_hold_return = 0.0

        # SPY benchmark
        try:
            spy_df = self.data_fetcher.fetch_historical("SPY", period=period)
            if not spy_df.empty:
                spy_prices = spy_df['close'].copy()
                if spy_prices.index.tz is not None:
                    spy_prices.index = spy_prices.index.tz_localize(None)
                spy_prices = spy_prices.reindex(eq_index, method='ffill').dropna()
                if len(spy_prices) > 0:
                    spy_normalized = spy_prices / spy_prices.iloc[0]
                    spy_equity = spy_normalized * self.initial_capital
                    spy_return = float(spy_normalized.iloc[-1] - 1.0)
                else:
                    spy_equity = pd.Series(self.initial_capital, index=eq_index)
                    spy_return = 0.0
            else:
                spy_equity = pd.Series(self.initial_capital, index=eq_index)
                spy_return = 0.0
        except Exception:
            spy_equity = pd.Series(self.initial_capital, index=eq_index)
            spy_return = 0.0

        return buy_hold_return, spy_return, buy_hold_equity, spy_equity

    def walk_forward_ml(
        self,
        symbols: List[str],
        train_period: int = 252,
        test_period: int = 63,
        step: int = 63,
        confidence_threshold: float = 0.6,
        sequence_length: int = 20,
        max_positions: int = 5,
        use_ensemble: bool = True,
        model_params: Optional[Dict] = None,
        optimized_params_path: Optional[str] = None,
        use_regime: bool = False,
        regime_mode: str = "adjust",
        enable_stop_loss: bool = False,
        enable_trailing_stop: bool = False,
        enable_kelly: bool = False,
        enable_circuit_breaker: bool = False,
        trailing_atr_multiplier: float = 2.0,
        circuit_breaker_threshold: float = 0.15,
        circuit_breaker_recovery: float = 0.05,
    ) -> BacktestResult:
        """
        Walk-forward ML backtest with model retraining at each step.

        For each window:
          1. Train fresh models on train_period days
          2. Test on next test_period days (out-of-sample)
          3. Advance by step days and repeat

        Args:
            symbols: List of stock symbols.
            train_period: Training window in days.
            test_period: Test window in days.
            step: Step size in days.
            confidence_threshold: Probability threshold for entry.
            sequence_length: Sequence length for LSTM/CNN.
            max_positions: Maximum concurrent positions.
            use_ensemble: If True, train XGBoost+LSTM+CNN; else XGBoost only.
            model_params: Optional dict of XGBoost hyperparameters.
            optimized_params_path: Optional path to optimized_params.json.

        Returns:
            Combined BacktestResult across all out-of-sample windows.
        """
        from src.ml.models.xgboost_model import XGBoostModel
        from src.ml.models.ensemble_model import EnsembleModel

        # Load optimized params if path provided
        if optimized_params_path:
            import json
            with open(optimized_params_path, 'r') as f:
                opt_data = json.load(f)
            model_params = opt_data.get("best_model_params", model_params)
            if "best_trading_params" in opt_data:
                confidence_threshold = opt_data["best_trading_params"].get(
                    "confidence_threshold", confidence_threshold)
                max_positions = opt_data["best_trading_params"].get(
                    "max_positions", max_positions)
            logger.info(f"Loaded optimized params from {optimized_params_path}")

        logger.info(f"Starting walk-forward backtest: train={train_period}d, test={test_period}d, step={step}d")

        # Fetch and prepare all data
        all_data = {}
        for symbol in symbols:
            try:
                df = self.data_fetcher.fetch_historical(symbol, period="5y")
                if df.empty:
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
                df = df.ffill().fillna(0)
                df = df[df['close'] > 0]
                if len(df) > train_period + test_period:
                    all_data[symbol] = df
                    logger.info(f"  {symbol}: {len(df)} samples loaded")
            except Exception as e:
                logger.warning(f"  {symbol}: failed to load - {e}")

        if not all_data:
            raise ValueError("No valid data for walk-forward backtest")

        # Pre-compute regime series if enabled
        regime_series = None
        if use_regime:
            from src.risk.regime_detector import compute_regime_series
            try:
                spy_data = self.data_fetcher.fetch_historical("SPY", period="5y")
                if not spy_data.empty and len(spy_data) >= 200:
                    if spy_data.index.tz is not None:
                        spy_data.index = spy_data.index.tz_localize(None)
                    regime_series = compute_regime_series(spy_data)
                    regime_counts = regime_series.value_counts()
                    logger.info(f"Regime series computed: {len(regime_series)} days")
                    for regime, count in regime_counts.items():
                        logger.info(f"  {regime.value.upper()}: {count} days ({count/len(regime_series)*100:.1f}%)")
                else:
                    logger.warning("Insufficient SPY data for regime detection")
            except Exception as e:
                logger.warning(f"Regime detection failed: {e}. Running without regime.")

        # Find common date range
        common_dates = None
        for df in all_data.values():
            dates = set(df.index)
            common_dates = dates if common_dates is None else common_dates.intersection(dates)
        common_dates = sorted(common_dates)
        total_days = len(common_dates)

        logger.info(f"Common dates: {total_days} days ({common_dates[0].date()} to {common_dates[-1].date()})")

        # Walk-forward loop
        all_equity = []
        all_trades = []
        capital = self.initial_capital
        window_count = 0

        for window_start in range(0, total_days - train_period - test_period + 1, step):
            train_end = window_start + train_period
            test_end = min(train_end + test_period, total_days)

            train_dates = common_dates[window_start:train_end]
            test_dates = common_dates[train_end:test_end]

            if len(test_dates) == 0:
                break

            window_count += 1
            logger.info(f"  Window {window_count}: train {train_dates[0].date()}-{train_dates[-1].date()}, "
                       f"test {test_dates[0].date()}-{test_dates[-1].date()}")

            # Prepare training data
            X_train_list, y_train_list = [], []
            feature_cols = None
            for symbol, df in all_data.items():
                train_df = df.loc[df.index.isin(train_dates)]
                if len(train_df) < 50:
                    continue
                if feature_cols is None:
                    feature_cols = self._get_feature_columns(train_df)
                X_sym = train_df[feature_cols].copy()
                # Create labels: price up in 5 days
                y_sym = (train_df['close'].shift(-5) / train_df['close'] - 1 > 0).astype(int)
                # Drop last 5 rows (no future label)
                X_sym = X_sym.iloc[:-5]
                y_sym = y_sym.iloc[:-5]
                X_train_list.append(X_sym.reset_index(drop=True))
                y_train_list.append(y_sym.reset_index(drop=True))

            if not X_train_list:
                continue

            X_train = pd.concat(X_train_list, ignore_index=True)
            y_train = pd.concat(y_train_list, ignore_index=True)

            # Train models
            try:
                if model_params:
                    xgb_params = {k: v for k, v in model_params.items()
                                  if k in ["n_estimators", "max_depth", "learning_rate",
                                           "subsample", "colsample_bytree", "min_child_weight",
                                           "gamma", "reg_alpha", "reg_lambda"]}
                    xgb_model = XGBoostModel(**xgb_params)
                else:
                    xgb_model = XGBoostModel()
                xgb_model.train(X_train, y_train)

                if use_ensemble:
                    from src.ml.models.lstm_model import LSTMModel
                    from src.ml.models.cnn_model import CNNModel

                    # Create sequences for LSTM/CNN
                    X_seq, y_seq = create_sequences(
                        X_train.values, y_train.values, sequence_length
                    )

                    lstm_model = LSTMModel(
                        sequence_length=sequence_length,
                        n_features=len(feature_cols)
                    )
                    lstm_model.train(X_seq, y_seq, epochs=15)

                    cnn_model = CNNModel(
                        sequence_length=sequence_length,
                        n_features=len(feature_cols)
                    )
                    cnn_model.train(X_seq, y_seq, epochs=15)

                    # Create ensemble
                    model = EnsembleModel(sequence_length=sequence_length)
                    model.xgboost_model = xgb_model
                    model.lstm_model = lstm_model
                    model.cnn_model = cnn_model
                    model.active_models = ["xgboost", "lstm", "cnn"]
                    model.is_loaded = True
                else:
                    model = xgb_model

            except Exception as e:
                logger.error(f"  Training failed for window {window_count}: {e}")
                continue

            # Generate signals on test window
            test_signals = []
            test_prices = {}
            test_highs = {}
            test_lows = {}
            test_atr = {}
            for symbol, df in all_data.items():
                test_df = df.loc[df.index.isin(test_dates)]
                if len(test_df) < 1:
                    continue

                try:
                    X_test = test_df[feature_cols]

                    if use_ensemble and hasattr(model, 'predict_proba'):
                        # For ensemble, need sequences
                        # Get history before test window for sequences
                        pre_test_idx = df.index.get_loc(test_dates[0])
                        history_start = max(0, pre_test_idx - sequence_length)
                        full_test = df.iloc[history_start:pre_test_idx + len(test_df)]
                        X_full = full_test[feature_cols].values

                        if len(X_full) >= sequence_length + 1:
                            X_seq_test, _ = create_sequences(
                                X_full,
                                np.zeros(len(X_full)),
                                sequence_length
                            )
                            # Align: sequences start at index sequence_length
                            n_seq = len(X_seq_test)
                            # Flat features aligned with sequences
                            X_flat = pd.DataFrame(
                                X_full[sequence_length:sequence_length + n_seq],
                                columns=feature_cols
                            )
                            probas = model.predict_proba(X_flat, X_seq_test)
                        else:
                            # Not enough history, use XGBoost only
                            probas = xgb_model.predict_proba(X_test)
                    else:
                        probas = model.predict_proba(X_test)

                    prob_up = probas[:, 1] if probas.ndim > 1 else probas
                    test_prices_sym = test_df['close']

                    for i, (date, price) in enumerate(test_prices_sym.items()):
                        if i < len(prob_up):
                            test_signals.append({
                                "date": date,
                                "symbol": symbol,
                                "price": price,
                                "prob_up": prob_up[i]
                            })

                    test_prices[symbol] = test_prices_sym
                    # Risk management data
                    if 'high' in test_df.columns:
                        test_highs[symbol] = test_df['high']
                    if 'low' in test_df.columns:
                        test_lows[symbol] = test_df['low']
                    if 'atr_14' in test_df.columns:
                        test_atr[symbol] = test_df['atr_14']

                except Exception as e:
                    logger.debug(f"  Signal generation failed for {symbol}: {e}")

            if not test_signals:
                continue

            # Simulate portfolio on this test window
            signals_df = pd.DataFrame(test_signals)
            window_result = self._simulate_portfolio_window(
                signals_df, test_prices, confidence_threshold, max_positions, capital,
                regime_series=regime_series, regime_mode=regime_mode,
                highs=test_highs, lows=test_lows, atr=test_atr,
                enable_stop_loss=enable_stop_loss,
                enable_trailing_stop=enable_trailing_stop,
                enable_kelly=enable_kelly,
                enable_circuit_breaker=enable_circuit_breaker,
                trailing_atr_multiplier=trailing_atr_multiplier,
                circuit_breaker_threshold=circuit_breaker_threshold,
                circuit_breaker_recovery=circuit_breaker_recovery,
            )

            capital = window_result.final_value
            if window_result.equity_curve is not None:
                all_equity.append(window_result.equity_curve)
            if len(window_result.trades) > 0:
                all_trades.append(window_result.trades)

            logger.info(f"  Window {window_count} result: {window_result.total_return:.2%} "
                       f"({window_result.total_trades} trades)")

        # Combine results
        if all_equity:
            combined_equity = pd.concat(all_equity)
        else:
            combined_equity = pd.Series([self.initial_capital], index=[common_dates[0]])

        if all_trades:
            combined_trades = pd.concat(all_trades, ignore_index=True)
        else:
            combined_trades = pd.DataFrame()

        # Calculate combined metrics
        total_return = (capital - self.initial_capital) / self.initial_capital
        daily_returns = combined_equity.pct_change().dropna()

        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        sortino = self._calculate_sortino(daily_returns)

        if len(combined_equity) > 0:
            peak = combined_equity.expanding().max()
            drawdown = (combined_equity - peak) / peak
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0.0

        if len(combined_trades) > 0:
            wins = combined_trades[combined_trades["return"] > 0]
            losses = combined_trades[combined_trades["return"] <= 0]
            win_rate = len(wins) / len(combined_trades)
            total_wins = wins["pnl"].sum() if len(wins) > 0 else 0
            total_losses = abs(losses["pnl"].sum()) if len(losses) > 0 else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        else:
            win_rate = 0
            profit_factor = 0

        days = (combined_equity.index[-1] - combined_equity.index[0]).days if len(combined_equity) > 1 else 0
        years = days / 365.25 if days > 0 else 1
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        result = BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(combined_trades),
            avg_trade_duration=0,
            final_value=capital,
            trades=combined_trades,
            equity_curve=combined_equity,
            daily_returns=daily_returns
        )

        # Add benchmarks
        if len(combined_equity) > 0:
            bh_return, spy_return, bh_equity, spy_equity = self._calculate_benchmarks(
                combined_equity, symbols, "5y"
            )
            result.benchmark_return = bh_return
            result.spy_return = spy_return
            self._last_bh_equity = bh_equity
            self._last_spy_equity = spy_equity

        logger.info(f"Walk-forward complete: {window_count} windows, {total_return:.2%} total return")
        return result

    def _lookup_regime(self, regime_series, date):
        """Look up regime for a given date, handling index mismatches."""
        from src.risk.regime_detector import MarketRegime
        # Normalize timezone
        lookup_date = date
        if hasattr(lookup_date, 'tz') and lookup_date.tz is not None:
            lookup_date = lookup_date.tz_localize(None)

        if lookup_date in regime_series.index:
            return regime_series.loc[lookup_date]

        # Find nearest previous date
        valid_dates = regime_series.index[regime_series.index <= lookup_date]
        if len(valid_dates) > 0:
            return regime_series.loc[valid_dates[-1]]

        return MarketRegime.BULL

    def _simulate_portfolio_window(
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
    ) -> BacktestResult:
        """Simulate portfolio trading for a single window with custom starting capital."""
        from src.risk.regime_detector import MarketRegime, DEFAULT_REGIME_PARAMS

        capital = starting_capital
        positions = {}
        trades = []
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
            day_stop_loss_pct = 0.05  # default
            day_trailing_enabled = True

            if regime_series is not None:
                regime = self._lookup_regime(regime_series, date)
                regime_params = DEFAULT_REGIME_PARAMS[regime]

                if regime_mode == "filter":
                    if regime in (MarketRegime.BEAR, MarketRegime.CHOPPY):
                        day_confidence = 0.99  # Block entries
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
            equity_history.append({"date": date, "equity": equity})

            # Circuit breaker
            if enable_circuit_breaker:
                if equity > peak_equity:
                    peak_equity = equity
                current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                if current_dd >= circuit_breaker_threshold:
                    circuit_breaker_active = True
                elif circuit_breaker_active and current_dd <= circuit_breaker_recovery:
                    circuit_breaker_active = False

            # Stop-loss exits (checked before signal exits)
            if enable_stop_loss or enable_trailing_stop:
                for sym in list(positions.keys()):
                    pos = positions[sym]

                    # Get daily high/low/atr
                    day_low = lows[sym].loc[date] if (lows and sym in lows and date in lows[sym].index) else None
                    day_high = highs[sym].loc[date] if (highs and sym in highs and date in highs[sym].index) else None
                    day_atr = atr[sym].loc[date] if (atr and sym in atr and date in atr[sym].index) else None

                    if day_low is None:
                        sym_signal = day_signals[day_signals["symbol"] == sym]
                        day_low = sym_signal.iloc[0]["price"] if not sym_signal.empty else None
                    if day_low is None:
                        continue

                    # Fixed stop
                    fixed_stop = None
                    if enable_stop_loss:
                        fixed_stop = pos["entry_price"] * (1 - day_stop_loss_pct)

                    # Trailing stop (ATR-based)
                    trail_stop = None
                    if enable_trailing_stop and day_trailing_enabled and day_atr is not None and day_atr > 0:
                        trail_distance = day_atr * trailing_atr_multiplier
                        if sym not in trailing_stops:
                            trailing_stops[sym] = pos["entry_price"] - trail_distance
                        if day_high is not None:
                            trailing_stops[sym] = max(trailing_stops[sym], day_high - trail_distance)
                        trail_stop = trailing_stops[sym]

                    # Use tighter (higher) stop
                    effective_stop = None
                    if fixed_stop is not None and trail_stop is not None:
                        effective_stop = max(fixed_stop, trail_stop)
                    elif fixed_stop is not None:
                        effective_stop = fixed_stop
                    elif trail_stop is not None:
                        effective_stop = trail_stop

                    # Check trigger
                    if effective_stop is not None and day_low <= effective_stop:
                        pos = positions.pop(sym)
                        if sym in trailing_stops:
                            del trailing_stops[sym]
                        exit_price = min(effective_stop, day_low) if day_low < effective_stop else effective_stop
                        proceeds = pos["shares"] * exit_price * (1 - self.commission - self.slippage)
                        capital += proceeds
                        trade_record = {
                            "symbol": sym,
                            "entry_date": pos["entry_date"],
                            "entry_price": pos["entry_price"],
                            "exit_date": date,
                            "exit_price": exit_price,
                            "shares": pos["shares"],
                            "pnl": proceeds - (pos["shares"] * pos["entry_price"] * (1 + self.commission + self.slippage)),
                            "return": (exit_price - pos["entry_price"]) / pos["entry_price"],
                            "exit_reason": "stop_loss"
                        }
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
                        trade_record = {
                            "symbol": sym,
                            "entry_date": pos["entry_date"],
                            "entry_price": pos["entry_price"],
                            "exit_date": date,
                            "exit_price": price,
                            "shares": pos["shares"],
                            "pnl": proceeds - (pos["shares"] * pos["entry_price"] * (1 + self.commission + self.slippage)),
                            "return": (price - pos["entry_price"]) / pos["entry_price"],
                            "exit_reason": "signal"
                        }
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
                        win_rate = len(wins) / len(recent)
                        avg_win = np.mean([t["return"] for t in wins]) if wins else 0.0
                        avg_loss = abs(np.mean([t["return"] for t in losses])) if losses else 1.0
                        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
                        kelly_f = win_rate - (1 - win_rate) / win_loss_ratio if win_loss_ratio > 0 else -1.0
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
                                "entry_date": date
                            }
                            # Initialize trailing stop
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
                trade_record = {
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "exit_date": prices[sym].index[-1],
                    "exit_price": last_price,
                    "shares": pos["shares"],
                    "pnl": proceeds - entry_cost,
                    "return": (last_price - pos["entry_price"]) / pos["entry_price"],
                    "exit_reason": "window_end"
                }
                trades.append(trade_record)
                completed_trades.append(trade_record)

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_history).set_index("date")

        total_return = (capital - starting_capital) / starting_capital
        equity_series = equity_df["equity"] if len(equity_df) > 0 else pd.Series(dtype=float)

        return BacktestResult(
            total_return=total_return,
            annualized_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            win_rate=0,
            profit_factor=0,
            total_trades=len(trades_df),
            avg_trade_duration=0,
            final_value=capital,
            trades=trades_df,
            equity_curve=equity_series
        )

    @staticmethod
    def plot_results(
        result: BacktestResult,
        buy_hold_equity: Optional[pd.Series] = None,
        spy_equity: Optional[pd.Series] = None,
        title: str = "Backtest Results",
        output_path: Optional[str] = None
    ) -> str:
        """
        Plot equity curve with benchmarks and save to file.

        Args:
            result: BacktestResult with equity_curve populated.
            buy_hold_equity: Optional equal-weight buy-hold equity curve.
            spy_equity: Optional SPY equity curve.
            title: Chart title.
            output_path: File path to save (default: data/backtest_results.png).

        Returns:
            Path to saved chart file.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if result.equity_curve is None or len(result.equity_curve) == 0:
            raise ValueError("BacktestResult has no equity_curve data")

        fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                                 gridspec_kw={'height_ratios': [3, 1, 1]})

        # Panel 1: Equity curves
        ax1 = axes[0]
        ax1.plot(result.equity_curve.index, result.equity_curve.values,
                 label=f'Strategy ({result.total_return:.1%})',
                 linewidth=1.5, color='#1f77b4')
        if buy_hold_equity is not None and len(buy_hold_equity) > 0:
            bh_return = buy_hold_equity.iloc[-1] / buy_hold_equity.iloc[0] - 1
            ax1.plot(buy_hold_equity.index, buy_hold_equity.values,
                     label=f'Buy & Hold ({bh_return:.1%})',
                     linewidth=1.0, color='#2ca02c', linestyle='--')
        if spy_equity is not None and len(spy_equity) > 0:
            spy_return = spy_equity.iloc[-1] / spy_equity.iloc[0] - 1
            ax1.plot(spy_equity.index, spy_equity.values,
                     label=f'SPY ({spy_return:.1%})',
                     linewidth=1.0, color='#ff7f0e', linestyle='-.')
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        # Panel 2: Drawdown
        ax2 = axes[1]
        peak = result.equity_curve.expanding().max()
        drawdown = (result.equity_curve - peak) / peak * 100
        ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.4, color='red')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Rolling Sharpe
        ax3 = axes[2]
        if result.daily_returns is not None and len(result.daily_returns) > 63:
            rolling_sharpe = result.daily_returns.rolling(63).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0,
                raw=False
            )
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values,
                     color='purple', linewidth=1.0)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_ylabel('Rolling Sharpe (63d)')
        else:
            ax3.set_ylabel('Rolling Sharpe')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Date')

        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Metrics text
        metrics_text = (
            f"Return: {result.total_return:.2%}  |  "
            f"Sharpe: {result.sharpe_ratio:.2f}  |  "
            f"Sortino: {result.sortino_ratio:.2f}  |  "
            f"MaxDD: {result.max_drawdown:.2%}  |  "
            f"Win Rate: {result.win_rate:.0%}  |  "
            f"Trades: {result.total_trades}"
        )
        fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=9,
                 style='italic', color='#555555')

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        if output_path is None:
            from config.settings import DATA_DIR
            output_path = str(DATA_DIR / "backtest_results.png")

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved backtest chart to {output_path}")
        return output_path
