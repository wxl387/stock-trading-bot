#!/usr/bin/env python3
"""
Script to run backtests on the ML trading strategy.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from config.settings import setup_logging, DATA_DIR

logger = setup_logging()


def main():
    """Run backtest."""
    parser = argparse.ArgumentParser(description="Backtest the ML trading strategy")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA",
                        help="Comma-separated list of symbols")
    parser.add_argument("--period", type=str, default="1y",
                        help="Historical period (e.g., 6mo, 1y, 2y)")
    parser.add_argument("--capital", type=float, default=100000,
                        help="Initial capital")
    parser.add_argument("--confidence", type=float, default=0.6,
                        help="Confidence threshold for trades")
    parser.add_argument("--max-positions", type=int, default=5,
                        help="Maximum concurrent positions")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use ensemble model (XGBoost+LSTM+CNN)")
    parser.add_argument("--single", action="store_true",
                        help="Backtest each symbol separately (vs portfolio)")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Use walk-forward validation (retrains models, no look-ahead bias)")
    parser.add_argument("--train-period", type=int, default=252,
                        help="Training window in days for walk-forward (default: 252)")
    parser.add_argument("--test-period", type=int, default=63,
                        help="Test window in days for walk-forward (default: 63)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip generating equity curve chart")
    parser.add_argument("--use-optimized-params", type=str, default=None,
                        help="Path to optimized_params.json from walk-forward optimization")

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    print("\n" + "=" * 60)
    print("ML STRATEGY BACKTEST")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {args.period}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Confidence Threshold: {args.confidence:.0%}")
    print(f"Max Positions: {args.max_positions}")
    if args.walk_forward:
        print(f"Mode: WALK-FORWARD (train={args.train_period}d, test={args.test_period}d)")
        print(f"Model: {'ENSEMBLE (XGBoost+LSTM+CNN)' if args.ensemble else 'XGBoost'}")
    else:
        print(f"Model: {'ENSEMBLE (XGBoost+LSTM+CNN)' if args.ensemble else 'XGBoost'}")
    print("=" * 60)

    # Import here to avoid slow startup
    from src.backtest.backtester import Backtester

    # Create backtester
    backtester = Backtester(
        initial_capital=args.capital,
        commission=0.0,
        slippage=0.001
    )

    if args.walk_forward:
        # Walk-forward: retrains models at each step (no look-ahead bias)
        if args.use_optimized_params:
            print(f"\nUsing optimized params from: {args.use_optimized_params}")
        print(f"Running walk-forward backtest (retraining at each step)...")
        result = backtester.walk_forward_ml(
            symbols=symbols,
            train_period=args.train_period,
            test_period=args.test_period,
            step=args.test_period,
            confidence_threshold=args.confidence,
            sequence_length=20,
            max_positions=args.max_positions,
            use_ensemble=args.ensemble,
            optimized_params_path=args.use_optimized_params
        )

        Backtester.print_results(result)

        # Trade summary
        if len(result.trades) > 0:
            print("\nTRADE SUMMARY BY SYMBOL")
            print("-" * 50)
            trades_by_symbol = result.trades.groupby("symbol").agg({
                "pnl": ["sum", "count"],
                "return": "mean"
            })
            trades_by_symbol.columns = ["Total P&L", "Trades", "Avg Return"]
            trades_by_symbol["Total P&L"] = trades_by_symbol["Total P&L"].apply(lambda x: f"${x:,.2f}")
            trades_by_symbol["Avg Return"] = trades_by_symbol["Avg Return"].apply(lambda x: f"{x:.2%}")
            print(trades_by_symbol.to_string())

            print(f"\nBest Trade:  {result.trades['return'].max():.2%}")
            print(f"Worst Trade: {result.trades['return'].min():.2%}")

    else:
        # Load pre-trained model
        if args.ensemble:
            from src.ml.models.ensemble_model import EnsembleModel
            print("\nLoading ensemble model...")
            model = EnsembleModel()
            loaded = model.load_models(
                xgboost_name="trading_model",
                lstm_name="lstm_trading_model",
                cnn_name="cnn_trading_model"
            )
            print(f"Loaded: {loaded}")
        else:
            from src.ml.models.xgboost_model import XGBoostModel
            print("\nLoading XGBoost model...")
            model = XGBoostModel()
            model.load("trading_model")
            print("Loaded XGBoost model")

        print(f"\nFetching data and running backtest...")

        if args.single:
            # Backtest each symbol separately
            results = backtester.run_ml_strategy(
                symbols=symbols,
                model=model,
                period=args.period,
                confidence_threshold=args.confidence,
                sequence_length=20
            )

            print("\n" + "=" * 60)
            print("INDIVIDUAL SYMBOL RESULTS")
            print("=" * 60)
            print(f"{'Symbol':<8} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8} {'Trades':>7}")
            print("-" * 60)

            total_return = 0
            for symbol, result in results.items():
                print(f"{symbol:<8} {result.total_return:>10.2%} {result.sharpe_ratio:>8.2f} "
                      f"{result.max_drawdown:>8.2%} {result.win_rate:>8.2%} {result.total_trades:>7}")
                total_return += result.total_return

            avg_return = total_return / len(results) if results else 0
            print("-" * 60)
            print(f"{'Average':<8} {avg_return:>10.2%}")
            result = None  # No single result for charting

        else:
            # Portfolio backtest
            result = backtester.run_ml_portfolio(
                symbols=symbols,
                model=model,
                period=args.period,
                confidence_threshold=args.confidence,
                sequence_length=20,
                max_positions=args.max_positions
            )

            Backtester.print_results(result)

            # Trade summary
            if len(result.trades) > 0:
                print("\nTRADE SUMMARY BY SYMBOL")
                print("-" * 50)
                trades_by_symbol = result.trades.groupby("symbol").agg({
                    "pnl": ["sum", "count"],
                    "return": "mean"
                })
                trades_by_symbol.columns = ["Total P&L", "Trades", "Avg Return"]
                trades_by_symbol["Total P&L"] = trades_by_symbol["Total P&L"].apply(lambda x: f"${x:,.2f}")
                trades_by_symbol["Avg Return"] = trades_by_symbol["Avg Return"].apply(lambda x: f"{x:.2%}")
                print(trades_by_symbol.to_string())

                print(f"\nBest Trade:  {result.trades['return'].max():.2%}")
                print(f"Worst Trade: {result.trades['return'].min():.2%}")

    # Generate chart
    if not args.no_chart and result is not None and result.equity_curve is not None:
        try:
            chart_path = Backtester.plot_results(
                result,
                buy_hold_equity=getattr(backtester, '_last_bh_equity', None),
                spy_equity=getattr(backtester, '_last_spy_equity', None),
                title=f"ML Strategy Backtest ({'Walk-Forward' if args.walk_forward else args.period})",
                output_path=str(DATA_DIR / "backtest_results.png")
            )
            print(f"\nEquity curve chart saved to: {chart_path}")
        except Exception as e:
            print(f"\nWarning: Could not generate chart: {e}")

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
