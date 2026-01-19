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
import pandas as pd
from config.settings import setup_logging

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
    print(f"Model: {'ENSEMBLE (XGBoost+LSTM+CNN)' if args.ensemble else 'XGBoost'}")
    print("=" * 60)

    # Import here to avoid slow startup
    from src.backtest.backtester import Backtester

    # Load model
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

    # Create backtester
    backtester = Backtester(
        initial_capital=args.capital,
        commission=0.0,
        slippage=0.001
    )

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

        # Show trade summary
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

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
