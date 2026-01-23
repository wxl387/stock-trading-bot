#!/usr/bin/env python3
"""
Walk-forward hyperparameter optimization for the ML trading strategy.

Usage:
    python scripts/run_wf_optimize.py --symbols AAPL,MSFT,GOOGL --n-trials 50
    python scripts/run_wf_optimize.py --n-trials 100 --optimize-features --metric sortino_ratio
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from config.settings import setup_logging, MODELS_DIR

logger = setup_logging()


def main():
    """Run walk-forward hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description="Walk-forward hyperparameter optimization for XGBoost"
    )
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL,AMZN,NVDA",
                        help="Comma-separated list of symbols")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--n-windows", type=int, default=4,
                        help="Number of walk-forward windows (default: 4)")
    parser.add_argument("--train-period", type=int, default=252,
                        help="Training window in days (default: 252)")
    parser.add_argument("--test-period", type=int, default=63,
                        help="Test window in days (default: 63)")
    parser.add_argument("--step", type=int, default=126,
                        help="Step size in days (default: 126)")
    parser.add_argument("--metric", type=str, default="sharpe_ratio",
                        choices=["sharpe_ratio", "sortino_ratio", "total_return", "profit_factor"],
                        help="Metric to optimize (default: sharpe_ratio)")
    parser.add_argument("--optimize-trading", action="store_true", default=True,
                        help="Also optimize trading params (default: True)")
    parser.add_argument("--no-optimize-trading", action="store_true",
                        help="Don't optimize trading params")
    parser.add_argument("--optimize-features", action="store_true",
                        help="Also optimize feature group toggles")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for params JSON (default: models/optimized_params.json)")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel trials (default: 1)")
    parser.add_argument("--use-regime", action="store_true",
                        help="Enable regime-adaptive trading in optimization windows")
    parser.add_argument("--regime-mode", type=str, default="adjust",
                        choices=["adjust", "filter"],
                        help="Regime mode for optimization")

    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",")]
    optimize_trading = args.optimize_trading and not args.no_optimize_trading

    print("\n" + "=" * 60)
    print("WALK-FORWARD HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Trials: {args.n_trials}")
    print(f"Windows: {args.n_windows} (train={args.train_period}d, test={args.test_period}d, step={args.step}d)")
    print(f"Metric: {args.metric}")
    print(f"Optimize trading params: {optimize_trading}")
    print(f"Optimize features: {args.optimize_features}")
    if args.use_regime:
        print(f"Regime detection: ENABLED (mode={args.regime_mode})")
    print("=" * 60)

    from src.ml.walk_forward_optimizer import WalkForwardOptimizer

    optimizer = WalkForwardOptimizer(
        symbols=symbols,
        train_period=args.train_period,
        test_period=args.test_period,
        step=args.step,
        n_windows=args.n_windows,
        n_trials=args.n_trials,
        optimize_metric=args.metric,
        optimize_trading_params=optimize_trading,
        optimize_features=args.optimize_features,
        use_regime=args.use_regime,
        regime_mode=args.regime_mode,
    )

    # Run optimization
    results = optimizer.optimize(n_jobs=args.n_jobs)

    # Save results
    output_path = args.output or str(MODELS_DIR / "optimized_params.json")
    optimizer.save_results(output_path)

    # Print usage instructions
    print(f"\nTo use optimized params in walk-forward backtest:")
    print(f"  python scripts/run_backtest.py --walk-forward --use-optimized-params {output_path}")


if __name__ == "__main__":
    main()
