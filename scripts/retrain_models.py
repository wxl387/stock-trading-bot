#!/usr/bin/env python3
"""
Script to retrain ML trading models with fresh data.
Supports XGBoost, LSTM, and CNN models.

Usage:
    python scripts/retrain_models.py --models all --deploy
    python scripts/retrain_models.py --models xgboost --tune
    python scripts/retrain_models.py --models lstm,cnn --epochs 30
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

from config.settings import setup_logging, Settings
from src.ml.retraining import RetrainingPipeline

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(
        description="Retrain ML trading models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/retrain_models.py --models all --deploy
  python scripts/retrain_models.py --models xgboost --tune
  python scripts/retrain_models.py --models lstm --epochs 30 --no-cache
        """
    )
    parser.add_argument(
        "--models", type=str, default="all",
        help="Models to retrain: all, xgboost, lstm, cnn (comma-separated)"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Tune all models with Optuna"
    )
    parser.add_argument(
        "--tune-xgboost", action="store_true",
        help="Tune XGBoost hyperparameters with Optuna"
    )
    parser.add_argument(
        "--tune-lstm", action="store_true",
        help="Tune LSTM hyperparameters with Optuna"
    )
    parser.add_argument(
        "--tune-cnn", action="store_true",
        help="Tune CNN hyperparameters with Optuna"
    )
    parser.add_argument(
        "--tune-transformer", action="store_true",
        help="Tune Transformer hyperparameters with Optuna"
    )
    parser.add_argument(
        "--tune-trials", type=int, default=20,
        help="Number of Optuna trials for tuning (default: 20)"
    )
    parser.add_argument(
        "--deploy", action="store_true",
        help="Deploy if better than production model"
    )
    parser.add_argument(
        "--force-deploy", action="store_true",
        help="Deploy regardless of performance comparison"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force fresh data download"
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols to train on"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Training epochs for LSTM/CNN (default: 50 for LSTM, 30 for CNN)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.01,
        help="Minimum accuracy improvement threshold for deployment (default: 0.01)"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show retraining status and exit"
    )
    parser.add_argument(
        "--list-versions", action="store_true",
        help="List all model versions and exit"
    )

    args = parser.parse_args()

    # Load config
    config = Settings.load_trading_config()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = config.get("trading", {}).get("symbols", [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"
        ])

    # Initialize pipeline
    pipeline = RetrainingPipeline(
        symbols=symbols,
        prediction_horizon=config.get("ml_model", {}).get("prediction_horizon", 5),
        train_period_days=config.get("ml_model", {}).get("train_window_days", 252),
        sequence_length=config.get("ml_model", {}).get("sequence_length", 20),
        min_improvement_threshold=args.threshold
    )

    # Handle status/list commands
    if args.status:
        status = pipeline.get_retraining_status()
        print("\n" + "=" * 60)
        print("RETRAINING STATUS")
        print("=" * 60)
        print(f"Total versions: {status['version_count']}")
        print(f"Recent versions: {', '.join(status['recent_versions'])}")
        print("\nProduction models:")
        for model_type, info in status["production_models"].items():
            print(f"  {model_type}: {info['version']}")
            print(f"    Deployed: {info['deployed_at']}")
            print(f"    Accuracy: {info['metrics'].get('accuracy', 0):.4f}")
        print("=" * 60)
        return

    if args.list_versions:
        versions = pipeline.list_versions()
        print("\n" + "=" * 60)
        print("MODEL VERSIONS")
        print("=" * 60)
        for v in versions:
            print(f"  {v['version']}")
            print(f"    Type: {v['model_type']}, Accuracy: {v['accuracy']:.4f}")
            print(f"    Created: {v['created_at']}")
        print("=" * 60)
        return

    # Parse models to train
    if args.models.lower() == "all":
        models_to_train = ["xgboost", "lstm", "cnn"]
    else:
        models_to_train = [m.strip().lower() for m in args.models.split(",")]

    # Determine tuning flags
    tune_xgboost = args.tune or args.tune_xgboost
    tune_lstm = args.tune or args.tune_lstm
    tune_cnn = args.tune or args.tune_cnn
    tune_transformer = args.tune or args.tune_transformer

    # Print configuration
    print("\n" + "=" * 60)
    print("MODEL RETRAINING PIPELINE")
    print("=" * 60)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Models: {', '.join(models_to_train)}")
    print(f"Tune XGBoost: {tune_xgboost}")
    print(f"Tune LSTM: {tune_lstm}")
    print(f"Tune CNN: {tune_cnn}")
    print(f"Tune Transformer: {tune_transformer}")
    if any([tune_xgboost, tune_lstm, tune_cnn, tune_transformer]):
        print(f"Tune trials: {args.tune_trials}")
    print(f"Auto-deploy: {args.deploy}")
    print(f"Force deploy: {args.force_deploy}")
    print(f"Use cache: {not args.no_cache}")
    print(f"Improvement threshold: {args.threshold:.2%}")
    print("=" * 60 + "\n")

    # Run retraining
    results = pipeline.retrain_all(
        use_cache=not args.no_cache,
        tune_xgboost=tune_xgboost,
        tune_lstm=tune_lstm,
        tune_cnn=tune_cnn,
        tune_transformer=tune_transformer,
        n_trials=args.tune_trials,
        models_to_train=models_to_train
    )

    # Print results
    print("\n" + "=" * 60)
    print("RETRAINING RESULTS")
    print("=" * 60)

    for model_type, result in results.items():
        if "error" in result:
            print(f"\n{model_type.upper()}: FAILED")
            print(f"  Error: {result['error']}")
        else:
            metrics = result["metrics"]
            print(f"\n{model_type.upper()}:")
            print(f"  Version: {result['version']}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")

            # Deploy if requested
            if (args.deploy or args.force_deploy) and "error" not in result:
                prod_names = {
                    "xgboost": "trading_model",
                    "lstm": "lstm_trading_model",
                    "cnn": "cnn_trading_model",
                    "transformer": "transformer_trading_model"
                }

                if args.force_deploy:
                    # Force deploy without comparison
                    pipeline._deploy_model(
                        model_type=model_type,
                        version=result["version"],
                        production_name=prod_names[model_type]
                    )
                    print(f"  Deployed: Yes (forced)")
                else:
                    deployed = pipeline.compare_and_deploy(
                        model_type=model_type,
                        new_version=result["version"],
                        production_name=prod_names[model_type]
                    )
                    print(f"  Deployed: {'Yes' if deployed else 'No (not better)'}")

    print("\n" + "=" * 60)

    # Show final status
    status = pipeline.get_retraining_status()
    print("\nCURRENT PRODUCTION MODELS:")
    for model_type, info in status["production_models"].items():
        print(f"  {model_type}: {info['version']} (accuracy: {info['metrics'].get('accuracy', 0):.4f})")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
