#!/usr/bin/env python3
"""
Script to start the trading bot.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

# Import TF BEFORE any pandas-importing module to avoid model.predict() deadlock
# (pandas 2.3+ / TF 2.20+ import order bug on macOS)
from src.ml.device_config import configure_tensorflow_device
configure_tensorflow_device()

from config.settings import setup_logging, Settings
from src.core.trading_engine import TradingEngine, run_bot

logger = setup_logging()


def main():
    """Start the trading bot."""
    parser = argparse.ArgumentParser(description="Start the auto trading bot")
    parser.add_argument("--live", action="store_true", help="Run in live trading mode (USE WITH CAUTION)")
    parser.add_argument("--simulated", action="store_true", help="Run in simulated mode (no broker credentials needed)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital for simulated mode")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--force", action="store_true", help="Ignore market hours (for testing)")
    parser.add_argument("--ensemble", action="store_true", help="Use ensemble model (XGBoost + LSTM + CNN)")

    args = parser.parse_args()

    # Load config
    config = Settings.load_trading_config()

    # Get symbols from config
    symbols = config.get("trading", {}).get("symbols", [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"
    ])

    paper_trading = not args.live

    # Warning for live trading
    if args.live:
        logger.warning("=" * 60)
        logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!")
        logger.warning("=" * 60)
        confirm = input("Type 'CONFIRM' to proceed with live trading: ")
        if confirm != "CONFIRM":
            logger.info("Live trading cancelled")
            return

    # Determine mode
    if args.simulated:
        mode_str = f"SIMULATED (${args.capital:,.2f})"
    elif paper_trading:
        mode_str = "PAPER (WeBull)"
    else:
        mode_str = "LIVE"

    logger.info("=" * 60)
    logger.info(f"Starting Trading Bot")
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Model: {'ENSEMBLE (XGBoost+LSTM+CNN)' if args.ensemble else 'XGBoost'}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Interval: {args.interval} seconds")
    logger.info("=" * 60)

    # Run the bot
    run_bot(
        symbols=symbols,
        paper_trading=paper_trading,
        simulated=args.simulated,
        initial_capital=args.capital,
        interval_seconds=args.interval,
        ignore_market_hours=args.force,
        use_ensemble=args.ensemble
    )


if __name__ == "__main__":
    main()
