#!/bin/bash
# 24/7 Trading Bot Daemon Wrapper

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

cd /Users/wenbiaoli/Desktop/trading_bot/stock-trading-bot

# Load .env file
set -a
source .env 2>/dev/null || true
set +a

# Run with venv python directly
/Users/wenbiaoli/Desktop/trading_bot/stock-trading-bot/venv311/bin/python \
    scripts/start_trading.py \
    --simulated \
    --interval 120 \
    --ensemble
