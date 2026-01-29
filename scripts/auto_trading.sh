#!/bin/bash
# Auto Trading Script - Runs at market open, stops at market close
# Called by launchd

cd /Users/wenbiaoli/Desktop/trading_bot/stock-trading-bot
source venv311/bin/activate

LOG_FILE="logs/trading_live.log"
PID_FILE="logs/trading_bot.pid"

start_bot() {
    echo "$(date): Starting trading bot..." >> "$LOG_FILE"
    python scripts/start_trading.py --simulated --interval 120 --ensemble >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "$(date): Bot started with PID $(cat $PID_FILE)" >> "$LOG_FILE"
}

stop_bot() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "$(date): Stopping trading bot (PID: $PID)..." >> "$LOG_FILE"
            kill $PID
            rm "$PID_FILE"
            echo "$(date): Bot stopped" >> "$LOG_FILE"
        fi
    fi
    # Also kill any orphaned processes
    pkill -f "start_trading.py" 2>/dev/null
}

case "$1" in
    start)
        stop_bot  # Stop any existing instance first
        start_bot
        ;;
    stop)
        stop_bot
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac
