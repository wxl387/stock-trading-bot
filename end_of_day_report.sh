#!/bin/bash
# End of Day Report Generator
# Run this at market close (4:00 PM ET)

cd /Users/wenbiaoli/Desktop/trading_bot/stock-trading-bot

echo "======================================================================="
echo "                    END OF DAY TRADING SUMMARY"
echo "======================================================================="
echo ""
echo "Market Close Time: $(date)"
echo ""

# Activate virtual environment
source venv311/bin/activate

echo "Generating daily performance report..."
echo ""

# Generate the daily report
python scripts/scheduled_reporter.py --test daily

echo ""
echo "======================================================================="
echo "                    QUICK STATISTICS"
echo "======================================================================="
echo ""

# Count total signals generated
SIGNALS=$(grep "Signal for" logs/trading_bot.log | grep "2026-01-28" | wc -l | tr -d ' ')
echo "📊 Total Signals Generated Today: $SIGNALS"

# Count trades executed
TRADES=$(grep "EXECUTED" logs/trading_bot.log | grep "2026-01-28" | wc -l | tr -d ' ')
echo "💼 Trades Executed Today: $TRADES"

# Count BUY vs SELL signals
BUY_SIGNALS=$(grep "Signal for" logs/trading_bot.log | grep "2026-01-28" | grep "BUY" | wc -l | tr -d ' ')
SELL_SIGNALS=$(grep "Signal for" logs/trading_bot.log | grep "2026-01-28" | grep "SELL" | wc -l | tr -d ' ')
echo "🟢 BUY Signals: $BUY_SIGNALS"
echo "🔴 SELL Signals: $SELL_SIGNALS"

echo ""
echo "======================================================================="
echo "                    REPORTS GENERATED"
echo "======================================================================="
echo ""

# List today's reports
LATEST_TXT=$(ls -t data/reports/daily_report_*.txt 2>/dev/null | head -1)
LATEST_HTML=$(ls -t data/reports/daily_report_*.html 2>/dev/null | head -1)

if [ -n "$LATEST_TXT" ]; then
    echo "📄 Text Report: $LATEST_TXT"
fi

if [ -n "$LATEST_HTML" ]; then
    echo "🌐 HTML Report: $LATEST_HTML"
    echo ""
    echo "View HTML report: open $LATEST_HTML"
fi

echo ""
echo "======================================================================="
echo "                    NEXT STEPS"
echo "======================================================================="
echo ""
echo "1. Review the reports above"
echo "2. Check the HTML report: open data/reports/$(basename $LATEST_HTML)"
echo "3. View full performance: python scripts/check_performance.py"
echo "4. Stop the bot: pkill -f start_trading.py"
echo ""
echo "Decide if you want to:"
echo "  • Continue tomorrow (same settings)"
echo "  • Adjust parameters (interval, confidence thresholds)"
echo "  • Move to paper trading (Phase 22)"
echo ""
echo "======================================================================="
