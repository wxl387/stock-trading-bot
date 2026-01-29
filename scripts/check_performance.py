#!/usr/bin/env python3
"""
Quick performance check script for the trading bot.
Shows current status, P&L, and key metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config.settings import DATA_DIR

def load_broker_state():
    """Load the simulated broker state."""
    state_file = DATA_DIR / "simulated_broker_state.json"
    if not state_file.exists():
        return None

    with open(state_file) as f:
        return json.load(f)

def get_current_prices(symbols):
    """Fetch current prices for symbols."""
    prices = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                prices[symbol] = hist['Close'].iloc[-1]
        except:
            prices[symbol] = 0
    return prices

def calculate_portfolio_value(state, current_prices):
    """Calculate total portfolio value."""
    cash = state.get('cash', 0)
    positions = state.get('positions', {})

    position_value = 0
    for symbol, pos in positions.items():
        # Support both 'shares'/'entry_price' and 'quantity'/'avg_cost' formats
        shares = pos.get('shares', pos.get('quantity', 0))
        price = current_prices.get(symbol, 0)
        position_value += shares * price

    return cash + position_value

def analyze_trades(trades):
    """Analyze trade history."""
    if not trades:
        return {}

    df = pd.DataFrame(trades)

    # Filter to closed positions (SELL trades)
    sells = df[df['action'] == 'SELL'].copy()

    if sells.empty:
        return {
            'total_trades': len(df),
            'closed_positions': 0,
            'realized_pnl': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0
        }

    # Calculate metrics
    realized_pnl = sells['pnl'].sum() if 'pnl' in sells.columns else 0
    winning_trades = len(sells[sells['pnl'] > 0])
    losing_trades = len(sells[sells['pnl'] < 0])
    total_closed = winning_trades + losing_trades

    win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0

    wins = sells[sells['pnl'] > 0]['pnl']
    losses = sells[sells['pnl'] < 0]['pnl']

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0

    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

    return {
        'total_trades': len(df),
        'closed_positions': total_closed,
        'realized_pnl': realized_pnl,
        'win_rate': win_rate,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }

def get_spy_performance(start_date):
    """Get SPY performance since start date."""
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(start=start_date)
        if hist.empty:
            return None

        initial_price = hist['Close'].iloc[0]
        final_price = hist['Close'].iloc[-1]
        return_pct = (final_price - initial_price) / initial_price * 100

        return {
            'initial': initial_price,
            'current': final_price,
            'return_pct': return_pct
        }
    except:
        return None

def main():
    print("=" * 70)
    print(" " * 20 + "TRADING BOT PERFORMANCE")
    print("=" * 70)

    # Load broker state
    state = load_broker_state()
    if not state:
        print("\n❌ No broker state found!")
        print("   Run the trading bot first: python scripts/start_trading.py --simulated")
        return

    # Get initial capital from state or use default
    initial_capital = state.get('initial_capital', 100000)
    current_cash = state.get('cash', 0)
    positions = state.get('positions', {})
    trades = state.get('trade_history', [])

    # Get current prices
    symbols = list(positions.keys())
    current_prices = get_current_prices(symbols) if symbols else {}

    # Calculate portfolio value
    portfolio_value = calculate_portfolio_value(state, current_prices)

    # Calculate returns
    total_return_pct = ((portfolio_value - initial_capital) / initial_capital * 100)

    # Analyze trades
    trade_stats = analyze_trades(trades)

    # Display summary
    print(f"\n📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💼 Initial Capital: ${initial_capital:,.2f}")
    print(f"💰 Current Cash: ${current_cash:,.2f}")
    print(f"📊 Position Value: ${portfolio_value - current_cash:,.2f}")
    print(f"💵 Total Portfolio Value: ${portfolio_value:,.2f}")

    print("\n" + "-" * 70)
    print("RETURNS")
    print("-" * 70)
    print(f"{'Total Return:':<30} ${portfolio_value - initial_capital:>12,.2f}")
    print(f"{'Return %:':<30} {total_return_pct:>12.2f}%")

    # Compare to SPY
    if trades:
        first_trade_date = pd.to_datetime(trades[0]['timestamp']).date()
        spy_perf = get_spy_performance(first_trade_date)

        if spy_perf:
            print(f"\n{'SPY (S&P 500) Return:':<30} {spy_perf['return_pct']:>12.2f}%")
            outperformance = total_return_pct - spy_perf['return_pct']
            print(f"{'Outperformance vs SPY:':<30} {outperformance:>12.2f}%")

    print("\n" + "-" * 70)
    print("POSITIONS")
    print("-" * 70)

    if positions:
        print(f"{'Symbol':<10} {'Shares':>10} {'Entry':>12} {'Current':>12} {'Value':>12} {'P&L':>12}")
        print("-" * 70)

        total_unrealized = 0
        for symbol, pos in positions.items():
            # Support both 'shares'/'entry_price' and 'quantity'/'avg_cost' formats
            shares = pos.get('shares', pos.get('quantity', 0))
            entry_price = pos.get('entry_price', pos.get('avg_cost', 0))
            current_price = current_prices.get(symbol, 0)
            position_value = shares * current_price
            unrealized_pnl = (current_price - entry_price) * shares
            total_unrealized += unrealized_pnl

            print(f"{symbol:<10} {shares:>10} ${entry_price:>11.2f} ${current_price:>11.2f} "
                  f"${position_value:>11,.2f} ${unrealized_pnl:>11,.2f}")

        print("-" * 70)
        print(f"{'Total Unrealized P&L:':<30} ${total_unrealized:>12,.2f}")
    else:
        print("No open positions")

    print("\n" + "-" * 70)
    print("TRADING STATISTICS")
    print("-" * 70)
    print(f"{'Total Trades:':<30} {trade_stats.get('total_trades', 0):>12}")
    print(f"{'Closed Positions:':<30} {trade_stats.get('closed_positions', 0):>12}")
    print(f"{'Realized P&L:':<30} ${trade_stats.get('realized_pnl', 0):>12,.2f}")

    if trade_stats.get('closed_positions', 0) > 0:
        print(f"\n{'Win Rate:':<30} {trade_stats.get('win_rate', 0):>12.1f}%")
        print(f"{'Winning Trades:':<30} {trade_stats.get('winning_trades', 0):>12}")
        print(f"{'Losing Trades:':<30} {trade_stats.get('losing_trades', 0):>12}")
        print(f"{'Average Win:':<30} ${trade_stats.get('avg_win', 0):>12,.2f}")
        print(f"{'Average Loss:':<30} ${trade_stats.get('avg_loss', 0):>12,.2f}")
        print(f"{'Profit Factor:':<30} {trade_stats.get('profit_factor', 0):>12.2f}")
        print(f"{'Gross Profit:':<30} ${trade_stats.get('gross_profit', 0):>12,.2f}")
        print(f"{'Gross Loss:':<30} ${trade_stats.get('gross_loss', 0):>12,.2f}")

    print("\n" + "=" * 70)

    # Recommendations
    print("\n💡 QUICK ACTIONS:")
    print("   • View dashboard: streamlit run src/dashboard/app.py")
    print("   • Check logs: tail -f logs/trading.log")
    print("   • Run backtest: python scripts/run_backtest.py --walk-forward --ensemble")
    print("   • View trades: cat data/simulated_broker_state.json | python -m json.tool")
    print("\n")

if __name__ == "__main__":
    main()
