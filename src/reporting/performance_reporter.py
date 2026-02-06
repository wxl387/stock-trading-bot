"""
Performance Reporter - Automated trading bot performance reports.
Generates comprehensive reports with metrics, charts, and recommendations.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class PerformanceReporter:
    """Generate automated performance reports for the trading bot."""

    def __init__(self, data_dir: Path, initial_capital: float = 100000):
        """
        Initialize the performance reporter.

        Args:
            data_dir: Directory containing broker state and cache
            initial_capital: Initial capital for return calculations
        """
        self.data_dir = Path(data_dir)
        self.initial_capital = initial_capital
        self.state_file = self.data_dir / "simulated_broker_state.json"

    def load_broker_state(self) -> Optional[Dict]:
        """Load the current broker state."""
        if not self.state_file.exists():
            logger.warning(f"Broker state file not found: {self.state_file}")
            return None

        try:
            with open(self.state_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading broker state: {e}")
            return None

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices for symbols."""
        prices = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    prices[symbol] = hist['Close'].iloc[-1]
                else:
                    logger.warning(f"No price data for {symbol}")
                    prices[symbol] = 0
            except Exception as e:
                logger.warning(f"Error fetching price for {symbol}: {e}")
                prices[symbol] = 0
        return prices

    def calculate_portfolio_value(self, state: Dict, current_prices: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate total portfolio value.

        Returns:
            Tuple of (total_value, position_value)
        """
        cash = state.get('cash', 0)
        positions = state.get('positions', {})

        position_value = 0
        for symbol, pos in positions.items():
            # Support both 'shares'/'entry_price' and 'quantity'/'avg_cost' formats
            shares = pos.get('shares', pos.get('quantity', 0))
            price = current_prices.get(symbol, 0)
            position_value += shares * price

        total_value = cash + position_value
        return total_value, position_value

    def analyze_trades(self, trades: List[Dict]) -> Dict:
        """Analyze trade history and calculate statistics."""
        if not trades:
            return {
                'total_trades': 0,
                'closed_positions': 0,
                'realized_pnl': 0,
                'win_rate': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_trade_duration': 0
            }

        df = pd.DataFrame(trades)

        # Filter to closed positions (SELL trades)
        sells = df[df['action'] == 'SELL'].copy()

        if sells.empty:
            return {
                'total_trades': len(df),
                'closed_positions': 0,
                'realized_pnl': 0,
                'win_rate': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'gross_profit': 0,
                'gross_loss': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'avg_trade_duration': 0
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

        best_trade = sells['pnl'].max() if 'pnl' in sells.columns else 0
        worst_trade = sells['pnl'].min() if 'pnl' in sells.columns else 0

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
            'gross_loss': gross_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_trade_duration': 0  # TODO: Calculate from entry/exit times
        }

    def get_spy_performance(self, start_date: datetime) -> Optional[Dict]:
        """Get SPY (S&P 500) performance since start date."""
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
        except Exception as e:
            logger.warning(f"Error fetching SPY performance: {e}")
            return None

    def calculate_sharpe_ratio(self, trades: List[Dict], risk_free_rate: float = 0.05) -> float:
        """
        Calculate Sharpe ratio from trade history.

        Args:
            trades: List of trade dictionaries
            risk_free_rate: Annual risk-free rate (default: 5%)

        Returns:
            Sharpe ratio
        """
        if not trades:
            return 0.0

        df = pd.DataFrame(trades)
        sells = df[df['action'] == 'SELL'].copy()

        if sells.empty or 'pnl' not in sells.columns:
            return 0.0

        # Calculate percentage returns (normalize P&L by initial capital)
        returns = sells['pnl'].values / self.initial_capital if self.initial_capital > 0 else sells['pnl'].values

        if len(returns) < 2:
            return 0.0

        mean_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        # Annualized Sharpe: (mean_return - daily_rf) / std * sqrt(trades_per_year)
        daily_rf = risk_free_rate / 252
        trades_per_year = min(252, len(returns) * 252 / max(1, self._trading_days(trades)))
        sharpe = (mean_return - daily_rf) / std_return * np.sqrt(trades_per_year)

        return sharpe

    def _trading_days(self, trades: List[Dict]) -> int:
        """Estimate number of trading days spanned by trades."""
        try:
            timestamps = [pd.to_datetime(t['timestamp']) for t in trades if 'timestamp' in t]
            if len(timestamps) >= 2:
                span = (max(timestamps) - min(timestamps)).days
                return max(1, int(span * 252 / 365))
        except Exception:
            pass
        return max(1, len(trades))

    def calculate_max_drawdown(self, trades: List[Dict]) -> Tuple[float, float]:
        """
        Calculate maximum drawdown from trade history.

        Returns:
            Tuple of (max_drawdown_pct, max_drawdown_amount)
        """
        if not trades:
            return 0.0, 0.0

        df = pd.DataFrame(trades)

        # Create equity curve
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        equity = self.initial_capital
        equity_curve = [equity]

        for _, trade in df.iterrows():
            if trade['action'] == 'SELL' and 'pnl' in df.columns:
                equity += trade['pnl']
            equity_curve.append(equity)

        # Calculate drawdown
        equity_series = pd.Series(equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax.replace(0, np.nan) * 100

        max_dd_pct = abs(drawdown.min())
        max_dd_amount = abs((equity_series - cummax).min())

        return max_dd_pct, max_dd_amount

    def generate_text_report(self, report_type: str = "daily") -> str:
        """
        Generate a text-based performance report.

        Args:
            report_type: Type of report ("daily", "weekly", "monthly")

        Returns:
            Formatted text report
        """
        state = self.load_broker_state()
        if not state:
            return "❌ Unable to generate report: No broker state found."

        # Get data
        initial_capital = state.get('initial_capital', self.initial_capital)
        current_cash = state.get('cash', 0)
        positions = state.get('positions', {})
        trades = state.get('trade_history', [])

        # Get current prices
        symbols = list(positions.keys())
        current_prices = self.get_current_prices(symbols) if symbols else {}

        # Calculate metrics
        portfolio_value, position_value = self.calculate_portfolio_value(state, current_prices)
        total_return_pct = ((portfolio_value - initial_capital) / initial_capital * 100)
        total_return_amount = portfolio_value - initial_capital

        trade_stats = self.analyze_trades(trades)
        max_dd_pct, max_dd_amount = self.calculate_max_drawdown(trades)
        sharpe = self.calculate_sharpe_ratio(trades)

        # Get benchmark performance
        spy_perf = None
        if trades:
            first_trade = min(trades, key=lambda x: x['timestamp'])
            first_trade_date = pd.to_datetime(first_trade['timestamp']).date()
            spy_perf = self.get_spy_performance(first_trade_date)

        # Build report
        report = []
        report.append("=" * 70)
        report.append(f" {report_type.upper()} TRADING BOT PERFORMANCE REPORT".center(70))
        report.append("=" * 70)
        report.append(f"\n📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"📊 Report Type: {report_type.title()}")

        # Portfolio Summary
        report.append("\n" + "-" * 70)
        report.append("PORTFOLIO SUMMARY")
        report.append("-" * 70)
        report.append(f"💼 Initial Capital:        ${initial_capital:>15,.2f}")
        report.append(f"💰 Current Cash:           ${current_cash:>15,.2f}")
        report.append(f"📊 Position Value:         ${position_value:>15,.2f}")
        report.append(f"💵 Total Portfolio Value:  ${portfolio_value:>15,.2f}")
        report.append(f"📈 Total Return:           ${total_return_amount:>15,.2f} ({total_return_pct:>6.2f}%)")

        # Performance Metrics
        report.append("\n" + "-" * 70)
        report.append("PERFORMANCE METRICS")
        report.append("-" * 70)
        report.append(f"📊 Sharpe Ratio:           {sharpe:>15.2f}")
        report.append(f"📉 Max Drawdown:           {max_dd_pct:>14.2f}% (${max_dd_amount:,.2f})")
        report.append(f"🎯 Win Rate:               {trade_stats['win_rate']:>14.1f}%")
        report.append(f"💰 Profit Factor:          {trade_stats['profit_factor']:>15.2f}")
        report.append(f"💵 Realized P&L:           ${trade_stats['realized_pnl']:>15,.2f}")

        # Benchmark Comparison
        if spy_perf:
            report.append("\n" + "-" * 70)
            report.append("BENCHMARK COMPARISON (SPY)")
            report.append("-" * 70)
            report.append(f"📈 SPY Return:             {spy_perf['return_pct']:>14.2f}%")
            outperformance = total_return_pct - spy_perf['return_pct']
            symbol = "🟢" if outperformance > 0 else "🔴"
            report.append(f"{symbol} Outperformance:        {outperformance:>14.2f}%")

        # Trading Activity
        report.append("\n" + "-" * 70)
        report.append("TRADING ACTIVITY")
        report.append("-" * 70)
        report.append(f"📊 Total Trades:           {trade_stats['total_trades']:>15}")
        report.append(f"✅ Closed Positions:       {trade_stats['closed_positions']:>15}")
        report.append(f"🟢 Winning Trades:         {trade_stats['winning_trades']:>15}")
        report.append(f"🔴 Losing Trades:          {trade_stats['losing_trades']:>15}")
        report.append(f"💰 Average Win:            ${trade_stats['avg_win']:>15,.2f}")
        report.append(f"💸 Average Loss:           ${trade_stats['avg_loss']:>15,.2f}")
        report.append(f"🏆 Best Trade:             ${trade_stats['best_trade']:>15,.2f}")
        report.append(f"💔 Worst Trade:            ${trade_stats['worst_trade']:>15,.2f}")

        # Current Positions
        report.append("\n" + "-" * 70)
        report.append("CURRENT POSITIONS")
        report.append("-" * 70)

        if positions:
            report.append(f"{'Symbol':<10} {'Shares':>10} {'Entry':>12} {'Current':>12} {'Value':>14} {'P&L':>12}")
            report.append("-" * 70)

            total_unrealized = 0
            for symbol, pos in positions.items():
                # Support both 'shares'/'entry_price' and 'quantity'/'avg_cost' formats
                shares = pos.get('shares', pos.get('quantity', 0))
                entry_price = pos.get('entry_price', pos.get('avg_cost', 0))
                current_price = current_prices.get(symbol, 0)
                pos_value = shares * current_price
                unrealized_pnl = (current_price - entry_price) * shares
                total_unrealized += unrealized_pnl

                pnl_symbol = "🟢" if unrealized_pnl > 0 else "🔴"
                report.append(
                    f"{symbol:<10} {shares:>10} ${entry_price:>11.2f} ${current_price:>11.2f} "
                    f"${pos_value:>13,.2f} {pnl_symbol}${unrealized_pnl:>10,.2f}"
                )

            report.append("-" * 70)
            report.append(f"{'Total Unrealized P&L:':<45} ${total_unrealized:>15,.2f}")
        else:
            report.append("No open positions")

        # Risk Assessment
        report.append("\n" + "-" * 70)
        report.append("RISK ASSESSMENT")
        report.append("-" * 70)

        # Risk level based on metrics
        risk_level = "LOW"
        risk_factors = []

        if max_dd_pct > 15:
            risk_level = "HIGH"
            risk_factors.append(f"⚠️  High drawdown: {max_dd_pct:.1f}%")
        elif max_dd_pct > 10:
            risk_level = "MEDIUM"
            risk_factors.append(f"⚠️  Moderate drawdown: {max_dd_pct:.1f}%")

        if trade_stats['win_rate'] < 45:
            risk_level = "HIGH" if risk_level == "MEDIUM" else risk_level
            risk_factors.append(f"⚠️  Low win rate: {trade_stats['win_rate']:.1f}%")

        if sharpe < 0.5:
            risk_level = "HIGH" if risk_level == "MEDIUM" else risk_level
            risk_factors.append(f"⚠️  Low Sharpe ratio: {sharpe:.2f}")

        if len(positions) > 7:
            risk_factors.append(f"⚠️  High number of positions: {len(positions)}")

        report.append(f"Risk Level: {risk_level}")
        if risk_factors:
            report.append("\nRisk Factors:")
            for factor in risk_factors:
                report.append(f"  {factor}")
        else:
            report.append("✅ No significant risk factors identified")

        # Recommendations
        report.append("\n" + "-" * 70)
        report.append("RECOMMENDATIONS")
        report.append("-" * 70)

        recommendations = []

        if total_return_pct < 0:
            recommendations.append("📉 Portfolio in drawdown - consider reviewing strategy")

        if trade_stats['win_rate'] < 50 and trade_stats['closed_positions'] > 10:
            recommendations.append("🎯 Win rate below 50% - review trade selection criteria")

        if sharpe < 1.0 and trade_stats['closed_positions'] > 5:
            recommendations.append("📊 Sharpe ratio below 1.0 - focus on risk-adjusted returns")

        if len(positions) == 0 and current_cash > initial_capital * 0.9:
            recommendations.append("💼 No positions open - system may be too conservative")

        if max_dd_pct > 15:
            recommendations.append("⚠️  Drawdown exceeds 15% - consider reducing position sizes")

        if spy_perf and total_return_pct < spy_perf['return_pct'] * 0.8:
            recommendations.append("📈 Underperforming SPY - consider strategy review")

        if not recommendations:
            recommendations.append("✅ System performing well - continue monitoring")

        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")

        report.append("\n" + "=" * 70)
        report.append("")

        return "\n".join(report)

    def generate_html_report(self, report_type: str = "daily") -> str:
        """Generate an HTML performance report with charts."""
        # Get text report data
        text_report = self.generate_text_report(report_type)

        # Build HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_type.title()} Performance Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .section {{
                    background: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric {{
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid #eee;
                }}
                .metric:last-child {{
                    border-bottom: none;
                }}
                .metric-label {{
                    font-weight: bold;
                    color: #555;
                }}
                .metric-value {{
                    color: #333;
                }}
                .positive {{
                    color: #10b981;
                }}
                .negative {{
                    color: #ef4444;
                }}
                .table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .table th {{
                    background-color: #667eea;
                    color: white;
                    padding: 10px;
                    text-align: left;
                }}
                .table td {{
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                }}
                pre {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                    font-family: 'Courier New', monospace;
                    font-size: 12px;
                    line-height: 1.5;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 {report_type.title()} Trading Bot Performance Report</h1>
                <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <div class="section">
                <pre>{text_report}</pre>
            </div>
        </body>
        </html>
        """

        return html

    def save_report(self, report: str, filename: str, format: str = "txt"):
        """
        Save report to file.

        Args:
            report: Report content
            filename: Output filename (without extension)
            format: Report format ("txt", "html")
        """
        reports_dir = self.data_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        filepath = reports_dir / f"{filename}.{format}"

        try:
            with open(filepath, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            return None
