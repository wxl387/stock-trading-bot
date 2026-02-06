"""
Streamlit dashboard for trading bot monitoring.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dashboard.data_provider import get_data_provider
from config.settings import settings

# Page config
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide"
)


def main():
    """Main dashboard application."""

    # Get data provider
    data_provider = get_data_provider()

    # Sidebar
    st.sidebar.title("📈 Trading Bot")

    # Mode indicator
    mode = "SIMULATED"
    mode_color = "blue"
    st.sidebar.markdown(f"**Mode:** :{mode_color}[{mode}]")

    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Portfolio Overview", "Positions", "Trade History", "Analytics", "Portfolio Optimization", "Backtest", "Settings"]
    )

    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        data_provider.refresh()
        st.rerun()

    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")

    # Last updated
    metrics = data_provider.get_portfolio_metrics()
    st.sidebar.markdown(f"*Last updated: {metrics.get('last_updated', 'N/A')[:19]}*")

    # Main content
    if page == "Portfolio Overview":
        render_overview(data_provider)
    elif page == "Positions":
        render_positions(data_provider)
    elif page == "Trade History":
        render_trade_history(data_provider)
    elif page == "Analytics":
        render_analytics(data_provider)
    elif page == "Portfolio Optimization":
        render_portfolio_optimization(data_provider)
    elif page == "Backtest":
        render_backtest()
    elif page == "Settings":
        render_settings()

    # Auto-refresh
    if auto_refresh:
        import time
        time.sleep(30)
        data_provider.refresh()
        st.rerun()


def render_overview(data_provider):
    """Render portfolio overview page."""
    st.title("Portfolio Overview")

    # Get real metrics
    metrics = data_provider.get_portfolio_metrics()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Portfolio Value",
            value=f"${metrics['portfolio_value']:,.2f}",
            delta=f"${metrics['total_pnl']:,.2f}"
        )

    with col2:
        delta_color = "normal" if metrics['total_pnl'] >= 0 else "inverse"
        st.metric(
            label="Total P&L",
            value=f"${metrics['total_pnl']:,.2f}",
            delta=f"{metrics['total_pnl_pct']:.2f}%"
        )

    with col3:
        st.metric(
            label="Cash Available",
            value=f"${metrics['cash']:,.2f}",
            delta=None
        )

    with col4:
        st.metric(
            label="Positions",
            value=f"{metrics['num_positions']}",
            delta=f"{metrics['total_trades']} trades"
        )

    st.divider()

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portfolio Value Over Time")
        history_df = data_provider.get_portfolio_history()
        if not history_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history_df["Date"],
                y=history_df["Portfolio Value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color="#1f77b4", width=2)
            ))
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Value ($)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical data available yet.")

    with col2:
        st.subheader("Position Allocation")
        allocation_df = data_provider.get_allocation_data()
        if not allocation_df.empty:
            fig = px.pie(
                allocation_df,
                values="Market Value",
                names="Symbol",
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No positions to display.")

    st.divider()

    # Recent trades table
    st.subheader("Recent Trades")
    trades_df = data_provider.get_trade_history(limit=10)
    if not trades_df.empty:
        st.dataframe(
            trades_df,
            use_container_width=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Total": st.column_config.NumberColumn(format="$%.2f"),
            }
        )
    else:
        st.info("No trades executed yet.")

    # Save snapshot button (for manual history tracking)
    if st.button("📸 Save Portfolio Snapshot"):
        data_provider.save_portfolio_snapshot()
        st.success("Portfolio snapshot saved!")


def render_positions(data_provider):
    """Render positions page."""
    st.title("Current Positions")

    positions_df = data_provider.get_positions()

    if positions_df.empty:
        st.info("No positions currently held.")
        return

    # Summary metrics
    total_value = positions_df["Market Value"].sum()
    total_pnl = positions_df["P&L ($)"].sum()
    avg_pnl_pct = positions_df["P&L (%)"].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Position Value", f"${total_value:,.2f}")
    with col2:
        st.metric("Unrealized P&L", f"${total_pnl:,.2f}")
    with col3:
        st.metric("Avg P&L %", f"{avg_pnl_pct:.2f}%")

    st.divider()

    # Positions table
    st.dataframe(
        positions_df,
        use_container_width=True,
        column_config={
            "Avg Cost": st.column_config.NumberColumn(format="$%.2f"),
            "Current Price": st.column_config.NumberColumn(format="$%.2f"),
            "Market Value": st.column_config.NumberColumn(format="$%.2f"),
            "P&L ($)": st.column_config.NumberColumn(format="$%.2f"),
            "P&L (%)": st.column_config.NumberColumn(format="%.2f%%"),
        }
    )

    st.divider()

    # Position breakdown chart
    st.subheader("Position Breakdown by Value")
    fig = px.bar(
        positions_df,
        x="Symbol",
        y="Market Value",
        color="P&L (%)",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # P&L breakdown chart
    st.subheader("P&L by Position")
    colors = ["green" if x >= 0 else "red" for x in positions_df["P&L ($)"]]
    fig = go.Figure(data=[
        go.Bar(
            x=positions_df["Symbol"],
            y=positions_df["P&L ($)"],
            marker_color=colors
        )
    ])
    fig.update_layout(
        height=400,
        xaxis_title="Symbol",
        yaxis_title="P&L ($)"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trade_history(data_provider):
    """Render trade history page."""
    st.title("Trade History")

    trades_df = data_provider.get_trade_history(limit=100)

    if trades_df.empty:
        st.info("No trades executed yet.")
        return

    # Symbol filter
    symbols = ["All"] + sorted(trades_df["Symbol"].unique().tolist())
    selected_symbol = st.selectbox("Filter by Symbol", symbols)

    if selected_symbol != "All":
        trades_df = trades_df[trades_df["Symbol"] == selected_symbol]

    # Display trades
    st.dataframe(
        trades_df,
        use_container_width=True,
        column_config={
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Total": st.column_config.NumberColumn(format="$%.2f"),
        }
    )

    # Summary metrics
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", len(trades_df))
    with col2:
        buy_count = len(trades_df[trades_df["Action"] == "BUY"])
        st.metric("Buy Orders", buy_count)
    with col3:
        sell_count = len(trades_df[trades_df["Action"] == "SELL"])
        st.metric("Sell Orders", sell_count)
    with col4:
        total_volume = trades_df["Total"].sum()
        st.metric("Total Volume", f"${total_volume:,.2f}")

    # Trade distribution chart
    st.divider()
    st.subheader("Trades by Symbol")
    trade_counts = trades_df.groupby("Symbol").size().reset_index(name="Count")
    fig = px.bar(trade_counts, x="Symbol", y="Count", color="Count", color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)


def render_analytics(data_provider):
    """Render portfolio analytics page."""
    st.title("Portfolio Analytics")

    # Import analytics modules
    try:
        from src.analytics import MetricsCalculator, BenchmarkComparison, PerformanceAttribution
        from src.analytics.data_aggregator import get_data_aggregator
        import numpy as np
    except ImportError as e:
        st.error(f"Analytics modules not available: {e}")
        return

    # Get data aggregator
    aggregator = get_data_aggregator()

    # Date range selector
    st.subheader("Analysis Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=90)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    st.divider()

    # Get analytics data
    returns = aggregator.get_daily_returns(start_dt, end_dt)
    equity_curve = aggregator.get_equity_curve(start_dt, end_dt)
    trades_df = aggregator.get_trade_history(start_dt, end_dt)
    positions = aggregator.get_positions()

    if len(returns) < 2:
        st.warning("Not enough historical data for analytics. Save more portfolio snapshots to enable analysis.")
        if st.button("Save Portfolio Snapshot Now"):
            data_provider.save_portfolio_snapshot()
            st.success("Snapshot saved! Continue saving daily snapshots for meaningful analytics.")
        return

    # Calculate metrics
    metrics_calc = MetricsCalculator(returns)
    all_metrics = metrics_calc.get_all_metrics()

    # Metrics cards row 1
    st.subheader("Risk-Adjusted Returns")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sharpe = all_metrics.get('sharpe_ratio', 0)
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}",
            help="Risk-adjusted return (excess return / volatility)"
        )

    with col2:
        sortino = all_metrics.get('sortino_ratio', 0)
        st.metric(
            label="Sortino Ratio",
            value=f"{sortino:.2f}",
            help="Downside risk-adjusted return"
        )

    with col3:
        calmar = all_metrics.get('calmar_ratio', 0)
        st.metric(
            label="Calmar Ratio",
            value=f"{calmar:.2f}",
            help="Annualized return / Max drawdown"
        )

    with col4:
        max_dd = all_metrics.get('max_drawdown', 0)
        st.metric(
            label="Max Drawdown",
            value=f"{max_dd:.2%}",
            help="Largest peak-to-trough decline"
        )

    # Metrics cards row 2
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ann_return = all_metrics.get('annualized_return', 0)
        st.metric(
            label="Annualized Return",
            value=f"{ann_return:.2%}"
        )

    with col2:
        volatility = all_metrics.get('volatility', 0) or all_metrics.get('annualized_volatility', 0)
        st.metric(
            label="Volatility",
            value=f"{volatility:.2%}"
        )

    with col3:
        total_return = all_metrics.get('total_return', 0)
        st.metric(
            label="Total Return",
            value=f"{total_return:.2%}"
        )

    with col4:
        trading_days = all_metrics.get('trading_days', 0)
        st.metric(
            label="Trading Days",
            value=f"{trading_days}"
        )

    st.divider()

    # Benchmark comparison section
    st.subheader("Benchmark Comparison (vs SPY)")

    if len(equity_curve) >= 10:
        benchmark = BenchmarkComparison(equity_curve, benchmark_symbol="SPY")
        benchmark_metrics = benchmark.get_all_metrics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            alpha = benchmark_metrics.get('alpha', 0)
            st.metric(
                label="Alpha",
                value=f"{alpha:.2%}",
                help="Excess return vs market (annualized)"
            )

        with col2:
            beta = benchmark_metrics.get('beta', 1)
            st.metric(
                label="Beta",
                value=f"{beta:.2f}",
                help="Sensitivity to market movements"
            )

        with col3:
            info_ratio = benchmark_metrics.get('information_ratio', 0)
            st.metric(
                label="Information Ratio",
                value=f"{info_ratio:.2f}",
                help="Active return / Tracking error"
            )

        with col4:
            rel_perf = benchmark_metrics.get('relative_performance', 0)
            st.metric(
                label="vs SPY",
                value=f"{rel_perf:+.2%}",
                help="Total outperformance vs benchmark"
            )

        # Cumulative performance chart
        cumulative = benchmark.cumulative_comparison()
        if not cumulative.empty:
            st.subheader("Cumulative Performance")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative['portfolio'] * 100,
                mode='lines',
                name='Portfolio',
                line=dict(color='#1f77b4', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative['benchmark'] * 100,
                mode='lines',
                name='SPY',
                line=dict(color='#ff7f0e', width=2)
            ))
            fig.update_layout(
                height=400,
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 10 days of data for benchmark comparison.")

    st.divider()

    # Rolling Sharpe chart
    st.subheader("Rolling Sharpe Ratio (63-day)")
    rolling_sharpe = metrics_calc.rolling_sharpe(window=63)
    if not rolling_sharpe.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='Rolling Sharpe',
            line=dict(color='#2ca02c', width=2)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=1, line_dash="dot", line_color="green", annotation_text="Good (1.0)")
        fig.add_hline(y=2, line_dash="dot", line_color="blue", annotation_text="Excellent (2.0)")
        fig.update_layout(
            height=350,
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for rolling Sharpe calculation (need 63+ days).")

    st.divider()

    # Monthly returns heatmap
    st.subheader("Monthly Returns")
    monthly = metrics_calc.monthly_returns()
    if not monthly.empty:
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=monthly.values * 100,
            x=monthly.columns,
            y=monthly.index,
            colorscale='RdYlGn',
            zmid=0,
            text=[[f"{v*100:.1f}%" if not np.isnan(v) else "" for v in row] for row in monthly.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>"
        ))
        fig.update_layout(
            height=300,
            xaxis_title="Month",
            yaxis_title="Year"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for monthly returns heatmap.")

    st.divider()

    # Performance Attribution
    st.subheader("Performance Attribution")

    if not trades_df.empty or positions:
        # Prepare trades DataFrame for attribution
        if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
            # Standardize column names
            trades_for_attr = trades_df.copy()
            if 'Symbol' in trades_for_attr.columns:
                trades_for_attr = trades_for_attr.rename(columns={'Symbol': 'symbol'})
            if 'Action' in trades_for_attr.columns:
                trades_for_attr = trades_for_attr.rename(columns={'Action': 'side'})
            if 'Shares' in trades_for_attr.columns:
                trades_for_attr = trades_for_attr.rename(columns={'Shares': 'quantity'})
            if 'Price' in trades_for_attr.columns:
                trades_for_attr = trades_for_attr.rename(columns={'Price': 'price'})
            if 'Time' in trades_for_attr.columns:
                trades_for_attr = trades_for_attr.rename(columns={'Time': 'timestamp'})
        else:
            trades_for_attr = pd.DataFrame()

        # Prepare positions dict
        positions_dict = {}
        if isinstance(positions, pd.DataFrame) and not positions.empty:
            for _, row in positions.iterrows():
                symbol = row.get('Symbol', '')
                if symbol:
                    positions_dict[symbol] = {
                        'unrealized_pnl': row.get('P&L ($)', 0),
                        'realized_pnl': 0
                    }
        elif isinstance(positions, dict):
            positions_dict = positions

        metrics = data_provider.get_portfolio_metrics()
        total_pnl = metrics.get('total_pnl', 0)

        attribution = PerformanceAttribution(trades_for_attr, positions_dict, total_pnl)

        # Winner/Loser analysis
        wl_analysis = attribution.winner_loser_analysis()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Win Rate", f"{wl_analysis.get('win_rate', 0):.1%}")
        with col2:
            st.metric("Avg Win", f"${wl_analysis.get('avg_win', 0):,.2f}")
        with col3:
            st.metric("Avg Loss", f"${wl_analysis.get('avg_loss', 0):,.2f}")
        with col4:
            st.metric("Profit Factor", f"{wl_analysis.get('profit_factor', 0):.2f}")

        # Position contribution chart
        contrib = attribution.position_contribution()
        if not contrib.empty:
            st.subheader("P&L by Position")
            colors = ['green' if x >= 0 else 'red' for x in contrib['pnl']]
            fig = go.Figure(data=[
                go.Bar(
                    x=contrib['symbol'],
                    y=contrib['pnl'],
                    marker_color=colors
                )
            ])
            fig.update_layout(
                height=350,
                xaxis_title="Symbol",
                yaxis_title="P&L ($)"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Realized vs Unrealized
        pnl_breakdown = attribution.realized_vs_unrealized()
        st.subheader("P&L Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Realized P&L", f"${pnl_breakdown.get('realized_pnl', 0):,.2f}")
        with col2:
            st.metric("Unrealized P&L", f"${pnl_breakdown.get('unrealized_pnl', 0):,.2f}")
        with col3:
            st.metric("Total P&L", f"${pnl_breakdown.get('total_pnl', 0):,.2f}")
    else:
        st.info("No trade or position data for attribution analysis.")

    st.divider()

    # PDF Report download
    st.subheader("Export Report")

    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF report..."):
            try:
                from src.analytics import ReportGenerator

                # Prepare data for report - recalculate to ensure availability
                report_metrics = all_metrics

                # Benchmark metrics
                report_benchmark = None
                if len(equity_curve) >= 10:
                    try:
                        bench = BenchmarkComparison(equity_curve, benchmark_symbol="SPY")
                        report_benchmark = bench.get_all_metrics()
                    except Exception:
                        pass

                # Monthly returns
                report_monthly = metrics_calc.monthly_returns()
                if report_monthly.empty:
                    report_monthly = None

                # Trade analysis and attribution
                report_trade = None
                report_attr = None
                if not trades_df.empty or positions:
                    # Prepare trades for attribution
                    trades_for_report = trades_df.copy() if isinstance(trades_df, pd.DataFrame) and not trades_df.empty else pd.DataFrame()
                    if not trades_for_report.empty:
                        if 'Symbol' in trades_for_report.columns:
                            trades_for_report = trades_for_report.rename(columns={'Symbol': 'symbol'})
                        if 'Action' in trades_for_report.columns:
                            trades_for_report = trades_for_report.rename(columns={'Action': 'side'})
                        if 'Shares' in trades_for_report.columns:
                            trades_for_report = trades_for_report.rename(columns={'Shares': 'quantity'})
                        if 'Price' in trades_for_report.columns:
                            trades_for_report = trades_for_report.rename(columns={'Price': 'price'})
                        if 'Time' in trades_for_report.columns:
                            trades_for_report = trades_for_report.rename(columns={'Time': 'timestamp'})

                    # Prepare positions
                    positions_for_report = {}
                    if isinstance(positions, pd.DataFrame) and not positions.empty:
                        for _, row in positions.iterrows():
                            symbol = row.get('Symbol', '')
                            if symbol:
                                positions_for_report[symbol] = {
                                    'unrealized_pnl': row.get('P&L ($)', 0),
                                    'realized_pnl': 0
                                }
                    elif isinstance(positions, dict):
                        positions_for_report = positions

                    report_metrics_full = data_provider.get_portfolio_metrics()
                    total_pnl_report = report_metrics_full.get('total_pnl', 0)

                    attr_for_report = PerformanceAttribution(trades_for_report, positions_for_report, total_pnl_report)
                    report_trade = attr_for_report.winner_loser_analysis()
                    report_attr = attr_for_report.get_summary()

                # Generate report
                generator = ReportGenerator(title="Portfolio Performance Report")
                pdf_bytes = generator.generate_report(
                    metrics=report_metrics,
                    benchmark_metrics=report_benchmark,
                    monthly_returns=report_monthly,
                    attribution_data=report_attr,
                    trade_analysis=report_trade,
                    start_date=start_dt,
                    end_date=end_dt
                )

                if pdf_bytes:
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("Report generated successfully!")
                else:
                    st.error("Failed to generate report. Make sure reportlab is installed: pip install reportlab")

            except Exception as e:
                st.error(f"Error generating report: {e}")


def render_backtest():
    """Render backtest page."""
    st.title("Backtest Results")

    # Backtest parameters
    st.subheader("Run New Backtest")

    col1, col2, col3 = st.columns(3)
    with col1:
        symbols = st.text_input("Symbols (comma-separated)", "AAPL,MSFT,GOOGL,NVDA")
    with col2:
        period = st.selectbox("Period", ["6mo", "1y", "2y"])
    with col3:
        use_ensemble = st.checkbox("Use Ensemble Model", value=True)

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            try:
                from src.backtest.backtester import Backtester

                if use_ensemble:
                    from src.ml.models.ensemble_model import EnsembleModel
                    model = EnsembleModel()
                    model.load_models(
                        xgboost_name="trading_model",
                        lstm_name="lstm_trading_model",
                        cnn_name="cnn_trading_model"
                    )
                else:
                    from src.ml.models.xgboost_model import XGBoostModel
                    model = XGBoostModel()
                    model.load("trading_model")

                backtester = Backtester(initial_capital=100000)
                symbol_list = [s.strip() for s in symbols.split(",")]

                result = backtester.run_ml_portfolio(
                    symbols=symbol_list,
                    model=model,
                    period=period,
                    confidence_threshold=0.6,
                    sequence_length=20,
                    max_positions=5
                )

                # Store result in session state
                st.session_state.backtest_result = result
                st.success("Backtest complete!")

            except Exception as e:
                st.error(f"Backtest failed: {e}")

    st.divider()

    # Display results
    st.subheader("Backtest Results")

    if "backtest_result" in st.session_state:
        result = st.session_state.backtest_result

        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

        with metrics_col1:
            st.metric("Total Return", f"{result.total_return:.2%}")
            st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

        with metrics_col2:
            st.metric("Max Drawdown", f"{result.max_drawdown:.2%}")
            st.metric("Win Rate", f"{result.win_rate:.2%}")

        with metrics_col3:
            st.metric("Total Trades", result.total_trades)
            st.metric("Profit Factor", f"{result.profit_factor:.2f}")

        with metrics_col4:
            st.metric("Final Value", f"${result.final_value:,.2f}")
            st.metric("Annualized Return", f"{result.annualized_return:.2%}")

        # Trade details
        if len(result.trades) > 0:
            st.divider()
            st.subheader("Trade Details")
            st.dataframe(result.trades, use_container_width=True)
    else:
        st.info("Run a backtest to see results.")


def render_settings():
    """Render settings page."""
    st.title("Settings")

    # Model Status Section
    st.subheader("Model Status")
    data_provider = get_data_provider()
    model_status = data_provider.get_model_status()

    # Production models
    col1, col2, col3 = st.columns(3)
    models = model_status.get("production_models", {})

    with col1:
        xgb = models.get("xgboost", {})
        st.metric(
            "XGBoost",
            f"{xgb.get('accuracy', 0):.1f}%",
            help=f"Version: {xgb.get('version', 'N/A')}"
        )

    with col2:
        lstm = models.get("lstm", {})
        st.metric(
            "LSTM",
            f"{lstm.get('accuracy', 0):.1f}%",
            help=f"Version: {lstm.get('version', 'N/A')}"
        )

    with col3:
        cnn = models.get("cnn", {})
        st.metric(
            "CNN",
            f"{cnn.get('accuracy', 0):.1f}%",
            help=f"Version: {cnn.get('version', 'N/A')}"
        )

    # Retraining schedule
    st.divider()
    st.subheader("Scheduled Retraining")

    retrain_col1, retrain_col2 = st.columns(2)

    with retrain_col1:
        enabled = model_status.get("retraining_enabled", False)
        status_text = "Enabled" if enabled else "Disabled"
        status_color = "green" if enabled else "red"
        st.markdown(f"**Status:** :{status_color}[{status_text}]")

        schedule = model_status.get("schedule", "weekly")
        day = model_status.get("day_of_week", "sun").capitalize()
        hour = model_status.get("hour", 2)
        st.markdown(f"**Schedule:** {schedule.capitalize()} ({day} at {hour}:00)")

    with retrain_col2:
        last_retrain = model_status.get("last_retrain")
        if last_retrain:
            st.markdown(f"**Last Retrain:** {last_retrain[:19]}")
        else:
            st.markdown("**Last Retrain:** Never")

    # Recent retraining history
    recent = model_status.get("recent_retrains", [])
    if recent:
        st.markdown("**Recent Retraining History:**")
        for entry in recent[:3]:
            completed = entry.get("completed_at", "")[:19] if entry.get("completed_at") else "N/A"
            duration = entry.get("duration_seconds", 0)
            deployments = sum(entry.get("deployments", {}).values())
            total = len(entry.get("models", {}))
            st.markdown(f"- {completed} ({duration:.0f}s) - Deployed: {deployments}/{total}")

    st.divider()

    st.subheader("Trading Configuration")

    # Trading mode
    trading_mode = st.radio("Trading Mode", ["Simulated", "Paper Trading (WeBull)", "Live Trading"])
    if trading_mode == "Live Trading":
        st.warning("⚠️ Live trading uses real money. Use with caution!")

    st.divider()

    # Risk settings
    st.subheader("Risk Management")

    col1, col2 = st.columns(2)

    with col1:
        max_position = st.slider("Max Position Size (%)", 1, 25, 10)
        max_daily_loss = st.slider("Max Daily Loss (%)", 1, 10, 5)

    with col2:
        max_positions = st.number_input("Max Positions", 1, 50, 7)
        max_exposure = st.slider("Max Total Exposure (%)", 50, 100, 80)

    st.divider()

    # Model settings
    st.subheader("Model Configuration")
    model_type = st.selectbox("Model Type", ["Ensemble (XGBoost+LSTM+CNN)", "XGBoost Only"])
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.6)

    st.divider()

    # Symbols
    st.subheader("Trading Symbols")
    symbols = st.text_area(
        "Symbols (one per line)",
        "AAPL\nMSFT\nGOOGL\nNVDA\nMETA\nAMZN\nTSLA"
    )

    st.divider()

    # Quick actions
    st.subheader("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Save Portfolio Snapshot"):
            data_provider = get_data_provider()
            data_provider.save_portfolio_snapshot()
            st.success("Snapshot saved!")

    with col2:
        if st.button("🔄 Refresh All Data"):
            data_provider = get_data_provider()
            data_provider.refresh()
            st.success("Data refreshed!")

    with col3:
        if st.button("📊 View Bot Status"):
            st.info("Trading bot is running in background.")

    if st.button("Save Settings"):
        st.success("Settings saved successfully!")
        st.info("Note: Settings changes require bot restart to take effect.")


def render_portfolio_optimization(data_provider):
    """Render portfolio optimization page."""
    st.title("Portfolio Optimization")

    # Check if optimization is enabled
    from config.settings import Settings
    config = Settings.load_trading_config()
    portfolio_config = config.get("portfolio_optimization", {})

    if not portfolio_config.get("enabled", False):
        st.warning("⚠️ Portfolio optimization is not enabled")
        st.info("Enable it in `config/trading_config.yaml` by setting `portfolio_optimization.enabled: true`")

        with st.expander("Configuration Example"):
            st.code("""
portfolio_optimization:
  enabled: true
  method: "max_sharpe"
  lookback_days: 252
  min_weight: 0.05
  max_weight: 0.30
  rebalancing:
    enabled: true
    drift_threshold: 0.10
    frequency: "monthly"
""", language="yaml")
        return

    # Import required modules
    try:
        from src.portfolio.efficient_frontier import EfficientFrontier
        from src.portfolio.correlation_analyzer import CorrelationAnalyzer
    except ImportError as e:
        st.error(f"Portfolio modules not available: {e}")
        return

    # Optimization settings
    st.subheader("⚙️ Optimization Settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Method", portfolio_config.get("method", "max_sharpe").replace("_", " ").title())
    with col2:
        st.metric("Lookback Period", f"{portfolio_config.get('lookback_days', 252)} days")
    with col3:
        rebalancing_config = portfolio_config.get("rebalancing", {})
        rebalance_enabled = "Enabled" if rebalancing_config.get("enabled", True) else "Disabled"
        st.metric("Rebalancing", rebalance_enabled)

    # Display current vs target allocation
    st.subheader("📊 Current vs Target Allocation")

    # Note: In a full implementation, we'd fetch actual target weights from the trading engine
    # For now, we show a placeholder that this feature is available when the bot is running
    st.info("💡 Target weights are calculated dynamically when the trading bot is running with portfolio optimization enabled.")

    # Show current positions
    positions = data_provider.get_positions()
    metrics = data_provider.get_portfolio_metrics()
    portfolio_value = metrics.get('portfolio_value', 100000)

    if not positions.empty:
        # Calculate current weights (using correct column names from data_provider)
        positions['value'] = positions['Shares'] * positions['Current Price']
        positions['weight'] = positions['value'] / portfolio_value

        # Create visualization
        fig = go.Figure()

        # Current allocation
        fig.add_trace(go.Bar(
            name='Current Allocation',
            x=positions['Symbol'],
            y=positions['weight'] * 100,
            marker_color='lightblue'
        ))

        fig.update_layout(
            title="Current Portfolio Allocation",
            xaxis_title="Symbol",
            yaxis_title="Weight (%)",
            yaxis_range=[0, max(40, positions['weight'].max() * 110)],
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Allocation table
        st.dataframe(
            positions[['Symbol', 'Shares', 'Current Price', 'value', 'weight']]
            .rename(columns={
                'Current Price': 'Price',
                'value': 'Value',
                'weight': 'Weight'
            })
            .style.format({
                'Price': '${:.2f}',
                'Value': '${:,.2f}',
                'Weight': '{:.2%}'
            }),
            hide_index=True
        )
    else:
        st.info("No positions currently held")

    # Optimization metrics
    st.subheader("📈 Optimization Metrics")

    # Get trading symbols from config
    symbols = config.get("symbols", ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA"])

    # Try to calculate optimization metrics
    try:
        # Initialize analyzers
        lookback_days = portfolio_config.get("lookback_days", 252)
        risk_free_rate = portfolio_config.get("risk_free_rate", 0.05)
        min_weight = portfolio_config.get("min_weight", 0.05)
        max_weight = portfolio_config.get("max_weight", 0.30)

        frontier_calculator = EfficientFrontier(
            risk_free_rate=risk_free_rate,
            use_shrinkage=True
        )
        correlation_analyzer = CorrelationAnalyzer(
            lookback_days=lookback_days,
            correlation_threshold=portfolio_config.get("correlation_threshold", 0.8)
        )

        # Fetch returns data
        from src.data.data_fetcher import DataFetcher
        data_fetcher = DataFetcher()

        returns_data = {}
        for symbol in symbols:
            try:
                df = data_fetcher.fetch_historical(symbol, period=f"{lookback_days + 30}d")
                if df is not None and not df.empty:
                    close_col = 'Close' if 'Close' in df.columns else 'close'
                    if close_col in df.columns:
                        returns = df[close_col].pct_change().dropna()
                        returns_data[symbol] = returns
            except Exception:
                continue

        if len(returns_data) >= 2:
            returns_df = pd.DataFrame(returns_data).dropna()
            if len(returns_df) > lookback_days:
                returns_df = returns_df.tail(lookback_days)

            # Calculate tangency portfolio (Max Sharpe)
            tangency = frontier_calculator.find_tangency_portfolio(
                returns_df, min_weight, max_weight
            )

            # Calculate minimum variance portfolio
            min_var = frontier_calculator.find_minimum_variance_portfolio(
                returns_df, min_weight, max_weight
            )

            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                sharpe = tangency.get('sharpe', 0) if tangency else 0
                st.metric(
                    "Max Sharpe Ratio",
                    f"{sharpe:.2f}",
                    help="Maximum achievable Sharpe ratio"
                )

            with col2:
                exp_return = tangency.get('expected_return', 0) if tangency else 0
                st.metric(
                    "Expected Return",
                    f"{exp_return:.1%}",
                    help="Annualized expected return (tangency portfolio)"
                )

            with col3:
                volatility = tangency.get('volatility', 0) if tangency else 0
                st.metric(
                    "Volatility",
                    f"{volatility:.1%}",
                    help="Annualized portfolio volatility (tangency portfolio)"
                )

            with col4:
                # Calculate diversification ratio for current positions
                if not positions.empty:
                    current_weights = dict(zip(positions['symbol'], positions['weight']))
                    div_ratio = correlation_analyzer.calculate_diversification_ratio(
                        current_weights, returns_df
                    )
                else:
                    div_ratio = 0
                st.metric(
                    "Diversification Ratio",
                    f"{div_ratio:.2f}",
                    help="Higher is better (1.0 = no diversification)"
                )

            st.divider()

            # Efficient Frontier Visualization
            st.subheader("🎯 Efficient Frontier")

            with st.spinner("Calculating efficient frontier..."):
                frontier_df = frontier_calculator.calculate_frontier(
                    returns_df,
                    num_points=100,
                    min_weight=min_weight,
                    max_weight=max_weight
                )

                if not frontier_df.empty:
                    # Calculate current portfolio metrics if positions exist
                    current_portfolio = None
                    if not positions.empty:
                        current_weights_dict = dict(zip(positions['symbol'], positions['weight']))
                        # Calculate current portfolio stats
                        mean_returns = returns_df.mean() * 252
                        cov_matrix = returns_df.cov() * 252

                        weights_array = np.array([current_weights_dict.get(s, 0) for s in returns_df.columns])
                        current_return = np.dot(weights_array, mean_returns)
                        current_vol = np.sqrt(np.dot(weights_array, np.dot(cov_matrix, weights_array)))

                        current_portfolio = {
                            'expected_return': current_return,
                            'volatility': current_vol
                        }

                    # Plot efficient frontier
                    fig = frontier_calculator.plot_frontier(
                        frontier_df,
                        current_portfolio=current_portfolio,
                        tangency_portfolio=tangency,
                        min_var_portfolio=min_var
                    )

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                        # Show tangency portfolio weights
                        if tangency and 'weights' in tangency:
                            with st.expander("📊 Tangency Portfolio Weights (Max Sharpe)"):
                                weights_df = pd.DataFrame([
                                    {"Symbol": sym, "Weight": weight}
                                    for sym, weight in tangency['weights'].items()
                                    if weight > 0.001
                                ]).sort_values("Weight", ascending=False)

                                st.dataframe(
                                    weights_df.style.format({"Weight": "{:.2%}"}),
                                    hide_index=True
                                )
                else:
                    st.warning("Could not calculate efficient frontier. Need more historical data.")

            st.divider()

            # Correlation Heatmap
            st.subheader("🔗 Correlation Matrix")

            corr_matrix = correlation_analyzer.calculate_correlation_matrix(
                symbols, returns_df
            )

            if not corr_matrix.empty:
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu_r',
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=[[f"{v:.2f}" for v in row] for row in corr_matrix.values],
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
                    colorbar=dict(title="Correlation")
                ))

                fig.update_layout(
                    title="Asset Correlation Matrix",
                    xaxis_title="",
                    yaxis_title="",
                    height=500,
                    width=600
                )

                st.plotly_chart(fig, use_container_width=True)

                # Correlation statistics
                corr_stats = correlation_analyzer.get_correlation_stats(corr_matrix)

                if corr_stats:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Mean Correlation",
                            f"{corr_stats.get('mean_correlation', 0):.2f}",
                            help="Average pairwise correlation"
                        )

                    with col2:
                        st.metric(
                            "Max Correlation",
                            f"{corr_stats.get('max_correlation', 0):.2f}",
                            help="Highest correlation between any pair"
                        )

                    with col3:
                        high_corr_pairs = corr_stats.get('n_high_correlation_pairs', 0)
                        total_pairs = corr_stats.get('n_total_pairs', 1)
                        st.metric(
                            "High Correlation Pairs",
                            f"{high_corr_pairs}/{total_pairs}",
                            help=f"Pairs with |correlation| > {portfolio_config.get('correlation_threshold', 0.8)}"
                        )

                    with col4:
                        # Find clusters
                        clusters = correlation_analyzer.find_correlated_clusters(corr_matrix)
                        st.metric(
                            "Asset Clusters",
                            f"{len(clusters)}",
                            help="Number of correlated asset groups"
                        )

                # Show clusters
                clusters = correlation_analyzer.find_correlated_clusters(corr_matrix)
                if clusters:
                    with st.expander("📦 Correlation Clusters"):
                        for cluster_id, cluster_symbols in clusters.items():
                            st.markdown(f"**Cluster {cluster_id + 1}:** {', '.join(cluster_symbols)}")

                # Check concentration risk
                if not positions.empty:
                    current_weights = dict(zip(positions['symbol'], positions['weight']))
                    concentration_check = correlation_analyzer.check_concentration_risk(
                        current_weights, corr_matrix
                    )

                    if concentration_check.get('has_concentration_risk', False):
                        st.warning("⚠️ **Concentration Risk Detected**")
                        for warning in concentration_check.get('warnings', []):
                            st.warning(f"• {warning}")
        else:
            st.info("Need at least 2 symbols with historical data to calculate optimization metrics.")

    except Exception as e:
        st.error(f"Error calculating optimization metrics: {e}")
        st.markdown("""
        **Metrics will be available when:**
        - Portfolio optimization is enabled
        - Historical data is available for symbols
        - Trading bot is actively running
        """)

    # Rebalancing status
    if rebalancing_config.get("enabled", True):
        st.subheader("🔄 Rebalancing Status")

        col1, col2 = st.columns(2)
        with col1:
            drift_threshold = rebalancing_config.get("drift_threshold", 0.10)
            st.metric("Drift Threshold", f"{drift_threshold:.1%}")
        with col2:
            frequency = rebalancing_config.get("frequency", "monthly")
            st.metric("Rebalancing Frequency", frequency.title())

        st.info("Rebalancing is triggered when portfolio drift exceeds the threshold OR on the scheduled frequency (combined mode).")

        # Transaction cost estimation
        st.divider()
        st.subheader("💰 Transaction Cost Estimate")

        if not positions.empty:
            try:
                from src.portfolio.transaction_costs import TransactionCostModel

                # Initialize cost model
                slippage_bps = rebalancing_config.get("slippage_pct", 0.001) * 10000
                cost_model = TransactionCostModel(
                    base_slippage_bps=slippage_bps,
                    commission_per_trade=0.0,
                    min_trade_value=rebalancing_config.get("min_trade_value", 200.0)
                )

                # Get current and target weights (example: use portfolio optimization weights if available)
                current_weights_dict = dict(zip(positions['symbol'], positions['weight']))

                # For demonstration, calculate costs for current allocation
                # In real scenario, this would use target weights from optimizer
                costs = cost_model.estimate_rebalancing_costs(
                    current_weights=current_weights_dict,
                    target_weights=current_weights_dict,  # Same = no cost
                    portfolio_value=portfolio_value
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Estimated Total Cost",
                        f"${costs.total_cost:.2f}",
                        help="Total transaction cost for rebalancing"
                    )

                with col2:
                    st.metric(
                        "Cost %",
                        f"{costs.total_cost_pct:.3%}",
                        help="Cost as percentage of portfolio value"
                    )

                with col3:
                    st.metric(
                        "Expected Trades",
                        f"{costs.expected_trades}",
                        help="Number of trades needed for rebalancing"
                    )

                with col4:
                    st.metric(
                        "Portfolio Turnover",
                        f"{costs.turnover_pct:.1f}%",
                        help="Percentage of portfolio being traded"
                    )

                # Cost breakdown
                with st.expander("💵 Cost Breakdown"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Slippage Cost", f"${costs.slippage_cost:.2f}")
                    with col2:
                        st.metric("Market Impact", f"${costs.market_impact_cost:.2f}")
                    with col3:
                        st.metric("Commission", f"${costs.commission_cost:.2f}")

                    st.markdown("""
                    **Transaction Costs Include:**
                    - **Slippage**: Difference between expected and actual execution price
                    - **Market Impact**: Price movement caused by the trade itself
                    - **Commission**: Trading fees (typically $0 for US stocks)
                    """)

            except Exception as e:
                st.warning(f"Could not calculate transaction costs: {e}")

    # Documentation
    with st.expander("📚 How Portfolio Optimization Works"):
        st.markdown("""
        ### Portfolio Optimization Overview

        The portfolio optimization module uses Modern Portfolio Theory to construct optimal portfolios:

        1. **Data Collection**: Fetches historical returns for all symbols
        2. **Optimization**: Calculates optimal weights using the selected method:
           - **Max Sharpe**: Maximizes risk-adjusted returns (recommended)
           - **Risk Parity**: Equal risk contribution from each asset
           - **Minimum Variance**: Minimizes portfolio volatility
           - **Equal Weight**: Baseline allocation (1/N for N assets)
        3. **Signal Integration**: Optionally tilts weights based on ML signal confidence
        4. **Rebalancing**: Periodically adjusts positions to maintain target allocation

        ### Benefits
        - ✅ Improved risk-adjusted returns (Sharpe ratio)
        - ✅ Better diversification
        - ✅ Systematic rebalancing discipline
        - ✅ Integration with ML signals

        ### Configuration
        Edit `config/trading_config.yaml` to customize:
        - Optimization method
        - Weight constraints (min/max per asset)
        - Rebalancing triggers and frequency
        - Signal integration strength
        """)


if __name__ == "__main__":
    main()
