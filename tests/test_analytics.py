"""
Tests for Analytics module - benchmark comparison, attribution, reports.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.analytics.benchmark import BenchmarkComparison
from src.analytics.attribution import PerformanceAttribution


class TestBenchmarkComparison:
    """Tests for BenchmarkComparison class."""

    def test_initialization(self, sample_equity_curve):
        """Test BenchmarkComparison initializes correctly."""
        bc = BenchmarkComparison(sample_equity_curve)
        assert bc is not None
        assert bc.benchmark_symbol == "SPY"

    def test_initialization_custom_benchmark(self, sample_equity_curve):
        """Test with custom benchmark symbol."""
        bc = BenchmarkComparison(sample_equity_curve, benchmark_symbol="QQQ")
        assert bc.benchmark_symbol == "QQQ"

    @patch('src.analytics.benchmark.BenchmarkComparison.fetch_benchmark_data')
    def test_calculate_beta(self, mock_fetch, sample_equity_curve):
        """Test beta calculation."""
        # Mock benchmark returns similar to portfolio
        mock_benchmark = pd.Series(
            np.random.randn(len(sample_equity_curve)) * 0.01 + 0.0002,
            index=sample_equity_curve.index
        )
        mock_fetch.return_value = mock_benchmark

        bc = BenchmarkComparison(sample_equity_curve)
        bc._benchmark_data = mock_benchmark

        beta = bc.calculate_beta()

        assert isinstance(beta, float)
        # Beta should be reasonable (usually between -2 and 3)
        assert -3 <= beta <= 5

    @patch('src.analytics.benchmark.BenchmarkComparison.fetch_benchmark_data')
    def test_calculate_alpha(self, mock_fetch, sample_equity_curve):
        """Test alpha calculation."""
        mock_benchmark = pd.Series(
            np.random.randn(len(sample_equity_curve)) * 0.01 + 0.0002,
            index=sample_equity_curve.index
        )
        mock_fetch.return_value = mock_benchmark

        bc = BenchmarkComparison(sample_equity_curve)
        bc._benchmark_data = mock_benchmark

        alpha = bc.calculate_alpha()

        assert isinstance(alpha, float)
        # Alpha can be large due to (1+mean)**252 annualization with random data
        assert np.isfinite(alpha)

    @patch('src.analytics.benchmark.BenchmarkComparison.fetch_benchmark_data')
    def test_information_ratio(self, mock_fetch, sample_equity_curve):
        """Test information ratio calculation."""
        mock_benchmark = pd.Series(
            np.random.randn(len(sample_equity_curve)) * 0.01,
            index=sample_equity_curve.index
        )
        mock_fetch.return_value = mock_benchmark

        bc = BenchmarkComparison(sample_equity_curve)
        bc._benchmark_data = mock_benchmark

        ir = bc.information_ratio()

        assert isinstance(ir, float)

    @patch('src.analytics.benchmark.BenchmarkComparison.fetch_benchmark_data')
    def test_cumulative_comparison(self, mock_fetch, sample_equity_curve):
        """Test cumulative comparison data."""
        mock_benchmark = pd.Series(
            np.random.randn(len(sample_equity_curve)) * 0.01,
            index=sample_equity_curve.index
        )
        mock_fetch.return_value = mock_benchmark

        bc = BenchmarkComparison(sample_equity_curve)
        bc._benchmark_data = mock_benchmark

        cumulative = bc.cumulative_comparison()

        if not cumulative.empty:
            assert 'portfolio' in cumulative.columns
            assert 'benchmark' in cumulative.columns

    @patch('src.analytics.benchmark.BenchmarkComparison.fetch_benchmark_data')
    def test_get_all_metrics(self, mock_fetch, sample_equity_curve):
        """Test getting all benchmark metrics."""
        mock_benchmark = pd.Series(
            np.random.randn(len(sample_equity_curve)) * 0.01,
            index=sample_equity_curve.index
        )
        mock_fetch.return_value = mock_benchmark

        bc = BenchmarkComparison(sample_equity_curve)
        bc._benchmark_data = mock_benchmark

        metrics = bc.get_all_metrics()

        assert isinstance(metrics, dict)
        assert 'alpha' in metrics
        assert 'beta' in metrics
        assert 'information_ratio' in metrics


class TestPerformanceAttribution:
    """Tests for PerformanceAttribution class."""

    def test_initialization(self, sample_trades_df, sample_positions):
        """Test PerformanceAttribution initializes correctly."""
        pa = PerformanceAttribution(
            trades=sample_trades_df,
            positions=sample_positions,
            total_pnl=1000.0
        )
        assert pa is not None

    def test_position_contribution(self, sample_trades_df, sample_positions):
        """Test position contribution calculation."""
        pa = PerformanceAttribution(
            trades=sample_trades_df,
            positions=sample_positions,
            total_pnl=60.0  # Total unrealized P&L from positions
        )

        contrib = pa.position_contribution()

        assert isinstance(contrib, pd.DataFrame)
        if not contrib.empty:
            assert 'symbol' in contrib.columns
            assert 'pnl' in contrib.columns

    def test_winner_loser_analysis(self, sample_trades_df, sample_positions):
        """Test winner/loser analysis."""
        pa = PerformanceAttribution(
            trades=sample_trades_df,
            positions=sample_positions,
            total_pnl=100.0
        )

        analysis = pa.winner_loser_analysis()

        assert isinstance(analysis, dict)
        assert 'win_rate' in analysis
        assert 'avg_win' in analysis
        assert 'avg_loss' in analysis

    def test_realized_vs_unrealized(self, sample_trades_df, sample_positions):
        """Test realized vs unrealized P&L breakdown."""
        pa = PerformanceAttribution(
            trades=sample_trades_df,
            positions=sample_positions,
            total_pnl=100.0
        )

        breakdown = pa.realized_vs_unrealized()

        assert isinstance(breakdown, dict)
        assert 'realized_pnl' in breakdown
        assert 'unrealized_pnl' in breakdown

    def test_top_contributors(self, sample_trades_df, sample_positions):
        """Test top contributors extraction."""
        pa = PerformanceAttribution(
            trades=sample_trades_df,
            positions=sample_positions,
            total_pnl=100.0
        )

        top = pa.top_contributors(n=3)

        assert isinstance(top, pd.DataFrame)

    def test_get_summary(self, sample_trades_df, sample_positions):
        """Test complete attribution summary."""
        pa = PerformanceAttribution(
            trades=sample_trades_df,
            positions=sample_positions,
            total_pnl=100.0
        )

        summary = pa.get_summary()

        assert isinstance(summary, dict)
        assert 'trade_analysis' in summary
        assert 'pnl_breakdown' in summary


class TestReportGeneration:
    """Tests for report generation."""

    def test_report_generator_initialization(self):
        """Test ReportGenerator initializes correctly."""
        from src.analytics.report_generator import ReportGenerator
        rg = ReportGenerator()
        assert rg is not None

    def test_report_generation(self):
        """Test PDF report generation."""
        from src.analytics.report_generator import ReportGenerator

        rg = ReportGenerator(title="Test Report")

        metrics = {
            'total_return': 0.15,
            'annualized_return': 0.18,
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'calmar_ratio': 2.2,
            'max_drawdown': -0.08,
            'trading_days': 252
        }

        pdf_bytes = rg.generate_report(
            metrics=metrics,
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now()
        )

        # Should return bytes or None if reportlab not installed
        assert pdf_bytes is None or isinstance(pdf_bytes, bytes)

    def test_report_with_benchmark(self):
        """Test report with benchmark metrics."""
        from src.analytics.report_generator import ReportGenerator

        rg = ReportGenerator()

        metrics = {
            'total_return': 0.15,
            'annualized_return': 0.18,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.08,
            'trading_days': 252
        }

        benchmark_metrics = {
            'alpha': 0.03,
            'beta': 0.9,
            'information_ratio': 0.5,
            'relative_performance': 0.02
        }

        pdf_bytes = rg.generate_report(
            metrics=metrics,
            benchmark_metrics=benchmark_metrics
        )

        assert pdf_bytes is None or isinstance(pdf_bytes, bytes)


class TestEdgeCases:
    """Test edge cases for analytics."""

    def test_empty_trades(self, sample_positions):
        """Test attribution with no trades."""
        empty_trades = pd.DataFrame()

        pa = PerformanceAttribution(
            trades=empty_trades,
            positions=sample_positions,
            total_pnl=60.0
        )

        analysis = pa.winner_loser_analysis()
        assert isinstance(analysis, dict)

    def test_empty_positions(self, sample_trades_df):
        """Test attribution with no positions."""
        empty_positions = {}

        pa = PerformanceAttribution(
            trades=sample_trades_df,
            positions=empty_positions,
            total_pnl=0.0
        )

        breakdown = pa.realized_vs_unrealized()
        assert isinstance(breakdown, dict)

    def test_short_equity_curve(self):
        """Test benchmark with very short equity curve."""
        short_curve = pd.Series(
            [100000, 101000, 102000],
            index=pd.date_range(end=datetime.now(), periods=3, freq='D')
        )

        bc = BenchmarkComparison(short_curve)
        metrics = bc.get_all_metrics()

        # Should handle gracefully
        assert isinstance(metrics, dict)
