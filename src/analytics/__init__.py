"""
Portfolio Analytics Module.
Provides metrics calculation, benchmark comparison, performance attribution, and report generation.
"""
from src.analytics.metrics import MetricsCalculator, calculate_returns, calculate_cumulative_returns
from src.analytics.data_aggregator import DataAggregator, get_data_aggregator
from src.analytics.benchmark import BenchmarkComparison
from src.analytics.attribution import PerformanceAttribution
from src.analytics.report_generator import ReportGenerator

__all__ = [
    "MetricsCalculator",
    "calculate_returns",
    "calculate_cumulative_returns",
    "DataAggregator",
    "get_data_aggregator",
    "BenchmarkComparison",
    "PerformanceAttribution",
    "ReportGenerator",
]
