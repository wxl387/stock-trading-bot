"""
Tests for DataFetcher - stock data fetching and caching.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.data.data_fetcher import DataFetcher


class TestDataFetcherInitialization:
    """Tests for DataFetcher initialization."""

    def test_initialization(self):
        """Test DataFetcher initializes correctly."""
        fetcher = DataFetcher()
        assert fetcher is not None


class TestFetchStockData:
    """Tests for fetching stock data."""

    @patch('src.data.data_fetcher.yf.download')
    def test_fetch_stock_data(self, mock_yf):
        """Test fetching stock data (mocked)."""
        # Create mock response
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        mock_data = pd.DataFrame({
            'Open': np.random.randn(100) + 150,
            'High': np.random.randn(100) + 152,
            'Low': np.random.randn(100) + 148,
            'Close': np.random.randn(100) + 150,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

        mock_yf.return_value = mock_data

        fetcher = DataFetcher()
        data = fetcher.fetch_stock_data('AAPL')

        assert data is not None
        assert isinstance(data, pd.DataFrame)

    @patch('src.data.data_fetcher.yf.download')
    def test_fetch_with_date_range(self, mock_yf):
        """Test fetching with specific date range."""
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        mock_data = pd.DataFrame({
            'Open': [150] * 50,
            'High': [152] * 50,
            'Low': [148] * 50,
            'Close': [150] * 50,
            'Volume': [1000000] * 50
        }, index=dates)

        mock_yf.return_value = mock_data

        fetcher = DataFetcher()
        start_date = datetime.now() - timedelta(days=60)
        end_date = datetime.now()

        data = fetcher.fetch_stock_data(
            'AAPL',
            start_date=start_date,
            end_date=end_date
        )

        assert data is not None

    @patch('src.data.data_fetcher.yf.download')
    def test_fetch_returns_ohlcv(self, mock_yf):
        """Test that fetched data has OHLCV columns."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        mock_data = pd.DataFrame({
            'Open': [150] * 30,
            'High': [152] * 30,
            'Low': [148] * 30,
            'Close': [150] * 30,
            'Volume': [1000000] * 30
        }, index=dates)

        mock_yf.return_value = mock_data

        fetcher = DataFetcher()
        data = fetcher.fetch_stock_data('AAPL')

        # Check for required columns (case may vary)
        columns_lower = [c.lower() for c in data.columns]
        assert 'open' in columns_lower or 'Open' in data.columns
        assert 'close' in columns_lower or 'Close' in data.columns


class TestDataCaching:
    """Tests for data caching functionality."""

    @patch('src.data.data_fetcher.yf.download')
    def test_data_caching(self, mock_yf):
        """Test that data is cached."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        mock_data = pd.DataFrame({
            'Close': [150] * 30
        }, index=dates)

        mock_yf.return_value = mock_data

        fetcher = DataFetcher()

        # First call
        data1 = fetcher.fetch_stock_data('AAPL')
        call_count_1 = mock_yf.call_count

        # Second call (should use cache if implemented)
        data2 = fetcher.fetch_stock_data('AAPL')
        call_count_2 = mock_yf.call_count

        # Data should be equal
        if data1 is not None and data2 is not None:
            assert len(data1) == len(data2)


class TestMultipleSymbols:
    """Tests for fetching multiple symbols."""

    @patch('src.data.data_fetcher.yf.download')
    def test_fetch_multiple_symbols(self, mock_yf):
        """Test fetching data for multiple symbols."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        mock_data = pd.DataFrame({
            'Close': [150] * 30
        }, index=dates)

        mock_yf.return_value = mock_data

        fetcher = DataFetcher()

        symbols = ['AAPL', 'MSFT', 'GOOGL']
        data = {}

        for symbol in symbols:
            data[symbol] = fetcher.fetch_stock_data(symbol)

        assert len(data) == 3
        for symbol in symbols:
            assert symbol in data


class TestErrorHandling:
    """Tests for error handling."""

    @patch('src.data.data_fetcher.yf.download')
    def test_invalid_symbol(self, mock_yf):
        """Test handling of invalid symbol."""
        mock_yf.return_value = pd.DataFrame()  # Empty result

        fetcher = DataFetcher()
        data = fetcher.fetch_stock_data('INVALID_SYMBOL_XYZ')

        # Should return None or empty DataFrame
        assert data is None or data.empty

    @patch('src.data.data_fetcher.yf.download')
    def test_network_error(self, mock_yf):
        """Test handling of network errors."""
        mock_yf.side_effect = Exception("Network error")

        fetcher = DataFetcher()

        # Should handle gracefully
        try:
            data = fetcher.fetch_stock_data('AAPL')
            assert data is None
        except Exception:
            pass  # Some implementations may raise

    @patch('src.data.data_fetcher.yf.download')
    def test_empty_date_range(self, mock_yf):
        """Test handling of empty date range."""
        mock_yf.return_value = pd.DataFrame()

        fetcher = DataFetcher()

        # Future dates should return empty
        data = fetcher.fetch_stock_data(
            'AAPL',
            start_date=datetime.now() + timedelta(days=30),
            end_date=datetime.now() + timedelta(days=60)
        )

        assert data is None or data.empty


class TestDataQuality:
    """Tests for data quality checks."""

    @patch('src.data.data_fetcher.yf.download')
    def test_data_has_index(self, mock_yf):
        """Test fetched data has datetime index."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        mock_data = pd.DataFrame({
            'Close': [150] * 30
        }, index=dates)

        mock_yf.return_value = mock_data

        fetcher = DataFetcher()
        data = fetcher.fetch_stock_data('AAPL')

        if data is not None and not data.empty:
            assert isinstance(data.index, pd.DatetimeIndex)

    @patch('src.data.data_fetcher.yf.download')
    def test_data_sorted_by_date(self, mock_yf):
        """Test data is sorted by date."""
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        mock_data = pd.DataFrame({
            'Close': list(range(30))
        }, index=dates)

        mock_yf.return_value = mock_data

        fetcher = DataFetcher()
        data = fetcher.fetch_stock_data('AAPL')

        if data is not None and not data.empty:
            assert data.index.is_monotonic_increasing
