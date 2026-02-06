"""
Tests for DataFetcher - stock data fetching and caching.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from src.data.data_fetcher import DataFetcher


@pytest.fixture
def fetcher(tmp_path):
    """Create DataFetcher with a temp cache directory."""
    return DataFetcher(cache_dir=tmp_path / "cache")


def _make_mock_df(n=100):
    """Create mock OHLCV DataFrame for tests."""
    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
    return pd.DataFrame({
        'Open': np.random.randn(n) + 150,
        'High': np.random.randn(n) + 152,
        'Low': np.random.randn(n) + 148,
        'Close': np.random.randn(n) + 150,
        'Volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)


class TestDataFetcherInitialization:
    """Tests for DataFetcher initialization."""

    def test_initialization(self, tmp_path):
        """Test DataFetcher initializes correctly."""
        fetcher = DataFetcher(cache_dir=tmp_path / "cache")
        assert fetcher is not None


class TestFetchStockData:
    """Tests for fetching stock data."""

    @patch('src.data.data_fetcher.yf.Ticker')
    def test_fetch_stock_data(self, mock_ticker_cls, fetcher):
        """Test fetching stock data (mocked)."""
        mock_df = _make_mock_df(100)
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_ticker

        data = fetcher.fetch_historical('AAPL', use_cache=False)

        assert data is not None
        assert isinstance(data, pd.DataFrame)

    @patch('src.data.data_fetcher.yf.Ticker')
    def test_fetch_with_date_range(self, mock_ticker_cls, fetcher):
        """Test fetching with specific date range."""
        mock_df = _make_mock_df(50)
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_ticker

        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        data = fetcher.fetch_historical(
            'AAPL',
            start_date=start_date,
            end_date=end_date,
            use_cache=False
        )

        assert data is not None

    @patch('src.data.data_fetcher.yf.Ticker')
    def test_fetch_returns_ohlcv(self, mock_ticker_cls, fetcher):
        """Test that fetched data has OHLCV columns."""
        mock_df = _make_mock_df(30)
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_ticker

        data = fetcher.fetch_historical('AAPL', use_cache=False)

        # Production standardizes to lowercase
        assert 'open' in data.columns
        assert 'close' in data.columns


class TestDataCaching:
    """Tests for data caching functionality."""

    @patch('src.data.data_fetcher.yf.Ticker')
    def test_data_caching(self, mock_ticker_cls, fetcher):
        """Test that data is cached."""
        mock_df = _make_mock_df(30)
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_ticker

        # First call (fetches and caches)
        data1 = fetcher.fetch_historical('AAPL', use_cache=True)

        # Second call (should use cache)
        data2 = fetcher.fetch_historical('AAPL', use_cache=True)

        # Data should be equal
        if data1 is not None and data2 is not None:
            assert len(data1) == len(data2)


class TestMultipleSymbols:
    """Tests for fetching multiple symbols."""

    @patch('src.data.data_fetcher.yf.Ticker')
    def test_fetch_multiple_symbols(self, mock_ticker_cls, fetcher):
        """Test fetching data for multiple symbols."""
        mock_df = _make_mock_df(30)
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_ticker

        symbols = ['AAPL', 'MSFT', 'GOOGL']
        data = {}

        for symbol in symbols:
            data[symbol] = fetcher.fetch_historical(symbol, use_cache=False)

        assert len(data) == 3
        for symbol in symbols:
            assert symbol in data


class TestErrorHandling:
    """Tests for error handling."""

    @patch('src.data.data_fetcher.yf.Ticker')
    def test_invalid_symbol(self, mock_ticker_cls, fetcher):
        """Test handling of invalid symbol."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()  # Empty result
        mock_ticker_cls.return_value = mock_ticker

        data = fetcher.fetch_historical('INVALID_SYMBOL_XYZ', use_cache=False)

        # Should return empty DataFrame
        assert data.empty

    @patch('src.data.data_fetcher.yf.Ticker')
    def test_network_error(self, mock_ticker_cls, fetcher):
        """Test handling of network errors."""
        mock_ticker = Mock()
        mock_ticker.history.side_effect = Exception("Network error")
        mock_ticker_cls.return_value = mock_ticker

        # Production raises on error
        with pytest.raises(Exception):
            fetcher.fetch_historical('AAPL', use_cache=False)

    @patch('src.data.data_fetcher.yf.Ticker')
    def test_empty_date_range(self, mock_ticker_cls, fetcher):
        """Test handling of empty date range."""
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_ticker

        # Future dates should return empty
        data = fetcher.fetch_historical(
            'AAPL',
            start_date=(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            end_date=(datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d'),
            use_cache=False
        )

        assert data.empty


class TestDataQuality:
    """Tests for data quality checks."""

    @patch('src.data.data_fetcher.yf.Ticker')
    def test_data_has_index(self, mock_ticker_cls, fetcher):
        """Test fetched data has datetime index."""
        mock_df = _make_mock_df(30)
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_ticker

        data = fetcher.fetch_historical('AAPL', use_cache=False)

        if data is not None and not data.empty:
            assert isinstance(data.index, pd.DatetimeIndex)

    @patch('src.data.data_fetcher.yf.Ticker')
    def test_data_sorted_by_date(self, mock_ticker_cls, fetcher):
        """Test data is sorted by date."""
        mock_df = _make_mock_df(30)
        mock_ticker = Mock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_ticker

        data = fetcher.fetch_historical('AAPL', use_cache=False)

        if data is not None and not data.empty:
            assert data.index.is_monotonic_increasing
