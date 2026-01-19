"""
Shared pytest fixtures for stock trading bot tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_ohlcv_data():
    """
    Generate sample OHLCV DataFrame for testing.

    Returns DataFrame with 100 days of realistic stock data.
    """
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    # Generate realistic price data
    base_price = 100.0
    returns = np.random.randn(n_days) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    high = prices * (1 + np.abs(np.random.randn(n_days)) * 0.01)
    low = prices * (1 - np.abs(np.random.randn(n_days)) * 0.01)
    open_prices = low + (high - low) * np.random.rand(n_days)
    volume = np.random.randint(1000000, 10000000, n_days)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)

    return df


@pytest.fixture
def sample_ohlcv_data_long():
    """
    Generate longer OHLCV DataFrame (252 trading days) for backtesting.
    """
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')  # Business days

    base_price = 100.0
    returns = np.random.randn(n_days) * 0.015  # 1.5% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    high = prices * (1 + np.abs(np.random.randn(n_days)) * 0.01)
    low = prices * (1 - np.abs(np.random.randn(n_days)) * 0.01)
    open_prices = low + (high - low) * np.random.rand(n_days)
    volume = np.random.randint(1000000, 10000000, n_days)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume
    }, index=dates)

    return df


@pytest.fixture
def sample_returns():
    """
    Generate sample daily returns series for metrics testing.

    Returns Series with 252 days (1 year) of returns.
    """
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    # Generate returns with slight positive drift
    returns = np.random.randn(n_days) * 0.01 + 0.0003  # ~7.5% annual return

    return pd.Series(returns, index=dates, name='returns')


@pytest.fixture
def sample_returns_negative():
    """
    Generate sample returns with negative performance for testing.
    """
    np.random.seed(123)
    n_days = 252
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    # Negative drift
    returns = np.random.randn(n_days) * 0.015 - 0.0005

    return pd.Series(returns, index=dates, name='returns')


@pytest.fixture
def sample_equity_curve():
    """
    Generate sample equity curve (portfolio values over time).
    """
    np.random.seed(42)
    n_days = 252
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    initial_value = 100000
    returns = np.random.randn(n_days) * 0.01 + 0.0003
    values = initial_value * np.exp(np.cumsum(returns))

    return pd.Series(values, index=dates, name='portfolio_value')


@pytest.fixture
def mock_broker():
    """
    Create SimulatedBroker instance with initial capital.
    """
    from src.broker.simulated_broker import SimulatedBroker
    return SimulatedBroker(initial_capital=100000)


@pytest.fixture
def sample_trades_df():
    """
    Generate sample trade history DataFrame.
    """
    trades = [
        {'timestamp': datetime.now() - timedelta(days=10), 'symbol': 'AAPL', 'side': 'BUY', 'quantity': 10, 'price': 150.00},
        {'timestamp': datetime.now() - timedelta(days=8), 'symbol': 'MSFT', 'side': 'BUY', 'quantity': 5, 'price': 300.00},
        {'timestamp': datetime.now() - timedelta(days=5), 'symbol': 'AAPL', 'side': 'SELL', 'quantity': 10, 'price': 155.00},
        {'timestamp': datetime.now() - timedelta(days=3), 'symbol': 'GOOGL', 'side': 'BUY', 'quantity': 3, 'price': 140.00},
        {'timestamp': datetime.now() - timedelta(days=1), 'symbol': 'MSFT', 'side': 'SELL', 'quantity': 5, 'price': 295.00},
    ]
    return pd.DataFrame(trades)


@pytest.fixture
def sample_positions():
    """
    Generate sample positions dictionary.
    """
    return {
        'AAPL': {
            'quantity': 10,
            'avg_cost': 150.00,
            'current_price': 155.00,
            'unrealized_pnl': 50.00,
            'realized_pnl': 0.0
        },
        'GOOGL': {
            'quantity': 5,
            'avg_cost': 140.00,
            'current_price': 142.00,
            'unrealized_pnl': 10.00,
            'realized_pnl': 0.0
        }
    }


@pytest.fixture
def sample_features_df(sample_ohlcv_data):
    """
    Generate DataFrame with features for ML model testing.
    """
    from src.data.feature_engineer import FeatureEngineer

    fe = FeatureEngineer()
    features = fe.create_features(sample_ohlcv_data)
    return features


@pytest.fixture
def small_training_data():
    """
    Generate small training dataset for quick ML model tests.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple linear decision boundary

    return X, y


@pytest.fixture
def sequence_data():
    """
    Generate sequence data for LSTM/CNN testing.
    """
    np.random.seed(42)
    n_samples = 50
    sequence_length = 20
    n_features = 10

    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randint(0, 2, n_samples)

    return X, y


@pytest.fixture
def tmp_model_dir(tmp_path):
    """
    Create temporary directory for model saving/loading tests.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def risk_config():
    """
    Sample risk management configuration.
    """
    return {
        'max_position_pct': 0.10,
        'max_portfolio_pct': 0.80,
        'stop_loss_pct': 0.05,
        'take_profit_levels': [0.05, 0.10, 0.15],
        'max_daily_loss_pct': 0.05,
        'max_drawdown_pct': 0.10
    }


@pytest.fixture
def mock_market_data():
    """
    Generate mock market data for multiple symbols.
    """
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
    data = {}

    np.random.seed(42)
    n_days = 100
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

    base_prices = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 140, 'NVDA': 450}

    for symbol in symbols:
        returns = np.random.randn(n_days) * 0.02
        prices = base_prices[symbol] * np.exp(np.cumsum(returns))

        high = prices * (1 + np.abs(np.random.randn(n_days)) * 0.01)
        low = prices * (1 - np.abs(np.random.randn(n_days)) * 0.01)
        open_prices = low + (high - low) * np.random.rand(n_days)
        volume = np.random.randint(1000000, 10000000, n_days)

        data[symbol] = pd.DataFrame({
            'open': open_prices,
            'high': high,
            'low': low,
            'close': prices,
            'volume': volume
        }, index=dates)

    return data
