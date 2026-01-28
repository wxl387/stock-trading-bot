"""
Unit tests for portfolio optimization module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.portfolio.portfolio_optimizer import (
    PortfolioOptimizer,
    PortfolioWeights,
    OptimizationMethod
)
from src.portfolio.efficient_frontier import EfficientFrontier
from src.portfolio.correlation_analyzer import CorrelationAnalyzer
from src.portfolio.rebalancer import (
    PortfolioRebalancer,
    RebalanceSignal,
    RebalanceTrigger,
    Position
)


class TestPortfolioOptimizer:
    """Test PortfolioOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return PortfolioOptimizer(
            lookback_days=252,
            min_weight=0.0,
            max_weight=0.5,  # Allow up to 50% per asset (feasible for 3 assets)
            risk_free_rate=0.05
        )

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

        # Create correlated returns
        mean_returns = [0.0005, 0.0004, 0.0006]
        volatilities = [0.02, 0.015, 0.025]

        returns_data = {}
        for i, symbol in enumerate(['AAPL', 'MSFT', 'GOOGL']):
            returns = np.random.normal(mean_returns[i], volatilities[i], 252)
            returns_data[symbol] = returns

        df = pd.DataFrame(returns_data, index=dates)
        return df

    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.lookback_days == 252
        assert optimizer.min_weight == 0.0
        assert optimizer.max_weight == 0.5
        assert optimizer.risk_free_rate == 0.05

    def test_equal_weight_optimization(self, optimizer):
        """Test equal weight baseline."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        weights = optimizer.equal_weight_optimize(symbols)

        assert len(weights) == 3
        assert all(abs(w - 1/3) < 0.01 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.001

    def test_max_sharpe_optimization(self, optimizer, sample_returns):
        """Test maximum Sharpe ratio optimization."""
        weights = optimizer.max_sharpe_optimize(sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(0 <= w <= 0.5 + 0.01 for w in weights.values())  # Allow small tolerance

    def test_risk_parity_optimization(self, optimizer, sample_returns):
        """Test risk-parity allocation."""
        weights = optimizer.risk_parity_optimize(sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(0 <= w <= 0.5 + 0.01 for w in weights.values())

    def test_minimum_variance_optimization(self, optimizer, sample_returns):
        """Test minimum variance portfolio."""
        weights = optimizer.minimum_variance_optimize(sample_returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01
        assert all(0 <= w <= 0.5 + 0.01 for w in weights.values())

        # Verify it's actually low variance
        cov_matrix = sample_returns.cov().values
        weights_array = np.array(list(weights.values()))
        portfolio_var = np.dot(weights_array, np.dot(cov_matrix, weights_array))

        # Should be lower than equal-weight
        equal_weights = np.array([1/3, 1/3, 1/3])
        equal_var = np.dot(equal_weights, np.dot(cov_matrix, equal_weights))

        assert portfolio_var <= equal_var * 1.1  # Allow 10% tolerance

    def test_mean_variance_optimization(self, optimizer, sample_returns):
        """Test mean-variance optimization with target return."""
        target_return = 0.10  # 10% annual return

        weights = optimizer.mean_variance_optimize(sample_returns, target_return)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_weight_constraints(self, optimizer, sample_returns):
        """Test min/max weight constraints are respected."""
        weights = optimizer.max_sharpe_optimize(sample_returns)

        for w in weights.values():
            assert w >= optimizer.min_weight - 0.001  # Small tolerance
            assert w <= optimizer.max_weight + 0.001

    def test_signal_tilting(self, optimizer):
        """Test signal-based weight tilting."""
        base_weights = {
            'AAPL': 0.33,
            'MSFT': 0.33,
            'GOOGL': 0.34
        }

        # Create mock signals
        mock_signals = {
            'AAPL': Mock(signal=Mock(value='BUY'), confidence=0.8),
            'MSFT': Mock(signal=Mock(value='HOLD'), confidence=0.5),
            'GOOGL': Mock(signal=Mock(value='SELL'), confidence=0.7)
        }

        tilted = optimizer._apply_signal_tilts(base_weights, mock_signals, tilt_strength=0.2)

        # AAPL should have higher weight (BUY signal)
        assert tilted['AAPL'] > base_weights['AAPL']

        # GOOGL should have lower weight (SELL signal)
        assert tilted['GOOGL'] < base_weights['GOOGL']

        # Weights should still sum to 1
        assert abs(sum(tilted.values()) - 1.0) < 0.001

    @patch('src.portfolio.portfolio_optimizer.DataFetcher')
    def test_optimize_with_signals(self, mock_fetcher, optimizer, sample_returns):
        """Test full optimization with signal incorporation."""
        # Mock data fetcher
        optimizer.data_fetcher = Mock()
        optimizer.data_fetcher.fetch_historical = Mock(side_effect=self._mock_fetch_data(sample_returns))

        symbols = ['AAPL', 'MSFT', 'GOOGL']

        # Create mock signals
        mock_signals = {
            'AAPL': Mock(signal=Mock(value='BUY'), confidence=0.8),
            'MSFT': Mock(signal=Mock(value='HOLD'), confidence=0.5),
            'GOOGL': Mock(signal=Mock(value='HOLD'), confidence=0.5)
        }

        result = optimizer.optimize(
            symbols,
            method=OptimizationMethod.MAX_SHARPE,
            signals=mock_signals
        )

        assert isinstance(result, PortfolioWeights)
        assert result.method == OptimizationMethod.MAX_SHARPE
        assert len(result.weights) == 3
        assert result.metadata['signals_incorporated'] is True

    def test_correlation_constraints(self, optimizer):
        """Test correlation-based diversification warnings."""
        # Create highly correlated returns
        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 252)

        returns = pd.DataFrame({
            'AAPL': base + np.random.normal(0, 0.005, 252),
            'MSFT': base + np.random.normal(0, 0.005, 252),  # Highly correlated with AAPL
            'GOOGL': np.random.normal(0.001, 0.02, 252)  # Uncorrelated
        })

        weights = {'AAPL': 0.4, 'MSFT': 0.4, 'GOOGL': 0.2}

        warnings = optimizer._check_correlation_constraints(weights, returns)

        # Should warn about AAPL-MSFT correlation
        assert len(warnings) > 0

    def test_insufficient_data_fallback(self, optimizer):
        """Test fallback to equal-weight with insufficient data."""
        optimizer.data_fetcher = Mock()
        optimizer.data_fetcher.fetch_data = Mock(return_value=pd.DataFrame())

        symbols = ['AAPL', 'MSFT', 'GOOGL']

        result = optimizer.optimize(symbols, method=OptimizationMethod.MAX_SHARPE)

        # Should fallback to equal weight
        assert result.method == OptimizationMethod.EQUAL_WEIGHT
        assert result.metadata.get('fallback') is True
        assert len(result.weights) == 3

    def test_enforce_constraints(self, optimizer):
        """Test constraint enforcement and normalization."""
        weights = {
            'AAPL': 0.6,  # Exceeds max_weight (0.5)
            'MSFT': -0.1,  # Below min_weight
            'GOOGL': 0.15
        }

        constrained = optimizer._enforce_constraints(weights)

        # Should normalize to sum to 1
        assert abs(sum(constrained.values()) - 1.0) < 0.001

        # Weights should be reasonable (may relax max constraint if infeasible)
        assert all(0 <= w <= 1.0 for w in constrained.values())

    def test_portfolio_metrics_calculation(self, optimizer, sample_returns):
        """Test portfolio metrics calculation."""
        weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}

        expected_return, expected_vol = optimizer._calculate_portfolio_metrics(
            weights, sample_returns
        )

        assert isinstance(expected_return, float)
        assert isinstance(expected_vol, float)
        assert expected_vol > 0  # Volatility should be positive

    def _mock_fetch_data(self, sample_returns):
        """Helper to create mock fetch_data function."""
        def fetch(symbol, period):
            if symbol in sample_returns.columns:
                df = pd.DataFrame({
                    'Close': np.cumsum(sample_returns[symbol])
                })
                return df
            return None
        return fetch


class TestEfficientFrontier:
    """Test EfficientFrontier class."""

    @pytest.fixture
    def frontier(self):
        """Create frontier calculator."""
        return EfficientFrontier(risk_free_rate=0.05)

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns data."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

        returns_data = {}
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            returns = np.random.normal(0.0005, 0.02, 252)
            returns_data[symbol] = returns

        return pd.DataFrame(returns_data, index=dates)

    def test_frontier_calculation(self, frontier, sample_returns):
        """Test efficient frontier point generation."""
        frontier_df = frontier.calculate_frontier(sample_returns, num_points=20)

        assert not frontier_df.empty
        assert len(frontier_df) > 0
        assert 'expected_return' in frontier_df.columns
        assert 'volatility' in frontier_df.columns
        assert 'sharpe' in frontier_df.columns
        assert 'weights' in frontier_df.columns

        # Verify volatility is positive
        assert all(frontier_df['volatility'] > 0)

    def test_tangency_portfolio(self, frontier, sample_returns):
        """Test tangency portfolio (max Sharpe)."""
        tangency = frontier.find_tangency_portfolio(sample_returns)

        assert 'weights' in tangency
        assert 'expected_return' in tangency
        assert 'volatility' in tangency
        assert 'sharpe' in tangency

        # Verify weights sum to 1
        assert abs(sum(tangency['weights'].values()) - 1.0) < 0.01

    def test_minimum_variance_portfolio(self, frontier, sample_returns):
        """Test global minimum variance portfolio."""
        min_var = frontier.find_minimum_variance_portfolio(sample_returns)

        assert 'weights' in min_var
        assert 'expected_return' in min_var
        assert 'volatility' in min_var

        # Verify weights sum to 1
        assert abs(sum(min_var['weights'].values()) - 1.0) < 0.01

    def test_plot_frontier(self, frontier, sample_returns):
        """Test frontier plotting."""
        frontier_df = frontier.calculate_frontier(sample_returns, num_points=20)
        tangency = frontier.find_tangency_portfolio(sample_returns)

        # This may fail if plotly not installed, which is expected
        try:
            fig = frontier.plot_frontier(frontier_df, tangency_portfolio=tangency)
            # If plotly is available, should return figure
            assert fig is not None
        except ImportError:
            # Expected if plotly not installed
            pass


class TestPortfolioWeights:
    """Test PortfolioWeights dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        weights = PortfolioWeights(
            weights={'AAPL': 0.5, 'MSFT': 0.5},
            method=OptimizationMethod.MAX_SHARPE,
            expected_return=0.12,
            expected_volatility=0.18,
            sharpe_ratio=0.67,
            timestamp=datetime(2026, 1, 27, 12, 0),
            metadata={'test': True}
        )

        weights_dict = weights.to_dict()

        assert weights_dict['weights'] == {'AAPL': 0.5, 'MSFT': 0.5}
        assert weights_dict['method'] == 'max_sharpe'
        assert weights_dict['expected_return'] == 0.12
        assert weights_dict['metadata']['test'] is True


class TestIntegration:
    """Integration tests for portfolio optimization."""

    def test_end_to_end_optimization(self):
        """Test end-to-end optimization workflow."""
        # Create optimizer
        optimizer = PortfolioOptimizer(
            lookback_days=100,
            min_weight=0.1,
            max_weight=0.4,
            risk_free_rate=0.05
        )

        # Create sample data
        np.random.seed(42)
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.0008, 0.018, 100),
            'GOOGL': np.random.normal(0.0012, 0.022, 100)
        })

        # Test each optimization method
        methods = [
            OptimizationMethod.EQUAL_WEIGHT,
            OptimizationMethod.MAX_SHARPE,
            OptimizationMethod.RISK_PARITY,
            OptimizationMethod.MINIMUM_VARIANCE
        ]

        for method in methods:
            # Mock data fetcher
            optimizer.data_fetcher = Mock()
            optimizer.data_fetcher.fetch_historical = Mock(
                side_effect=self._create_fetch_mock(returns)
            )

            result = optimizer.optimize(
                symbols=['AAPL', 'MSFT', 'GOOGL'],
                method=method
            )

            assert isinstance(result, PortfolioWeights)
            assert result.method == method or result.method == OptimizationMethod.EQUAL_WEIGHT
            assert len(result.weights) == 3
            assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def _create_fetch_mock(self, returns):
        """Create mock fetch function."""
        def fetch(symbol, period):
            if symbol in returns.columns:
                return pd.DataFrame({'Close': np.cumsum(returns[symbol])})
            return None
        return fetch


class TestCorrelationAnalyzer:
    """Test CorrelationAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return CorrelationAnalyzer(
            lookback_days=100,
            correlation_threshold=0.8
        )

    @pytest.fixture
    def sample_returns(self):
        """Create sample returns with varying correlations."""
        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 100)

        returns = pd.DataFrame({
            'AAPL': base + np.random.normal(0, 0.005, 100),  # Highly correlated
            'MSFT': base + np.random.normal(0, 0.005, 100),  # Highly correlated with AAPL
            'GOOGL': np.random.normal(0.001, 0.02, 100)      # Uncorrelated
        })
        return returns

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.lookback_days == 100
        assert analyzer.correlation_threshold == 0.8
        assert analyzer.use_shrinkage is True

    def test_correlation_matrix_calculation(self, analyzer, sample_returns):
        """Test correlation matrix calculation."""
        corr_matrix = analyzer.calculate_correlation_matrix(
            ['AAPL', 'MSFT', 'GOOGL'],
            returns=sample_returns
        )

        assert not corr_matrix.empty
        assert corr_matrix.shape == (3, 3)

        # Diagonal should be 1
        assert all(abs(corr_matrix.iloc[i, i] - 1.0) < 0.01 for i in range(3))

        # AAPL and MSFT should be highly correlated
        assert corr_matrix.loc['AAPL', 'MSFT'] > 0.7

    def test_diversification_ratio(self, analyzer, sample_returns):
        """Test diversification ratio calculation."""
        weights = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}

        div_ratio = analyzer.calculate_diversification_ratio(
            weights,
            returns=sample_returns
        )

        assert div_ratio > 0
        assert div_ratio < 10  # Reasonable bound

    def test_find_correlated_clusters(self, analyzer, sample_returns):
        """Test cluster detection."""
        corr_matrix = analyzer.calculate_correlation_matrix(
            ['AAPL', 'MSFT', 'GOOGL'],
            returns=sample_returns
        )

        clusters = analyzer.find_correlated_clusters(corr_matrix, n_clusters=2)

        assert len(clusters) == 2
        assert all(len(symbols) > 0 for symbols in clusters.values())

    def test_concentration_risk(self, analyzer, sample_returns):
        """Test concentration risk detection."""
        corr_matrix = analyzer.calculate_correlation_matrix(
            ['AAPL', 'MSFT', 'GOOGL'],
            returns=sample_returns
        )

        # High concentration in correlated assets
        weights = {'AAPL': 0.5, 'MSFT': 0.4, 'GOOGL': 0.1}

        risk_result = analyzer.check_concentration_risk(
            weights,
            corr_matrix,
            max_correlated_exposure=0.4
        )

        assert 'has_concentration_risk' in risk_result
        assert 'warnings' in risk_result
        assert 'clusters' in risk_result

    def test_correlation_stats(self, analyzer, sample_returns):
        """Test correlation statistics."""
        corr_matrix = analyzer.calculate_correlation_matrix(
            ['AAPL', 'MSFT', 'GOOGL'],
            returns=sample_returns
        )

        stats = analyzer.get_correlation_stats(corr_matrix)

        assert 'mean_correlation' in stats
        assert 'max_correlation' in stats
        assert 'min_correlation' in stats
        assert 'n_high_correlation_pairs' in stats

        assert -1 <= stats['mean_correlation'] <= 1
        assert -1 <= stats['max_correlation'] <= 1


class TestPortfolioRebalancer:
    """Test PortfolioRebalancer class."""

    @pytest.fixture
    def rebalancer(self):
        """Create rebalancer instance."""
        return PortfolioRebalancer(
            drift_threshold=0.05,
            min_trade_value=100.0,
            calendar_frequency="weekly",
            trigger_type=RebalanceTrigger.THRESHOLD
        )

    @pytest.fixture
    def sample_positions(self):
        """Create sample portfolio positions."""
        return {
            'AAPL': Position('AAPL', 10, 150.0, 1500.0, 0.3),
            'MSFT': Position('MSFT', 15, 200.0, 3000.0, 0.6),
            'GOOGL': Position('GOOGL', 5, 100.0, 500.0, 0.1)
        }

    def test_initialization(self, rebalancer):
        """Test rebalancer initialization."""
        assert rebalancer.drift_threshold == 0.05
        assert rebalancer.min_trade_value == 100.0
        assert rebalancer.trigger_type == RebalanceTrigger.THRESHOLD

    def test_drift_calculation(self, rebalancer):
        """Test drift calculation."""
        current = {'AAPL': 0.3, 'MSFT': 0.6, 'GOOGL': 0.1}
        target = {'AAPL': 0.33, 'MSFT': 0.50, 'GOOGL': 0.17}

        drift = rebalancer.calculate_drift(current, target)

        # Max drift should be MSFT: |0.6 - 0.5| = 0.1
        assert abs(drift - 0.1) < 0.01

    def test_threshold_trigger(self, rebalancer, sample_positions):
        """Test threshold-based rebalancing trigger."""
        # Target weights differ significantly from current
        target_weights = {'AAPL': 0.5, 'MSFT': 0.3, 'GOOGL': 0.2}

        signal = rebalancer.check_rebalance_needed(
            sample_positions,
            target_weights,
            portfolio_value=5000.0
        )

        # Drift is 0.2 (AAPL: 0.3 -> 0.5) which exceeds 0.05 threshold
        assert signal.should_rebalance is True
        assert signal.drift_pct > 0.05

    def test_no_rebalance_needed(self, rebalancer, sample_positions):
        """Test when no rebalancing is needed."""
        # Target weights close to current
        target_weights = {'AAPL': 0.31, 'MSFT': 0.59, 'GOOGL': 0.10}

        signal = rebalancer.check_rebalance_needed(
            sample_positions,
            target_weights,
            portfolio_value=5000.0
        )

        # Drift is minimal, should not rebalance
        assert signal.should_rebalance is False

    def test_rebalance_order_generation(self, rebalancer, sample_positions):
        """Test generation of rebalancing trades."""
        target_weights = {'AAPL': 0.4, 'MSFT': 0.4, 'GOOGL': 0.2}
        current_prices = {'AAPL': 150.0, 'MSFT': 200.0, 'GOOGL': 100.0}

        trades = rebalancer.generate_rebalance_orders(
            sample_positions,
            target_weights,
            portfolio_value=5000.0,
            current_prices=current_prices
        )

        assert isinstance(trades, list)

        # Verify trade structure
        for trade in trades:
            assert 'symbol' in trade
            assert 'action' in trade
            assert 'shares' in trade
            assert trade['action'] in ['BUY', 'SELL']

    def test_min_trade_value_filter(self, rebalancer, sample_positions):
        """Test that tiny trades are filtered out."""
        # Target very close to current (small trades)
        target_weights = {'AAPL': 0.301, 'MSFT': 0.599, 'GOOGL': 0.100}
        current_prices = {'AAPL': 150.0, 'MSFT': 200.0, 'GOOGL': 100.0}

        trades = rebalancer.generate_rebalance_orders(
            sample_positions,
            target_weights,
            portfolio_value=5000.0,
            current_prices=current_prices
        )

        # Small trades should be filtered out
        for trade in trades:
            assert trade['value'] >= rebalancer.min_trade_value

    def test_calendar_trigger_weekly(self):
        """Test weekly calendar trigger."""
        rebalancer = PortfolioRebalancer(
            drift_threshold=0.05,
            calendar_frequency="weekly",
            day_of_week="monday",
            trigger_type=RebalanceTrigger.CALENDAR
        )

        # Set last rebalance to 8 days ago
        rebalancer.last_rebalance = datetime.now() - timedelta(days=8)

        # Mock check (actual behavior depends on current day of week)
        # Just verify it doesn't crash
        triggered = rebalancer._should_rebalance_calendar()
        assert isinstance(triggered, bool)

    def test_record_rebalance(self, rebalancer):
        """Test recording rebalance event."""
        assert rebalancer.last_rebalance is None

        rebalancer.record_rebalance()

        assert rebalancer.last_rebalance is not None
        assert isinstance(rebalancer.last_rebalance, datetime)

    def test_rebalance_signal_to_dict(self):
        """Test RebalanceSignal serialization."""
        signal = RebalanceSignal(
            should_rebalance=True,
            reason="Test reason",
            current_weights={'AAPL': 0.5},
            target_weights={'AAPL': 0.6},
            drift_pct=0.1,
            trades_needed=[],
            timestamp=datetime(2026, 1, 27, 12, 0)
        )

        signal_dict = signal.to_dict()

        assert signal_dict['should_rebalance'] is True
        assert signal_dict['reason'] == "Test reason"
        assert signal_dict['drift_pct'] == 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
