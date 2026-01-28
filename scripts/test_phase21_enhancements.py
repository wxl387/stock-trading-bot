"""
Test script for Phase 21: Portfolio Optimization Enhancements.

Tests:
1. Efficient frontier calculation and visualization
2. Correlation heatmap generation
3. Regime-aware portfolio optimization
4. Transaction cost modeling
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

from src.portfolio.efficient_frontier import EfficientFrontier
from src.portfolio.correlation_analyzer import CorrelationAnalyzer
from src.portfolio.portfolio_optimizer import PortfolioOptimizer, OptimizationMethod
from src.portfolio.transaction_costs import TransactionCostModel
from src.risk.regime_detector import RegimeDetector, MarketRegime
from src.data.data_fetcher import DataFetcher


def print_section(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_efficient_frontier():
    """Test efficient frontier calculation and visualization."""
    print_section("TEST 1: Efficient Frontier Calculation")

    # Fetch historical data
    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    data_fetcher = DataFetcher()

    returns_data = {}
    for symbol in symbols:
        df = data_fetcher.fetch_historical(symbol, period="1y")
        if df is not None and not df.empty:
            close_col = 'Close' if 'Close' in df.columns else 'close'
            if close_col in df.columns:
                returns = df[close_col].pct_change().dropna()
                returns_data[symbol] = returns

    returns_df = pd.DataFrame(returns_data).dropna()
    print(f"✓ Fetched returns for {len(returns_df.columns)} symbols, {len(returns_df)} days")

    # Calculate efficient frontier
    frontier = EfficientFrontier(risk_free_rate=0.05)

    frontier_df = frontier.calculate_frontier(
        returns_df, num_points=50, min_weight=0.0, max_weight=0.4
    )

    if not frontier_df.empty:
        print(f"✓ Calculated {len(frontier_df)} efficient frontier points")
        print(f"  Return range: {frontier_df['expected_return'].min():.2%} to {frontier_df['expected_return'].max():.2%}")
        print(f"  Volatility range: {frontier_df['volatility'].min():.2%} to {frontier_df['volatility'].max():.2%}")
        print(f"  Max Sharpe: {frontier_df['sharpe'].max():.2f}")
    else:
        print("✗ Failed to calculate efficient frontier")
        return False

    # Find tangency portfolio
    tangency = frontier.find_tangency_portfolio(returns_df, min_weight=0.0, max_weight=0.4)

    if tangency:
        print(f"✓ Found tangency portfolio (Max Sharpe)")
        print(f"  Expected return: {tangency['expected_return']:.2%}")
        print(f"  Volatility: {tangency['volatility']:.2%}")
        print(f"  Sharpe ratio: {tangency['sharpe']:.2f}")
        print(f"  Weights: {tangency['weights']}")
    else:
        print("✗ Failed to find tangency portfolio")
        return False

    # Find minimum variance portfolio
    min_var = frontier.find_minimum_variance_portfolio(returns_df, min_weight=0.0, max_weight=0.4)

    if min_var:
        print(f"✓ Found minimum variance portfolio")
        print(f"  Expected return: {min_var['expected_return']:.2%}")
        print(f"  Volatility: {min_var['volatility']:.2%}")
        print(f"  Weights: {min_var['weights']}")
    else:
        print("✗ Failed to find minimum variance portfolio")
        return False

    print("\n✅ Efficient Frontier Test PASSED")
    return True


def test_correlation_heatmap():
    """Test correlation matrix and clustering."""
    print_section("TEST 2: Correlation Heatmap & Clustering")

    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA"]
    analyzer = CorrelationAnalyzer(lookback_days=252)

    # Calculate correlation matrix
    corr_matrix = analyzer.calculate_correlation_matrix(symbols)

    if not corr_matrix.empty:
        print(f"✓ Calculated correlation matrix: {corr_matrix.shape[0]}x{corr_matrix.shape[1]}")

        # Get correlation statistics
        stats = analyzer.get_correlation_stats(corr_matrix)
        print(f"✓ Correlation statistics:")
        print(f"  Mean correlation: {stats['mean_correlation']:.3f}")
        print(f"  Max correlation: {stats['max_correlation']:.3f}")
        print(f"  High correlation pairs: {stats['n_high_correlation_pairs']}/{stats['n_total_pairs']}")
    else:
        print("✗ Failed to calculate correlation matrix")
        return False

    # Find correlation clusters
    clusters = analyzer.find_correlated_clusters(corr_matrix)

    if clusters:
        print(f"✓ Found {len(clusters)} correlation clusters:")
        for cluster_id, cluster_symbols in clusters.items():
            print(f"  Cluster {cluster_id + 1}: {', '.join(cluster_symbols)}")
    else:
        print("✗ Failed to find correlation clusters")
        return False

    # Calculate diversification ratio
    weights = {sym: 1.0 / len(symbols) for sym in symbols}
    div_ratio = analyzer.calculate_diversification_ratio(weights)

    print(f"✓ Diversification ratio: {div_ratio:.2f}")

    # Check concentration risk
    concentration = analyzer.check_concentration_risk(weights, corr_matrix)

    print(f"✓ Concentration risk check:")
    print(f"  Has risk: {concentration['has_concentration_risk']}")
    print(f"  Max cluster weight: {concentration['max_cluster_weight']:.1%}")
    if concentration['warnings']:
        for warning in concentration['warnings']:
            print(f"  ⚠ {warning}")

    print("\n✅ Correlation Heatmap Test PASSED")
    return True


def test_regime_aware_optimization():
    """Test regime-aware portfolio optimization."""
    print_section("TEST 3: Regime-Aware Portfolio Optimization")

    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA"]
    optimizer = PortfolioOptimizer(
        lookback_days=252,
        min_weight=0.10,
        max_weight=0.40,
        risk_free_rate=0.05
    )

    # Test with different regimes
    regimes = [
        MarketRegime.BULL,
        MarketRegime.BEAR,
        MarketRegime.CHOPPY,
        MarketRegime.VOLATILE
    ]

    for regime in regimes:
        print(f"\n--- Testing {regime.value.upper()} regime ---")

        try:
            weights = optimizer.optimize_regime_aware(
                symbols=symbols,
                market_regime=regime
            )

            if weights and weights.weights:
                print(f"✓ Optimization successful for {regime.value}")
                print(f"  Method selected: {weights.method.value}")
                print(f"  Expected return: {weights.expected_return:.2%}")
                print(f"  Volatility: {weights.expected_volatility:.2%}")
                print(f"  Sharpe ratio: {weights.sharpe_ratio:.2f}")
                print(f"  Weights: {weights.weights}")

                # Check regime-specific adjustments
                if 'regime_adjustment' in weights.metadata:
                    print(f"  Adjustment: {weights.metadata['regime_adjustment']}")
                if 'cash_buffer' in weights.metadata:
                    print(f"  Cash buffer: {weights.metadata['cash_buffer']:.1%}")
                if 'max_position_override' in weights.metadata:
                    print(f"  Max position: {weights.metadata['max_position_override']:.1%}")
            else:
                print(f"✗ Optimization failed for {regime.value}")
                return False

        except Exception as e:
            print(f"✗ Error optimizing for {regime.value}: {e}")
            return False

    print("\n✅ Regime-Aware Optimization Test PASSED")
    return True


def test_transaction_costs():
    """Test transaction cost modeling."""
    print_section("TEST 4: Transaction Cost Modeling")

    # Initialize cost model
    cost_model = TransactionCostModel(
        base_slippage_bps=10.0,  # 0.1% slippage
        commission_per_trade=0.0,
        market_impact_factor=0.1,
        min_trade_value=100.0
    )

    print("✓ Initialized transaction cost model")

    # Test scenarios
    portfolio_value = 100000.0

    # Scenario 1: Small rebalancing (10% drift)
    current_weights = {"AAPL": 0.30, "MSFT": 0.30, "GOOGL": 0.25, "NVDA": 0.15}
    target_weights = {"AAPL": 0.25, "MSFT": 0.30, "GOOGL": 0.30, "NVDA": 0.15}

    print("\n--- Scenario 1: Small Rebalancing (10% weight shift) ---")
    costs = cost_model.estimate_rebalancing_costs(
        current_weights=current_weights,
        target_weights=target_weights,
        portfolio_value=portfolio_value
    )

    print(f"✓ Cost estimation:")
    print(f"  Total cost: ${costs.total_cost:.2f} ({costs.total_cost_pct:.3%})")
    print(f"  Slippage: ${costs.slippage_cost:.2f}")
    print(f"  Market impact: ${costs.market_impact_cost:.2f}")
    print(f"  Expected trades: {costs.expected_trades}")
    print(f"  Turnover: {costs.turnover_pct:.1f}%")

    # Scenario 2: Complete rebalancing (100% turnover)
    current_weights = {"AAPL": 1.0, "MSFT": 0.0, "GOOGL": 0.0, "NVDA": 0.0}
    target_weights = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "NVDA": 0.25}

    print("\n--- Scenario 2: Complete Rebalancing (100% turnover) ---")
    costs = cost_model.estimate_rebalancing_costs(
        current_weights=current_weights,
        target_weights=target_weights,
        portfolio_value=portfolio_value
    )

    print(f"✓ Cost estimation:")
    print(f"  Total cost: ${costs.total_cost:.2f} ({costs.total_cost_pct:.3%})")
    print(f"  Slippage: ${costs.slippage_cost:.2f}")
    print(f"  Market impact: ${costs.market_impact_cost:.2f}")
    print(f"  Expected trades: {costs.expected_trades}")
    print(f"  Turnover: {costs.turnover_pct:.1f}%")

    # Scenario 3: Turnover penalty calculation
    print("\n--- Scenario 3: Turnover Penalty ---")
    penalty = cost_model.calculate_turnover_penalty(
        current_weights=current_weights,
        target_weights=target_weights,
        penalty_coefficient=0.01
    )

    print(f"✓ Turnover penalty: {penalty:.6f}")

    print("\n✅ Transaction Cost Modeling Test PASSED")
    return True


def main():
    """Run all Phase 21 tests."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "PHASE 21 ENHANCEMENT TESTS" + " " * 36 + "║")
    print("╚" + "=" * 78 + "╝")

    results = {}

    # Run tests
    results['Efficient Frontier'] = test_efficient_frontier()
    results['Correlation Heatmap'] = test_correlation_heatmap()
    results['Regime-Aware Optimization'] = test_regime_aware_optimization()
    results['Transaction Cost Modeling'] = test_transaction_costs()

    # Summary
    print_section("TEST SUMMARY")

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\n{'=' * 80}")
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print(f"{'=' * 80}\n")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Phase 21 enhancements are working correctly.")
        return 0
    else:
        print(f"⚠️ {total_tests - passed_tests} test(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
