"""
Manual test script for portfolio optimization module.

Demonstrates:
- Portfolio optimization (equal-weight, max Sharpe, risk-parity, min variance)
- Correlation analysis and clustering
- Diversification metrics
- Rebalancing logic
"""

import sys
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, '/Users/wenbiaoli/Desktop/trading_bot/stock-trading-bot')

from src.portfolio.portfolio_optimizer import PortfolioOptimizer, OptimizationMethod
from src.portfolio.correlation_analyzer import CorrelationAnalyzer
from src.portfolio.rebalancer import PortfolioRebalancer, RebalanceTrigger, Position
from src.portfolio.efficient_frontier import EfficientFrontier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_portfolio_optimization():
    """Test portfolio optimization with different methods."""
    print_section("PORTFOLIO OPTIMIZATION")

    # Symbols to optimize
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    print(f"\nSymbols: {', '.join(symbols)}")

    # Create optimizer
    optimizer = PortfolioOptimizer(
        lookback_days=60,  # 60 days for faster testing
        min_weight=0.1,
        max_weight=0.4,
        risk_free_rate=0.05
    )

    print("\nOptimizer Settings:")
    print(f"  Lookback: {optimizer.lookback_days} days")
    print(f"  Weight range: {optimizer.min_weight:.1%} - {optimizer.max_weight:.1%}")
    print(f"  Risk-free rate: {optimizer.risk_free_rate:.1%}")

    # Test each optimization method
    methods = [
        OptimizationMethod.EQUAL_WEIGHT,
        OptimizationMethod.MAX_SHARPE,
        OptimizationMethod.RISK_PARITY,
        OptimizationMethod.MINIMUM_VARIANCE,
    ]

    results = {}

    for method in methods:
        print(f"\n{'─' * 80}")
        print(f"Method: {method.value.upper()}")
        print(f"{'─' * 80}")

        try:
            portfolio_weights = optimizer.optimize(
                symbols=symbols,
                method=method
            )

            results[method.value] = portfolio_weights

            # Print results
            print(f"\nOptimized Weights:")
            for symbol, weight in sorted(portfolio_weights.weights.items(),
                                        key=lambda x: x[1], reverse=True):
                print(f"  {symbol:8s}: {weight:6.2%}")

            print(f"\nPortfolio Metrics:")
            print(f"  Expected Return:     {portfolio_weights.expected_return:6.2%}")
            print(f"  Expected Volatility: {portfolio_weights.expected_volatility:6.2%}")
            print(f"  Sharpe Ratio:        {portfolio_weights.sharpe_ratio:6.2f}")

            if portfolio_weights.metadata.get('correlation_warnings'):
                print(f"\n⚠️  Correlation Warnings:")
                for warning in portfolio_weights.metadata['correlation_warnings']:
                    print(f"  - {warning}")

        except Exception as e:
            print(f"❌ Error: {e}")
            logger.exception(f"Optimization failed for {method.value}")

    return results


def test_correlation_analysis():
    """Test correlation analysis and clustering."""
    print_section("CORRELATION ANALYSIS")

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    weights = {'AAPL': 0.25, 'MSFT': 0.25, 'GOOGL': 0.20, 'AMZN': 0.15, 'NVDA': 0.15}

    print(f"\nAnalyzing: {', '.join(symbols)}")
    print(f"Portfolio weights: {weights}")

    # Create analyzer
    analyzer = CorrelationAnalyzer(
        lookback_days=60,
        correlation_threshold=0.7
    )

    try:
        # Calculate correlation matrix
        print("\n1. Calculating correlation matrix...")
        corr_matrix = analyzer.calculate_correlation_matrix(symbols)

        if not corr_matrix.empty:
            print("\nCorrelation Matrix:")
            print(corr_matrix.round(2).to_string())

            # Correlation statistics
            print("\n2. Correlation Statistics:")
            stats = analyzer.get_correlation_stats(corr_matrix)
            print(f"  Mean correlation:        {stats.get('mean_correlation', 0):.3f}")
            print(f"  Max correlation:         {stats.get('max_correlation', 0):.3f}")
            print(f"  Min correlation:         {stats.get('min_correlation', 0):.3f}")
            print(f"  High correlation pairs:  {stats.get('n_high_correlation_pairs', 0)}")

            # Find correlated clusters
            print("\n3. Correlated Asset Clusters:")
            clusters = analyzer.find_correlated_clusters(corr_matrix)
            for cluster_id, cluster_symbols in clusters.items():
                print(f"  Cluster {cluster_id}: {', '.join(cluster_symbols)}")

            # Calculate diversification ratio
            print("\n4. Diversification Metrics:")
            div_ratio = analyzer.calculate_diversification_ratio(weights)
            print(f"  Diversification Ratio: {div_ratio:.3f}")
            print(f"  (Higher is better, 1.0 = no diversification benefit)")

            # Check concentration risk
            print("\n5. Concentration Risk Check:")
            risk_result = analyzer.check_concentration_risk(
                weights,
                corr_matrix,
                max_correlated_exposure=0.5
            )

            if risk_result['has_concentration_risk']:
                print("  ⚠️  CONCENTRATION RISK DETECTED!")
                for warning in risk_result['warnings']:
                    print(f"    - {warning}")
            else:
                print("  ✅ No concentration risk detected")

            print(f"\n  Max cluster weight: {risk_result.get('max_cluster_weight', 0):.1%}")

        else:
            print("⚠️  Could not calculate correlation matrix (insufficient data)")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Correlation analysis failed")


def test_efficient_frontier():
    """Test efficient frontier calculation."""
    print_section("EFFICIENT FRONTIER")

    symbols = ['AAPL', 'MSFT', 'GOOGL']
    print(f"\nSymbols: {', '.join(symbols)}")

    # Create frontier calculator
    frontier_calc = EfficientFrontier(risk_free_rate=0.05)

    # Create sample data (for demo purposes)
    print("\n⚠️  Note: Using sample data for demonstration")
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    returns_data = {
        'AAPL': np.random.normal(0.0008, 0.02, 100),
        'MSFT': np.random.normal(0.0006, 0.018, 100),
        'GOOGL': np.random.normal(0.001, 0.022, 100)
    }
    returns = pd.DataFrame(returns_data, index=dates)

    try:
        # Calculate efficient frontier
        print("\n1. Calculating Efficient Frontier (20 points)...")
        frontier_df = frontier_calc.calculate_frontier(
            returns,
            num_points=20,
            min_weight=0.1,
            max_weight=0.6
        )

        if not frontier_df.empty:
            print(f"\n✅ Generated {len(frontier_df)} frontier points")
            print("\nFrontier Summary:")
            print(f"  Return range: {frontier_df['expected_return'].min():.2%} to "
                  f"{frontier_df['expected_return'].max():.2%}")
            print(f"  Volatility range: {frontier_df['volatility'].min():.2%} to "
                  f"{frontier_df['volatility'].max():.2%}")

            # Find tangency portfolio (max Sharpe)
            print("\n2. Finding Tangency Portfolio (Max Sharpe)...")
            tangency = frontier_calc.find_tangency_portfolio(returns)

            if tangency:
                print("\nTangency Portfolio:")
                print("  Weights:")
                for symbol, weight in tangency['weights'].items():
                    print(f"    {symbol:8s}: {weight:6.2%}")
                print(f"\n  Expected Return:  {tangency['expected_return']:6.2%}")
                print(f"  Volatility:       {tangency['volatility']:6.2%}")
                print(f"  Sharpe Ratio:     {tangency['sharpe']:6.2f}")

            # Find minimum variance portfolio
            print("\n3. Finding Minimum Variance Portfolio...")
            min_var = frontier_calc.find_minimum_variance_portfolio(returns)

            if min_var:
                print("\nMinimum Variance Portfolio:")
                print("  Weights:")
                for symbol, weight in min_var['weights'].items():
                    print(f"    {symbol:8s}: {weight:6.2%}")
                print(f"\n  Expected Return:  {min_var['expected_return']:6.2%}")
                print(f"  Volatility:       {min_var['volatility']:6.2%}")
                print(f"  Sharpe Ratio:     {min_var['sharpe']:6.2f}")

        else:
            print("⚠️  Could not calculate efficient frontier")

    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Efficient frontier calculation failed")


def test_rebalancing():
    """Test portfolio rebalancing logic."""
    print_section("PORTFOLIO REBALANCING")

    # Create current portfolio positions
    current_positions = {
        'AAPL': Position('AAPL', 10, 150.0, 1500.0, 0.30),
        'MSFT': Position('MSFT', 12, 250.0, 3000.0, 0.60),
        'GOOGL': Position('GOOGL', 5, 100.0, 500.0, 0.10),
    }
    portfolio_value = 5000.0

    print("\nCurrent Portfolio:")
    print(f"  Total Value: ${portfolio_value:,.2f}")
    print("\n  Holdings:")
    for symbol, pos in current_positions.items():
        print(f"    {symbol:8s}: {pos.shares:3d} shares @ ${pos.price:6.2f} = "
              f"${pos.value:8.2f} ({pos.weight:5.1%})")

    # Target weights (from optimization)
    target_weights = {
        'AAPL': 0.35,  # Increase from 30%
        'MSFT': 0.45,  # Decrease from 60%
        'GOOGL': 0.20  # Increase from 10%
    }

    print("\nTarget Weights:")
    for symbol, weight in target_weights.items():
        current_weight = current_positions[symbol].weight
        diff = weight - current_weight
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"  {symbol:8s}: {weight:5.1%} {arrow} (currently {current_weight:5.1%})")

    # Create rebalancer
    rebalancer = PortfolioRebalancer(
        drift_threshold=0.05,  # 5% drift threshold
        min_trade_value=100.0,
        calendar_frequency="weekly",
        trigger_type=RebalanceTrigger.THRESHOLD
    )

    print("\nRebalancer Settings:")
    print(f"  Trigger type: {rebalancer.trigger_type.value}")
    print(f"  Drift threshold: {rebalancer.drift_threshold:.1%}")
    print(f"  Min trade value: ${rebalancer.min_trade_value:.2f}")

    # Check if rebalancing is needed
    print("\n" + "─" * 80)
    print("Checking Rebalancing Triggers...")
    print("─" * 80)

    current_prices = {symbol: pos.price for symbol, pos in current_positions.items()}

    signal = rebalancer.check_rebalance_needed(
        current_positions,
        target_weights,
        portfolio_value,
        current_prices
    )

    print(f"\nDrift Analysis:")
    print(f"  Maximum drift: {signal.drift_pct:.2%}")
    print(f"  Threshold: {rebalancer.drift_threshold:.2%}")

    if signal.should_rebalance:
        print(f"\n✅ REBALANCING RECOMMENDED")
        print(f"   Reason: {signal.reason}")

        if signal.trades_needed:
            print(f"\n📊 Rebalancing Trades ({len(signal.trades_needed)} orders):")
            print("\n  " + "─" * 76)
            print(f"  {'Symbol':<8} {'Action':<6} {'Shares':>7} {'Price':>8} {'Value':>10} {'Weight Change'}")
            print("  " + "─" * 76)

            total_buy = 0
            total_sell = 0

            for trade in signal.trades_needed:
                weight_change = f"{trade['current_weight']:.1%} → {trade['target_weight']:.1%}"
                print(f"  {trade['symbol']:<8} {trade['action']:<6} "
                      f"{trade['shares']:7d} ${trade['price']:7.2f} "
                      f"${trade['value']:9.2f} {weight_change}")

                if trade['action'] == 'BUY':
                    total_buy += trade['value']
                else:
                    total_sell += trade['value']

            print("  " + "─" * 76)
            print(f"\n  Total BUY orders:  ${total_buy:,.2f}")
            print(f"  Total SELL orders: ${total_sell:,.2f}")
            print(f"  Net cash flow:     ${total_buy - total_sell:+,.2f}")

            # Record the rebalance
            rebalancer.record_rebalance()
            print(f"\n✓ Rebalance recorded at {rebalancer.last_rebalance}")

        else:
            print("\n⚠️  No trades generated (all adjustments below minimum trade value)")

    else:
        print(f"\n❌ NO REBALANCING NEEDED")
        print(f"   Drift ({signal.drift_pct:.2%}) is below threshold ({rebalancer.drift_threshold:.2%})")


def test_signal_tilting():
    """Test ML signal incorporation into portfolio weights."""
    print_section("ML SIGNAL TILTING")

    symbols = ['AAPL', 'MSFT', 'GOOGL']
    print(f"\nSymbols: {', '.join(symbols)}")

    # Create optimizer
    optimizer = PortfolioOptimizer(
        lookback_days=60,
        min_weight=0.2,
        max_weight=0.5,
        risk_free_rate=0.05
    )

    # Create mock signals
    from unittest.mock import Mock

    signals = {
        'AAPL': Mock(signal=Mock(value='BUY'), confidence=0.85),   # Strong buy
        'MSFT': Mock(signal=Mock(value='HOLD'), confidence=0.50),  # Neutral
        'GOOGL': Mock(signal=Mock(value='SELL'), confidence=0.70)  # Moderate sell
    }

    print("\nML Trading Signals:")
    for symbol, signal in signals.items():
        signal_type = signal.signal.value
        confidence = signal.confidence
        emoji = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"
        print(f"  {emoji} {symbol:8s}: {signal_type:4s} (confidence: {confidence:.0%})")

    # Optimize without signals
    print("\n1. Optimization WITHOUT Signals:")
    base_weights = optimizer.optimize(
        symbols=symbols,
        method=OptimizationMethod.MAX_SHARPE,
        signals=None
    )

    print("  Base Weights:")
    for symbol, weight in sorted(base_weights.weights.items()):
        print(f"    {symbol:8s}: {weight:6.2%}")

    # Optimize with signals
    print("\n2. Optimization WITH Signals:")
    tilted_weights = optimizer.optimize(
        symbols=symbols,
        method=OptimizationMethod.MAX_SHARPE,
        signals=signals
    )

    print("  Tilted Weights:")
    for symbol, weight in sorted(tilted_weights.weights.items()):
        base_w = base_weights.weights[symbol]
        diff = weight - base_w
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"    {symbol:8s}: {weight:6.2%} {arrow} ({diff:+.2%} from base)")

    print("\n3. Impact Analysis:")
    print(f"  AAPL (BUY signal): "
          f"{base_weights.weights['AAPL']:.2%} → {tilted_weights.weights['AAPL']:.2%}")
    print(f"  GOOGL (SELL signal): "
          f"{base_weights.weights['GOOGL']:.2%} → {tilted_weights.weights['GOOGL']:.2%}")

    if tilted_weights.metadata.get('signals_incorporated'):
        print("\n✅ Signals successfully incorporated into portfolio weights")


def main():
    """Run all tests."""
    print("\n" + "▓" * 80)
    print("  PORTFOLIO OPTIMIZATION MODULE - MANUAL TEST SUITE")
    print("▓" * 80)
    print(f"\nTest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script demonstrates the complete portfolio optimization pipeline:")
    print("  • Multiple optimization methods (equal-weight, max Sharpe, risk-parity, min variance)")
    print("  • Correlation analysis and asset clustering")
    print("  • Diversification metrics")
    print("  • Rebalancing logic with drift detection")
    print("  • ML signal integration")
    print("  • Efficient frontier calculation")

    try:
        # Run tests
        test_portfolio_optimization()
        test_correlation_analysis()
        test_efficient_frontier()
        test_rebalancing()
        test_signal_tilting()

        # Summary
        print_section("TEST SUMMARY")
        print("\n✅ All manual tests completed successfully!")
        print("\nNext Steps:")
        print("  1. Integrate with trading engine (Phase 4)")
        print("  2. Add backtesting support (Phase 5)")
        print("  3. Build dashboard visualization (Phase 6)")

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        logger.exception("Test suite failed")
        raise


if __name__ == '__main__':
    main()
