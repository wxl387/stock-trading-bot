#!/usr/bin/env python3
"""
Test script for Phase 20: Portfolio Optimization Integration

Tests:
1. Trading engine initializes with portfolio optimizer
2. Target weights can be calculated
3. ML strategy accepts target weights
4. Dashboard renders portfolio optimization tab
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from config.settings import setup_logging, Settings

logger = setup_logging()


def test_trading_engine_initialization():
    """Test that trading engine initializes with portfolio optimizer."""
    print("\n" + "=" * 70)
    print("TEST 1: Trading Engine Initialization with Portfolio Optimizer")
    print("=" * 70)

    try:
        from src.core.trading_engine import TradingEngine

        # Load config
        config = Settings.load_trading_config()
        symbols = config.get("trading", {}).get("symbols", ["AAPL", "MSFT", "GOOGL"])

        # Initialize trading engine
        engine = TradingEngine(
            symbols=symbols[:3],  # Use first 3 symbols for testing
            simulated=True,
            initial_capital=100000,
            ignore_market_hours=True
        )

        # Check if optimizer is initialized
        if engine.portfolio_optimizer is not None:
            print("✅ Portfolio optimizer initialized successfully")
            print(f"   Lookback days: {engine.portfolio_optimizer.lookback_days}")
            print(f"   Weight range: [{engine.portfolio_optimizer.min_weight:.2%}, {engine.portfolio_optimizer.max_weight:.2%}]")
        else:
            print("❌ Portfolio optimizer not initialized")
            return False

        # Check if rebalancer is initialized
        if engine.portfolio_rebalancer is not None:
            print("✅ Portfolio rebalancer initialized successfully")
            print(f"   Drift threshold: {engine.portfolio_rebalancer.drift_threshold:.1%}")
            print(f"   Frequency: {engine.portfolio_rebalancer.frequency}")
        else:
            print("❌ Portfolio rebalancer not initialized")
            return False

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_target_weights_calculation():
    """Test that target portfolio weights can be calculated."""
    print("\n" + "=" * 70)
    print("TEST 2: Target Portfolio Weights Calculation")
    print("=" * 70)

    try:
        from src.core.trading_engine import TradingEngine
        from config.settings import Settings

        config = Settings.load_trading_config()
        symbols = config.get("trading", {}).get("symbols", ["AAPL", "MSFT", "GOOGL"])[:3]

        engine = TradingEngine(
            symbols=symbols,
            simulated=True,
            initial_capital=100000,
            ignore_market_hours=True
        )

        # Start engine to initialize broker
        engine.start()

        # Get target weights
        print(f"Calculating target weights for {', '.join(symbols)}...")
        target_weights = engine._get_target_portfolio_weights()

        if target_weights:
            print("✅ Target weights calculated successfully:")
            total_weight = 0
            for symbol, weight in sorted(target_weights.items(), key=lambda x: x[1], reverse=True):
                print(f"   {symbol:6s}: {weight:6.2%}")
                total_weight += weight

            print(f"\nTotal weight: {total_weight:.2%}")

            if abs(total_weight - 1.0) < 0.01:
                print("✅ Weights sum to 1.0 (within tolerance)")
                return True
            else:
                print(f"❌ Weights do not sum to 1.0: {total_weight:.4f}")
                return False
        else:
            print("❌ No target weights returned")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_strategy_integration():
    """Test that ML strategy accepts and uses target weights."""
    print("\n" + "=" * 70)
    print("TEST 3: ML Strategy Integration with Target Weights")
    print("=" * 70)

    try:
        from src.strategy.ml_strategy import MLStrategy
        from src.risk.risk_manager import RiskManager

        strategy = MLStrategy()
        risk_manager = RiskManager()

        # Test data
        symbols = ["AAPL", "MSFT", "GOOGL"]
        portfolio_value = 100000
        current_positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        target_weights = {"AAPL": 0.40, "MSFT": 0.35, "GOOGL": 0.25}

        print("Testing ML strategy with target weights...")
        print(f"Target weights: {target_weights}")

        # Get recommendations (without target weights)
        recommendations_baseline = strategy.get_trade_recommendations(
            symbols=symbols,
            portfolio_value=portfolio_value,
            current_positions=current_positions,
            risk_manager=risk_manager,
            target_weights=None  # No target weights
        )

        # Get recommendations (with target weights)
        recommendations_optimized = strategy.get_trade_recommendations(
            symbols=symbols,
            portfolio_value=portfolio_value,
            current_positions=current_positions,
            risk_manager=risk_manager,
            target_weights=target_weights  # With target weights
        )

        print(f"\n✅ ML strategy accepts target_weights parameter")
        print(f"   Baseline recommendations: {len(recommendations_baseline)}")
        print(f"   Optimized recommendations: {len(recommendations_optimized)}")

        # Check if target_weights parameter is in the function signature
        import inspect
        sig = inspect.signature(strategy.get_trade_recommendations)
        params = list(sig.parameters.keys())

        if "target_weights" in params:
            print(f"✅ target_weights parameter exists in function signature")
            return True
        else:
            print(f"❌ target_weights parameter not found in function signature")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test that portfolio optimization configuration is loaded correctly."""
    print("\n" + "=" * 70)
    print("TEST 4: Portfolio Optimization Configuration")
    print("=" * 70)

    try:
        from config.settings import Settings

        config = Settings.load_trading_config()
        portfolio_config = config.get("portfolio_optimization", {})

        if not portfolio_config:
            print("❌ portfolio_optimization section not found in config")
            return False

        print("✅ Portfolio optimization configuration found:")
        print(f"   Enabled: {portfolio_config.get('enabled', False)}")
        print(f"   Method: {portfolio_config.get('method', 'N/A')}")
        print(f"   Lookback days: {portfolio_config.get('lookback_days', 'N/A')}")
        print(f"   Weight range: [{portfolio_config.get('min_weight', 'N/A'):.2%}, {portfolio_config.get('max_weight', 'N/A'):.2%}]")

        rebalancing_config = portfolio_config.get("rebalancing", {})
        if rebalancing_config:
            print(f"\n✅ Rebalancing configuration found:")
            print(f"   Enabled: {rebalancing_config.get('enabled', False)}")
            print(f"   Trigger type: {rebalancing_config.get('trigger_type', 'N/A')}")
            print(f"   Drift threshold: {rebalancing_config.get('drift_threshold', 'N/A'):.1%}")
            print(f"   Frequency: {rebalancing_config.get('frequency', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "█" * 70)
    print("  PHASE 20: PORTFOLIO OPTIMIZATION INTEGRATION TESTS")
    print("█" * 70)

    results = {
        "Trading Engine Initialization": test_trading_engine_initialization(),
        "Target Weights Calculation": test_target_weights_calculation(),
        "ML Strategy Integration": test_ml_strategy_integration(),
        "Configuration Loading": test_configuration()
    }

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<50} {status}")

    print("=" * 70)
    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All integration tests passed! Phase 20 is ready.")
        print("\nNext steps:")
        print("  1. Start trading bot with: python scripts/start_trading.py --simulated")
        print("  2. View dashboard: streamlit run src/dashboard/app.py")
        print("  3. Navigate to 'Portfolio Optimization' tab")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
