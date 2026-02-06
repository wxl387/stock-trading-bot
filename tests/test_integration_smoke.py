"""
Integration smoke tests for the stock trading bot.

These tests exercise the main components end-to-end using the simulated broker
and mocked ML strategy so they run without trained models, network access, or
real broker credentials.  Every test is independent and targets < 5 s wall time.

Run with:
    pytest tests/test_integration_smoke.py -v -m integration
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.broker.base_broker import OrderSide, OrderStatus, OrderType
from src.broker.simulated_broker import SimulatedBroker
from src.risk.risk_manager import (
    RiskManager,
    StopLoss,
    StopLossType,
    TakeProfitOrder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_TRADING_CONFIG = {
    "ml_model": {
        "confidence_threshold": 0.55,
        "min_confidence_sell": 0.55,
    },
    "risk_management": {
        "max_position_pct": 0.10,
        "max_daily_loss_pct": 0.05,
        "max_total_exposure": 0.80,
        "volatility_sizing": {"enabled": False},
        "drawdown_protection": {"enabled": False},
        "regime_detection": {"enabled": False},
        "take_profit": {
            "enabled": True,
            "levels": [(0.05, 0.33), (0.10, 0.50), (0.15, 1.0)],
        },
    },
    "retraining": {"enabled": False},
    "notifications": {"enabled": False},
    "portfolio_optimization": {"enabled": False},
    "agents": {"enabled": False},
    "dynamic_symbols": {"enabled": False},
}


def _make_broker_with_position(symbol: str, qty: int, price: float,
                                initial_capital: float = 100_000.0):
    """Return a SimulatedBroker that already holds *qty* shares of *symbol*."""
    broker = SimulatedBroker.__new__(SimulatedBroker)
    broker.initial_capital = initial_capital
    broker.cash = initial_capital - (price * qty)
    broker.positions = {
        symbol: {
            "quantity": qty,
            "avg_cost": price,
            "realized_pnl": 0.0,
            "last_price": price,
        }
    }
    broker.orders = {}
    broker.trades = []
    broker.realized_pnl = 0.0
    broker._connected = True
    broker._account_id = "SIM-SMOKETEST"
    return broker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_state_dir(tmp_path):
    """Return a fresh temporary directory for risk-manager state files."""
    return tmp_path


@pytest.fixture()
def isolated_broker(tmp_path):
    """SimulatedBroker whose state file lives inside a tmp directory."""
    state_file = tmp_path / "simulated_broker_state.json"
    with patch.object(SimulatedBroker, "STATE_FILE", state_file):
        broker = SimulatedBroker(initial_capital=100_000.0)
        yield broker


# ---------------------------------------------------------------------------
# 1. Engine lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEngineLifecycle:
    """Create a TradingEngine, start it, run cycles, stop it."""

    @patch("src.core.trading_engine.Settings.load_trading_config",
           return_value=MINIMAL_TRADING_CONFIG)
    def test_start_run_stop(self, _mock_cfg, tmp_path):
        """Engine boots, runs 2 cycles, then shuts down without error."""
        # Patch RiskManager state file to tmp directory
        risk_state = tmp_path / "risk_manager_state.json"
        broker_state = tmp_path / "simulated_broker_state.json"

        with (
            patch.object(RiskManager, "STATE_FILE", risk_state),
            patch.object(SimulatedBroker, "STATE_FILE", broker_state),
        ):
            from src.core.trading_engine import TradingEngine

            engine = TradingEngine(
                symbols=["AAPL", "MSFT"],
                simulated=True,
                initial_capital=100_000.0,
                ignore_market_hours=True,
            )

            # The model will not be found -- that is expected; the engine
            # sets self.model_loaded = False and skips signal generation.
            engine.start()
            assert engine.is_running is True

            # Mock broker price lookups so no network calls happen.
            fake_quote = MagicMock()
            fake_quote.last = 150.0
            engine.broker.get_quote = MagicMock(return_value=fake_quote)
            engine.broker.get_quotes = MagicMock(
                return_value={"AAPL": fake_quote, "MSFT": fake_quote}
            )

            r1 = engine.run_trading_cycle()
            r2 = engine.run_trading_cycle()
            assert r1["status"] in ("success", "blocked")
            assert r2["status"] in ("success", "blocked")

            engine.stop()
            assert engine.is_running is False

    @patch("src.core.trading_engine.Settings.load_trading_config",
           return_value=MINIMAL_TRADING_CONFIG)
    def test_status_after_start(self, _mock_cfg, tmp_path):
        """get_status() returns well-formed dict while engine is running."""
        risk_state = tmp_path / "risk_manager_state.json"
        broker_state = tmp_path / "simulated_broker_state.json"

        with (
            patch.object(RiskManager, "STATE_FILE", risk_state),
            patch.object(SimulatedBroker, "STATE_FILE", broker_state),
        ):
            from src.core.trading_engine import TradingEngine

            engine = TradingEngine(
                symbols=["AAPL"],
                simulated=True,
                initial_capital=50_000.0,
                ignore_market_hours=True,
            )
            engine.start()

            status = engine.get_status()
            assert status["is_running"] is True
            assert "account" in status
            assert status["account"]["portfolio_value"] == pytest.approx(
                50_000.0, rel=0.01
            )

            engine.stop()


# ---------------------------------------------------------------------------
# 2. RiskManager round-trip persistence
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRiskManagerRoundTrip:
    """Persist and reload stop-losses / take-profits across instances."""

    def test_save_and_load_state(self, tmp_state_dir):
        state_file = tmp_state_dir / "risk_manager_state.json"

        with patch.object(RiskManager, "STATE_FILE", state_file):
            rm = RiskManager(max_position_pct=0.10)

            # Set a trailing stop-loss
            rm.set_stop_loss(
                symbol="AAPL",
                entry_price=150.0,
                stop_type=StopLossType.TRAILING,
                stop_price=142.50,
            )

            # Set take-profit levels
            rm.set_take_profit(
                symbol="AAPL",
                entry_price=150.0,
                quantity=100,
                tp_levels=[(0.05, 0.33), (0.10, 0.50), (0.15, 1.0)],
            )

            # Set a second symbol
            rm.set_stop_loss(
                symbol="MSFT",
                entry_price=300.0,
                stop_type=StopLossType.FIXED,
                stop_price=285.0,
            )

            # Track drawdown peak
            rm.peak_portfolio_value = 200_000.0
            rm.drawdown_recovery_mode = True

            rm.save_state()
            assert state_file.exists()

        # --- Create a brand-new RiskManager and reload ---
        with patch.object(RiskManager, "STATE_FILE", state_file):
            rm2 = RiskManager(max_position_pct=0.10)
            rm2.load_state()

            # Verify stop-losses
            assert "AAPL" in rm2.stop_losses
            assert "MSFT" in rm2.stop_losses
            assert rm2.stop_losses["AAPL"].stop_price == pytest.approx(142.50)
            assert rm2.stop_losses["AAPL"].stop_type == StopLossType.TRAILING
            assert rm2.stop_losses["MSFT"].stop_type == StopLossType.FIXED

            # Verify take-profits
            assert "AAPL" in rm2.take_profits
            tp = rm2.take_profits["AAPL"]
            assert tp.total_quantity == 100
            assert len(tp.levels) == 3
            assert tp.levels[0].target_pct == pytest.approx(0.05)

            # Verify drawdown state
            assert rm2.peak_portfolio_value == pytest.approx(200_000.0)
            assert rm2.drawdown_recovery_mode is True

    def test_load_prunes_stale_symbols(self, tmp_state_dir):
        """Only symbols in *active_symbols* are restored."""
        state_file = tmp_state_dir / "risk_manager_state.json"

        with patch.object(RiskManager, "STATE_FILE", state_file):
            rm = RiskManager()
            rm.set_stop_loss("AAPL", 150.0, StopLossType.FIXED, 142.0)
            rm.set_stop_loss("MSFT", 300.0, StopLossType.FIXED, 285.0)
            rm.set_stop_loss("TSLA", 200.0, StopLossType.FIXED, 190.0)
            rm.save_state()

        with patch.object(RiskManager, "STATE_FILE", state_file):
            rm2 = RiskManager()
            rm2.load_state(active_symbols=["AAPL", "MSFT"])

            assert "AAPL" in rm2.stop_losses
            assert "MSFT" in rm2.stop_losses
            assert "TSLA" not in rm2.stop_losses


# ---------------------------------------------------------------------------
# 3. Simulated broker basic buy / sell flow
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSimulatedBrokerFlow:
    """Place orders, verify position tracking and cash accounting."""

    def test_buy_then_sell(self, tmp_path):
        state_file = tmp_path / "simulated_broker_state.json"
        with patch.object(SimulatedBroker, "STATE_FILE", state_file):
            broker = SimulatedBroker(initial_capital=100_000.0)
            broker.connect()
            assert broker.is_connected()

            # -- BUY 10 shares at $150 --
            with patch.object(broker, "_get_current_price", return_value=150.0):
                buy_order = broker.place_order(
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    quantity=10,
                    order_type=OrderType.MARKET,
                )
            assert buy_order.status == OrderStatus.FILLED
            assert buy_order.filled_quantity == 10

            pos = broker.get_position("AAPL")
            assert pos is not None
            assert pos.quantity == 10
            assert pos.avg_cost == pytest.approx(150.0)

            expected_cash = 100_000.0 - (10 * 150.0)
            assert broker.cash == pytest.approx(expected_cash)

            # -- SELL 10 shares at $160 --
            with patch.object(broker, "_get_current_price", return_value=160.0):
                sell_order = broker.place_order(
                    symbol="AAPL",
                    side=OrderSide.SELL,
                    quantity=10,
                    order_type=OrderType.MARKET,
                )
            assert sell_order.status == OrderStatus.FILLED

            pos_after = broker.get_position("AAPL")
            assert pos_after is None  # position fully closed

            expected_cash_after = expected_cash + (10 * 160.0)
            assert broker.cash == pytest.approx(expected_cash_after)
            assert broker.realized_pnl == pytest.approx(100.0)  # 10 * (160-150)

            broker.disconnect()
            assert not broker.is_connected()

    def test_insufficient_funds_rejected(self, tmp_path):
        state_file = tmp_path / "simulated_broker_state.json"
        with patch.object(SimulatedBroker, "STATE_FILE", state_file):
            broker = SimulatedBroker(initial_capital=1_000.0)
            broker.connect()

            with patch.object(broker, "_get_current_price", return_value=500.0):
                order = broker.place_order(
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    quantity=10,  # $5000 > $1000 cash
                    order_type=OrderType.MARKET,
                )
            assert order.status == OrderStatus.REJECTED

    def test_sell_more_than_held_rejected(self, tmp_path):
        state_file = tmp_path / "simulated_broker_state.json"
        with patch.object(SimulatedBroker, "STATE_FILE", state_file):
            broker = SimulatedBroker(initial_capital=100_000.0)
            broker.connect()

            # Buy 5 shares first
            with patch.object(broker, "_get_current_price", return_value=100.0):
                broker.place_order("AAPL", OrderSide.BUY, 5, OrderType.MARKET)

            # Attempt to sell 10
            with patch.object(broker, "_get_current_price", return_value=100.0):
                order = broker.place_order(
                    "AAPL", OrderSide.SELL, 10, OrderType.MARKET
                )
            assert order.status == OrderStatus.REJECTED


# ---------------------------------------------------------------------------
# 4. Stop-loss trigger flow
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestStopLossTrigger:
    """Set a stop-loss, simulate a price drop, verify trigger."""

    def test_fixed_stop_triggers_on_drop(self):
        rm = RiskManager()

        rm.set_stop_loss(
            symbol="AAPL",
            entry_price=150.0,
            stop_type=StopLossType.FIXED,
            stop_price=142.50,
        )

        # Price still above stop -- should NOT trigger
        triggered = rm.check_stop_losses({"AAPL": 145.0})
        assert triggered == []

        # Price drops below stop -- SHOULD trigger
        triggered = rm.check_stop_losses({"AAPL": 140.0})
        assert "AAPL" in triggered

    def test_trailing_stop_ratchets_up(self):
        rm = RiskManager()

        rm.set_stop_loss(
            symbol="AAPL",
            entry_price=150.0,
            stop_type=StopLossType.TRAILING,
            stop_price=142.50,  # trailing distance = 7.50
        )

        assert rm.stop_losses["AAPL"].trailing_distance == pytest.approx(7.50)

        # Price rises to 160 -- trailing stop should ratchet to 152.50
        rm.update_trailing_stop("AAPL", 160.0)
        assert rm.stop_losses["AAPL"].stop_price == pytest.approx(152.50)

        # Price dips to 155 -- stop should NOT move down
        rm.update_trailing_stop("AAPL", 155.0)
        assert rm.stop_losses["AAPL"].stop_price == pytest.approx(152.50)

        # Price falls to 151 -- below trailing stop, should trigger
        triggered = rm.check_stop_losses({"AAPL": 151.0})
        assert "AAPL" in triggered

    def test_stop_loss_full_round_trip_with_broker(self, tmp_path):
        """End-to-end: buy, set stop, simulate drop, sell on stop."""
        state_file = tmp_path / "simulated_broker_state.json"
        with patch.object(SimulatedBroker, "STATE_FILE", state_file):
            broker = SimulatedBroker(initial_capital=100_000.0)
            broker.connect()

            # Buy position
            with patch.object(broker, "_get_current_price", return_value=150.0):
                broker.place_order("AAPL", OrderSide.BUY, 20, OrderType.MARKET)

            rm = RiskManager()
            rm.set_stop_loss("AAPL", 150.0, StopLossType.FIXED, 140.0)

            # Price drops
            triggered = rm.check_stop_losses({"AAPL": 138.0})
            assert "AAPL" in triggered

            # Execute the stop-loss sell
            with patch.object(broker, "_get_current_price", return_value=138.0):
                sell_order = broker.place_order(
                    "AAPL", OrderSide.SELL, 20, OrderType.MARKET
                )
            assert sell_order.status == OrderStatus.FILLED

            rm.remove_stop_loss("AAPL")
            assert "AAPL" not in rm.stop_losses

            pos = broker.get_position("AAPL")
            assert pos is None


# ---------------------------------------------------------------------------
# 5. Take-profit trigger flow
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTakeProfitTrigger:
    """Set take-profit levels, simulate price rise, verify partial exits."""

    def test_partial_exits_at_each_level(self):
        rm = RiskManager()
        rm.set_take_profit(
            symbol="AAPL",
            entry_price=100.0,
            quantity=100,
            tp_levels=[(0.05, 0.33), (0.10, 0.50), (0.15, 1.0)],
        )

        tp = rm.take_profits["AAPL"]
        assert tp.quantity_remaining == 100

        # --- Level 1: price hits 105 (5% gain) => sell ~33 shares ---
        hits = rm.check_take_profits({"AAPL": 105.0})
        assert len(hits) == 1
        sym, qty, price = hits[0]
        assert sym == "AAPL"
        assert qty == 33  # int(100 * 0.33)
        assert price == pytest.approx(105.0)

        tp = rm.take_profits["AAPL"]
        assert tp.quantity_remaining == 100 - 33  # 67

        # --- Level 2: price hits 110.50 (above 10% gain) => sell ~50% of remaining ---
        # Use price clearly above target to avoid floating-point edge cases
        # (100 * 1.10 may produce 110.00000000000001).
        hits = rm.check_take_profits({"AAPL": 110.50})
        assert len(hits) == 1
        sym, qty, price = hits[0]
        assert qty == 33  # int(67 * 0.50) = 33
        assert price == pytest.approx(110.50)

        tp = rm.take_profits["AAPL"]
        assert tp.quantity_remaining == 67 - 33  # 34

        # --- Level 3: price hits 115.50 (above 15% gain) => sell rest ---
        hits = rm.check_take_profits({"AAPL": 115.50})
        assert len(hits) == 1
        sym, qty, price = hits[0]
        assert qty == 34
        assert price == pytest.approx(115.50)

        # TP order should be fully removed
        assert "AAPL" not in rm.take_profits

    def test_no_trigger_below_first_level(self):
        rm = RiskManager()
        rm.set_take_profit("AAPL", 100.0, 50, [(0.10, 0.50), (0.20, 1.0)])

        hits = rm.check_take_profits({"AAPL": 105.0})  # only 5%, first level at 10%
        assert hits == []

    def test_take_profit_with_broker_sell(self, tmp_path):
        """End-to-end: buy, set TP, price rises, partial sell."""
        state_file = tmp_path / "simulated_broker_state.json"
        with patch.object(SimulatedBroker, "STATE_FILE", state_file):
            broker = SimulatedBroker(initial_capital=100_000.0)
            broker.connect()

            entry_price = 100.0
            qty = 100

            with patch.object(broker, "_get_current_price", return_value=entry_price):
                broker.place_order("AAPL", OrderSide.BUY, qty, OrderType.MARKET)

            rm = RiskManager()
            rm.set_take_profit(
                "AAPL", entry_price, qty,
                tp_levels=[(0.05, 0.50), (0.10, 1.0)],
            )

            # Price hits first TP level (5% of 100 = 105)
            hits = rm.check_take_profits({"AAPL": 105.50})
            assert len(hits) == 1
            _, sell_qty, _ = hits[0]

            with patch.object(broker, "_get_current_price", return_value=105.50):
                sell_order = broker.place_order(
                    "AAPL", OrderSide.SELL, sell_qty, OrderType.MARKET,
                )
            assert sell_order.status == OrderStatus.FILLED

            pos = broker.get_position("AAPL")
            assert pos is not None
            assert pos.quantity == qty - sell_qty

            # Price hits second TP level (10% of 100 = 110)
            # Use price above target to avoid float precision edge case
            hits = rm.check_take_profits({"AAPL": 110.50})
            assert len(hits) == 1
            _, sell_qty2, _ = hits[0]

            with patch.object(broker, "_get_current_price", return_value=110.50):
                sell_order2 = broker.place_order(
                    "AAPL", OrderSide.SELL, sell_qty2, OrderType.MARKET,
                )
            assert sell_order2.status == OrderStatus.FILLED

            pos = broker.get_position("AAPL")
            assert pos is None  # fully exited
