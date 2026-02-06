"""
Comprehensive unit tests for TradingEngine.

Covers initialization, market hours, trade execution, stop-loss execution,
take-profit execution, and the full trading cycle.
"""
import pytest
from datetime import datetime, time as dt_time
from unittest.mock import MagicMock, patch, PropertyMock
from zoneinfo import ZoneInfo

from src.broker.base_broker import (
    AccountInfo,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    TimeInForce,
)
from src.risk.risk_manager import RiskCheckResult, StopLossType


# ---------------------------------------------------------------------------
# Helpers -- factory functions for data-classes used throughout the tests
# ---------------------------------------------------------------------------

def _make_order(
    order_id="ORD-001",
    symbol="AAPL",
    side=OrderSide.BUY,
    quantity=10,
    price=150.0,
    status=OrderStatus.FILLED,
    filled_quantity=10,
    filled_price=150.0,
):
    """Return a fully-populated Order object."""
    now = datetime.now()
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity,
        price=price,
        stop_price=None,
        status=status,
        filled_quantity=filled_quantity,
        filled_price=filled_price,
        time_in_force=TimeInForce.DAY,
        created_at=now,
        updated_at=now,
    )


def _make_position(
    symbol="AAPL",
    quantity=10,
    avg_cost=150.0,
    current_price=155.0,
):
    """Return a Position with computed PnL fields."""
    market_value = quantity * current_price
    unrealized_pnl = (current_price - avg_cost) * quantity
    unrealized_pnl_pct = (current_price - avg_cost) / avg_cost if avg_cost else 0.0
    return Position(
        symbol=symbol,
        quantity=quantity,
        avg_cost=avg_cost,
        current_price=current_price,
        market_value=market_value,
        unrealized_pnl=unrealized_pnl,
        unrealized_pnl_pct=unrealized_pnl_pct,
        realized_pnl=0.0,
    )


def _make_account(portfolio_value=100_000.0, cash=50_000.0):
    """Return an AccountInfo."""
    return AccountInfo(
        account_id="ACC-001",
        cash=cash,
        buying_power=cash * 2,
        portfolio_value=portfolio_value,
        day_trades_remaining=3,
        positions_count=2,
    )


def _make_quote(symbol="AAPL", last=155.0):
    """Return a Quote."""
    return Quote(
        symbol=symbol,
        bid=last - 0.05,
        ask=last + 0.05,
        last=last,
        volume=1_000_000,
        timestamp=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Minimal test config returned by the mocked Settings.load_trading_config
# ---------------------------------------------------------------------------

MINIMAL_CONFIG = {
    "ml_model": {
        "confidence_threshold": 0.60,
        "min_confidence_sell": 0.55,
    },
    "risk_management": {
        "max_position_pct": 0.10,
        "max_daily_loss_pct": 0.05,
        "max_total_exposure": 0.80,
        "volatility_sizing": {
            "enabled": True,
            "vix_thresholds": {"low": 15, "normal": 25, "high": 35, "extreme": 50},
            "size_multipliers": {"low": 1.2, "normal": 1.0, "high": 0.7, "extreme": 0.5},
        },
        "drawdown_protection": {
            "enabled": True,
            "max_drawdown_pct": 0.10,
            "recovery_threshold": 0.95,
            "recovery_size_multiplier": 0.5,
        },
        "regime_detection": {
            "enabled": False,  # disabled by default in tests
        },
        "take_profit": {
            "enabled": True,
            "levels": [(0.05, 0.33), (0.10, 0.50), (0.15, 1.0)],
        },
    },
    "notifications": {"enabled": False},
    "retraining": {"enabled": False},
    "agents": {"enabled": False},
    "dynamic_symbols": {"enabled": False},
    "portfolio_optimization": {"enabled": False},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def _patch_settings():
    """Patch Settings.load_trading_config to return MINIMAL_CONFIG."""
    with patch(
        "src.core.trading_engine.Settings.load_trading_config",
        return_value=MINIMAL_CONFIG,
    ):
        yield


@pytest.fixture()
def engine(_patch_settings):
    """Return a TradingEngine with a mock broker wired in."""
    from src.core.trading_engine import TradingEngine

    eng = TradingEngine(
        symbols=["AAPL", "MSFT"],
        simulated=True,
        initial_capital=100_000.0,
        ignore_market_hours=True,
    )
    # Replace broker with a MagicMock so no real I/O happens
    eng.broker = MagicMock()
    eng.is_running = True
    eng.model_loaded = True
    return eng


@pytest.fixture()
def engine_market_hours(_patch_settings):
    """Engine with ignore_market_hours=False for market-hours tests."""
    from src.core.trading_engine import TradingEngine

    eng = TradingEngine(
        symbols=["AAPL"],
        simulated=True,
        ignore_market_hours=False,
    )
    return eng


# ===================================================================
# 1. Initialization tests
# ===================================================================
class TestInitialization:
    """Verify __init__ wires config values into sub-components."""

    def test_vix_sizing_loaded(self, engine):
        rm = engine.risk_manager
        assert rm.vix_sizing_enabled is True
        assert rm.vix_thresholds["low"] == 15
        assert rm.vix_multipliers["extreme"] == 0.5

    def test_drawdown_protection_loaded(self, engine):
        rm = engine.risk_manager
        assert rm.max_drawdown_pct == 0.10
        assert rm.recovery_threshold == 0.95
        assert rm.recovery_size_multiplier == 0.5

    def test_regime_detector_disabled(self, engine):
        """Regime detector should be None when disabled in config."""
        assert engine.regime_detector is None

    def test_regime_detector_created_when_enabled(self):
        """When regime_detection.enabled=True, a detector instance is created."""
        config_with_regime = {
            **MINIMAL_CONFIG,
            "risk_management": {
                **MINIMAL_CONFIG["risk_management"],
                "regime_detection": {"enabled": True},
            },
        }
        with patch(
            "src.core.trading_engine.Settings.load_trading_config",
            return_value=config_with_regime,
        ), patch(
            "src.core.trading_engine.get_regime_detector",
        ) as mock_get_rd:
            mock_detector = MagicMock()
            mock_get_rd.return_value = mock_detector

            from src.core.trading_engine import TradingEngine

            eng = TradingEngine(symbols=["AAPL"], simulated=True)
            assert eng.regime_detector is mock_detector

    def test_default_values_empty_config(self):
        """When config is empty, defaults are applied."""
        with patch(
            "src.core.trading_engine.Settings.load_trading_config",
            return_value={},
        ):
            from src.core.trading_engine import TradingEngine

            eng = TradingEngine(symbols=["AAPL"], simulated=True)
            assert eng.risk_manager.max_position_pct == 0.10
            assert eng.risk_manager.max_daily_loss_pct == 0.05
            assert eng.risk_manager.max_total_exposure == 0.80
            # VIX sizing wired with defaults
            assert eng.risk_manager.vix_sizing_enabled is True

    def test_symbols_stored(self, engine):
        assert engine.symbols == ["AAPL", "MSFT"]

    def test_ignore_market_hours_flag(self, engine):
        assert engine.ignore_market_hours is True

    def test_strategy_confidence_from_config(self, engine):
        assert engine.strategy.confidence_threshold == 0.60
        assert engine.strategy.min_confidence_sell == 0.55

    def test_notifier_none_when_disabled(self, engine):
        assert engine.notifier is None

    def test_retrainer_none_when_disabled(self, engine):
        assert engine.retrainer is None

    def test_agent_orchestrator_none_when_disabled(self, engine):
        assert engine.agent_orchestrator is None

    def test_is_running_defaults_false(self, _patch_settings):
        from src.core.trading_engine import TradingEngine

        eng = TradingEngine(symbols=["AAPL"], simulated=True)
        assert eng.is_running is False

    def test_custom_risk_params_from_config(self):
        """Custom risk params in config are forwarded to RiskManager."""
        custom_config = {
            **MINIMAL_CONFIG,
            "risk_management": {
                **MINIMAL_CONFIG["risk_management"],
                "max_position_pct": 0.20,
                "max_daily_loss_pct": 0.03,
                "max_total_exposure": 0.60,
            },
        }
        with patch(
            "src.core.trading_engine.Settings.load_trading_config",
            return_value=custom_config,
        ):
            from src.core.trading_engine import TradingEngine

            eng = TradingEngine(symbols=["AAPL"], simulated=True)
            assert eng.risk_manager.max_position_pct == 0.20
            assert eng.risk_manager.max_daily_loss_pct == 0.03
            assert eng.risk_manager.max_total_exposure == 0.60

    def test_vix_custom_thresholds_from_config(self):
        custom_config = {
            **MINIMAL_CONFIG,
            "risk_management": {
                **MINIMAL_CONFIG["risk_management"],
                "volatility_sizing": {
                    "enabled": True,
                    "vix_thresholds": {"low": 12, "normal": 20, "high": 30, "extreme": 45},
                    "size_multipliers": {"low": 1.3, "normal": 1.0, "high": 0.6, "extreme": 0.4},
                },
            },
        }
        with patch(
            "src.core.trading_engine.Settings.load_trading_config",
            return_value=custom_config,
        ):
            from src.core.trading_engine import TradingEngine

            eng = TradingEngine(symbols=["AAPL"], simulated=True)
            assert eng.risk_manager.vix_thresholds["low"] == 12
            assert eng.risk_manager.vix_multipliers["extreme"] == 0.4

    def test_drawdown_protection_disabled(self):
        """When drawdown_protection.enabled=False, defaults remain."""
        config_no_dd = {
            **MINIMAL_CONFIG,
            "risk_management": {
                **MINIMAL_CONFIG["risk_management"],
                "drawdown_protection": {"enabled": False},
            },
        }
        with patch(
            "src.core.trading_engine.Settings.load_trading_config",
            return_value=config_no_dd,
        ):
            from src.core.trading_engine import TradingEngine

            eng = TradingEngine(symbols=["AAPL"], simulated=True)
            # RiskManager defaults: max_drawdown_pct=0.10
            assert eng.risk_manager.max_drawdown_pct == 0.10


# ===================================================================
# 2. Market hours tests
# ===================================================================
class TestMarketHours:
    """Test _is_market_open with timezone-aware datetimes."""

    def test_returns_true_during_market_hours(self, engine_market_hours):
        """A Wednesday at 10:00 ET should be open."""
        # 2025-01-08 is a Wednesday
        mock_dt = datetime(2025, 1, 8, 10, 0, 0, tzinfo=ZoneInfo("US/Eastern"))
        with patch("src.core.trading_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **k: datetime(*a, **k)
            assert engine_market_hours._is_market_open() is True

    def test_returns_false_before_open(self, engine_market_hours):
        """A weekday at 08:00 ET is before market open."""
        mock_dt = datetime(2025, 1, 8, 8, 0, 0, tzinfo=ZoneInfo("US/Eastern"))
        with patch("src.core.trading_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **k: datetime(*a, **k)
            assert engine_market_hours._is_market_open() is False

    def test_returns_false_after_close(self, engine_market_hours):
        """A weekday at 17:00 ET is after market close."""
        mock_dt = datetime(2025, 1, 8, 17, 0, 0, tzinfo=ZoneInfo("US/Eastern"))
        with patch("src.core.trading_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **k: datetime(*a, **k)
            assert engine_market_hours._is_market_open() is False

    def test_returns_false_on_saturday(self, engine_market_hours):
        """Saturday should be closed."""
        # 2025-01-11 is a Saturday
        mock_dt = datetime(2025, 1, 11, 11, 0, 0, tzinfo=ZoneInfo("US/Eastern"))
        with patch("src.core.trading_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **k: datetime(*a, **k)
            assert engine_market_hours._is_market_open() is False

    def test_returns_false_on_sunday(self, engine_market_hours):
        """Sunday should be closed."""
        # 2025-01-12 is a Sunday
        mock_dt = datetime(2025, 1, 12, 11, 0, 0, tzinfo=ZoneInfo("US/Eastern"))
        with patch("src.core.trading_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **k: datetime(*a, **k)
            assert engine_market_hours._is_market_open() is False

    def test_ignore_market_hours_bypasses_check(self, engine):
        """When ignore_market_hours=True, always returns True."""
        # engine fixture already has ignore_market_hours=True
        assert engine._is_market_open() is True

    def test_boundary_at_open(self, engine_market_hours):
        """Exactly 09:30 ET should be open."""
        mock_dt = datetime(2025, 1, 8, 9, 30, 0, tzinfo=ZoneInfo("US/Eastern"))
        with patch("src.core.trading_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **k: datetime(*a, **k)
            assert engine_market_hours._is_market_open() is True

    def test_boundary_at_close(self, engine_market_hours):
        """Exactly 16:00 ET should be open (<=)."""
        mock_dt = datetime(2025, 1, 8, 16, 0, 0, tzinfo=ZoneInfo("US/Eastern"))
        with patch("src.core.trading_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **k: datetime(*a, **k)
            assert engine_market_hours._is_market_open() is True

    def test_one_minute_before_open(self, engine_market_hours):
        """09:29 ET should still be closed."""
        mock_dt = datetime(2025, 1, 8, 9, 29, 0, tzinfo=ZoneInfo("US/Eastern"))
        with patch("src.core.trading_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **k: datetime(*a, **k)
            assert engine_market_hours._is_market_open() is False

    def test_one_minute_after_close(self, engine_market_hours):
        """16:01 ET should be closed."""
        mock_dt = datetime(2025, 1, 8, 16, 1, 0, tzinfo=ZoneInfo("US/Eastern"))
        with patch("src.core.trading_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.side_effect = lambda *a, **k: datetime(*a, **k)
            assert engine_market_hours._is_market_open() is False


# ===================================================================
# 3. Trade execution tests -- _execute_trade
# ===================================================================
class TestExecuteTrade:
    """Test _execute_trade covering buys, sells, risk checks, and edge cases."""

    def _buy_recommendation(self, **overrides):
        rec = {
            "symbol": "AAPL",
            "action": "BUY",
            "shares": 10,
            "price": 150.0,
            "stop_loss": 142.50,
            "confidence": 0.75,
        }
        rec.update(overrides)
        return rec

    def _sell_recommendation(self, **overrides):
        rec = {
            "symbol": "AAPL",
            "action": "SELL",
            "shares": 10,
            "price": 160.0,
            "confidence": 0.70,
        }
        rec.update(overrides)
        return rec

    # -- BUY order sets stop-loss and take-profit -----------------------
    def test_buy_order_sets_stop_loss_and_take_profit(self, engine):
        order = _make_order()
        engine.broker.place_order.return_value = order

        result = engine._execute_trade(
            recommendation=self._buy_recommendation(),
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        assert result is not None
        assert result["action"] == "BUY"
        assert result["shares"] == 10
        assert result["symbol"] == "AAPL"
        assert result["order_id"] == "ORD-001"
        # Stop-loss should have been set
        assert "AAPL" in engine.risk_manager.stop_losses
        sl = engine.risk_manager.stop_losses["AAPL"]
        assert sl.stop_price == 142.50
        assert sl.stop_type == StopLossType.TRAILING
        # Take-profit should have been set
        assert "AAPL" in engine.risk_manager.take_profits

    # -- SELL order executes normally -----------------------------------
    def test_sell_order_executes_normally(self, engine):
        order = _make_order(side=OrderSide.SELL, filled_price=160.0, price=160.0)
        engine.broker.place_order.return_value = order

        result = engine._execute_trade(
            recommendation=self._sell_recommendation(),
            portfolio_value=100_000.0,
            current_positions={"AAPL": 10},
            position_market_values={"AAPL": 1500.0},
        )

        assert result is not None
        assert result["action"] == "SELL"
        # Sells should NOT set stop-loss or take-profit
        assert "AAPL" not in engine.risk_manager.stop_losses

    def test_sell_order_does_not_go_through_risk_check(self, engine):
        """SELL orders should bypass the risk_manager.check_position call."""
        order = _make_order(side=OrderSide.SELL, filled_price=160.0, price=160.0)
        engine.broker.place_order.return_value = order

        engine.risk_manager.check_position = MagicMock(
            return_value=RiskCheckResult(approved=False, reason="blocked")
        )

        result = engine._execute_trade(
            recommendation=self._sell_recommendation(),
            portfolio_value=100_000.0,
            current_positions={"AAPL": 10},
            position_market_values={"AAPL": 1500.0},
        )

        # SELL should succeed even though check_position would reject
        assert result is not None
        assert result["action"] == "SELL"
        engine.risk_manager.check_position.assert_not_called()

    # -- Risk check rejects oversized position, adjusts quantity --------
    def test_risk_check_adjusts_quantity(self, engine):
        """If risk manager rejects but provides an adjusted quantity, use it."""
        # Request 200 shares at $150 = $30k -> exceeds 10% of $100k
        rec = self._buy_recommendation(shares=200, price=150.0)

        # The broker returns a filled order with the adjusted quantity
        adjusted_order = _make_order(quantity=66, filled_quantity=66)
        engine.broker.place_order.return_value = adjusted_order

        result = engine._execute_trade(
            recommendation=rec,
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        # The risk manager should have adjusted quantity; the trade should proceed
        assert result is not None
        # The order placed to broker uses the adjusted quantity
        call_kwargs = engine.broker.place_order.call_args
        placed_qty = call_kwargs.kwargs.get("quantity") or call_kwargs[1].get("quantity", None)
        if placed_qty is None:
            # positional arg
            placed_qty = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None
        assert placed_qty is not None
        assert placed_qty <= 66  # risk-manager cap

    def test_risk_check_rejects_no_adjusted(self, engine):
        """If risk manager rejects with adjusted_quantity=0, return None."""
        rec = self._buy_recommendation(shares=200, price=150.0)

        # Patch risk_manager.check_position to reject entirely
        engine.risk_manager.check_position = MagicMock(
            return_value=RiskCheckResult(
                approved=False,
                reason="Total exposure exceeded",
                adjusted_quantity=0,
            )
        )

        result = engine._execute_trade(
            recommendation=rec,
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        assert result is None

    def test_risk_check_rejects_no_adjusted_quantity_field(self, engine):
        """If risk manager rejects with no adjusted_quantity at all, return None."""
        rec = self._buy_recommendation(shares=200, price=150.0)

        engine.risk_manager.check_position = MagicMock(
            return_value=RiskCheckResult(
                approved=False,
                reason="Total exposure exceeded",
                adjusted_quantity=None,
            )
        )

        result = engine._execute_trade(
            recommendation=rec,
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        assert result is None

    # -- Agent halt blocks trade ----------------------------------------
    def test_agent_halt_blocks_trade(self, engine):
        mock_orch = MagicMock()
        mock_orch.is_trading_halted.return_value = True
        engine.agent_orchestrator = mock_orch

        result = engine._execute_trade(
            recommendation=self._buy_recommendation(),
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        assert result is None
        # Broker should NOT have been called
        engine.broker.place_order.assert_not_called()

    # -- Malformed recommendation returns None --------------------------
    def test_malformed_recommendation_missing_all(self, engine):
        bad_rec = {"symbol": "AAPL"}  # missing action, shares, price
        result = engine._execute_trade(
            recommendation=bad_rec,
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )
        assert result is None

    def test_malformed_recommendation_missing_price(self, engine):
        rec = {"symbol": "AAPL", "action": "BUY", "shares": 10}
        result = engine._execute_trade(
            recommendation=rec,
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )
        assert result is None

    def test_malformed_recommendation_missing_shares(self, engine):
        rec = {"symbol": "AAPL", "action": "BUY", "price": 150.0}
        result = engine._execute_trade(
            recommendation=rec,
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )
        assert result is None

    def test_malformed_recommendation_missing_action(self, engine):
        rec = {"symbol": "AAPL", "shares": 10, "price": 150.0}
        result = engine._execute_trade(
            recommendation=rec,
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )
        assert result is None

    def test_empty_recommendation(self, engine):
        result = engine._execute_trade(
            recommendation={},
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )
        assert result is None

    # -- Partial fill adjusts shares count ------------------------------
    def test_partial_fill_adjusts_shares(self, engine):
        partial_order = _make_order(
            status=OrderStatus.PARTIALLY_FILLED,
            quantity=10,
            filled_quantity=7,
            filled_price=150.0,
        )
        engine.broker.place_order.return_value = partial_order

        result = engine._execute_trade(
            recommendation=self._buy_recommendation(),
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        assert result is not None
        assert result["shares"] == 7  # adjusted to filled quantity

    def test_partial_fill_with_stop_loss_uses_adjusted_qty(self, engine):
        """Take-profit quantity should use the partial-fill count."""
        partial_order = _make_order(
            status=OrderStatus.PARTIALLY_FILLED,
            quantity=10,
            filled_quantity=7,
            filled_price=150.0,
        )
        engine.broker.place_order.return_value = partial_order

        engine._execute_trade(
            recommendation=self._buy_recommendation(),
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        # Take-profit should be set with filled quantity (7), not requested (10)
        tp = engine.risk_manager.take_profits.get("AAPL")
        assert tp is not None
        assert tp.total_quantity == 7

    def test_rejected_order_returns_none(self, engine):
        rejected_order = _make_order(
            status=OrderStatus.REJECTED, filled_quantity=0, filled_price=None
        )
        engine.broker.place_order.return_value = rejected_order

        result = engine._execute_trade(
            recommendation=self._buy_recommendation(),
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        assert result is None

    def test_pending_order_returns_none(self, engine):
        """An order stuck in PENDING with 0 fills should return None."""
        pending_order = _make_order(
            status=OrderStatus.PENDING, filled_quantity=0, filled_price=None
        )
        engine.broker.place_order.return_value = pending_order

        result = engine._execute_trade(
            recommendation=self._buy_recommendation(),
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        assert result is None

    # -- Notifier called on successful trade ----------------------------
    def test_notifier_called_on_buy(self, engine):
        engine.notifier = MagicMock()
        order = _make_order()
        engine.broker.place_order.return_value = order

        engine._execute_trade(
            recommendation=self._buy_recommendation(),
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        engine.notifier.notify_trade.assert_called_once()
        call_kwargs = engine.notifier.notify_trade.call_args.kwargs
        assert call_kwargs["action"] == "BUY"
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["confidence"] == 0.75

    def test_notifier_not_called_when_none(self, engine):
        """When notifier is None, no exception is raised."""
        engine.notifier = None
        order = _make_order()
        engine.broker.place_order.return_value = order

        # Should not raise
        result = engine._execute_trade(
            recommendation=self._buy_recommendation(),
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )
        assert result is not None

    # -- Buy without stop_loss key does not set stop --------------------
    def test_buy_without_stop_loss_key_skips_stop(self, engine):
        order = _make_order()
        engine.broker.place_order.return_value = order

        rec = self._buy_recommendation()
        del rec["stop_loss"]

        engine._execute_trade(
            recommendation=rec,
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        assert "AAPL" not in engine.risk_manager.stop_losses

    # -- Take-profit disabled in config ---------------------------------
    def test_buy_take_profit_disabled(self, engine):
        engine.config["risk_management"]["take_profit"]["enabled"] = False
        order = _make_order()
        engine.broker.place_order.return_value = order

        engine._execute_trade(
            recommendation=self._buy_recommendation(),
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        assert "AAPL" not in engine.risk_manager.take_profits

    def test_result_contains_expected_fields(self, engine):
        order = _make_order()
        engine.broker.place_order.return_value = order

        result = engine._execute_trade(
            recommendation=self._buy_recommendation(),
            portfolio_value=100_000.0,
            current_positions={},
            position_market_values={},
        )

        assert "order_id" in result
        assert "action" in result
        assert "symbol" in result
        assert "shares" in result
        assert "price" in result
        assert "timestamp" in result


# ===================================================================
# 4. Stop-loss execution tests -- _execute_stop_loss
# ===================================================================
class TestExecuteStopLoss:
    """Test _execute_stop_loss selling, cleanup, PnL, and notifications."""

    def _setup_position_and_stop(self, engine, symbol="AAPL"):
        """Helper: wire broker to return a position and set a stop/TP."""
        pos = _make_position(
            symbol=symbol, quantity=10, avg_cost=150.0, current_price=140.0
        )
        engine.broker.get_position.return_value = pos
        sell_order = _make_order(
            side=OrderSide.SELL,
            symbol=symbol,
            filled_price=139.50,
            quantity=10,
            filled_quantity=10,
        )
        engine.broker.place_order.return_value = sell_order
        # Pre-set stop loss and take profit
        engine.risk_manager.set_stop_loss(symbol, entry_price=150.0, stop_price=142.0)
        engine.risk_manager.set_take_profit(symbol, entry_price=150.0, quantity=10)
        return pos, sell_order

    def test_sells_entire_position(self, engine):
        self._setup_position_and_stop(engine)
        engine._execute_stop_loss("AAPL")

        engine.broker.place_order.assert_called_once_with(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET,
        )

    def test_removes_stop_loss_and_take_profit(self, engine):
        self._setup_position_and_stop(engine)

        engine._execute_stop_loss("AAPL")

        assert "AAPL" not in engine.risk_manager.stop_losses
        assert "AAPL" not in engine.risk_manager.take_profits

    def test_pnl_calculated_from_fill_price(self, engine):
        self._setup_position_and_stop(engine)

        engine._execute_stop_loss("AAPL")

        # fill_price=139.50, entry=150.0, qty=10 -> pnl = (139.50-150)*10 = -105
        expected_pnl = (139.50 - 150.0) * 10
        assert engine.risk_manager.daily_pnl == pytest.approx(expected_pnl, abs=0.01)

    def test_notifier_called(self, engine):
        engine.notifier = MagicMock()
        self._setup_position_and_stop(engine)

        engine._execute_stop_loss("AAPL")

        engine.notifier.notify_stop_loss.assert_called_once()
        call_kwargs = engine.notifier.notify_stop_loss.call_args.kwargs
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["exit_price"] == pytest.approx(139.50)
        assert call_kwargs["loss_amount"] < 0  # it is a loss

    def test_notifier_loss_pct_is_correct(self, engine):
        engine.notifier = MagicMock()
        self._setup_position_and_stop(engine)

        engine._execute_stop_loss("AAPL")

        call_kwargs = engine.notifier.notify_stop_loss.call_args.kwargs
        # loss_pct = pnl / (qty * entry) = -105 / (10*150) = -0.07
        expected_loss_pct = ((139.50 - 150.0) * 10) / (10 * 150.0)
        assert call_kwargs["loss_pct"] == pytest.approx(expected_loss_pct, abs=0.001)

    def test_no_position_is_noop(self, engine):
        """If broker returns no position, nothing happens."""
        engine.broker.get_position.return_value = None
        engine._execute_stop_loss("AAPL")
        engine.broker.place_order.assert_not_called()

    def test_zero_quantity_position_is_noop(self, engine):
        """Position with zero quantity should not trigger a sell."""
        pos = _make_position(quantity=0)
        engine.broker.get_position.return_value = pos
        engine._execute_stop_loss("AAPL")
        engine.broker.place_order.assert_not_called()

    def test_pnl_uses_current_price_when_no_fill_price(self, engine):
        """If order has no filled_price, fallback to position.current_price."""
        pos = _make_position(
            symbol="AAPL", quantity=10, avg_cost=150.0, current_price=140.0
        )
        engine.broker.get_position.return_value = pos
        sell_order = _make_order(side=OrderSide.SELL)
        # Remove filled_price so getattr returns None
        sell_order.filled_price = None
        engine.broker.place_order.return_value = sell_order
        engine.risk_manager.set_stop_loss(
            "AAPL", entry_price=150.0, stop_price=142.0
        )

        engine._execute_stop_loss("AAPL")

        # Fallback to current_price=140.0: pnl = (140-150)*10 = -100
        expected_pnl = (140.0 - 150.0) * 10
        assert engine.risk_manager.daily_pnl == pytest.approx(expected_pnl, abs=0.01)

    def test_pnl_marked_as_loss(self, engine):
        """Negative PnL should be flagged as a loss in risk_manager."""
        self._setup_position_and_stop(engine)
        engine._execute_stop_loss("AAPL")

        # After a loss, consecutive_losses should increment
        assert engine.risk_manager.consecutive_losses >= 1

    def test_positive_pnl_stop_loss(self, engine):
        """Stop-loss that results in a gain (trailing stop profit lock-in)."""
        pos = _make_position(
            symbol="AAPL", quantity=10, avg_cost=100.0, current_price=120.0
        )
        engine.broker.get_position.return_value = pos
        sell_order = _make_order(
            side=OrderSide.SELL,
            symbol="AAPL",
            filled_price=118.0,
        )
        engine.broker.place_order.return_value = sell_order
        engine.risk_manager.set_stop_loss("AAPL", entry_price=100.0, stop_price=115.0)

        engine._execute_stop_loss("AAPL")

        # pnl = (118-100)*10 = 180 (positive)
        assert engine.risk_manager.daily_pnl == pytest.approx(180.0, abs=0.01)
        # Positive PnL should reset consecutive losses
        assert engine.risk_manager.consecutive_losses == 0


# ===================================================================
# 5. Take-profit execution tests -- _execute_take_profit
# ===================================================================
class TestExecuteTakeProfit:
    """Test _execute_take_profit for partial, full, and rejected orders."""

    def test_partial_sell_correct_quantity(self, engine):
        """Sell quantity is capped by position quantity."""
        pos = _make_position(
            symbol="AAPL", quantity=100, avg_cost=100.0, current_price=110.0
        )
        engine.broker.get_position.return_value = pos
        fill_order = _make_order(side=OrderSide.SELL, filled_price=110.0)
        engine.broker.sell.return_value = fill_order

        # Set stop-loss so we can verify it stays
        engine.risk_manager.set_stop_loss("AAPL", entry_price=100.0, stop_price=95.0)

        engine._execute_take_profit("AAPL", quantity=33, target_price=110.0)

        engine.broker.sell.assert_called_once_with("AAPL", 33)
        # PnL = (110-100)*33 = 330
        assert engine.risk_manager.daily_pnl == pytest.approx(330.0, abs=0.01)
        # Stop-loss should still exist (partial sell, position not fully closed)
        assert "AAPL" in engine.risk_manager.stop_losses

    def test_full_exit_removes_stop_loss(self, engine):
        """When sell_qty >= position.quantity, stop-loss should be removed."""
        pos = _make_position(
            symbol="AAPL", quantity=10, avg_cost=100.0, current_price=115.0
        )
        engine.broker.get_position.return_value = pos
        fill_order = _make_order(side=OrderSide.SELL, filled_price=115.0)
        engine.broker.sell.return_value = fill_order

        engine.risk_manager.set_stop_loss("AAPL", entry_price=100.0, stop_price=95.0)

        # Request to sell all 10 shares
        engine._execute_take_profit("AAPL", quantity=10, target_price=115.0)

        # Position fully exited -> stop-loss removed
        assert "AAPL" not in engine.risk_manager.stop_losses

    def test_overshoot_sell_removes_stop_loss(self, engine):
        """If requested quantity > position, sell all and remove stop-loss."""
        pos = _make_position(
            symbol="AAPL", quantity=10, avg_cost=100.0, current_price=115.0
        )
        engine.broker.get_position.return_value = pos
        fill_order = _make_order(side=OrderSide.SELL, filled_price=115.0)
        engine.broker.sell.return_value = fill_order

        engine.risk_manager.set_stop_loss("AAPL", entry_price=100.0, stop_price=95.0)

        engine._execute_take_profit("AAPL", quantity=20, target_price=115.0)

        # Capped to 10, fully exited -> stop removed
        assert "AAPL" not in engine.risk_manager.stop_losses

    def test_rejected_order_handled(self, engine):
        """Rejected order should exit early without updating PnL."""
        pos = _make_position(symbol="AAPL", quantity=10, avg_cost=100.0)
        engine.broker.get_position.return_value = pos
        rejected_order = _make_order(status=OrderStatus.REJECTED)
        engine.broker.sell.return_value = rejected_order

        initial_pnl = engine.risk_manager.daily_pnl
        engine._execute_take_profit("AAPL", quantity=5, target_price=110.0)

        # PnL should NOT change because the order was rejected
        assert engine.risk_manager.daily_pnl == initial_pnl

    def test_none_order_handled(self, engine):
        """When broker.sell returns None, should exit gracefully."""
        pos = _make_position(symbol="AAPL", quantity=10)
        engine.broker.get_position.return_value = pos
        engine.broker.sell.return_value = None

        initial_pnl = engine.risk_manager.daily_pnl
        engine._execute_take_profit("AAPL", quantity=5, target_price=110.0)

        assert engine.risk_manager.daily_pnl == initial_pnl

    def test_no_position_is_noop(self, engine):
        """If no position, nothing happens."""
        engine.broker.get_position.return_value = None

        engine._execute_take_profit("AAPL", quantity=5, target_price=110.0)
        engine.broker.sell.assert_not_called()

    def test_zero_quantity_position_is_noop(self, engine):
        pos = _make_position(quantity=0)
        engine.broker.get_position.return_value = pos

        engine._execute_take_profit("AAPL", quantity=5, target_price=110.0)
        engine.broker.sell.assert_not_called()

    def test_sell_qty_capped_to_position(self, engine):
        """If requested quantity > position quantity, sell only what we have."""
        pos = _make_position(
            symbol="AAPL", quantity=5, avg_cost=100.0, current_price=110.0
        )
        engine.broker.get_position.return_value = pos
        fill_order = _make_order(side=OrderSide.SELL, filled_price=110.0)
        engine.broker.sell.return_value = fill_order

        engine._execute_take_profit("AAPL", quantity=20, target_price=110.0)

        engine.broker.sell.assert_called_once_with("AAPL", 5)

    def test_notifier_called_on_take_profit(self, engine):
        engine.notifier = MagicMock()
        pos = _make_position(
            symbol="AAPL", quantity=100, avg_cost=100.0, current_price=110.0
        )
        engine.broker.get_position.return_value = pos
        fill_order = _make_order(side=OrderSide.SELL, filled_price=110.0)
        engine.broker.sell.return_value = fill_order

        engine._execute_take_profit("AAPL", quantity=33, target_price=110.0)

        engine.notifier.notify_take_profit.assert_called_once()
        call_kwargs = engine.notifier.notify_take_profit.call_args.kwargs
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["quantity"] == 33
        assert call_kwargs["gain_amount"] > 0

    def test_pnl_uses_target_price_when_no_fill_price(self, engine):
        """If order has no filled_price, use target_price for PnL calc."""
        pos = _make_position(
            symbol="AAPL", quantity=100, avg_cost=100.0, current_price=110.0
        )
        engine.broker.get_position.return_value = pos
        fill_order = _make_order(side=OrderSide.SELL)
        fill_order.filled_price = None
        engine.broker.sell.return_value = fill_order

        engine._execute_take_profit("AAPL", quantity=33, target_price=112.0)

        # pnl = (target_price - avg_cost) * sell_qty = (112-100)*33 = 396
        assert engine.risk_manager.daily_pnl == pytest.approx(396.0, abs=0.01)

    def test_zero_sell_qty_is_noop(self, engine):
        """If calculated sell_qty is 0 (e.g., quantity=0 requested), do nothing."""
        pos = _make_position(symbol="AAPL", quantity=10)
        engine.broker.get_position.return_value = pos

        engine._execute_take_profit("AAPL", quantity=0, target_price=110.0)

        engine.broker.sell.assert_not_called()


# ===================================================================
# 6. Trading cycle tests -- run_trading_cycle
# ===================================================================
class TestRunTradingCycle:
    """Test full run_trading_cycle method."""

    def test_not_running_returns_status(self, engine):
        engine.is_running = False
        result = engine.run_trading_cycle()
        assert result["status"] == "not_running"

    def test_market_closed_returns_status(self, _patch_settings):
        """When market is closed (and not ignored), status is market_closed."""
        from src.core.trading_engine import TradingEngine

        eng = TradingEngine(
            symbols=["AAPL"],
            simulated=True,
            ignore_market_hours=False,
        )
        eng.broker = MagicMock()
        eng.is_running = True

        # Force market to be closed
        with patch.object(eng, "_is_market_open", return_value=False):
            result = eng.run_trading_cycle()
        assert result["status"] == "market_closed"

    def test_handles_stop_loss_triggers(self, engine):
        """If stop-loss triggers, the trade list includes STOP_LOSS entries."""
        account = _make_account()
        aapl_pos = _make_position(
            symbol="AAPL", quantity=10, current_price=140.0
        )
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = [aapl_pos]
        engine.broker.get_quotes.return_value = {
            "AAPL": _make_quote("AAPL", last=140.0),
        }

        # Pre-set a stop-loss that is already triggered (stop > current price)
        engine.risk_manager.set_stop_loss(
            "AAPL", entry_price=150.0, stop_price=145.0
        )

        # Wire _execute_stop_loss to track calls
        engine.broker.get_position.return_value = aapl_pos
        sell_order = _make_order(side=OrderSide.SELL, filled_price=140.0)
        engine.broker.place_order.return_value = sell_order

        # No ML recommendations
        engine.strategy = MagicMock()
        engine.strategy.get_trade_recommendations.return_value = []

        result = engine.run_trading_cycle()

        assert result["status"] == "success"
        stop_trades = [t for t in result["trades"] if t["type"] == "STOP_LOSS"]
        assert len(stop_trades) == 1
        assert stop_trades[0]["symbol"] == "AAPL"

    def test_saves_risk_state_after_trades(self, engine):
        """After trades are executed, risk_manager.save_state is called."""
        account = _make_account()
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []
        engine.broker.get_quotes.return_value = {}

        mock_strategy = MagicMock()
        rec = {
            "symbol": "AAPL",
            "action": "BUY",
            "shares": 10,
            "price": 150.0,
            "stop_loss": 142.50,
            "confidence": 0.75,
        }
        mock_strategy.get_trade_recommendations.return_value = [rec]
        engine.strategy = mock_strategy

        order = _make_order()
        engine.broker.place_order.return_value = order

        with patch.object(engine.risk_manager, "save_state") as mock_save:
            result = engine.run_trading_cycle()

        assert result["status"] == "success"
        assert len(result["trades"]) > 0
        mock_save.assert_called_once()

    def test_no_save_when_no_trades(self, engine):
        """If no trades are made, save_state is NOT called."""
        account = _make_account()
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []
        engine.broker.get_quotes.return_value = {}
        engine.strategy = MagicMock()
        engine.strategy.get_trade_recommendations.return_value = []

        with patch.object(engine.risk_manager, "save_state") as mock_save:
            result = engine.run_trading_cycle()

        assert result["status"] == "success"
        mock_save.assert_not_called()

    def test_error_handling_produces_error_status(self, engine):
        """If broker blows up, result has status=error."""
        engine.broker.get_account_info.side_effect = RuntimeError("broker down")

        result = engine.run_trading_cycle()

        assert result["status"] == "error"
        assert len(result["errors"]) > 0
        assert "broker down" in result["errors"][0]

    def test_error_sends_notification(self, engine):
        engine.notifier = MagicMock()
        engine.broker.get_account_info.side_effect = RuntimeError("broker down")

        engine.run_trading_cycle()

        engine.notifier.notify_error.assert_called_once()

    def test_agent_halt_blocks_cycle(self, engine):
        """When agent orchestrator halts trading, cycle is blocked."""
        mock_orch = MagicMock()
        mock_orch.is_trading_halted.return_value = True
        mock_orch.get_halt_reason.return_value = "Risk limit breached by agent"
        engine.agent_orchestrator = mock_orch

        account = _make_account()
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []

        result = engine.run_trading_cycle()

        assert result["status"] == "blocked"
        assert "Agent halt" in result["block_reason"]

    def test_risk_check_blocks_cycle(self, engine):
        """When risk_manager.check_can_trade rejects, cycle is blocked."""
        engine.risk_manager.trading_halted = True

        account = _make_account()
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []

        result = engine.run_trading_cycle()

        assert result["status"] == "blocked"

    def test_drawdown_blocks_cycle(self, engine):
        """When drawdown protection fires, the cycle is blocked."""
        # Set peak high and current value low to trigger drawdown
        engine.risk_manager.peak_portfolio_value = 200_000.0
        engine.risk_manager.max_drawdown_pct = 0.10

        account = _make_account(portfolio_value=170_000.0)  # 15% drawdown
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []

        result = engine.run_trading_cycle()

        assert result["status"] == "blocked"
        assert "drawdown" in result["block_reason"].lower() or "Drawdown" in result["block_reason"]

    def test_drawdown_sends_risk_notification(self, engine):
        """Drawdown breach should notify via notifier."""
        engine.notifier = MagicMock()
        engine.risk_manager.peak_portfolio_value = 200_000.0
        engine.risk_manager.max_drawdown_pct = 0.10

        account = _make_account(portfolio_value=170_000.0)
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []

        engine.run_trading_cycle()

        engine.notifier.notify_risk_warning.assert_called_once()

    def test_regime_detected_in_results(self, engine):
        """When regime_detector is present, regime is included in results."""
        from src.risk.regime_detector import MarketRegime

        mock_detector = MagicMock()
        mock_detector.detect_regime.return_value = MarketRegime.BULL
        engine.regime_detector = mock_detector

        account = _make_account()
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []
        engine.broker.get_quotes.return_value = {}
        engine.strategy = MagicMock()
        engine.strategy.get_trade_recommendations.return_value = []

        result = engine.run_trading_cycle()

        assert result["status"] == "success"
        assert result["regime"] == "bull"

    def test_take_profit_triggers_in_cycle(self, engine):
        """If take-profit levels are hit, the cycle includes TAKE_PROFIT entries."""
        account = _make_account()
        aapl_pos = _make_position(
            symbol="AAPL", quantity=100, avg_cost=100.0, current_price=110.0
        )
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = [aapl_pos]
        engine.broker.get_quotes.return_value = {
            "AAPL": _make_quote("AAPL", last=110.0),
        }

        # Set take-profit that will trigger at current price
        engine.risk_manager.set_take_profit(
            "AAPL",
            entry_price=100.0,
            quantity=100,
            tp_levels=[(0.05, 0.33)],  # 5% = $105, current is $110 -> triggers
        )

        # Wire broker.sell and broker.get_position for the _execute_take_profit call
        engine.broker.get_position.return_value = aapl_pos
        fill_order = _make_order(side=OrderSide.SELL, filled_price=110.0)
        engine.broker.sell.return_value = fill_order

        engine.strategy = MagicMock()
        engine.strategy.get_trade_recommendations.return_value = []

        result = engine.run_trading_cycle()

        tp_trades = [t for t in result["trades"] if t["type"] == "TAKE_PROFIT"]
        assert len(tp_trades) == 1
        assert tp_trades[0]["symbol"] == "AAPL"

    def test_cycle_refreshes_positions_after_stops(self, engine):
        """After stop-loss/take-profit triggers, positions are refreshed."""
        account = _make_account()
        aapl_pos = _make_position(
            symbol="AAPL", quantity=10, current_price=140.0
        )
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = [aapl_pos]
        engine.broker.get_quotes.return_value = {
            "AAPL": _make_quote("AAPL", last=140.0),
        }

        # Set a stop-loss that triggers
        engine.risk_manager.set_stop_loss(
            "AAPL", entry_price=150.0, stop_price=145.0
        )
        engine.broker.get_position.return_value = aapl_pos
        sell_order = _make_order(side=OrderSide.SELL, filled_price=140.0)
        engine.broker.place_order.return_value = sell_order

        engine.strategy = MagicMock()
        engine.strategy.get_trade_recommendations.return_value = []

        engine.run_trading_cycle()

        # get_positions should be called at least twice: initial + refresh
        assert engine.broker.get_positions.call_count >= 2

    def test_no_model_skips_recommendations(self, engine):
        """When model is not loaded, recommendations list is empty."""
        engine.model_loaded = False

        account = _make_account()
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []
        engine.broker.get_quotes.return_value = {}

        result = engine.run_trading_cycle()

        assert result["status"] == "success"
        assert len(result["trades"]) == 0

    def test_trade_execution_error_captured(self, engine):
        """If a single trade throws, error is captured but cycle continues."""
        account = _make_account()
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []
        engine.broker.get_quotes.return_value = {}

        mock_strategy = MagicMock()
        mock_strategy.get_trade_recommendations.return_value = [
            {
                "symbol": "AAPL",
                "action": "BUY",
                "shares": 10,
                "price": 150.0,
                "stop_loss": 142.50,
                "confidence": 0.75,
            }
        ]
        engine.strategy = mock_strategy

        # Make broker.place_order raise for this trade
        engine.broker.place_order.side_effect = RuntimeError("order failed")

        result = engine.run_trading_cycle()

        assert result["status"] == "success"
        assert len(result["errors"]) == 1
        assert "order failed" in result["errors"][0]

    def test_multiple_stop_losses_in_single_cycle(self, engine):
        """Multiple stop-losses can trigger in a single cycle."""
        account = _make_account()
        aapl_pos = _make_position(
            symbol="AAPL", quantity=10, avg_cost=150.0, current_price=140.0
        )
        msft_pos = _make_position(
            symbol="MSFT", quantity=5, avg_cost=300.0, current_price=280.0
        )
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = [aapl_pos, msft_pos]
        engine.broker.get_quotes.return_value = {
            "AAPL": _make_quote("AAPL", last=140.0),
            "MSFT": _make_quote("MSFT", last=280.0),
        }

        # Set stop-losses that both trigger
        engine.risk_manager.set_stop_loss(
            "AAPL", entry_price=150.0, stop_price=145.0
        )
        engine.risk_manager.set_stop_loss(
            "MSFT", entry_price=300.0, stop_price=290.0
        )

        def get_position_side_effect(symbol):
            return {"AAPL": aapl_pos, "MSFT": msft_pos}.get(symbol)

        engine.broker.get_position.side_effect = get_position_side_effect
        sell_order = _make_order(side=OrderSide.SELL, filled_price=140.0)
        engine.broker.place_order.return_value = sell_order

        engine.strategy = MagicMock()
        engine.strategy.get_trade_recommendations.return_value = []

        result = engine.run_trading_cycle()

        stop_trades = [t for t in result["trades"] if t["type"] == "STOP_LOSS"]
        assert len(stop_trades) == 2

    def test_successful_cycle_returns_timestamp(self, engine):
        """Cycle results should include a timestamp."""
        account = _make_account()
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []
        engine.broker.get_quotes.return_value = {}
        engine.strategy = MagicMock()
        engine.strategy.get_trade_recommendations.return_value = []

        result = engine.run_trading_cycle()

        assert "timestamp" in result
        assert isinstance(result["timestamp"], datetime)

    def test_consecutive_loss_pause_blocks_cycle(self, engine):
        """After N consecutive losses, check_can_trade returns False."""
        engine.risk_manager.consecutive_losses = 3
        engine.risk_manager.pause_after_consecutive_losses = 3

        account = _make_account()
        engine.broker.get_account_info.return_value = account
        engine.broker.get_positions.return_value = []

        result = engine.run_trading_cycle()

        assert result["status"] == "blocked"
        assert "consecutive" in result["block_reason"].lower()
