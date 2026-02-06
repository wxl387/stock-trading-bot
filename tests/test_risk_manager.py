"""
Comprehensive unit tests for RiskManager -- the trading bot's core risk module.

Covers: position sizing, stop losses, take profits, daily limits,
risk checks, VIX-based sizing, drawdown protection, and state persistence.

All tests are independent, fast, and require no network access.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is on sys.path so `config.settings` resolves.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.risk.risk_manager import (
    RiskManager,
    StopLoss,
    StopLossType,
    TakeProfitLevel,
    TakeProfitOrder,
    RiskCheckResult,
    DEFAULT_VIX_THRESHOLDS,
    DEFAULT_VIX_MULTIPLIERS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rm():
    """Fresh RiskManager with sensible defaults for testing."""
    return RiskManager(
        max_position_pct=0.10,
        max_daily_loss_pct=0.05,
        max_total_exposure=0.80,
        max_sector_exposure=0.30,
        max_positions=20,
        max_daily_trades=50,
        pause_after_consecutive_losses=3,
    )


@pytest.fixture
def rm_no_vix():
    """RiskManager with VIX sizing disabled (deterministic sizes)."""
    m = RiskManager(max_position_pct=0.10)
    m.vix_sizing_enabled = False
    return m


@pytest.fixture
def rm_with_equity(rm):
    """RiskManager whose daily limits have been seeded with $100 000 equity."""
    rm.reset_daily_limits(100_000.0)
    return rm


# ===================================================================
# 1. Position Sizing -- calculate_position_size()
# ===================================================================

class TestPositionSizing:
    """Tests for calculate_position_size()."""

    def test_basic_sizing_without_stop_loss(self, rm_no_vix):
        """Without stop loss the result equals max-position-pct shares."""
        shares = rm_no_vix.calculate_position_size(
            portfolio_value=100_000,
            entry_price=50.0,
            stop_loss_price=None,
        )
        # max position = 10% of 100k = 10 000 => 10 000 / 50 = 200 shares
        assert shares == 200

    def test_sizing_with_stop_loss(self, rm_no_vix):
        """Stop-loss-based risk sizing returns fewer shares when risk is tight."""
        shares = rm_no_vix.calculate_position_size(
            portfolio_value=100_000,
            entry_price=100.0,
            stop_loss_price=95.0,
            risk_per_trade=0.01,
        )
        # risk amount = 1% of 100k = 1 000
        # risk per share = 100 - 95 = 5
        # shares by risk = 1000 / 5 = 200
        # max by value = 10% of 100k / 100 = 100  <-- tighter limit
        assert shares == 100

    def test_stop_loss_tighter_than_max_value(self, rm_no_vix):
        """When risk-based sizing is tighter, it wins over max-value sizing."""
        shares = rm_no_vix.calculate_position_size(
            portfolio_value=100_000,
            entry_price=100.0,
            stop_loss_price=99.0,     # very tight stop => 1 dollar risk per share
            risk_per_trade=0.005,     # 0.5% risk => $500 total risk
        )
        # shares by risk = 500 / 1 = 500
        # max by value   = 10 000 / 100 = 100  <-- tighter
        assert shares == 100

    def test_risk_based_is_binding(self, rm_no_vix):
        """When risk-based sizing is smaller, it is the binding constraint."""
        shares = rm_no_vix.calculate_position_size(
            portfolio_value=100_000,
            entry_price=10.0,
            stop_loss_price=9.0,     # $1 risk per share
            risk_per_trade=0.005,    # risk amount = $500
        )
        # shares by risk = 500 / 1 = 500
        # max by value   = 10 000 / 10 = 1000
        # risk is binding
        assert shares == 500

    def test_zero_entry_price_returns_zero(self, rm_no_vix):
        shares = rm_no_vix.calculate_position_size(
            portfolio_value=100_000, entry_price=0.0
        )
        assert shares == 0

    def test_negative_entry_price_returns_zero(self, rm_no_vix):
        shares = rm_no_vix.calculate_position_size(
            portfolio_value=100_000, entry_price=-5.0
        )
        assert shares == 0

    def test_stop_equal_to_entry_uses_max(self, rm_no_vix):
        """When stop == entry, risk_per_share is 0 so max_shares_by_value is used."""
        shares = rm_no_vix.calculate_position_size(
            portfolio_value=100_000,
            entry_price=50.0,
            stop_loss_price=50.0,
        )
        assert shares == 200  # falls back to max_shares_by_value

    @patch.object(RiskManager, "calculate_volatility_multiplier", return_value=0.5)
    def test_vix_multiplier_reduces_shares(self, mock_mult):
        """VIX multiplier of 0.5 should roughly halve position size."""
        rm = RiskManager(max_position_pct=0.10)
        rm.vix_sizing_enabled = True
        shares = rm.calculate_position_size(
            portfolio_value=100_000,
            entry_price=50.0,
            apply_vix_sizing=True,
        )
        # base = 200, multiplied by 0.5 => 100
        assert shares == 100

    @patch.object(RiskManager, "calculate_volatility_multiplier", return_value=1.2)
    def test_vix_multiplier_increases_shares(self, mock_mult):
        """VIX multiplier > 1 can increase position size."""
        rm = RiskManager(max_position_pct=0.10)
        rm.vix_sizing_enabled = True
        shares = rm.calculate_position_size(
            portfolio_value=100_000,
            entry_price=50.0,
            apply_vix_sizing=True,
        )
        # base = 200, * 1.2 = 240
        assert shares == 240

    @patch.object(RiskManager, "calculate_volatility_multiplier", return_value=0.001)
    def test_vix_multiplier_floors_at_one_share(self, mock_mult):
        """Very small multiplier should still return at least 1 if base > 0."""
        rm = RiskManager(max_position_pct=0.10)
        rm.vix_sizing_enabled = True
        shares = rm.calculate_position_size(
            portfolio_value=100_000,
            entry_price=50.0,
            apply_vix_sizing=True,
        )
        assert shares >= 1

    def test_max_position_clamping(self, rm_no_vix):
        """Shares should never exceed max_position_pct of portfolio."""
        shares = rm_no_vix.calculate_position_size(
            portfolio_value=100_000,
            entry_price=1.0,          # very cheap stock
            stop_loss_price=None,
        )
        max_allowed_value = 100_000 * rm_no_vix.max_position_pct
        assert shares * 1.0 <= max_allowed_value


# ===================================================================
# 2. Stop Losses -- set_stop_loss, check_stop_losses, trailing
# ===================================================================

class TestStopLosses:
    """Tests for stop-loss calculation, setting, checking, and trailing."""

    # -- calculate_stop_loss --

    def test_fixed_stop_loss(self, rm):
        price = rm.calculate_stop_loss(100.0, StopLossType.FIXED, fixed_pct=0.05)
        assert price == pytest.approx(95.0)

    def test_atr_based_stop_loss(self, rm):
        price = rm.calculate_stop_loss(
            100.0, StopLossType.ATR, atr=2.0, atr_multiplier=2.0
        )
        # 100 - 2*2 = 96
        assert price == pytest.approx(96.0)

    def test_atr_none_falls_back_to_fixed(self, rm):
        """If ATR is None the method falls back to fixed percentage."""
        price = rm.calculate_stop_loss(
            100.0, StopLossType.ATR, atr=None, fixed_pct=0.05
        )
        assert price == pytest.approx(95.0)

    def test_atr_zero_falls_back_to_fixed(self, rm):
        price = rm.calculate_stop_loss(
            100.0, StopLossType.ATR, atr=0.0, fixed_pct=0.05
        )
        assert price == pytest.approx(95.0)

    def test_atr_negative_falls_back_to_fixed(self, rm):
        price = rm.calculate_stop_loss(
            100.0, StopLossType.ATR, atr=-1.0, fixed_pct=0.05
        )
        assert price == pytest.approx(95.0)

    def test_trailing_stop_initial_equals_fixed(self, rm):
        price = rm.calculate_stop_loss(
            100.0, StopLossType.TRAILING, fixed_pct=0.05
        )
        assert price == pytest.approx(95.0)

    def test_stop_clamped_when_above_entry(self, rm):
        """If stop >= entry, the code clamps it to entry * 0.95."""
        price = rm.calculate_stop_loss(
            100.0, StopLossType.FIXED, fixed_pct=-0.10  # yields 110 > entry
        )
        assert price == pytest.approx(95.0)

    def test_negative_stop_clamped(self, rm):
        """Large ATR can produce negative stop; code clamps to entry * 0.95."""
        price = rm.calculate_stop_loss(
            10.0, StopLossType.ATR, atr=100.0, atr_multiplier=2.0
        )
        # 10 - 200 = -190 -> negative -> clamped to 10 * 0.95 = 9.5
        assert price == pytest.approx(9.5)

    # -- set_stop_loss --

    def test_set_stop_loss_fixed(self, rm):
        sl = rm.set_stop_loss("AAPL", 150.0, StopLossType.FIXED)
        assert sl.symbol == "AAPL"
        assert sl.stop_type == StopLossType.FIXED
        assert sl.stop_price < 150.0
        assert sl.entry_price == 150.0
        assert "AAPL" in rm.stop_losses

    def test_set_stop_loss_explicit_price(self, rm):
        sl = rm.set_stop_loss("MSFT", 300.0, stop_price=280.0)
        assert sl.stop_price == 280.0

    def test_set_stop_loss_trailing_has_distance(self, rm):
        sl = rm.set_stop_loss("NVDA", 200.0, StopLossType.TRAILING)
        assert sl.trailing_distance is not None
        assert sl.trailing_distance > 0
        assert sl.trailing_distance == pytest.approx(200.0 - sl.stop_price)

    # -- check_stop_losses --

    def test_stop_triggered(self, rm):
        rm.set_stop_loss("AAPL", 100.0, StopLossType.FIXED, stop_price=95.0)
        triggered = rm.check_stop_losses({"AAPL": 94.0})
        assert "AAPL" in triggered

    def test_stop_not_triggered(self, rm):
        rm.set_stop_loss("AAPL", 100.0, StopLossType.FIXED, stop_price=95.0)
        triggered = rm.check_stop_losses({"AAPL": 96.0})
        assert triggered == []

    def test_stop_exact_boundary(self, rm):
        """Price exactly at stop should trigger."""
        rm.set_stop_loss("AAPL", 100.0, StopLossType.FIXED, stop_price=95.0)
        triggered = rm.check_stop_losses({"AAPL": 95.0})
        assert "AAPL" in triggered

    def test_stop_multiple_symbols(self, rm):
        rm.set_stop_loss("AAPL", 100.0, stop_price=95.0)
        rm.set_stop_loss("MSFT", 300.0, stop_price=285.0)
        triggered = rm.check_stop_losses({"AAPL": 94.0, "MSFT": 290.0})
        assert "AAPL" in triggered
        assert "MSFT" not in triggered

    def test_stop_symbol_not_in_prices(self, rm):
        rm.set_stop_loss("AAPL", 100.0, stop_price=95.0)
        triggered = rm.check_stop_losses({"MSFT": 50.0})
        assert triggered == []

    # -- update_trailing_stop --

    def test_trailing_stop_moves_up(self, rm):
        rm.set_stop_loss("AAPL", 100.0, StopLossType.TRAILING, stop_price=95.0)
        new_stop = rm.update_trailing_stop("AAPL", 110.0)
        # trailing distance = 100 - 95 = 5, new stop = 110 - 5 = 105
        assert new_stop == pytest.approx(105.0)
        assert rm.stop_losses["AAPL"].stop_price == pytest.approx(105.0)

    def test_trailing_stop_does_not_move_down(self, rm):
        rm.set_stop_loss("AAPL", 100.0, StopLossType.TRAILING, stop_price=95.0)
        result = rm.update_trailing_stop("AAPL", 98.0)
        # new stop would be 98 - 5 = 93, which is < 95, so no update
        assert result is None
        assert rm.stop_losses["AAPL"].stop_price == pytest.approx(95.0)

    def test_trailing_stop_non_trailing_type(self, rm):
        rm.set_stop_loss("AAPL", 100.0, StopLossType.FIXED, stop_price=95.0)
        result = rm.update_trailing_stop("AAPL", 110.0)
        assert result is None

    def test_trailing_stop_unknown_symbol(self, rm):
        result = rm.update_trailing_stop("NOPE", 110.0)
        assert result is None

    # -- remove_stop_loss --

    def test_remove_stop_loss(self, rm):
        rm.set_stop_loss("AAPL", 100.0, stop_price=95.0)
        rm.remove_stop_loss("AAPL")
        assert "AAPL" not in rm.stop_losses

    def test_remove_nonexistent_stop_no_error(self, rm):
        rm.remove_stop_loss("NOPE")  # should not raise


# ===================================================================
# 3. Take Profits -- set_take_profit, check_take_profits
# ===================================================================

class TestTakeProfits:
    """Tests for take-profit levels and partial exits."""

    def test_default_levels_created(self, rm):
        tp = rm.set_take_profit("AAPL", 100.0, 100)
        assert len(tp.levels) == 3
        # default percentages: 5%, 10%, 15%
        assert tp.levels[0].target_pct == pytest.approx(0.05)
        assert tp.levels[1].target_pct == pytest.approx(0.10)
        assert tp.levels[2].target_pct == pytest.approx(0.15)

    def test_target_prices_correct(self, rm):
        tp = rm.set_take_profit("AAPL", 100.0, 100)
        assert tp.levels[0].target_price == pytest.approx(105.0)
        assert tp.levels[1].target_price == pytest.approx(110.0)
        assert tp.levels[2].target_price == pytest.approx(115.0)

    def test_custom_levels(self, rm):
        tp = rm.set_take_profit(
            "AAPL", 200.0, 50,
            tp_levels=[(0.02, 0.5), (0.08, 1.0)],
        )
        assert len(tp.levels) == 2
        assert tp.levels[0].target_price == pytest.approx(204.0)
        assert tp.levels[1].target_price == pytest.approx(216.0)

    def test_quantity_remaining_initialized(self, rm):
        tp = rm.set_take_profit("AAPL", 100.0, 100)
        assert tp.quantity_remaining == 100

    def test_tp_level1_triggers_partial_exit(self, rm):
        rm.set_take_profit("AAPL", 100.0, 100)  # default levels
        sales = rm.check_take_profits({"AAPL": 106.0})
        assert len(sales) == 1
        symbol, qty, price = sales[0]
        assert symbol == "AAPL"
        # exit_pct for level 0 = 0.33 => int(100 * 0.33) = 33
        assert qty == 33
        assert price == 106.0

    def test_tp_levels_trigger_in_order(self, rm):
        """If price jumps past two levels at once, both trigger sequentially."""
        rm.set_take_profit("AAPL", 100.0, 100)
        # Price at 111 exceeds both 105 (5%) and 110 (10%)
        sales = rm.check_take_profits({"AAPL": 111.0})
        assert len(sales) == 2
        assert sales[0][0] == "AAPL"
        assert sales[1][0] == "AAPL"
        # Level 0: int(100 * 0.33) = 33, remaining 67
        # Level 1: int(67 * 0.50) = 33, remaining 34
        assert sales[0][1] == 33
        assert sales[1][1] == 33

    def test_tp_full_exit_removes_order(self, rm):
        rm.set_take_profit("AAPL", 100.0, 100)
        # Trigger all three levels at once
        rm.check_take_profits({"AAPL": 120.0})
        # After full exit, TP order should be removed
        assert "AAPL" not in rm.take_profits

    def test_tp_partial_exit_preserves_order(self, rm):
        rm.set_take_profit("AAPL", 100.0, 100)
        rm.check_take_profits({"AAPL": 106.0})  # triggers level 0 only
        assert "AAPL" in rm.take_profits
        assert rm.take_profits["AAPL"].quantity_remaining == 67

    def test_tp_already_triggered_skipped(self, rm):
        rm.set_take_profit("AAPL", 100.0, 100)
        rm.check_take_profits({"AAPL": 106.0})  # trigger level 0
        sales = rm.check_take_profits({"AAPL": 106.0})  # same price again
        assert sales == []

    def test_tp_symbol_not_in_prices(self, rm):
        rm.set_take_profit("AAPL", 100.0, 100)
        sales = rm.check_take_profits({"MSFT": 999.0})
        assert sales == []

    def test_remove_take_profit(self, rm):
        rm.set_take_profit("AAPL", 100.0, 100)
        rm.remove_take_profit("AAPL")
        assert "AAPL" not in rm.take_profits

    def test_remove_nonexistent_tp_no_error(self, rm):
        rm.remove_take_profit("NOPE")

    def test_tp_small_quantity_ensures_at_least_1(self, rm):
        """If exit_pct rounds to 0 shares but remaining > 0, sell remaining."""
        rm.set_take_profit("AAPL", 100.0, 2, tp_levels=[(0.05, 0.01)])
        sales = rm.check_take_profits({"AAPL": 106.0})
        # int(2 * 0.01) = 0 -> fallback to quantity_remaining (2)
        assert len(sales) == 1
        assert sales[0][1] == 2


# ===================================================================
# 4. Daily Limits -- reset, update_pnl, check_can_trade
# ===================================================================

class TestDailyLimits:
    """Tests for daily loss halt, consecutive-loss pause, trade count."""

    def test_reset_daily_limits(self, rm):
        rm.daily_pnl = -999.0
        rm.daily_trades = 42
        rm.trading_halted = True
        rm.reset_daily_limits(100_000.0)
        assert rm.daily_pnl == 0.0
        assert rm.daily_trades == 0
        assert rm.trading_halted is False
        assert rm.starting_equity == 100_000.0

    def test_reset_seeds_peak_on_first_call(self, rm):
        assert rm.peak_portfolio_value == 0.0
        rm.reset_daily_limits(100_000.0)
        assert rm.peak_portfolio_value == 100_000.0

    def test_reset_does_not_overwrite_nonzero_peak(self, rm):
        rm.peak_portfolio_value = 120_000.0
        rm.reset_daily_limits(100_000.0)
        assert rm.peak_portfolio_value == 120_000.0

    def test_update_pnl_accumulates(self, rm_with_equity):
        rm_with_equity.update_pnl(500.0)
        rm_with_equity.update_pnl(300.0)
        assert rm_with_equity.daily_pnl == pytest.approx(800.0)
        assert rm_with_equity.daily_trades == 2

    def test_daily_loss_halt_triggers(self, rm_with_equity):
        """Loss exceeding max_daily_loss_pct halts trading."""
        # 5% of 100k = 5000
        result = rm_with_equity.update_pnl(-5500.0, is_loss=True)
        assert result is False
        assert rm_with_equity.trading_halted is True

    def test_daily_loss_halt_not_triggered_below_threshold(self, rm_with_equity):
        result = rm_with_equity.update_pnl(-3000.0, is_loss=True)
        assert result is True
        assert rm_with_equity.trading_halted is False

    def test_consecutive_losses_pause(self, rm_with_equity):
        rm_with_equity.update_pnl(-100.0, is_loss=True)
        rm_with_equity.update_pnl(-100.0, is_loss=True)
        rm_with_equity.update_pnl(-100.0, is_loss=True)
        assert rm_with_equity.consecutive_losses == 3
        check = rm_with_equity.check_can_trade()
        assert check.approved is False
        assert "consecutive" in check.reason.lower()

    def test_win_resets_consecutive_losses(self, rm_with_equity):
        rm_with_equity.update_pnl(-100.0, is_loss=True)
        rm_with_equity.update_pnl(-100.0, is_loss=True)
        rm_with_equity.update_pnl(200.0, is_loss=False)
        assert rm_with_equity.consecutive_losses == 0

    def test_trade_count_limit(self, rm):
        rm.max_daily_trades = 2
        rm.reset_daily_limits(100_000.0)
        rm.update_pnl(10.0)
        rm.update_pnl(10.0)
        check = rm.check_can_trade()
        assert check.approved is False
        assert "trade limit" in check.reason.lower()

    def test_check_can_trade_ok(self, rm_with_equity):
        check = rm_with_equity.check_can_trade()
        assert check.approved is True
        assert check.reason == "OK"

    def test_halted_blocks_trading(self, rm):
        rm.trading_halted = True
        check = rm.check_can_trade()
        assert check.approved is False


# ===================================================================
# 5. Risk Checks -- check_position
# ===================================================================

class TestRiskChecks:
    """Tests for the composite check_position() gate."""

    def test_approved_basic(self, rm_with_equity):
        result = rm_with_equity.check_position(
            symbol="AAPL",
            quantity=10,
            price=100.0,
            portfolio_value=100_000.0,
            current_positions={},
        )
        assert result.approved is True

    def test_position_size_exceeds_max(self, rm_with_equity):
        """Buying more than max_position_pct returns adjusted quantity."""
        result = rm_with_equity.check_position(
            symbol="AAPL",
            quantity=200,
            price=100.0,       # 200 * 100 = 20k = 20%
            portfolio_value=100_000.0,
            current_positions={},
        )
        assert result.approved is False
        assert result.adjusted_quantity is not None
        # adjusted = int(100000 * 0.10 / 100) = 100
        assert result.adjusted_quantity == 100

    def test_total_exposure_exceeded(self, rm_with_equity):
        positions = {"MSFT": 75_000.0}  # 75% invested
        result = rm_with_equity.check_position(
            symbol="AAPL",
            quantity=10,
            price=1000.0,      # 10k more would be 85%
            portfolio_value=100_000.0,
            current_positions=positions,
        )
        assert result.approved is False
        assert "exposure" in result.reason.lower()

    def test_max_positions_reached(self, rm):
        rm.max_positions = 2
        rm.reset_daily_limits(100_000.0)
        positions = {"AAPL": 5000.0, "MSFT": 5000.0}
        result = rm.check_position(
            symbol="GOOGL",   # new symbol
            quantity=1,
            price=100.0,
            portfolio_value=100_000.0,
            current_positions=positions,
        )
        assert result.approved is False
        assert "positions" in result.reason.lower()

    def test_existing_symbol_bypasses_max_positions(self, rm):
        """Adding to an existing position should not be blocked by max positions."""
        rm.max_positions = 2
        rm.reset_daily_limits(100_000.0)
        positions = {"AAPL": 5000.0, "MSFT": 5000.0}
        result = rm.check_position(
            symbol="AAPL",
            quantity=1,
            price=100.0,
            portfolio_value=100_000.0,
            current_positions=positions,
        )
        assert result.approved is True

    def test_zero_portfolio_value(self, rm_with_equity):
        result = rm_with_equity.check_position(
            symbol="AAPL",
            quantity=1,
            price=100.0,
            portfolio_value=0.0,
            current_positions={},
        )
        assert result.approved is False
        assert result.adjusted_quantity == 0

    def test_negative_portfolio_value(self, rm_with_equity):
        result = rm_with_equity.check_position(
            symbol="AAPL",
            quantity=1,
            price=100.0,
            portfolio_value=-10_000.0,
            current_positions={},
        )
        assert result.approved is False

    def test_halted_trading_propagates(self, rm):
        rm.trading_halted = True
        result = rm.check_position(
            symbol="AAPL",
            quantity=1,
            price=100.0,
            portfolio_value=100_000.0,
            current_positions={},
        )
        assert result.approved is False


# ===================================================================
# 6. VIX-Based Sizing -- calculate_volatility_multiplier, set_vix_sizing
# ===================================================================

class TestVIXSizing:
    """Tests for VIX bracket multipliers and configuration."""

    def test_low_vix_bracket(self, rm):
        assert rm.calculate_volatility_multiplier(vix=10.0) == DEFAULT_VIX_MULTIPLIERS["low"]

    def test_normal_vix_bracket(self, rm):
        assert rm.calculate_volatility_multiplier(vix=20.0) == DEFAULT_VIX_MULTIPLIERS["normal"]

    def test_high_vix_bracket(self, rm):
        assert rm.calculate_volatility_multiplier(vix=30.0) == DEFAULT_VIX_MULTIPLIERS["high"]

    def test_extreme_vix_bracket(self, rm):
        assert rm.calculate_volatility_multiplier(vix=50.0) == DEFAULT_VIX_MULTIPLIERS["extreme"]

    def test_vix_exactly_at_low_threshold(self, rm):
        # VIX == 15 falls into "normal" bracket (vix < 25)
        assert rm.calculate_volatility_multiplier(vix=15.0) == DEFAULT_VIX_MULTIPLIERS["normal"]

    def test_vix_exactly_at_normal_threshold(self, rm):
        # VIX == 25 falls into "high" bracket (vix < 35)
        assert rm.calculate_volatility_multiplier(vix=25.0) == DEFAULT_VIX_MULTIPLIERS["high"]

    def test_vix_exactly_at_high_threshold(self, rm):
        # VIX == 35 falls into "extreme" bracket
        assert rm.calculate_volatility_multiplier(vix=35.0) == DEFAULT_VIX_MULTIPLIERS["extreme"]

    def test_vix_none_returns_1(self, rm):
        """When VIX is unavailable, default multiplier is 1.0."""
        rm._cached_vix = None
        with patch.object(rm, "get_current_vix", return_value=None):
            assert rm.calculate_volatility_multiplier(vix=None) == 1.0

    def test_custom_thresholds(self, rm):
        rm.set_vix_sizing(
            enabled=True,
            thresholds={"low": 10, "normal": 20, "high": 30, "extreme": 40},
            multipliers={"low": 1.5, "normal": 1.0, "high": 0.6, "extreme": 0.3},
        )
        assert rm.calculate_volatility_multiplier(vix=5.0) == 1.5
        assert rm.calculate_volatility_multiplier(vix=15.0) == 1.0
        assert rm.calculate_volatility_multiplier(vix=25.0) == 0.6
        assert rm.calculate_volatility_multiplier(vix=35.0) == 0.3

    def test_disabled_returns_1(self, rm):
        """When VIX sizing is disabled, calculate_position_size skips multiplier."""
        rm.set_vix_sizing(enabled=False)
        assert rm.vix_sizing_enabled is False
        # With sizing disabled, calculate_position_size should not apply multiplier
        rm_no = RiskManager(max_position_pct=0.10)
        rm_no.set_vix_sizing(enabled=False)
        shares = rm_no.calculate_position_size(
            portfolio_value=100_000,
            entry_price=50.0,
            apply_vix_sizing=True,  # requested but disabled at instance level
        )
        # Even though apply_vix_sizing=True, vix_sizing_enabled=False skips it
        # So base_shares = int(10_000/50) = 200
        assert shares == 200

    def test_set_vix_sizing_partial_update(self, rm):
        """Passing only some keys should merge, not replace."""
        rm.set_vix_sizing(multipliers={"low": 1.5})
        assert rm.vix_multipliers["low"] == 1.5
        # Others unchanged
        assert rm.vix_multipliers["normal"] == DEFAULT_VIX_MULTIPLIERS["normal"]


# ===================================================================
# 7. Drawdown Protection -- update_portfolio_value, check_drawdown
# ===================================================================

class TestDrawdownProtection:
    """Tests for max-drawdown tracking, recovery mode, and trade blocking."""

    def test_peak_tracking_increases(self, rm):
        rm.update_portfolio_value(100_000.0)
        rm.update_portfolio_value(110_000.0)
        assert rm.peak_portfolio_value == 110_000.0

    def test_peak_tracking_does_not_decrease(self, rm):
        rm.update_portfolio_value(100_000.0)
        rm.update_portfolio_value(90_000.0)
        assert rm.peak_portfolio_value == 100_000.0

    def test_drawdown_pct_calculated(self, rm):
        rm.update_portfolio_value(100_000.0)
        result = rm.update_portfolio_value(90_000.0)
        assert result["drawdown_pct"] == pytest.approx(0.10)

    def test_recovery_mode_entry(self, rm):
        rm.max_drawdown_pct = 0.10
        rm.update_portfolio_value(100_000.0)
        result = rm.update_portfolio_value(89_000.0)
        assert result["recovery_mode"] is True
        assert rm.drawdown_recovery_mode is True

    def test_recovery_mode_exit_via_threshold(self, rm):
        rm.max_drawdown_pct = 0.10
        rm.recovery_threshold = 0.95
        rm.update_portfolio_value(100_000.0)
        rm.update_portfolio_value(85_000.0)  # enter recovery
        assert rm.drawdown_recovery_mode is True
        # Recovery target = 100k * 0.95 = 95k
        result = rm.update_portfolio_value(96_000.0)
        assert result["recovery_mode"] is False

    def test_recovery_mode_exit_via_new_peak(self, rm):
        rm.max_drawdown_pct = 0.10
        rm.update_portfolio_value(100_000.0)
        rm.update_portfolio_value(85_000.0)  # enter recovery
        assert rm.drawdown_recovery_mode is True
        result = rm.update_portfolio_value(101_000.0)  # new peak
        assert result["recovery_mode"] is False

    def test_check_drawdown_blocks_trades(self, rm):
        rm.update_portfolio_value(100_000.0)
        check = rm.check_drawdown(88_000.0)  # 12% drawdown > 10% default
        assert check.approved is False
        assert "drawdown" in check.reason.lower()

    def test_check_drawdown_allows_trades(self, rm):
        rm.update_portfolio_value(100_000.0)
        check = rm.check_drawdown(95_000.0)  # 5% < 10%
        assert check.approved is True

    def test_drawdown_adjusted_size_in_recovery(self, rm):
        rm.drawdown_recovery_mode = True
        rm.recovery_size_multiplier = 0.5
        adjusted = rm.get_drawdown_adjusted_size(100)
        assert adjusted == 50

    def test_drawdown_adjusted_size_normal(self, rm):
        rm.drawdown_recovery_mode = False
        adjusted = rm.get_drawdown_adjusted_size(100)
        assert adjusted == 100

    def test_drawdown_adjusted_size_floors_at_1(self, rm):
        rm.drawdown_recovery_mode = True
        rm.recovery_size_multiplier = 0.001
        adjusted = rm.get_drawdown_adjusted_size(10)
        assert adjusted >= 1

    def test_set_drawdown_protection(self, rm):
        rm.set_drawdown_protection(
            max_drawdown_pct=0.15,
            recovery_threshold=0.90,
            recovery_size_multiplier=0.4,
        )
        assert rm.max_drawdown_pct == 0.15
        assert rm.recovery_threshold == 0.90
        assert rm.recovery_size_multiplier == 0.4

    def test_zero_peak_no_divide_by_zero(self, rm):
        """If peak is 0 drawdown pct should be 0, not an error."""
        rm.peak_portfolio_value = 0.0
        result = rm.update_portfolio_value(0.0)
        assert result["drawdown_pct"] == 0.0


# ===================================================================
# 8. State Persistence -- save_state, load_state
# ===================================================================

class TestStatePersistence:
    """Tests for saving / loading risk state to JSON (disk round-trip)."""

    def _set_state_file(self, rm, path: Path):
        """Helper: point RiskManager.STATE_FILE to a tmp path."""
        # We monkey-patch the class-level attribute on the instance's class
        # to avoid side-effects across tests; instead patch at instance level.
        rm.__class__ = type(
            "RiskManagerTmp", (RiskManager,), {"STATE_FILE": path}
        )

    def test_round_trip_stop_losses(self, rm, tmp_path):
        state_file = tmp_path / "state.json"
        self._set_state_file(rm, state_file)

        rm.set_stop_loss("AAPL", 150.0, StopLossType.FIXED, stop_price=142.0)
        rm.set_stop_loss("MSFT", 300.0, StopLossType.TRAILING, stop_price=285.0)
        rm.save_state()

        rm2 = RiskManager()
        self._set_state_file(rm2, state_file)
        rm2.load_state()

        assert "AAPL" in rm2.stop_losses
        assert rm2.stop_losses["AAPL"].stop_price == pytest.approx(142.0)
        assert rm2.stop_losses["AAPL"].stop_type == StopLossType.FIXED

        assert "MSFT" in rm2.stop_losses
        assert rm2.stop_losses["MSFT"].stop_type == StopLossType.TRAILING
        assert rm2.stop_losses["MSFT"].trailing_distance is not None

    def test_round_trip_take_profits(self, rm, tmp_path):
        state_file = tmp_path / "state.json"
        self._set_state_file(rm, state_file)

        rm.set_take_profit("AAPL", 100.0, 100)
        rm.save_state()

        rm2 = RiskManager()
        self._set_state_file(rm2, state_file)
        rm2.load_state()

        assert "AAPL" in rm2.take_profits
        tp = rm2.take_profits["AAPL"]
        assert tp.total_quantity == 100
        assert tp.quantity_remaining == 100
        assert len(tp.levels) == 3

    def test_round_trip_drawdown_state(self, rm, tmp_path):
        state_file = tmp_path / "state.json"
        self._set_state_file(rm, state_file)

        rm.peak_portfolio_value = 120_000.0
        rm.drawdown_recovery_mode = True
        rm.save_state()

        rm2 = RiskManager()
        self._set_state_file(rm2, state_file)
        rm2.load_state()

        assert rm2.peak_portfolio_value == 120_000.0
        assert rm2.drawdown_recovery_mode is True

    def test_active_symbols_prunes_stale(self, rm, tmp_path):
        state_file = tmp_path / "state.json"
        self._set_state_file(rm, state_file)

        rm.set_stop_loss("AAPL", 150.0, stop_price=142.0)
        rm.set_stop_loss("STALE", 50.0, stop_price=47.0)
        rm.set_take_profit("AAPL", 150.0, 50)
        rm.set_take_profit("STALE", 50.0, 20)
        rm.save_state()

        rm2 = RiskManager()
        self._set_state_file(rm2, state_file)
        rm2.load_state(active_symbols=["AAPL"])

        assert "AAPL" in rm2.stop_losses
        assert "STALE" not in rm2.stop_losses
        assert "AAPL" in rm2.take_profits
        assert "STALE" not in rm2.take_profits

    def test_corrupt_file_handling(self, rm, tmp_path):
        state_file = tmp_path / "state.json"
        state_file.write_text("NOT VALID JSON {{{")
        self._set_state_file(rm, state_file)

        # Should not raise, just log an error and return
        rm.load_state()
        assert rm.stop_losses == {}
        assert rm.take_profits == {}

    def test_missing_file_returns_gracefully(self, rm, tmp_path):
        state_file = tmp_path / "no_such_file.json"
        self._set_state_file(rm, state_file)

        rm.load_state()  # should not raise
        assert rm.stop_losses == {}

    def test_partial_corrupt_entries_skipped(self, rm, tmp_path):
        """If one stop-loss entry is malformed, others still load."""
        state_file = tmp_path / "state.json"
        data = {
            "stop_losses": {
                "AAPL": {
                    "stop_type": "fixed",
                    "stop_price": 140.0,
                    "entry_price": 150.0,
                },
                "BAD": {
                    # missing required keys
                    "stop_type": "unknown_garbage",
                },
            },
            "take_profits": {},
            "peak_portfolio_value": 0.0,
            "drawdown_recovery_mode": False,
            "saved_at": datetime.now().isoformat(),
        }
        state_file.write_text(json.dumps(data))
        self._set_state_file(rm, state_file)
        rm.load_state()

        assert "AAPL" in rm.stop_losses
        assert "BAD" not in rm.stop_losses

    def test_save_creates_parent_dirs(self, rm, tmp_path):
        state_file = tmp_path / "deep" / "nested" / "state.json"
        self._set_state_file(rm, state_file)
        rm.set_stop_loss("AAPL", 100.0, stop_price=95.0)
        rm.save_state()
        assert state_file.exists()

    def test_empty_state_round_trip(self, rm, tmp_path):
        """Saving and loading with no positions is fine."""
        state_file = tmp_path / "state.json"
        self._set_state_file(rm, state_file)
        rm.save_state()

        rm2 = RiskManager()
        self._set_state_file(rm2, state_file)
        rm2.load_state()

        assert rm2.stop_losses == {}
        assert rm2.take_profits == {}


# ===================================================================
# Edge Cases and Regression Guards
# ===================================================================

class TestEdgeCases:
    """Boundary conditions and constructor clamping."""

    def test_max_position_pct_clamped_low(self):
        rm = RiskManager(max_position_pct=0.001)
        assert rm.max_position_pct == 0.01

    def test_max_position_pct_clamped_high(self):
        rm = RiskManager(max_position_pct=5.0)
        assert rm.max_position_pct == 1.0

    def test_max_daily_loss_pct_clamped_low(self):
        rm = RiskManager(max_daily_loss_pct=0.0001)
        assert rm.max_daily_loss_pct == 0.001

    def test_max_daily_loss_pct_clamped_high(self):
        rm = RiskManager(max_daily_loss_pct=2.0)
        assert rm.max_daily_loss_pct == 1.0

    def test_max_total_exposure_clamped(self):
        rm = RiskManager(max_total_exposure=0.001)
        assert rm.max_total_exposure == 0.01

    def test_risk_check_result_defaults(self):
        r = RiskCheckResult(approved=True, reason="OK")
        assert r.adjusted_quantity is None

    def test_take_profit_order_post_init(self):
        tp = TakeProfitOrder(
            symbol="X", entry_price=10.0, total_quantity=50, levels=[]
        )
        assert tp.quantity_remaining == 50

    def test_take_profit_order_explicit_remaining(self):
        tp = TakeProfitOrder(
            symbol="X",
            entry_price=10.0,
            total_quantity=50,
            levels=[],
            quantity_remaining=30,
        )
        assert tp.quantity_remaining == 30
