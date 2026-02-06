"""
Comprehensive unit tests for the trading bot's agent system.

Tests cover:
- RiskGuardianAgent: drawdown monitoring, daily loss tracking, trading halts
- OperationsAgent: action execution, config save, message processing, priority
- AgentOrchestrator: halt checks, lifecycle, agent coordination
"""

import os
import tempfile
import threading
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import agent data structures (these have no heavy dependencies)
# ---------------------------------------------------------------------------
from src.agents.base_agent import (
    AgentMessage,
    AgentRole,
    BaseAgent,
    MessagePriority,
    MessageType,
)


# ============================================================================
# Helpers / lightweight fakes
# ============================================================================

class FakeMessageQueue:
    """In-memory stand-in for the SQLite MessageQueue."""

    def __init__(self):
        self._messages = []

    def enqueue(self, message: AgentMessage) -> None:
        self._messages.append(message)

    def get_messages_for_recipient(self, recipient, processed=None, limit=100):
        results = []
        for m in self._messages:
            if m.recipient != recipient:
                continue
            if processed is not None and m.processed != (not processed is False and processed):
                # Match processed flag exactly
                if processed is True and not m.processed:
                    continue
                if processed is False and m.processed:
                    continue
            results.append(m)
        # Sort by priority desc, then timestamp asc (mimic real queue)
        results.sort(key=lambda m: (-m.priority.value, m.timestamp))
        return results[:limit]

    def mark_processed(self, message_id: str) -> None:
        for m in self._messages:
            if m.id == message_id:
                m.processed = True

    def get_conversation(self, agent1, agent2, limit=50):
        results = []
        for m in self._messages:
            if (m.sender == agent1 and m.recipient == agent2) or \
               (m.sender == agent2 and m.recipient == agent1):
                results.append(m)
        results.sort(key=lambda m: m.timestamp)
        return results[:limit]

    def get_stats(self):
        return {"total_messages": len(self._messages)}

    def close(self):
        pass

    def delete_old_messages(self, days):
        return 0


def _make_config(**overrides):
    """Build a minimal agent config dict."""
    cfg = {
        "enabled": True,
        "use_llm": False,
        "risk_guardian": {
            "risk_check_minutes": 30,
            "drawdown_monitor_minutes": 15,
            "correlation_check_hours": 4,
            "daily_report_time": "16:00",
            "thresholds": {
                "drawdown_warning": 0.05,
                "drawdown_critical": 0.10,
                "position_warning": 0.12,
                "position_critical": 0.15,
                "sector_warning": 0.25,
                "sector_critical": 0.35,
                "daily_loss_warning": 0.03,
                "daily_loss_critical": 0.05,
            },
            "emergency_actions": {
                "enabled": True,
                "auto_reduce_on_critical": True,
                "reduce_percentage": 0.25,
            },
        },
        "operations": {
            "process_messages_minutes": 15,
            "execution_quality_hours": 2,
            "system_health_hours": 4,
            "degradation_check_hours": 12,
            "cooldown_hours": 4,
            "auto_retrain_on_degradation": True,
        },
        "market_intelligence": {
            "alert_on_vix_spike": 25,
        },
    }
    cfg.update(overrides)
    return cfg


def _make_risk_guardian(config=None, mq=None, notifier=None, llm=None):
    """Instantiate a RiskGuardianAgent with all external deps mocked out."""
    from src.agents.risk_guardian import RiskGuardianAgent

    config = config or _make_config()
    mq = mq or FakeMessageQueue()
    agent = RiskGuardianAgent(
        config=config,
        message_queue=mq,
        notifier=notifier,
        llm_client=llm,
    )
    # Prevent lazy-loading from touching real imports
    agent._data_aggregator = MagicMock()
    agent._portfolio_manager = MagicMock()
    agent._symbol_manager = MagicMock()
    return agent


def _make_operations_agent(config=None, mq=None, notifier=None, llm=None):
    """Instantiate an OperationsAgent with all external deps mocked out."""
    from src.agents.operations_agent import OperationsAgent

    config = config or _make_config()
    mq = mq or FakeMessageQueue()
    agent = OperationsAgent(
        config=config,
        message_queue=mq,
        notifier=notifier,
        llm_client=llm,
    )
    # Prevent lazy-loading from touching real imports
    agent._retrainer = MagicMock()
    agent._symbol_manager = MagicMock()
    agent._data_aggregator = MagicMock()
    agent._degradation_monitor = MagicMock()
    return agent


# ============================================================================
# RiskGuardianAgent Tests
# ============================================================================

class TestRiskGuardianDailyLossTracking:
    """Daily loss tracking should reset when the date changes."""

    def test_daily_loss_resets_on_new_date(self):
        """When the date rolls over, daily_start_value should be updated and
        daily_loss should return to zero."""
        rg = _make_risk_guardian()

        # Simulate yesterday's state
        yesterday = date.today() - timedelta(days=1)
        rg._daily_start_date = yesterday
        rg._daily_start_value = 100_000.0
        rg._daily_loss = 0.02  # 2 % loss carried from yesterday

        # Supply a current portfolio value
        rg._portfolio_manager.get_total_value.return_value = 98_000.0

        # Also need a peak so drawdown math doesn't blow up
        rg._peak_portfolio_value = 100_000.0

        messages = rg.run_drawdown_monitor()

        # The date should have been reset to today
        assert rg._daily_start_date == date.today()
        # daily_start_value should now equal the fresh value
        assert rg._daily_start_value == 98_000.0
        # daily_loss recomputed relative to the new start: (98k-98k)/98k = 0
        assert rg._daily_loss == pytest.approx(0.0)

    def test_daily_loss_not_reset_same_date(self):
        """On the same date, daily_start_value should NOT be overwritten."""
        rg = _make_risk_guardian()

        today = date.today()
        rg._daily_start_date = today
        rg._daily_start_value = 100_000.0
        rg._peak_portfolio_value = 105_000.0

        # Portfolio dropped to 97k
        rg._portfolio_manager.get_total_value.return_value = 97_000.0

        rg.run_drawdown_monitor()

        # daily_start_value unchanged
        assert rg._daily_start_value == 100_000.0
        # daily_loss = (100k - 97k) / 100k = 3 %
        assert rg._daily_loss == pytest.approx(0.03)

    def test_daily_loss_reset_when_no_prior_date(self):
        """First invocation (no prior date) should initialise daily tracking."""
        rg = _make_risk_guardian()

        assert rg._daily_start_date is None
        rg._portfolio_manager.get_total_value.return_value = 50_000.0

        rg.run_drawdown_monitor()

        assert rg._daily_start_date == date.today()
        assert rg._daily_start_value == 50_000.0
        assert rg._daily_loss == pytest.approx(0.0)


class TestRiskGuardianTradingHalt:
    """Emergency actions must set the _trading_halted flag."""

    def test_trigger_emergency_sets_halted_flag(self):
        rg = _make_risk_guardian()
        assert rg._trading_halted is False

        msg = rg.trigger_emergency_action(reason="drawdown exceeded 10 %")

        assert rg._trading_halted is True
        assert msg.priority == MessagePriority.URGENT
        assert msg.recipient == AgentRole.OPERATIONS
        assert msg.context.get("halt_trading") is True

    def test_resume_clears_halted_flag(self):
        """Processing a resume_trading ACTION should clear the halt."""
        rg = _make_risk_guardian()
        rg._trading_halted = True

        resume_msg = AgentMessage(
            sender=AgentRole.OPERATIONS,
            recipient=AgentRole.RISK_GUARDIAN,
            message_type=MessageType.ACTION,
            subject="Resume Trading",
            content="resume",
            context={"action": "resume_trading"},
        )

        response = rg.process_message(resume_msg)

        assert rg._trading_halted is False
        assert response is not None
        assert response.message_type == MessageType.ACKNOWLEDGMENT


class TestRiskGuardianDrawdownMonitor:
    """Drawdown calculations based on portfolio value changes."""

    def test_drawdown_warning_generated(self):
        """When drawdown crosses the warning threshold a HIGH-priority
        message should be emitted."""
        rg = _make_risk_guardian()
        rg._peak_portfolio_value = 100_000.0
        rg._daily_start_date = date.today()
        rg._daily_start_value = 100_000.0

        # 6 % drawdown => above 5 % warning, below 10 % critical
        rg._portfolio_manager.get_total_value.return_value = 94_000.0

        messages = rg.run_drawdown_monitor()

        assert rg._current_drawdown == pytest.approx(0.06)
        # Should have a drawdown warning message
        drawdown_msgs = [m for m in messages if "Drawdown" in m.subject and "Warning" in m.subject]
        assert len(drawdown_msgs) == 1
        assert drawdown_msgs[0].priority == MessagePriority.HIGH

    def test_drawdown_critical_generated(self):
        """When drawdown crosses the critical threshold an URGENT message
        should be emitted."""
        rg = _make_risk_guardian()
        rg._peak_portfolio_value = 100_000.0
        rg._daily_start_date = date.today()
        rg._daily_start_value = 100_000.0

        # 12 % drawdown => above 10 % critical
        rg._portfolio_manager.get_total_value.return_value = 88_000.0

        messages = rg.run_drawdown_monitor()

        assert rg._current_drawdown == pytest.approx(0.12)
        critical_msgs = [m for m in messages if "CRITICAL" in m.subject and "Drawdown" in m.subject]
        assert len(critical_msgs) == 1
        assert critical_msgs[0].priority == MessagePriority.URGENT

    def test_peak_value_updates_on_new_high(self):
        """Peak portfolio value should track the high-water mark."""
        rg = _make_risk_guardian()
        rg._peak_portfolio_value = 100_000.0
        rg._daily_start_date = date.today()
        rg._daily_start_value = 100_000.0

        rg._portfolio_manager.get_total_value.return_value = 110_000.0

        rg.run_drawdown_monitor()

        assert rg._peak_portfolio_value == 110_000.0
        assert rg._current_drawdown == pytest.approx(0.0)

    def test_no_messages_when_within_thresholds(self):
        """No alerts when drawdown and daily loss are both within limits."""
        rg = _make_risk_guardian()
        rg._peak_portfolio_value = 100_000.0
        rg._daily_start_date = date.today()
        rg._daily_start_value = 100_000.0

        # 2 % drawdown -- well below warning
        rg._portfolio_manager.get_total_value.return_value = 98_000.0

        messages = rg.run_drawdown_monitor()
        assert len(messages) == 0

    def test_daily_loss_critical_triggers_halt_suggestion(self):
        """When daily loss exceeds the critical threshold a halt suggestion
        should be included."""
        rg = _make_risk_guardian()
        rg._peak_portfolio_value = 100_000.0
        rg._daily_start_date = date.today()
        rg._daily_start_value = 100_000.0

        # 6 % intraday loss (critical threshold is 5 %)
        rg._portfolio_manager.get_total_value.return_value = 94_000.0

        messages = rg.run_drawdown_monitor()

        daily_loss_msgs = [m for m in messages if "Daily Loss" in m.subject]
        assert len(daily_loss_msgs) == 1
        assert daily_loss_msgs[0].priority == MessagePriority.URGENT
        assert daily_loss_msgs[0].context.get("recommended_action") == "halt_trading"

    def test_portfolio_value_none_returns_empty(self):
        """If portfolio value cannot be retrieved, monitor returns no messages."""
        rg = _make_risk_guardian()
        rg._portfolio_manager.get_total_value.return_value = None
        # data_aggregator fallback also returns None
        rg._data_aggregator.prepare_analytics_data.return_value = {"portfolio_metrics": {}}

        messages = rg.run_drawdown_monitor()
        assert messages == []


# ============================================================================
# OperationsAgent Tests
# ============================================================================

class TestOperationsExecuteAction:
    """_execute_action must route correctly and handle unknown actions."""

    def test_unknown_action_returns_failure(self):
        ops = _make_operations_agent()
        result = ops._execute_action("totally_bogus_action", {})
        assert result["success"] is False
        assert "not_implemented" in result.get("status", "")

    def test_halt_trading_action(self):
        ops = _make_operations_agent()
        result = ops._execute_action("halt_trading", {"reason": "test halt"})
        assert result["success"] is True
        assert result["trading_halted"] is True
        assert ops._trading_halted is True
        assert ops._halt_reason == "test halt"

    def test_resume_trading_action(self):
        ops = _make_operations_agent()
        ops._trading_halted = True
        ops._halt_reason = "previous halt"

        result = ops._execute_action("resume_trading", {})
        assert result["success"] is True
        assert result["trading_halted"] is False
        assert ops._trading_halted is False
        assert ops._halt_reason is None

    def test_trigger_retrain_delegates_to_retrainer(self):
        ops = _make_operations_agent()
        ops._retrainer.run_retrain.return_value = {
            "models": {"xgb": {}},
            "deployments": {},
            "duration_seconds": 42,
        }

        result = ops._execute_action("trigger_retrain", {})
        assert result["success"] is True
        ops._retrainer.run_retrain.assert_called_once()

    def test_trigger_retrain_no_retrainer(self):
        from src.agents.operations_agent import OperationsAgent

        ops = _make_operations_agent()
        # Patch the property to return None (cannot just set _retrainer=None
        # because the property getter would try to lazy-import real modules)
        with patch.object(OperationsAgent, "retrainer", new_callable=PropertyMock, return_value=None):
            result = ops._execute_action("trigger_retrain", {})
        assert result["success"] is False
        assert "not available" in result.get("error", "").lower()


class TestOperationsAtomicConfigSave:
    """Config save should be atomic: write to temp, then os.replace."""

    def test_save_and_load_config_roundtrip(self):
        ops = _make_operations_agent()

        # Use a real temp dir for the config file
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "trading_config.yaml"
            # Seed the file
            config_path.write_text(yaml.dump({"ml_model": {"confidence_threshold": 0.55}}))
            ops._config_path = config_path

            # Perform a write
            ops._save_config({"ml_model": {"confidence_threshold": 0.60}, "extra": True})

            # Verify the file is valid YAML and contains the new value
            with open(config_path) as f:
                loaded = yaml.safe_load(f)
            assert loaded["ml_model"]["confidence_threshold"] == 0.60
            assert loaded["extra"] is True

    def test_save_config_no_partial_write(self):
        """If the write fails, the original config should remain untouched."""
        ops = _make_operations_agent()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "trading_config.yaml"
            original_data = {"ml_model": {"confidence_threshold": 0.55}}
            config_path.write_text(yaml.dump(original_data))
            ops._config_path = config_path

            # Force yaml.dump to raise inside _save_config.
            # yaml is imported locally inside the method, so we patch the
            # top-level yaml module's dump function.
            with patch("yaml.dump", side_effect=ValueError("boom")):
                with pytest.raises(ValueError):
                    ops._save_config({"bad": "data"})

            # Original file should be intact
            with open(config_path) as f:
                loaded = yaml.safe_load(f)
            assert loaded == original_data

    def test_config_lock_prevents_concurrent_modification(self):
        """Two threads writing config simultaneously should not corrupt it."""
        ops = _make_operations_agent()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "trading_config.yaml"
            config_path.write_text(yaml.dump({
                "ml_model": {"confidence_threshold": 0.55},
                "risk_management": {"max_position_pct": 0.10},
            }))
            ops._config_path = config_path

            errors = []
            results = []

            def adjust_confidence():
                try:
                    r = ops._adjust_confidence_threshold(increase=True)
                    results.append(r)
                except Exception as e:
                    errors.append(e)

            def adjust_position():
                try:
                    r = ops._adjust_position_size(decrease=True)
                    results.append(r)
                except Exception as e:
                    errors.append(e)

            t1 = threading.Thread(target=adjust_confidence)
            t2 = threading.Thread(target=adjust_position)
            t1.start()
            t2.start()
            t1.join(timeout=10)
            t2.join(timeout=10)

            assert len(errors) == 0, f"Errors during concurrent config writes: {errors}"
            # Both should succeed
            assert all(r["success"] for r in results)
            # The final file should be valid YAML
            with open(config_path) as f:
                loaded = yaml.safe_load(f)
            assert loaded is not None
            assert "ml_model" in loaded
            assert "risk_management" in loaded


class TestOperationsNotificationFailure:
    """A failing notifier must not prevent message processing from completing."""

    def test_notification_failure_does_not_block_processing(self):
        mq = FakeMessageQueue()
        bad_notifier = MagicMock()
        bad_notifier.notify_agent_message.side_effect = RuntimeError("Discord down")

        ops = _make_operations_agent(mq=mq, notifier=bad_notifier)

        # Plant a STATUS_UPDATE message for Operations
        status_msg = AgentMessage(
            sender=AgentRole.RISK_GUARDIAN,
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.STATUS_UPDATE,
            subject="Daily Risk Report",
            content="all good",
            context={},
        )
        mq.enqueue(status_msg)

        result = ops.run_cycle()

        # The message should still be processed despite notification failure
        assert result["messages_processed"] == 1
        assert len(result["errors"]) == 0
        # The notifier was called (and failed), but processing continued
        bad_notifier.notify_agent_message.assert_called()


class TestOperationsTradingHaltRejectsTradesuggestions:
    """When trading is halted, trade suggestions must be rejected."""

    def test_suggestion_rejected_when_halted(self):
        ops = _make_operations_agent()
        ops._trading_halted = True
        ops._halt_reason = "emergency drawdown"

        suggestion = AgentMessage(
            sender=AgentRole.PORTFOLIO_STRATEGIST,
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.SUGGESTION,
            subject="Buy AAPL",
            content="Recommend buying AAPL",
            context={"recommendations": [{"action": "add_symbol", "symbol": "AAPL"}]},
        )

        response = ops.process_message(suggestion)

        assert response is not None
        assert response.context.get("decision") == "rejected_halted"
        assert "halted" in response.content.lower()


class TestOperationsPriorityHandling:
    """Risk Guardian suggestions must not be deferred; other agents' should be
    deferred when urgent risk alerts are pending."""

    def test_risk_guardian_suggestion_not_deferred(self):
        mq = FakeMessageQueue()
        ops = _make_operations_agent(mq=mq)

        # Plant an urgent Risk Guardian message (simulates pending risk alert)
        urgent_risk_msg = AgentMessage(
            sender=AgentRole.RISK_GUARDIAN,
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.SUGGESTION,
            subject="Critical drawdown",
            content="drawdown alert",
            priority=MessagePriority.URGENT,
            context={"emergency_action_needed": True, "reason": "drawdown"},
        )
        mq.enqueue(urgent_risk_msg)

        # Now process a NEW suggestion FROM Risk Guardian
        new_rg_suggestion = AgentMessage(
            sender=AgentRole.RISK_GUARDIAN,
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.SUGGESTION,
            subject="Reduce positions",
            content="reduce",
            priority=MessagePriority.URGENT,
            context={"emergency_action_needed": True, "reason": "risk spike"},
        )

        response = ops._handle_suggestion(new_rg_suggestion)

        # Should NOT be deferred -- it's from Risk Guardian
        assert response.context.get("decision") != "deferred_risk_pending"

    def test_non_risk_guardian_deferred_when_risk_alerts_pending(self):
        mq = FakeMessageQueue()
        ops = _make_operations_agent(mq=mq)

        # Plant an unprocessed URGENT Risk Guardian message
        urgent_risk_msg = AgentMessage(
            sender=AgentRole.RISK_GUARDIAN,
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.SUGGESTION,
            subject="Critical drawdown",
            content="drawdown alert",
            priority=MessagePriority.URGENT,
            context={"emergency_action_needed": True},
        )
        mq.enqueue(urgent_risk_msg)

        # Now a Portfolio Strategist suggestion arrives
        ps_suggestion = AgentMessage(
            sender=AgentRole.PORTFOLIO_STRATEGIST,
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.SUGGESTION,
            subject="Rebalance portfolio",
            content="rebalance",
            priority=MessagePriority.NORMAL,
            context={"recommendations": [{"action": "adjust_allocation"}]},
        )

        response = ops._handle_suggestion(ps_suggestion)

        assert response is not None
        assert response.context.get("decision") == "deferred_risk_pending"


# ============================================================================
# AgentOrchestrator Tests
# ============================================================================

class TestOrchestratorHaltStatus:
    """is_trading_halted() must reflect both Risk Guardian and Operations."""

    def _make_orchestrator(self):
        """Build an orchestrator with fully mocked internals."""
        from src.agents.orchestrator import AgentOrchestrator

        config = {"agents": _make_config()}

        with patch.object(AgentOrchestrator, "__init__", lambda self, cfg: None):
            orch = AgentOrchestrator.__new__(AgentOrchestrator)

        orch.config = config
        orch.agents_config = config["agents"]
        orch.enabled = True
        orch.message_queue = FakeMessageQueue()
        orch.notifier = None
        orch.llm_client = None

        orch.risk_guardian = _make_risk_guardian(mq=orch.message_queue)
        orch.operations = _make_operations_agent(mq=orch.message_queue)
        orch.market_intelligence = MagicMock()
        orch.portfolio_strategist = MagicMock()

        orch.agents = {
            AgentRole.MARKET_INTELLIGENCE: orch.market_intelligence,
            AgentRole.RISK_GUARDIAN: orch.risk_guardian,
            AgentRole.PORTFOLIO_STRATEGIST: orch.portfolio_strategist,
            AgentRole.OPERATIONS: orch.operations,
        }

        orch.scheduler = MagicMock()
        orch._is_running = False
        orch._lock = threading.Lock()

        return orch

    def test_not_halted_by_default(self):
        orch = self._make_orchestrator()
        assert orch.is_trading_halted() is False

    def test_halted_by_risk_guardian(self):
        orch = self._make_orchestrator()
        orch.risk_guardian._trading_halted = True

        assert orch.is_trading_halted() is True

    def test_halted_by_operations(self):
        orch = self._make_orchestrator()
        orch.operations._trading_halted = True

        assert orch.is_trading_halted() is True

    def test_halted_by_both(self):
        orch = self._make_orchestrator()
        orch.risk_guardian._trading_halted = True
        orch.operations._trading_halted = True
        orch.operations._halt_reason = "manual halt"

        assert orch.is_trading_halted() is True

    def test_not_halted_when_disabled(self):
        orch = self._make_orchestrator()
        orch.enabled = False
        orch.risk_guardian._trading_halted = True

        assert orch.is_trading_halted() is False


class TestOrchestratorHaltReason:
    """get_halt_reason() must compose correct human-readable strings."""

    def _make_orchestrator(self):
        from src.agents.orchestrator import AgentOrchestrator

        config = {"agents": _make_config()}

        with patch.object(AgentOrchestrator, "__init__", lambda self, cfg: None):
            orch = AgentOrchestrator.__new__(AgentOrchestrator)

        orch.config = config
        orch.agents_config = config["agents"]
        orch.enabled = True
        orch.message_queue = FakeMessageQueue()
        orch.notifier = None
        orch.llm_client = None

        orch.risk_guardian = _make_risk_guardian(mq=orch.message_queue)
        orch.operations = _make_operations_agent(mq=orch.message_queue)
        orch.market_intelligence = MagicMock()
        orch.portfolio_strategist = MagicMock()
        orch.agents = {}
        orch.scheduler = MagicMock()
        orch._is_running = False
        orch._lock = threading.Lock()

        return orch

    def test_no_halt_returns_none(self):
        orch = self._make_orchestrator()
        assert orch.get_halt_reason() is None

    def test_risk_guardian_halt_reason(self):
        orch = self._make_orchestrator()
        orch.risk_guardian._trading_halted = True

        reason = orch.get_halt_reason()
        assert reason is not None
        assert "Risk Guardian" in reason

    def test_operations_halt_reason_includes_detail(self):
        orch = self._make_orchestrator()
        orch.operations._trading_halted = True
        orch.operations._halt_reason = "max daily loss exceeded"

        reason = orch.get_halt_reason()
        assert "Operations" in reason
        assert "max daily loss exceeded" in reason

    def test_both_halted_shows_both(self):
        orch = self._make_orchestrator()
        orch.risk_guardian._trading_halted = True
        orch.operations._trading_halted = True
        orch.operations._halt_reason = "manual"

        reason = orch.get_halt_reason()
        assert "Risk Guardian" in reason
        assert "Operations" in reason

    def test_disabled_returns_none(self):
        orch = self._make_orchestrator()
        orch.enabled = False
        orch.risk_guardian._trading_halted = True

        assert orch.get_halt_reason() is None


class TestOrchestratorLifecycle:
    """Start / stop lifecycle."""

    def _make_orchestrator(self):
        from src.agents.orchestrator import AgentOrchestrator

        config = {"agents": _make_config()}

        with patch.object(AgentOrchestrator, "__init__", lambda self, cfg: None):
            orch = AgentOrchestrator.__new__(AgentOrchestrator)

        orch.config = config
        orch.agents_config = config["agents"]
        orch.enabled = True
        orch.message_queue = FakeMessageQueue()
        orch.notifier = None
        orch.llm_client = None

        orch.risk_guardian = _make_risk_guardian(mq=orch.message_queue)
        orch.operations = _make_operations_agent(mq=orch.message_queue)
        orch.market_intelligence = MagicMock()
        orch.portfolio_strategist = MagicMock()
        orch.agents = {
            AgentRole.MARKET_INTELLIGENCE: orch.market_intelligence,
            AgentRole.RISK_GUARDIAN: orch.risk_guardian,
            AgentRole.PORTFOLIO_STRATEGIST: orch.portfolio_strategist,
            AgentRole.OPERATIONS: orch.operations,
        }

        orch.scheduler = MagicMock()
        orch._is_running = False
        orch._lock = threading.Lock()

        return orch

    def test_start_sets_running(self):
        orch = self._make_orchestrator()
        orch.start()

        assert orch._is_running is True
        orch.scheduler.start.assert_called_once()

    def test_start_when_disabled_is_noop(self):
        orch = self._make_orchestrator()
        orch.enabled = False

        orch.start()

        assert orch._is_running is False
        orch.scheduler.start.assert_not_called()

    def test_start_twice_does_not_double_start(self):
        orch = self._make_orchestrator()
        orch.start()
        orch.start()  # second call -- should be a no-op

        # scheduler.start only called once
        assert orch.scheduler.start.call_count == 1

    def test_stop_sets_not_running(self):
        orch = self._make_orchestrator()
        orch.start()
        assert orch._is_running is True

        orch.stop()
        assert orch._is_running is False
        orch.scheduler.shutdown.assert_called_once_with(wait=True)

    def test_stop_when_not_running_is_noop(self):
        orch = self._make_orchestrator()
        orch.stop()
        orch.scheduler.shutdown.assert_not_called()


class TestOrchestratorAgentCycles:
    """Orchestrator wrapper methods should delegate to agent methods and
    enqueue generated messages."""

    def _make_orchestrator(self):
        from src.agents.orchestrator import AgentOrchestrator

        config = {"agents": _make_config()}

        with patch.object(AgentOrchestrator, "__init__", lambda self, cfg: None):
            orch = AgentOrchestrator.__new__(AgentOrchestrator)

        orch.config = config
        orch.agents_config = config["agents"]
        orch.enabled = True
        orch.message_queue = FakeMessageQueue()
        orch.notifier = None
        orch.llm_client = None

        orch.risk_guardian = _make_risk_guardian(mq=orch.message_queue)
        orch.operations = _make_operations_agent(mq=orch.message_queue)
        orch.market_intelligence = MagicMock()
        orch.portfolio_strategist = MagicMock()
        orch.agents = {}
        orch.scheduler = MagicMock()
        orch._is_running = True
        orch._lock = threading.Lock()

        return orch

    def test_run_operations_cycle(self):
        orch = self._make_orchestrator()

        # Plant a message for operations to process
        msg = AgentMessage(
            sender=AgentRole.RISK_GUARDIAN,
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.STATUS_UPDATE,
            subject="Test",
            content="test",
        )
        orch.message_queue.enqueue(msg)

        # Run the operations cycle through the orchestrator wrapper
        orch._run_operations_cycle()

        # The planted message should now be processed
        remaining = orch.message_queue.get_messages_for_recipient(
            AgentRole.OPERATIONS, processed=False
        )
        assert len(remaining) == 0

    def test_run_drawdown_monitor_delegates(self):
        orch = self._make_orchestrator()

        # Set up Risk Guardian state so drawdown monitor has something to do
        orch.risk_guardian._peak_portfolio_value = 100_000.0
        orch.risk_guardian._daily_start_date = date.today()
        orch.risk_guardian._daily_start_value = 100_000.0
        orch.risk_guardian._portfolio_manager.get_total_value.return_value = 99_000.0

        # Should not raise
        orch._run_drawdown_monitor()

        # _last_drawdown_monitor should have been set
        assert orch.risk_guardian._last_drawdown_monitor is not None

    def test_run_risk_check_error_does_not_propagate(self):
        """If a scheduled agent method raises, the orchestrator wrapper
        catches it so APScheduler does not crash."""
        orch = self._make_orchestrator()

        # Force risk_check to blow up
        with patch.object(orch.risk_guardian, "run_risk_check", side_effect=RuntimeError("boom")):
            # Should not raise
            orch._run_risk_check()


# ============================================================================
# Additional edge-case tests
# ============================================================================

class TestRiskGuardianAnalyzeRiskLevels:
    """_analyze_risk_levels should produce correct alerts for various metric
    combinations."""

    def test_no_alerts_for_healthy_metrics(self):
        rg = _make_risk_guardian()
        metrics = {
            "current_drawdown": 0.01,
            "daily_loss": 0.005,
            "max_position_size": 0.05,
            "max_sector_exposure": 0.15,
        }
        alerts = rg._analyze_risk_levels(metrics)
        assert alerts == []

    def test_multiple_critical_alerts(self):
        rg = _make_risk_guardian()
        metrics = {
            "current_drawdown": 0.15,
            "daily_loss": 0.06,
            "max_position_size": 0.20,
            "max_sector_exposure": 0.40,
        }
        alerts = rg._analyze_risk_levels(metrics)
        critical_alerts = [a for a in alerts if a["severity"] == "critical"]
        assert len(critical_alerts) >= 3  # drawdown, daily_loss, position, sector

    def test_warning_vs_critical_distinction(self):
        rg = _make_risk_guardian()
        # Drawdown at warning level, daily loss at critical level
        metrics = {
            "current_drawdown": 0.07,  # between 0.05 warning and 0.10 critical
            "daily_loss": 0.06,        # above 0.05 critical
            "max_position_size": 0.0,
            "max_sector_exposure": 0.0,
        }
        alerts = rg._analyze_risk_levels(metrics)
        drawdown_alert = next((a for a in alerts if a["type"] == "drawdown"), None)
        daily_loss_alert = next((a for a in alerts if a["type"] == "daily_loss"), None)

        assert drawdown_alert is not None
        assert drawdown_alert["severity"] == "warning"
        assert daily_loss_alert is not None
        assert daily_loss_alert["severity"] == "critical"


class TestOperationsRunCycleEndToEnd:
    """Full run_cycle through the Operations agent."""

    def test_cycle_processes_and_responds(self):
        mq = FakeMessageQueue()
        ops = _make_operations_agent(mq=mq)

        # Enqueue a QUERY message for Operations
        query = AgentMessage(
            sender=AgentRole.RISK_GUARDIAN,
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.QUERY,
            subject="Trading Status",
            content="What is trading status?",
            context={"query_type": "trading_status"},
        )
        mq.enqueue(query)

        result = ops.run_cycle()

        assert result["messages_processed"] == 1
        assert result["messages_sent"] == 1
        assert len(result["errors"]) == 0

        # The response should be enqueued back
        responses = mq.get_messages_for_recipient(AgentRole.RISK_GUARDIAN)
        assert len(responses) == 1
        assert "Trading halted" in responses[0].content

    def test_cycle_empty_queue(self):
        mq = FakeMessageQueue()
        ops = _make_operations_agent(mq=mq)

        result = ops.run_cycle()
        assert result["messages_processed"] == 0
        assert result["messages_sent"] == 0
        assert len(result["errors"]) == 0


class TestOperationsEmergencyExecution:
    """_execute_emergency_action must halt trading and notify Risk Guardian."""

    def test_emergency_action_halts_and_responds(self):
        ops = _make_operations_agent()

        emergency_msg = AgentMessage(
            sender=AgentRole.RISK_GUARDIAN,
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.SUGGESTION,
            subject="Emergency",
            content="critical drawdown",
            context={
                "emergency_action_needed": True,
                "reason": "drawdown at 12 %",
                "reduce_percentage": 0.25,
            },
        )

        response = ops._execute_emergency_action(emergency_msg)

        assert ops._trading_halted is True
        assert ops._halt_reason == "drawdown at 12 %"
        assert response.recipient == AgentRole.RISK_GUARDIAN
        assert response.priority == MessagePriority.URGENT


class TestAgentMessageSerialization:
    """AgentMessage round-trip through to_dict / from_dict."""

    def test_roundtrip(self):
        msg = AgentMessage(
            sender=AgentRole.RISK_GUARDIAN,
            recipient=AgentRole.OPERATIONS,
            message_type=MessageType.SUGGESTION,
            subject="Test",
            content="content",
            priority=MessagePriority.HIGH,
            context={"key": "value"},
            requires_response=True,
        )

        d = msg.to_dict()
        restored = AgentMessage.from_dict(d)

        assert restored.sender == msg.sender
        assert restored.recipient == msg.recipient
        assert restored.message_type == msg.message_type
        assert restored.subject == msg.subject
        assert restored.priority == msg.priority
        assert restored.context == msg.context
        assert restored.requires_response == msg.requires_response


class TestOperationsCooldown:
    """Cooldown logic: _can_take_action should honour cooldown_hours."""

    def test_action_allowed_when_never_taken(self):
        ops = _make_operations_agent()
        assert ops._can_take_action("trigger_retrain") is True

    def test_action_blocked_during_cooldown(self):
        ops = _make_operations_agent()
        ops._action_history["trigger_retrain"] = datetime.now()
        assert ops._can_take_action("trigger_retrain") is False

    def test_action_allowed_after_cooldown_expires(self):
        ops = _make_operations_agent()
        ops._action_history["trigger_retrain"] = datetime.now() - timedelta(hours=5)
        # cooldown_hours = 4, so 5 hours later it should be allowed
        assert ops._can_take_action("trigger_retrain") is True

    def test_no_action_always_allowed(self):
        ops = _make_operations_agent()
        assert ops._can_take_action("no_action") is True
