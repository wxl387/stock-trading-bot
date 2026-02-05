"""
Operations Agent

Execution agent that implements configuration changes, triggers retraining,
monitors execution quality, and ensures system health.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_agent import (
    AgentMessage,
    AgentRole,
    BaseAgent,
    MessagePriority,
    MessageType,
)

logger = logging.getLogger(__name__)


class OperationsAgent(BaseAgent):
    """
    Operations Agent - implements changes and maintains system health.

    Responsibilities:
    - Implement configuration changes
    - Trigger model retraining
    - Monitor execution quality
    - System health checks
    - Symbol management execution
    - Degradation response

    Schedule:
    - Process messages: Every 15 minutes
    - Execution quality check: Every 2 hours
    - System health check: Every 4 hours
    - Degradation check: Every 12 hours

    Available Actions:
    - trigger_retrain: Retrain ML models
    - adjust_confidence: Modify confidence threshold
    - adjust_position_size: Change position limits
    - adjust_stop_loss: Modify stop loss settings
    - add_symbol: Add to trading universe
    - remove_symbol: Remove from universe
    - halt_trading: Emergency stop
    - resume_trading: Resume after halt

    Outputs To:
    - All agents: Action confirmations, status updates
    - Risk Guardian: Execution metrics for risk assessment
    """

    AVAILABLE_ACTIONS = [
        "trigger_retrain",
        "adjust_confidence_threshold",
        "adjust_position_size",
        "adjust_stop_loss",
        "toggle_degradation_detection",
        "toggle_auto_rollback",
        "add_symbol",
        "remove_symbol",
        "adjust_allocation",
        "halt_trading",
        "resume_trading",
        "no_action",
    ]

    def __init__(
        self,
        config: Dict[str, Any],
        message_queue,
        notifier=None,
        llm_client=None,
    ):
        """
        Initialize Operations agent.

        Args:
            config: Agent configuration
            message_queue: Shared message queue
            notifier: Optional Discord notifier
            llm_client: Optional LLM client for decision support
        """
        super().__init__(
            role=AgentRole.OPERATIONS,
            config=config,
            message_queue=message_queue,
            notifier=notifier,
            llm_client=llm_client,
        )

        # Load configuration
        ops_config = config.get("operations", {})
        self.process_messages_minutes = ops_config.get("process_messages_minutes", 15)
        self.execution_quality_hours = ops_config.get("execution_quality_hours", 2)
        self.system_health_hours = ops_config.get("system_health_hours", 4)
        self.degradation_check_hours = ops_config.get("degradation_check_hours", 12)
        self.cooldown_hours = ops_config.get("cooldown_hours", 4)
        self.auto_retrain_on_degradation = ops_config.get("auto_retrain_on_degradation", True)

        # Track last analysis times
        self._last_execution_quality_check: Optional[datetime] = None
        self._last_system_health_check: Optional[datetime] = None
        self._last_degradation_check: Optional[datetime] = None

        # Track action history for cooldowns
        self._action_history: Dict[str, datetime] = {}

        # Trading state
        self._trading_halted: bool = False
        self._halt_reason: Optional[str] = None

        # Lazy-loaded components
        self._retrainer = None
        self._config_path = None
        self._symbol_manager = None
        self._data_aggregator = None
        self._degradation_monitor = None

        logger.info(
            f"Operations agent initialized: "
            f"process_messages={self.process_messages_minutes}min, "
            f"cooldown={self.cooldown_hours}h"
        )

    @property
    def retrainer(self):
        """Lazy load scheduled retrainer."""
        if self._retrainer is None:
            try:
                from src.ml.scheduled_retrainer import ScheduledRetrainer
                from config.settings import Settings
                config = Settings.load_trading_config()
                retrain_config = config.get("retraining", {})

                self._retrainer = ScheduledRetrainer(
                    enabled=retrain_config.get("enabled", True),
                    schedule=retrain_config.get("schedule", "weekly"),
                    auto_deploy=retrain_config.get("auto_deploy", True),
                )
            except ImportError as e:
                logger.error(f"Failed to import ScheduledRetrainer: {e}")
        return self._retrainer

    @property
    def config_path(self) -> Path:
        """Get path to trading config file."""
        if self._config_path is None:
            try:
                from config.settings import CONFIG_DIR
                self._config_path = CONFIG_DIR / "trading_config.yaml"
            except Exception:
                self._config_path = Path("config/trading_config.yaml")
        return self._config_path

    @property
    def symbol_manager(self):
        """Lazy load symbol manager."""
        if self._symbol_manager is None:
            try:
                from src.core.symbol_manager import get_symbol_manager
                from config.settings import Settings
                config = Settings.load_trading_config()
                self._symbol_manager = get_symbol_manager(config)
            except ImportError as e:
                logger.error(f"Failed to import SymbolManager: {e}")
        return self._symbol_manager

    @property
    def data_aggregator(self):
        """Lazy load data aggregator."""
        if self._data_aggregator is None:
            try:
                from src.analytics.data_aggregator import DataAggregator
                self._data_aggregator = DataAggregator()
            except ImportError as e:
                logger.error(f"Failed to import DataAggregator: {e}")
        return self._data_aggregator

    @property
    def degradation_monitor(self):
        """Lazy load degradation monitor."""
        if self._degradation_monitor is None:
            try:
                from src.ml.degradation_monitor import DegradationMonitor
                from config.settings import Settings
                config = Settings.load_trading_config()
                deg_config = config.get("retraining", {}).get("degradation_detection", {})

                self._degradation_monitor = DegradationMonitor(
                    enabled=deg_config.get("enabled", True),
                    accuracy_drop_threshold=deg_config.get("accuracy_drop_threshold", 0.05),
                    confidence_collapse_threshold=deg_config.get("confidence_collapse_threshold", 0.55),
                    min_win_rate=deg_config.get("min_win_rate", 0.40),
                )
            except ImportError as e:
                logger.error(f"Failed to import DegradationMonitor: {e}")
        return self._degradation_monitor

    def analyze(self) -> List[AgentMessage]:
        """
        Operations agent is reactive - it responds to messages.

        Returns:
            Empty list (operations is reactive, not proactive)
        """
        return []

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming messages and take appropriate actions.

        Args:
            message: Incoming message from other agents

        Returns:
            Response message with action taken or decision
        """
        logger.info(f"Processing message from {message.sender.value}: {message.subject}")

        # Handle different message types
        if message.message_type == MessageType.SUGGESTION:
            return self._handle_suggestion(message)
        elif message.message_type == MessageType.OBSERVATION:
            return self._handle_observation(message)
        elif message.message_type == MessageType.ACTION:
            return self._handle_action_request(message)
        elif message.message_type == MessageType.STATUS_UPDATE:
            return self._handle_status_update(message)
        elif message.message_type == MessageType.QUERY:
            return self._handle_query(message)
        elif message.message_type == MessageType.ACKNOWLEDGMENT:
            logger.info(f"Received acknowledgment: {message.subject}")
            return None

        return None

    def _handle_suggestion(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle suggestion from other agents."""
        recommendations = message.context.get("recommendations", [])
        emergency_action_needed = message.context.get("emergency_action_needed", False)

        # Handle emergency situations first
        if emergency_action_needed:
            return self._execute_emergency_action(message)

        # Evaluate recommendations
        if not recommendations:
            return self._handle_general_suggestion(message)

        # Use LLM to evaluate action request
        if self.llm_client and self.llm_client.is_available():
            evaluation = self.llm_client.evaluate_action_request(
                str(message.content),
                self._get_system_state(),
                self._get_cooldown_status()
            )
            if evaluation:
                action = self._parse_llm_action(evaluation)
            else:
                action = self._rule_based_action_decision(recommendations)
        else:
            action = self._rule_based_action_decision(recommendations)

        # Execute the action
        if action != "no_action":
            result = self._execute_action(action, message.context)
            return self._create_action_response(message, action, result)

        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            subject="Suggestion Evaluated - No Action",
            content="Evaluated the suggestion. No immediate action required based on current state and cooldowns.",
            priority=MessagePriority.NORMAL,
            parent_message_id=message.id,
            context={"decision": "no_action"},
        )

    def _handle_observation(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle observation from other agents."""
        # Log the observation
        logger.info(f"Observation from {message.sender.value}: {message.subject}")

        # Check if observation requires action
        alerts = message.context.get("alerts", [])
        critical_alerts = [a for a in alerts if a.get("severity") == "critical"]

        if critical_alerts:
            # May need to take action on critical observations
            return self._handle_critical_observation(message, critical_alerts)

        # Just acknowledge
        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.ACKNOWLEDGMENT,
            subject=f"Acknowledged: {message.subject}",
            content="Observation received and logged.",
            priority=MessagePriority.LOW,
            parent_message_id=message.id,
        )

    def _handle_action_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle direct action request (e.g., emergency from Risk Guardian)."""
        action = message.context.get("action")
        halt_trading = message.context.get("halt_trading", False)

        if halt_trading:
            self._trading_halted = True
            self._halt_reason = message.context.get("reason", "Emergency action")
            logger.warning(f"TRADING HALTED: {self._halt_reason}")

        if action:
            result = self._execute_action(action, message.context)
            return self._create_action_response(message, action, result)

        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.ACKNOWLEDGMENT,
            subject="Action Request Received",
            content="Action request acknowledged.",
            priority=MessagePriority.NORMAL,
            parent_message_id=message.id,
        )

    def _handle_status_update(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle status updates (daily reviews, reports)."""
        # Log status updates
        logger.info(f"Status update from {message.sender.value}: {message.subject}")

        # Acknowledge
        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.ACKNOWLEDGMENT,
            subject=f"Status Acknowledged: {message.subject[:50]}",
            content="Status update received and logged.",
            priority=MessagePriority.LOW,
            parent_message_id=message.id,
        )

    def _handle_query(self, message: AgentMessage) -> AgentMessage:
        """Handle queries from other agents."""
        query_type = message.context.get("query_type", "general")

        if query_type == "action_history":
            content = self._format_action_history()
        elif query_type == "current_config":
            content = self._format_current_config()
        elif query_type == "cooldown_status":
            content = self._format_cooldown_status()
        elif query_type == "system_status":
            content = self._format_system_status()
        elif query_type == "trading_status":
            content = f"Trading halted: {self._trading_halted}\nReason: {self._halt_reason or 'N/A'}"
        else:
            content = f"Unknown query type: {query_type}"

        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            subject=f"Re: {message.subject}",
            content=content,
            priority=MessagePriority.NORMAL,
            parent_message_id=message.id,
        )

    def run_execution_quality_check(self) -> List[AgentMessage]:
        """
        Check execution quality of recent trades.

        Returns:
            List of messages with execution quality analysis
        """
        self._last_execution_quality_check = datetime.now()
        messages = []

        try:
            # Get recent trades
            trades = self._get_recent_trades(days=7)

            if not trades:
                logger.info("No recent trades for execution quality check")
                return messages

            # Analyze execution quality
            quality_metrics = self._analyze_execution_quality(trades)

            content = self._format_execution_quality_report(quality_metrics)

            # Use LLM for analysis
            if self.llm_client and self.llm_client.is_available():
                llm_analysis = self.llm_client.analyze_execution_quality(
                    quality_metrics, trades
                )
                if llm_analysis:
                    content += f"\n\n### AI Analysis\n{llm_analysis}"

            # Alert Risk Guardian if issues found
            if quality_metrics.get("issues"):
                messages.append(self.create_message(
                    recipient=AgentRole.RISK_GUARDIAN,
                    message_type=MessageType.OBSERVATION,
                    subject=f"Execution Quality Alert: {len(quality_metrics['issues'])} issues",
                    content=content,
                    priority=MessagePriority.NORMAL,
                    context={"quality_metrics": quality_metrics},
                ))

            logger.info(f"Execution quality check complete: {len(quality_metrics.get('issues', []))} issues")

        except Exception as e:
            logger.error(f"Error during execution quality check: {e}")

        return messages

    def run_system_health_check(self) -> List[AgentMessage]:
        """
        Check overall system health.

        Returns:
            List of messages with system health status
        """
        self._last_system_health_check = datetime.now()
        messages = []

        try:
            health_data = self._gather_health_data()

            # Analyze health
            issues = self._analyze_system_health(health_data)

            if issues:
                content = self._format_health_report(health_data, issues)

                # Use LLM for diagnosis
                if self.llm_client and self.llm_client.is_available():
                    llm_diagnosis = self.llm_client.diagnose_system_issue(
                        health_data, issues
                    )
                    if llm_diagnosis:
                        content += f"\n\n### AI Diagnosis\n{llm_diagnosis}"

                # Broadcast to all agents
                for agent_role in [AgentRole.RISK_GUARDIAN, AgentRole.PORTFOLIO_STRATEGIST, AgentRole.MARKET_INTELLIGENCE]:
                    messages.append(self.create_message(
                        recipient=agent_role,
                        message_type=MessageType.STATUS_UPDATE,
                        subject=f"System Health Alert: {len(issues)} issue(s)",
                        content=content,
                        priority=MessagePriority.HIGH if any(i.get("severity") == "critical" for i in issues) else MessagePriority.NORMAL,
                        context={"health_data": health_data, "issues": issues},
                    ))

            logger.info(f"System health check complete: {len(issues)} issues")

        except Exception as e:
            logger.error(f"Error during system health check: {e}")

        return messages

    def run_degradation_check(self) -> List[AgentMessage]:
        """
        Check for model degradation.

        Returns:
            List of messages with degradation alerts
        """
        self._last_degradation_check = datetime.now()
        messages = []

        if not self.degradation_monitor:
            logger.warning("Degradation monitor not available")
            return messages

        try:
            reports = self.degradation_monitor.check_all_models()
            degraded_models = [r for r in reports if r.is_degraded]

            if degraded_models:
                content = self._format_degradation_report(degraded_models)

                # Check if auto-retrain is enabled
                if self.auto_retrain_on_degradation and self._can_take_action("trigger_retrain"):
                    result = self._trigger_retrain(reason="model_degradation")
                    content += f"\n\n### Auto-Retrain Triggered\n{self._format_retrain_result(result)}"

                # Notify all agents
                for agent_role in [AgentRole.RISK_GUARDIAN, AgentRole.PORTFOLIO_STRATEGIST]:
                    messages.append(self.create_message(
                        recipient=agent_role,
                        message_type=MessageType.OBSERVATION,
                        subject=f"Model Degradation: {len(degraded_models)} model(s)",
                        content=content,
                        priority=MessagePriority.HIGH,
                        context={
                            "degraded_models": [r.to_dict() for r in degraded_models],
                            "auto_retrain_triggered": self.auto_retrain_on_degradation,
                        },
                    ))

            logger.info(f"Degradation check complete: {len(degraded_models)} degraded models")

        except Exception as e:
            logger.error(f"Error during degradation check: {e}")

        return messages

    def _execute_emergency_action(self, message: AgentMessage) -> AgentMessage:
        """Execute emergency action."""
        self._trading_halted = True
        self._halt_reason = message.context.get("reason", "Emergency from Risk Guardian")

        # Execute reduce if specified
        reduce_percentage = message.context.get("reduce_percentage", 0.25)

        result = {
            "trading_halted": True,
            "halt_reason": self._halt_reason,
            "reduce_percentage": reduce_percentage,
        }

        logger.warning(f"EMERGENCY ACTION: Trading halted - {self._halt_reason}")

        # Broadcast halt to all agents
        return self.create_message(
            recipient=AgentRole.RISK_GUARDIAN,
            message_type=MessageType.ACTION,
            subject=":rotating_light: EMERGENCY: Trading Halted",
            content=f"## Emergency Action Executed\n\n"
                    f"**Trading Status:** HALTED\n"
                    f"**Reason:** {self._halt_reason}\n"
                    f"**Reduce Percentage:** {reduce_percentage:.0%}\n\n"
                    f"All new trades suspended until resume command is issued.",
            priority=MessagePriority.URGENT,
            parent_message_id=message.id,
            context={"action": "emergency_halt", "result": result},
        )

    def _handle_critical_observation(self, message: AgentMessage, alerts: List[Dict]) -> AgentMessage:
        """Handle critical observation that may require action."""
        # Evaluate whether to take action
        action = self._rule_based_action_decision([
            {"action": "adjust_position_size" if "drawdown" in str(alerts) else "no_action"}
        ])

        if action != "no_action" and self._can_take_action(action):
            result = self._execute_action(action, message.context)
            return self._create_action_response(message, action, result)

        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            subject="Critical Observation Evaluated",
            content=f"Evaluated {len(alerts)} critical alert(s). "
                    f"Action decision: {action}. "
                    f"{'Action on cooldown.' if not self._can_take_action(action) else ''}",
            priority=MessagePriority.HIGH,
            parent_message_id=message.id,
            context={"decision": action},
        )

    def _handle_general_suggestion(self, message: AgentMessage) -> AgentMessage:
        """Handle general suggestion without specific recommendations."""
        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            subject="Suggestion Received",
            content="Suggestion received and logged. No specific action recommendations found.",
            priority=MessagePriority.LOW,
            parent_message_id=message.id,
        )

    def _rule_based_action_decision(self, recommendations: List[Dict]) -> str:
        """Make rule-based action decision."""
        for rec in recommendations:
            action = rec.get("action", "no_action")
            if action in self.AVAILABLE_ACTIONS and action != "no_action":
                if self._can_take_action(action):
                    return action

        return "no_action"

    def _parse_llm_action(self, llm_response: str) -> str:
        """Parse action from LLM response."""
        response_lower = llm_response.lower()

        for action in self.AVAILABLE_ACTIONS:
            if action.replace("_", " ") in response_lower or action in response_lower:
                if self._can_take_action(action):
                    return action

        return "no_action"

    def _can_take_action(self, action: str) -> bool:
        """Check if action is allowed (not on cooldown)."""
        if action == "no_action":
            return True

        last_time = self._action_history.get(action)
        if last_time is None:
            return True

        cooldown_end = last_time + timedelta(hours=self.cooldown_hours)
        return datetime.now() >= cooldown_end

    def _record_action(self, action: str) -> None:
        """Record that an action was taken."""
        if action != "no_action":
            self._action_history[action] = datetime.now()

    def _execute_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action."""
        if action == "trigger_retrain":
            return self._trigger_retrain(reason="agent_request")
        elif action == "adjust_confidence_threshold":
            return self._adjust_confidence_threshold(increase=True)
        elif action == "adjust_position_size":
            return self._adjust_position_size(decrease=True)
        elif action == "adjust_stop_loss":
            return self._adjust_stop_loss(tighter=True)
        elif action == "toggle_degradation_detection":
            return self._toggle_feature("degradation_detection")
        elif action == "toggle_auto_rollback":
            return self._toggle_feature("auto_rollback")
        elif action == "add_symbol":
            return self._add_symbol(context)
        elif action == "remove_symbol":
            return self._remove_symbol(context)
        elif action == "halt_trading":
            self._trading_halted = True
            self._halt_reason = context.get("reason", "Manual halt")
            return {"success": True, "trading_halted": True}
        elif action == "resume_trading":
            self._trading_halted = False
            self._halt_reason = None
            return {"success": True, "trading_halted": False}

        return {"action": action, "status": "not_implemented"}

    def _trigger_retrain(self, reason: str = "manual") -> Dict[str, Any]:
        """Trigger model retraining."""
        self._record_action("trigger_retrain")

        if not self.retrainer:
            return {"success": False, "error": "Retrainer not available"}

        try:
            result = self.retrainer.run_retrain(trigger_reason=reason)
            return {
                "success": True,
                "reason": reason,
                "result": {
                    "models_trained": list(result.get("models", {}).keys()),
                    "deployments": result.get("deployments", {}),
                    "duration": result.get("duration_seconds"),
                },
            }
        except Exception as e:
            logger.error(f"Failed to trigger retrain: {e}")
            return {"success": False, "error": str(e)}

    def _adjust_confidence_threshold(self, increase: bool = True) -> Dict[str, Any]:
        """Adjust the ML confidence threshold."""
        self._record_action("adjust_confidence_threshold")

        try:
            config = self._load_config()
            current = config.get("ml_model", {}).get("confidence_threshold", 0.55)

            adjustment = 0.02 if increase else -0.02
            new_value = max(0.50, min(0.70, current + adjustment))

            config.setdefault("ml_model", {})["confidence_threshold"] = new_value
            self._save_config(config)

            logger.info(f"Adjusted confidence threshold: {current:.2f} -> {new_value:.2f}")

            return {
                "success": True,
                "parameter": "confidence_threshold",
                "old_value": current,
                "new_value": new_value,
            }
        except Exception as e:
            logger.error(f"Failed to adjust confidence threshold: {e}")
            return {"success": False, "error": str(e)}

    def _adjust_position_size(self, decrease: bool = True) -> Dict[str, Any]:
        """Adjust maximum position size."""
        self._record_action("adjust_position_size")

        try:
            config = self._load_config()
            current = config.get("risk_management", {}).get("max_position_pct", 0.10)

            adjustment = -0.02 if decrease else 0.02
            new_value = max(0.02, min(0.20, current + adjustment))

            config.setdefault("risk_management", {})["max_position_pct"] = new_value
            self._save_config(config)

            logger.info(f"Adjusted max position: {current:.1%} -> {new_value:.1%}")

            return {
                "success": True,
                "parameter": "max_position_pct",
                "old_value": current,
                "new_value": new_value,
            }
        except Exception as e:
            logger.error(f"Failed to adjust position size: {e}")
            return {"success": False, "error": str(e)}

    def _adjust_stop_loss(self, tighter: bool = True) -> Dict[str, Any]:
        """Adjust stop loss settings."""
        self._record_action("adjust_stop_loss")

        try:
            config = self._load_config()
            stop_loss = config.get("risk_management", {}).get("stop_loss", {})
            current = stop_loss.get("fixed_pct", 0.05)

            adjustment = -0.01 if tighter else 0.01
            new_value = max(0.02, min(0.10, current + adjustment))

            config.setdefault("risk_management", {}).setdefault("stop_loss", {})["fixed_pct"] = new_value
            self._save_config(config)

            logger.info(f"Adjusted stop loss: {current:.1%} -> {new_value:.1%}")

            return {
                "success": True,
                "parameter": "stop_loss.fixed_pct",
                "old_value": current,
                "new_value": new_value,
            }
        except Exception as e:
            logger.error(f"Failed to adjust stop loss: {e}")
            return {"success": False, "error": str(e)}

    def _toggle_feature(self, feature: str) -> Dict[str, Any]:
        """Toggle a feature on/off."""
        self._record_action(f"toggle_{feature}")

        try:
            config = self._load_config()

            feature_paths = {
                "degradation_detection": ("retraining", "degradation_detection", "enabled"),
                "auto_rollback": ("retraining", "auto_rollback", "enabled"),
            }

            if feature not in feature_paths:
                return {"success": False, "error": f"Unknown feature: {feature}"}

            path = feature_paths[feature]
            current_config = config
            for key in path[:-1]:
                current_config = current_config.setdefault(key, {})

            current_value = current_config.get(path[-1], False)
            new_value = not current_value
            current_config[path[-1]] = new_value

            self._save_config(config)

            logger.info(f"Toggled {feature}: {current_value} -> {new_value}")

            return {
                "success": True,
                "feature": feature,
                "old_value": current_value,
                "new_value": new_value,
            }
        except Exception as e:
            logger.error(f"Failed to toggle feature {feature}: {e}")
            return {"success": False, "error": str(e)}

    def _add_symbol(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Add a symbol to the trading universe."""
        self._record_action("add_symbol")

        if not self.symbol_manager:
            return {"success": False, "error": "Symbol manager not available"}

        try:
            recommendations = context.get("recommendations", [])
            if not recommendations:
                return {"success": False, "error": "No symbol recommendations in context"}

            added_symbols = []
            skipped_symbols = []

            for rec in recommendations[:3]:
                symbol = rec.get("symbol")
                score = rec.get("score", 0)
                sector = rec.get("sector", "")
                reason = rec.get("reason", "Agent recommendation")

                if not symbol:
                    continue

                success = self.symbol_manager.add_symbol(
                    symbol=symbol,
                    reason=reason,
                    score=score,
                    sector=sector,
                )

                if success:
                    added_symbols.append(symbol)
                    logger.info(f"Added symbol {symbol}")
                else:
                    skipped_symbols.append(symbol)

            return {
                "success": len(added_symbols) > 0,
                "added_symbols": added_symbols,
                "skipped_symbols": skipped_symbols,
                "active_count": len(self.symbol_manager.get_active_symbols()),
            }
        except Exception as e:
            logger.error(f"Failed to add symbol: {e}")
            return {"success": False, "error": str(e)}

    def _remove_symbol(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove a symbol from the trading universe."""
        self._record_action("remove_symbol")

        if not self.symbol_manager:
            return {"success": False, "error": "Symbol manager not available"}

        try:
            recommendations = context.get("recommendations", [])
            removal_recs = [r for r in recommendations if r.get("action") in ("remove", "consider_removal")]

            if not removal_recs:
                return {"success": False, "error": "No removal recommendations"}

            removed_symbols = []
            skipped_symbols = []

            for rec in removal_recs:
                symbol = rec.get("symbol")
                action = rec.get("action")
                reason = rec.get("reason", "Agent recommendation")

                if not symbol:
                    continue

                if action == "remove":
                    success = self.symbol_manager.remove_symbol(
                        symbol=symbol,
                        reason=reason,
                        apply_cooldown=True,
                    )

                    if success:
                        removed_symbols.append(symbol)
                        logger.info(f"Removed symbol {symbol}")
                    else:
                        skipped_symbols.append(symbol)
                else:
                    skipped_symbols.append(symbol)

            return {
                "success": len(removed_symbols) > 0,
                "removed_symbols": removed_symbols,
                "skipped_symbols": skipped_symbols,
                "active_count": len(self.symbol_manager.get_active_symbols()),
            }
        except Exception as e:
            logger.error(f"Failed to remove symbol: {e}")
            return {"success": False, "error": str(e)}

    def _create_action_response(
        self,
        original_message: AgentMessage,
        action: str,
        result: Dict[str, Any]
    ) -> AgentMessage:
        """Create response message for action taken."""
        content = self._format_action_report(action, result)

        return self.create_message(
            recipient=original_message.sender,
            message_type=MessageType.ACTION,
            subject=f"Action Taken: {action.replace('_', ' ').title()}",
            content=content,
            priority=MessagePriority.NORMAL,
            parent_message_id=original_message.id,
            context={"action": action, "result": result},
        )

    def _get_system_state(self) -> Dict[str, Any]:
        """Get current system state."""
        return {
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "action_history": {
                action: ts.isoformat() for action, ts in self._action_history.items()
            },
        }

    def _get_cooldown_status(self) -> Dict[str, bool]:
        """Get cooldown status for all actions."""
        return {action: self._can_take_action(action) for action in self.AVAILABLE_ACTIONS}

    def _get_recent_trades(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent trades."""
        trades = []

        try:
            if self.data_aggregator:
                analytics = self.data_aggregator.prepare_analytics_data(
                    start_date=datetime.now() - timedelta(days=days),
                    end_date=datetime.now()
                )
                trades_df = analytics.get("trades", None)
                if trades_df is not None and not trades_df.empty:
                    trades = trades_df.to_dict("records")
        except Exception as e:
            logger.debug(f"Error getting recent trades: {e}")

        return trades

    def _analyze_execution_quality(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze execution quality of trades."""
        metrics = {
            "total_trades": len(trades),
            "avg_slippage": 0,
            "fill_rate": 100,
            "issues": [],
        }

        if not trades:
            return metrics

        # Calculate slippage if data available
        slippages = []
        for trade in trades:
            if "expected_price" in trade and "fill_price" in trade:
                slippage = abs(trade["fill_price"] - trade["expected_price"]) / trade["expected_price"]
                slippages.append(slippage)

        if slippages:
            metrics["avg_slippage"] = sum(slippages) / len(slippages)
            if metrics["avg_slippage"] > 0.005:  # >0.5% slippage
                metrics["issues"].append({
                    "type": "high_slippage",
                    "severity": "warning",
                    "value": metrics["avg_slippage"],
                })

        return metrics

    def _gather_health_data(self) -> Dict[str, Any]:
        """Gather system health data."""
        health = {
            "timestamp": datetime.now().isoformat(),
            "trading_halted": self._trading_halted,
            "components": {},
        }

        # Check data aggregator
        health["components"]["data_aggregator"] = self.data_aggregator is not None

        # Check symbol manager
        health["components"]["symbol_manager"] = self.symbol_manager is not None

        # Check retrainer
        health["components"]["retrainer"] = self.retrainer is not None

        # Check degradation monitor
        health["components"]["degradation_monitor"] = self.degradation_monitor is not None

        return health

    def _analyze_system_health(self, health_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze system health and identify issues."""
        issues = []

        for component, available in health_data.get("components", {}).items():
            if not available:
                issues.append({
                    "type": "component_unavailable",
                    "component": component,
                    "severity": "warning",
                })

        return issues

    def _load_config(self) -> Dict[str, Any]:
        """Load trading config."""
        import yaml
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f) or {}

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save trading config."""
        import yaml
        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def _format_action_report(self, action: str, result: Dict[str, Any]) -> str:
        """Format action report."""
        lines = [
            f"## Action: {action.replace('_', ' ').title()}",
            f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "### Result",
        ]

        if result.get("success"):
            lines.append(":white_check_mark: Action completed successfully")

            if "old_value" in result and "new_value" in result:
                lines.extend([
                    "",
                    f"- Parameter: `{result.get('parameter', action)}`",
                    f"- Previous: `{result['old_value']}`",
                    f"- New: `{result['new_value']}`",
                ])
            elif "added_symbols" in result:
                lines.extend([
                    "",
                    f"- Added: {', '.join(result['added_symbols']) or 'None'}",
                    f"- Skipped: {', '.join(result['skipped_symbols']) or 'None'}",
                    f"- Active symbols: {result['active_count']}",
                ])
            elif "removed_symbols" in result:
                lines.extend([
                    "",
                    f"- Removed: {', '.join(result['removed_symbols']) or 'None'}",
                    f"- Skipped: {', '.join(result['skipped_symbols']) or 'None'}",
                    f"- Active symbols: {result['active_count']}",
                ])
        else:
            lines.extend([
                ":x: Action failed",
                f"- Error: {result.get('error', 'Unknown error')}",
            ])

        return "\n".join(lines)

    def _format_action_history(self) -> str:
        """Format action history."""
        if not self._action_history:
            return "No actions taken yet."

        lines = ["## Recent Actions", ""]
        for action, timestamp in sorted(self._action_history.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"- {action}: {timestamp.strftime('%Y-%m-%d %H:%M')}")

        return "\n".join(lines)

    def _format_current_config(self) -> str:
        """Format current configuration summary."""
        try:
            config = self._load_config()
            ml_config = config.get("ml_model", {})
            risk_config = config.get("risk_management", {})

            return "\n".join([
                "## Current Configuration",
                "",
                "### ML Model",
                f"- Confidence Threshold: {ml_config.get('confidence_threshold', 0.55):.2f}",
                f"- Primary Model: {ml_config.get('primary_model', 'ensemble')}",
                "",
                "### Risk Management",
                f"- Max Position: {risk_config.get('max_position_pct', 0.10):.1%}",
                f"- Max Daily Loss: {risk_config.get('max_daily_loss_pct', 0.05):.1%}",
                f"- Stop Loss: {risk_config.get('stop_loss', {}).get('fixed_pct', 0.05):.1%}",
            ])
        except Exception as e:
            return f"Error loading config: {e}"

    def _format_cooldown_status(self) -> str:
        """Format cooldown status."""
        lines = ["## Action Cooldown Status", ""]
        now = datetime.now()

        for action in self.AVAILABLE_ACTIONS:
            if action == "no_action":
                continue

            last_time = self._action_history.get(action)
            if last_time is None:
                status = ":white_check_mark: Available"
            else:
                cooldown_end = last_time + timedelta(hours=self.cooldown_hours)
                if now >= cooldown_end:
                    status = ":white_check_mark: Available"
                else:
                    remaining = cooldown_end - now
                    status = f":hourglass: Cooldown ({remaining.seconds // 60}m remaining)"

            lines.append(f"- {action}: {status}")

        return "\n".join(lines)

    def _format_system_status(self) -> str:
        """Format system status."""
        health = self._gather_health_data()

        lines = [
            "## System Status",
            f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Trading Halted:** {self._trading_halted}",
            f"**Halt Reason:** {self._halt_reason or 'N/A'}",
            "",
            "### Components",
        ]

        for component, available in health.get("components", {}).items():
            status = ":white_check_mark:" if available else ":x:"
            lines.append(f"- {component}: {status}")

        return "\n".join(lines)

    def _format_execution_quality_report(self, metrics: Dict[str, Any]) -> str:
        """Format execution quality report."""
        lines = [
            "## Execution Quality Report",
            f"**Check Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "### Metrics",
            f"- Total Trades: {metrics.get('total_trades', 0)}",
            f"- Average Slippage: {metrics.get('avg_slippage', 0):.2%}",
            f"- Fill Rate: {metrics.get('fill_rate', 100):.1f}%",
        ]

        if metrics.get("issues"):
            lines.extend(["", "### Issues"])
            for issue in metrics["issues"]:
                lines.append(f"- {issue['type']}: {issue.get('value', '')}")

        return "\n".join(lines)

    def _format_health_report(
        self,
        health_data: Dict[str, Any],
        issues: List[Dict[str, Any]]
    ) -> str:
        """Format health report."""
        lines = [
            "## System Health Report",
            f"**Check Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        if issues:
            lines.append("### Issues Detected")
            for issue in issues:
                lines.append(f"- {issue['type']}: {issue.get('component', 'N/A')}")

        lines.extend([
            "",
            "### Component Status",
        ])

        for component, available in health_data.get("components", {}).items():
            status = ":white_check_mark:" if available else ":x:"
            lines.append(f"- {component}: {status}")

        return "\n".join(lines)

    def _format_degradation_report(self, reports: List) -> str:
        """Format degradation report."""
        lines = [
            "## Model Degradation Report",
            f"**Check Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        for report in reports:
            lines.extend([
                f"### {report.model_type.upper()}",
                f"- **Status:** Degraded",
                f"- **Recommendation:** {report.recommendation}",
                "",
                "**Degradation Reasons:**",
            ])
            for reason in report.degradation_reasons:
                lines.append(f"- {reason}")
            lines.append("")

        return "\n".join(lines)

    def _format_retrain_result(self, result: Dict[str, Any]) -> str:
        """Format retrain result."""
        if result.get("success"):
            return f"Retrain completed. Models: {', '.join(result.get('result', {}).get('models_trained', []))}"
        return f"Retrain failed: {result.get('error', 'Unknown error')}"

    def get_status(self) -> Dict[str, Any]:
        """Get agent status including action history."""
        status = super().get_status()
        status.update({
            "last_execution_quality_check": self._last_execution_quality_check.isoformat() if self._last_execution_quality_check else None,
            "last_system_health_check": self._last_system_health_check.isoformat() if self._last_system_health_check else None,
            "last_degradation_check": self._last_degradation_check.isoformat() if self._last_degradation_check else None,
            "action_history": {
                action: ts.isoformat() for action, ts in self._action_history.items()
            },
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "cooldown_hours": self.cooldown_hours,
            "auto_retrain_on_degradation": self.auto_retrain_on_degradation,
        })
        return status
