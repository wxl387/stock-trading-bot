"""
Developer Agent

AI-powered agent that receives suggestions, decides actions, and implements changes automatically.
"""

import json
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


class DeveloperAgent(BaseAgent):
    """
    Developer Agent - implements changes to the trading system.

    Responsibilities:
    - Evaluate suggestions from Stock Analyst
    - Trigger model retraining when needed
    - Adjust configuration parameters
    - Enable/disable features
    - Track action cooldowns to prevent over-reacting

    Available Actions:
    - trigger_retrain: Trigger model retraining
    - adjust_confidence: Adjust ML confidence threshold
    - adjust_risk_params: Modify risk management parameters
    - toggle_feature: Enable/disable features

    Cooldown: 4 hours between same action types (configurable)
    """

    # Available actions the developer can take
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
        Initialize Developer agent.

        Args:
            config: Agent configuration
            message_queue: Shared message queue
            notifier: Optional Discord notifier
            llm_client: Optional LLM client for decision support
        """
        super().__init__(
            role=AgentRole.DEVELOPER,
            config=config,
            message_queue=message_queue,
            notifier=notifier,
            llm_client=llm_client,
        )

        # Load settings from config
        developer_config = config.get("developer", {})
        self.auto_retrain_on_degradation = developer_config.get("auto_retrain_on_degradation", True)
        self.auto_adjust_confidence = developer_config.get("auto_adjust_confidence", True)
        self.cooldown_hours = developer_config.get("cooldown_hours", 4)

        # Track action history for cooldowns
        self._action_history: Dict[str, datetime] = {}

        # Cache for retrainer and config path
        self._retrainer = None
        self._config_path = None
        self._symbol_manager = None

        logger.info(f"Developer agent initialized with cooldown={self.cooldown_hours}h, "
                   f"auto_retrain={self.auto_retrain_on_degradation}, "
                   f"auto_adjust={self.auto_adjust_confidence}")

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

    def analyze(self) -> List[AgentMessage]:
        """
        Developer agent doesn't proactively analyze - it responds to messages.

        Returns:
            Empty list (developer is reactive, not proactive)
        """
        return []

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming messages and decide on actions.

        Args:
            message: Incoming message from Stock Analyst

        Returns:
            Response message with action taken or decision
        """
        logger.info(f"Processing message: {message.subject}")

        # Handle different message types
        if message.message_type == MessageType.OBSERVATION:
            return self._handle_observation(message)

        elif message.message_type == MessageType.SUGGESTION:
            return self._handle_suggestion(message)

        elif message.message_type == MessageType.STATUS_UPDATE:
            return self._handle_status_update(message)

        elif message.message_type == MessageType.QUERY:
            return self._handle_query(message)

        elif message.message_type == MessageType.ACKNOWLEDGMENT:
            # No response needed for acknowledgments
            logger.info(f"Received acknowledgment: {message.subject}")
            return None

        return None

    def _handle_observation(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle performance observation from Stock Analyst."""
        issues = message.context.get("issues", [])
        metrics = message.context.get("metrics", {})

        if not issues:
            # No issues, just acknowledge
            return self.create_message(
                recipient=AgentRole.STOCK_ANALYST,
                message_type=MessageType.ACKNOWLEDGMENT,
                subject="Observation Received",
                content="Performance metrics received. No immediate action required.",
                priority=MessagePriority.LOW,
                parent_message_id=message.id,
            )

        # Evaluate issues and decide on action
        action, reasoning = self._decide_action_for_issues(issues, metrics)

        if action == "no_action":
            return self.create_message(
                recipient=AgentRole.STOCK_ANALYST,
                message_type=MessageType.RESPONSE,
                subject="No Action Required",
                content=f"After evaluation:\n\n{reasoning}\n\nNo immediate action will be taken.",
                priority=MessagePriority.NORMAL,
                parent_message_id=message.id,
                context={"decision": "no_action", "reasoning": reasoning},
            )

        # Execute the action
        result = self._execute_action(action, message.context)

        return self.create_message(
            recipient=AgentRole.STOCK_ANALYST,
            message_type=MessageType.ACTION,
            subject=f"Action: {action.replace('_', ' ').title()}",
            content=self._format_action_report(action, result, reasoning),
            priority=MessagePriority.NORMAL,
            parent_message_id=message.id,
            context={"action": action, "result": result},
        )

    def _handle_suggestion(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle suggestion from Stock Analyst (e.g., degradation alert)."""
        recommendations = message.context.get("recommendations", [])
        degraded_models = message.context.get("degraded_models", [])

        # Determine action based on recommendations
        if "retrain" in recommendations:
            if self._can_take_action("trigger_retrain"):
                result = self._trigger_retrain(reason="degradation")
                action = "trigger_retrain"
            else:
                action = "no_action"
                result = {"skipped": True, "reason": "cooldown"}
        elif "rollback" in recommendations:
            # Rollback is handled automatically by AutoRollbackManager
            action = "no_action"
            result = {"note": "Rollback handled by AutoRollbackManager"}
        else:
            action = "no_action"
            result = {"note": "No actionable recommendation"}

        # Get LLM reasoning if available
        reasoning = self._get_llm_decision(message.content, degraded_models) or \
                   "Based on degradation signals and configured policies."

        return self.create_message(
            recipient=AgentRole.STOCK_ANALYST,
            message_type=MessageType.ACTION if action != "no_action" else MessageType.RESPONSE,
            subject=f"Action: {action.replace('_', ' ').title()}" if action != "no_action" else "Suggestion Evaluated",
            content=self._format_action_report(action, result, reasoning),
            priority=MessagePriority.NORMAL,
            parent_message_id=message.id,
            context={"action": action, "result": result},
        )

    def _handle_status_update(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle status updates (daily reviews, etc.)."""
        # For daily reviews, just acknowledge
        if "Daily Review" in message.subject:
            return self.create_message(
                recipient=AgentRole.STOCK_ANALYST,
                message_type=MessageType.ACKNOWLEDGMENT,
                subject="Daily Review Acknowledged",
                content="Daily review received and logged. Continue monitoring.",
                priority=MessagePriority.LOW,
                parent_message_id=message.id,
            )

        return None

    def _handle_query(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle queries from Stock Analyst."""
        query_type = message.context.get("query_type", "general")

        if query_type == "action_history":
            content = self._format_action_history()
        elif query_type == "current_config":
            content = self._format_current_config()
        elif query_type == "cooldown_status":
            content = self._format_cooldown_status()
        else:
            content = f"Unknown query type: {query_type}"

        return self.create_message(
            recipient=AgentRole.STOCK_ANALYST,
            message_type=MessageType.RESPONSE,
            subject=f"Re: {message.subject}",
            content=content,
            priority=MessagePriority.NORMAL,
            parent_message_id=message.id,
        )

    def _decide_action_for_issues(
        self,
        issues: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> tuple[str, str]:
        """
        Decide what action to take based on issues.

        Args:
            issues: List of detected issues
            metrics: Current performance metrics

        Returns:
            Tuple of (action, reasoning)
        """
        # Use LLM for decision if available
        if self.llm_client and self.llm_client.is_available():
            llm_response = self.llm_client.evaluate_suggestion(
                suggestion=f"Issues detected: {issues}",
                current_state=metrics,
                available_actions=self.AVAILABLE_ACTIONS,
            )
            if llm_response:
                # Parse LLM response to extract action
                action = self._parse_llm_action(llm_response)
                return action, llm_response

        # Rule-based fallback
        reasoning_parts = []

        # Check for high drawdown - most critical
        high_drawdown = any(i["type"] == "high_drawdown" and i.get("severity") == "high" for i in issues)
        if high_drawdown:
            reasoning_parts.append("High drawdown detected - considering risk adjustment.")
            if self._can_take_action("adjust_position_size") and self.auto_adjust_confidence:
                return "adjust_position_size", "\n".join(reasoning_parts)

        # Check for low Sharpe + low win rate - suggests model issues
        low_sharpe = any(i["type"] == "low_sharpe" for i in issues)
        low_win_rate = any(i["type"] == "low_win_rate" for i in issues)

        if low_sharpe and low_win_rate:
            reasoning_parts.append("Low Sharpe and win rate suggest model degradation.")
            if self._can_take_action("trigger_retrain") and self.auto_retrain_on_degradation:
                return "trigger_retrain", "\n".join(reasoning_parts)

        # Check for just low Sharpe - might be market conditions
        if low_sharpe:
            reasoning_parts.append("Low Sharpe ratio detected. May be due to market conditions.")
            if self._can_take_action("adjust_confidence_threshold") and self.auto_adjust_confidence:
                return "adjust_confidence_threshold", "\n".join(reasoning_parts)

        # Default: no action
        reasoning_parts.append("Issues detected but no automatic action configured or on cooldown.")
        return "no_action", "\n".join(reasoning_parts) if reasoning_parts else "No actionable issues."

    def _parse_llm_action(self, llm_response: str) -> str:
        """Parse action from LLM response."""
        response_lower = llm_response.lower()

        for action in self.AVAILABLE_ACTIONS:
            if action.replace("_", " ") in response_lower or action in response_lower:
                if self._can_take_action(action):
                    return action

        return "no_action"

    def _get_llm_decision(
        self,
        suggestion: str,
        context: Any
    ) -> Optional[str]:
        """Get LLM decision on a suggestion."""
        if not self.llm_client or not self.llm_client.is_available():
            return None

        return self.llm_client.evaluate_suggestion(
            suggestion=suggestion,
            current_state={"context": str(context)},
            available_actions=self.AVAILABLE_ACTIONS,
        )

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
        """
        Execute an action.

        Args:
            action: Action to execute
            context: Context from the triggering message

        Returns:
            Result dictionary
        """
        if action == "trigger_retrain":
            return self._trigger_retrain(reason="performance_issue")

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

        elif action == "adjust_allocation":
            return self._adjust_allocation(context)

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

            # Adjust by 0.02 (2%)
            adjustment = 0.02 if increase else -0.02
            new_value = max(0.50, min(0.70, current + adjustment))

            # Update config
            config.setdefault("ml_model", {})["confidence_threshold"] = new_value
            self._save_config(config)

            logger.info(f"Adjusted confidence threshold: {current:.2f} -> {new_value:.2f}")

            return {
                "success": True,
                "parameter": "confidence_threshold",
                "old_value": current,
                "new_value": new_value,
                "direction": "increased" if increase else "decreased",
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

            # Adjust by 0.02 (2%)
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
                "direction": "decreased" if decrease else "increased",
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

            # Adjust by 0.01 (1%)
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
                "direction": "tighter" if tighter else "looser",
            }
        except Exception as e:
            logger.error(f"Failed to adjust stop loss: {e}")
            return {"success": False, "error": str(e)}

    def _toggle_feature(self, feature: str) -> Dict[str, Any]:
        """Toggle a feature on/off."""
        self._record_action(f"toggle_{feature}")

        try:
            config = self._load_config()

            # Map feature names to config paths
            feature_paths = {
                "degradation_detection": ("retraining", "degradation_detection", "enabled"),
                "auto_rollback": ("retraining", "auto_rollback", "enabled"),
            }

            if feature not in feature_paths:
                return {"success": False, "error": f"Unknown feature: {feature}"}

            path = feature_paths[feature]

            # Navigate to the setting
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
            # Get recommendations from context
            recommendations = context.get("recommendations", [])
            if not recommendations:
                return {"success": False, "error": "No symbol recommendations in context"}

            # Process up to 3 symbols at a time (conservative approach)
            added_symbols = []
            skipped_symbols = []

            for rec in recommendations[:3]:
                symbol = rec.get("symbol")
                score = rec.get("score", 0)
                sector = rec.get("sector", "")
                reason = rec.get("reason", "Stock screening recommendation")

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
                    logger.info(f"Added symbol {symbol} (score={score:.1f})")
                else:
                    skipped_symbols.append(symbol)
                    logger.info(f"Skipped symbol {symbol}")

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
            # Get recommendations from context
            recommendations = context.get("recommendations", [])
            if not recommendations:
                return {"success": False, "error": "No removal recommendations in context"}

            # Filter for removal recommendations
            removal_recs = [
                r for r in recommendations
                if r.get("action") in ("remove", "consider_removal")
            ]

            if not removal_recs:
                return {"success": False, "error": "No removal actions in recommendations"}

            # Process only "remove" actions (not "consider_removal")
            removed_symbols = []
            skipped_symbols = []

            for rec in removal_recs:
                symbol = rec.get("symbol")
                action = rec.get("action")
                reason = rec.get("reason", "Portfolio review recommendation")

                if not symbol:
                    continue

                # Only auto-remove if action is "remove" (not "consider_removal")
                if action == "remove":
                    success = self.symbol_manager.remove_symbol(
                        symbol=symbol,
                        reason=reason,
                        apply_cooldown=True,
                    )

                    if success:
                        removed_symbols.append(symbol)
                        logger.info(f"Removed symbol {symbol} (reason: {reason})")
                    else:
                        skipped_symbols.append(symbol)
                else:
                    skipped_symbols.append(symbol)
                    logger.info(f"Skipped {symbol} (action was consider_removal, not remove)")

            return {
                "success": len(removed_symbols) > 0,
                "removed_symbols": removed_symbols,
                "skipped_symbols": skipped_symbols,
                "active_count": len(self.symbol_manager.get_active_symbols()),
            }

        except Exception as e:
            logger.error(f"Failed to remove symbol: {e}")
            return {"success": False, "error": str(e)}

    def _adjust_allocation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust allocation for a symbol (future implementation)."""
        self._record_action("adjust_allocation")

        # This would adjust target weights in portfolio optimization
        # For now, just acknowledge the request
        return {
            "success": False,
            "error": "Allocation adjustment not yet implemented. "
                    "Use portfolio optimization settings in config."
        }

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

    def _format_action_report(
        self,
        action: str,
        result: Dict[str, Any],
        reasoning: str
    ) -> str:
        """Format an action report."""
        lines = [
            f"## Action Taken: {action.replace('_', ' ').title()}",
            f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "### Reasoning",
            reasoning,
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
            elif "result" in result:
                for key, value in result["result"].items():
                    lines.append(f"- {key}: {value}")
        elif result.get("skipped"):
            lines.extend([
                ":hourglass: Action skipped",
                f"- Reason: {result.get('reason', 'Unknown')}",
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

            lines = [
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
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"Error loading config: {e}"

    def _format_cooldown_status(self) -> str:
        """Format cooldown status for all actions."""
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

    def get_status(self) -> Dict[str, Any]:
        """Get agent status including action history."""
        status = super().get_status()
        status.update({
            "action_history": {
                action: ts.isoformat()
                for action, ts in self._action_history.items()
            },
            "cooldown_hours": self.cooldown_hours,
            "auto_retrain_on_degradation": self.auto_retrain_on_degradation,
            "auto_adjust_confidence": self.auto_adjust_confidence,
        })
        return status
