"""
Agent Notifier Module

Discord integration for agent chat with colored embeds and conversation threading.
"""

import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from .base_agent import AgentMessage, AgentRole, MessagePriority, MessageType

logger = logging.getLogger(__name__)

# Module-level singleton
_agent_notifier: Optional["AgentNotifier"] = None
_lock = threading.Lock()


def get_agent_notifier(config: Optional[Dict[str, Any]] = None) -> "AgentNotifier":
    """
    Get or create the singleton AgentNotifier instance.

    Args:
        config: Configuration dictionary with notification settings

    Returns:
        AgentNotifier singleton instance
    """
    global _agent_notifier
    with _lock:
        if _agent_notifier is None:
            _agent_notifier = AgentNotifier(config or {})
        return _agent_notifier


# Discord embed colors for agents
COLORS = {
    AgentRole.STOCK_ANALYST: 0x3498DB,  # Blue
    AgentRole.DEVELOPER: 0x9B59B6,       # Purple
}

# Priority indicators
PRIORITY_EMOJI = {
    MessagePriority.LOW: "",
    MessagePriority.NORMAL: "",
    MessagePriority.HIGH: ":warning:",
    MessagePriority.URGENT: ":rotating_light:",
}

# Message type emoji
TYPE_EMOJI = {
    MessageType.OBSERVATION: ":mag:",
    MessageType.SUGGESTION: ":bulb:",
    MessageType.ACTION: ":hammer:",
    MessageType.RESPONSE: ":speech_balloon:",
    MessageType.STATUS_UPDATE: ":clipboard:",
    MessageType.QUERY: ":question:",
    MessageType.ACKNOWLEDGMENT: ":white_check_mark:",
}


class AgentNotifier:
    """
    Discord notification handler for agent conversations.

    Features:
    - Color-coded embeds per agent (blue for analyst, purple for developer)
    - Priority indicators (warning emoji for high priority)
    - Message type icons
    - Threaded conversations (reply references)
    - Conversation logging to file
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the agent notifier.

        Args:
            config: Configuration dictionary with optional keys:
                - discord_enabled: Whether to send Discord notifications
                - discord_webhook_url: Webhook URL (or uses env var)
        """
        self.config = config
        self.enabled = config.get("discord_enabled", True)

        # Get webhook URL from config or environment
        self.webhook_url = config.get(
            "discord_webhook_url",
            os.environ.get("DISCORD_WEBHOOK_URL")
        )

        if not self.webhook_url and self.enabled:
            logger.warning(
                "Discord webhook URL not configured. "
                "Agent Discord notifications disabled."
            )
            self.enabled = False

        # Set up conversation log file
        self._setup_conversation_log()

        if self.enabled:
            logger.info("Agent notifier initialized with Discord enabled")
        else:
            logger.info("Agent notifier initialized (Discord disabled)")

    def _setup_conversation_log(self) -> None:
        """Set up conversation log file."""
        try:
            from config.settings import LOGS_DIR
            self.log_path = LOGS_DIR / "agent_conversations.log"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.log_path = None
            logger.warning("Could not set up conversation log file")

    def _log_conversation(self, message: AgentMessage) -> None:
        """
        Log message to conversation log file.

        Args:
            message: The agent message to log
        """
        if not self.log_path:
            return

        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"\n{'='*60}\n")
                f.write(f"[{timestamp}] {message.sender.value.upper()} -> {message.recipient.value.upper()}\n")
                f.write(f"Type: {message.message_type.value} | Priority: {message.priority.name}\n")
                f.write(f"Subject: {message.subject}\n")
                f.write(f"{'-'*40}\n")
                f.write(f"{message.content}\n")
                f.write(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"Failed to log conversation: {e}")

    def notify_agent_message(self, message: AgentMessage) -> bool:
        """
        Send an agent message to Discord with appropriate formatting.

        Args:
            message: The agent message to notify about

        Returns:
            True if notification sent successfully (or disabled)
        """
        # Always log to file
        self._log_conversation(message)

        if not self.enabled:
            return True

        try:
            embed = self._create_agent_embed(message)
            payload = {"embeds": [embed]}

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            logger.debug(f"Sent Discord notification for message {message.id}")
            return True

        except requests.RequestException as e:
            logger.error(f"Failed to send Discord agent notification: {e}")
            return False

    def _create_agent_embed(self, message: AgentMessage) -> Dict[str, Any]:
        """
        Create Discord embed for an agent message.

        Args:
            message: The agent message

        Returns:
            Discord embed dictionary
        """
        # Get colors and emojis
        color = COLORS.get(message.sender, 0x808080)
        priority_emoji = PRIORITY_EMOJI.get(message.priority, "")
        type_emoji = TYPE_EMOJI.get(message.message_type, "")

        # Build title with sender info
        sender_name = message.sender.value.replace("_", " ").title()
        recipient_name = message.recipient.value.replace("_", " ").title()

        title = f"{type_emoji} {sender_name} -> {recipient_name}"
        if priority_emoji:
            title = f"{priority_emoji} {title}"

        # Build description (subject)
        description = f"**{message.subject}**"

        # Build fields
        fields = []

        # Main content field (truncate if too long)
        content = message.content
        if len(content) > 1024:
            content = content[:1021] + "..."

        fields.append({
            "name": "Message",
            "value": content,
            "inline": False
        })

        # Add context summary if available
        if message.context:
            context_summary = self._format_context_summary(message.context)
            if context_summary:
                fields.append({
                    "name": "Context",
                    "value": context_summary,
                    "inline": False
                })

        # Add metadata
        meta_parts = [
            f"Type: `{message.message_type.value}`",
            f"Priority: `{message.priority.name}`",
        ]
        if message.requires_response:
            meta_parts.append("Response Required: Yes")
        if message.parent_message_id:
            meta_parts.append(f"Reply to: `{message.parent_message_id[:8]}...`")

        fields.append({
            "name": "Details",
            "value": " | ".join(meta_parts),
            "inline": False
        })

        embed = {
            "title": title,
            "description": description,
            "color": color,
            "fields": fields,
            "timestamp": message.timestamp.isoformat(),
            "footer": {
                "text": f"Message ID: {message.id[:8]}..."
            }
        }

        return embed

    def _format_context_summary(self, context: Dict[str, Any]) -> str:
        """
        Format context dictionary into a brief summary.

        Args:
            context: Context dictionary

        Returns:
            Formatted summary string
        """
        if not context:
            return ""

        summary_parts = []

        # Extract key metrics if present
        if "metrics" in context:
            metrics = context["metrics"]
            if "sharpe_ratio" in metrics:
                summary_parts.append(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            if "max_drawdown" in metrics:
                summary_parts.append(f"DD: {metrics['max_drawdown']:.1%}")
            if "win_rate" in metrics:
                summary_parts.append(f"WR: {metrics['win_rate']:.1%}")

        # Extract action info if present
        if "action" in context:
            summary_parts.append(f"Action: {context['action']}")

        # Limit length
        summary = " | ".join(summary_parts)
        return summary[:500] if len(summary) > 500 else summary

    def notify_agent_startup(self, agents: List[str]) -> bool:
        """
        Notify that the agent system has started.

        Args:
            agents: List of agent names that are active

        Returns:
            True if notification sent successfully
        """
        if not self.enabled:
            return True

        embed = {
            "title": ":robot: Agent System Started",
            "description": "Multi-agent collaboration system is now active.",
            "color": 0x2ECC71,  # Green
            "fields": [
                {
                    "name": "Active Agents",
                    "value": "\n".join(f"- {agent}" for agent in agents),
                    "inline": False
                }
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._send_embed(embed)

    def notify_agent_shutdown(self, reason: str = "Manual shutdown") -> bool:
        """
        Notify that the agent system has stopped.

        Args:
            reason: Shutdown reason

        Returns:
            True if notification sent successfully
        """
        if not self.enabled:
            return True

        embed = {
            "title": ":stop_sign: Agent System Stopped",
            "description": reason,
            "color": 0xE74C3C,  # Red
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._send_embed(embed)

    def notify_agent_error(
        self,
        agent: AgentRole,
        error: str,
        details: Optional[str] = None
    ) -> bool:
        """
        Notify about an agent error.

        Args:
            agent: The agent that encountered the error
            error: Error message
            details: Optional error details

        Returns:
            True if notification sent successfully
        """
        if not self.enabled:
            return True

        agent_name = agent.value.replace("_", " ").title()

        fields = [
            {"name": "Agent", "value": agent_name, "inline": True},
            {"name": "Error", "value": error, "inline": False},
        ]

        if details:
            truncated = details[:500] + "..." if len(details) > 500 else details
            fields.append({
                "name": "Details",
                "value": f"```{truncated}```",
                "inline": False
            })

        embed = {
            "title": ":fire: Agent Error",
            "color": 0xE74C3C,  # Red
            "fields": fields,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._send_embed(embed)

    def notify_action_taken(
        self,
        agent: AgentRole,
        action: str,
        result: str,
        success: bool = True
    ) -> bool:
        """
        Notify about an action taken by an agent.

        Args:
            agent: The agent that took the action
            action: Description of the action
            result: Result of the action
            success: Whether the action was successful

        Returns:
            True if notification sent successfully
        """
        if not self.enabled:
            return True

        agent_name = agent.value.replace("_", " ").title()
        color = 0x2ECC71 if success else 0xE74C3C
        emoji = ":white_check_mark:" if success else ":x:"

        embed = {
            "title": f"{emoji} Action Taken by {agent_name}",
            "color": color,
            "fields": [
                {"name": "Action", "value": action, "inline": False},
                {"name": "Result", "value": result, "inline": False},
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        return self._send_embed(embed)

    def _send_embed(self, embed: Dict[str, Any]) -> bool:
        """
        Send an embed to Discord.

        Args:
            embed: Discord embed dictionary

        Returns:
            True if sent successfully
        """
        try:
            response = requests.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=10
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to send Discord embed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get notifier status."""
        return {
            "enabled": self.enabled,
            "discord_configured": bool(self.webhook_url),
            "log_path": str(self.log_path) if self.log_path else None,
        }
