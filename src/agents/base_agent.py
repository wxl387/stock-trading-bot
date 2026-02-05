"""
Base Agent Module

Provides the abstract base class and data structures for the multi-agent system.
"""

import uuid
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .message_queue import MessageQueue
    from .agent_notifier import AgentNotifier
    from .llm_client import LLMClient


class AgentRole(Enum):
    """Defines the roles of agents in the system."""
    STOCK_ANALYST = "stock_analyst"
    DEVELOPER = "developer"


class MessagePriority(Enum):
    """Priority levels for agent messages."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageType(Enum):
    """Types of messages agents can exchange."""
    OBSERVATION = "observation"      # Stock analyst reports what it sees
    SUGGESTION = "suggestion"        # Stock analyst suggests actions
    ACTION = "action"               # Developer takes an action
    RESPONSE = "response"           # Response to a previous message
    STATUS_UPDATE = "status_update"  # General status updates
    QUERY = "query"                 # Asking for information
    ACKNOWLEDGMENT = "acknowledgment"  # Simple acknowledgment


@dataclass
class AgentMessage:
    """
    Represents a message exchanged between agents.

    Attributes:
        id: Unique message identifier
        sender: The agent sending the message
        recipient: The agent receiving the message
        message_type: Type of message (observation, suggestion, action, etc.)
        subject: Brief subject line for the message
        content: Full message content (supports markdown)
        priority: Message priority level
        timestamp: When the message was created
        context: Additional context data (metrics, reports, etc.)
        requires_response: Whether a response is expected
        parent_message_id: ID of the message this is replying to (for threading)
        processed: Whether the message has been processed by recipient
        metadata: Additional metadata for tracking
    """
    sender: AgentRole
    recipient: AgentRole
    message_type: MessageType
    subject: str
    content: str
    priority: MessagePriority = MessagePriority.NORMAL
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = False
    parent_message_id: Optional[str] = None
    processed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for storage."""
        data = asdict(self)
        data["sender"] = self.sender.value
        data["recipient"] = self.recipient.value
        data["message_type"] = self.message_type.value
        data["priority"] = self.priority.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary."""
        data = data.copy()
        data["sender"] = AgentRole(data["sender"])
        data["recipient"] = AgentRole(data["recipient"])
        data["message_type"] = MessageType(data["message_type"])
        data["priority"] = MessagePriority(data["priority"])
        if isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def format_for_display(self) -> str:
        """Format message for human-readable display."""
        priority_emoji = {
            MessagePriority.LOW: "",
            MessagePriority.NORMAL: "",
            MessagePriority.HIGH: "WARNING ",
            MessagePriority.URGENT: "URGENT ",
        }

        lines = [
            f"[{self.sender.value.upper()} -> {self.recipient.value.upper()}] ({self.message_type.value})",
            f"{priority_emoji.get(self.priority, '')}Subject: {self.subject}",
            "",
            self.content,
            "=" * 60,
        ]
        return "\n".join(lines)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    Agents analyze data, process messages, and communicate with each other
    through a persistent message queue.
    """

    def __init__(
        self,
        role: AgentRole,
        config: Dict[str, Any],
        message_queue: "MessageQueue",
        notifier: Optional["AgentNotifier"] = None,
        llm_client: Optional["LLMClient"] = None,
    ):
        """
        Initialize the base agent.

        Args:
            role: The agent's role in the system
            config: Configuration dictionary
            message_queue: Shared message queue for agent communication
            notifier: Optional notifier for Discord integration
            llm_client: Optional LLM client for intelligent analysis
        """
        self.role = role
        self.config = config
        self.message_queue = message_queue
        self.notifier = notifier
        self.llm_client = llm_client
        self.logger = logging.getLogger(f"agent.{role.value}")

        # Track last run time for cooldowns
        self._last_run: Optional[datetime] = None
        self._run_count: int = 0

    @property
    def name(self) -> str:
        """Get the agent's display name."""
        return self.role.value.replace("_", " ").title()

    @abstractmethod
    def analyze(self) -> List[AgentMessage]:
        """
        Perform analysis and generate observations/suggestions.

        Returns:
            List of messages to send to other agents
        """
        pass

    @abstractmethod
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message from another agent.

        Args:
            message: The message to process

        Returns:
            Optional response message
        """
        pass

    def send_message(self, message: AgentMessage) -> None:
        """
        Send a message to the queue and optionally notify via Discord.

        Args:
            message: The message to send
        """
        # Store in persistent queue
        self.message_queue.enqueue(message)

        # Log the message
        self.logger.info(
            f"Sent message to {message.recipient.value}: {message.subject}"
        )

        # Send to Discord if enabled
        if self.notifier:
            try:
                self.notifier.notify_agent_message(message)
            except Exception as e:
                self.logger.error(f"Failed to send Discord notification: {e}")

    def get_pending_messages(self) -> List[AgentMessage]:
        """
        Get all unprocessed messages for this agent.

        Returns:
            List of pending messages
        """
        return self.message_queue.get_messages_for_recipient(
            self.role, processed=False
        )

    def mark_message_processed(self, message_id: str) -> None:
        """
        Mark a message as processed.

        Args:
            message_id: ID of the message to mark
        """
        self.message_queue.mark_processed(message_id)

    def run_cycle(self) -> Dict[str, Any]:
        """
        Run one complete agent cycle.

        This includes:
        1. Processing any pending messages
        2. Running analysis to generate new observations
        3. Sending any generated messages

        Returns:
            Dictionary with cycle results
        """
        self._run_count += 1
        self._last_run = datetime.now()

        results = {
            "agent": self.role.value,
            "timestamp": self._last_run.isoformat(),
            "messages_processed": 0,
            "messages_sent": 0,
            "errors": [],
        }

        # Process pending messages
        pending = self.get_pending_messages()
        for message in pending:
            try:
                response = self.process_message(message)
                self.mark_message_processed(message.id)
                results["messages_processed"] += 1

                if response:
                    self.send_message(response)
                    results["messages_sent"] += 1
            except Exception as e:
                self.logger.error(f"Error processing message {message.id}: {e}")
                results["errors"].append(str(e))

        # Run analysis and generate new messages
        try:
            new_messages = self.analyze()
            for message in new_messages:
                self.send_message(message)
                results["messages_sent"] += 1
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            results["errors"].append(str(e))

        return results

    def get_conversation_history(
        self,
        other_agent: AgentRole,
        limit: int = 10
    ) -> List[AgentMessage]:
        """
        Get recent conversation history with another agent.

        Args:
            other_agent: The other agent in the conversation
            limit: Maximum number of messages to retrieve

        Returns:
            List of messages in chronological order
        """
        return self.message_queue.get_conversation(
            self.role, other_agent, limit=limit
        )

    def create_message(
        self,
        recipient: AgentRole,
        message_type: MessageType,
        subject: str,
        content: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        context: Optional[Dict[str, Any]] = None,
        requires_response: bool = False,
        parent_message_id: Optional[str] = None,
    ) -> AgentMessage:
        """
        Create a new message from this agent.

        Args:
            recipient: Who the message is for
            message_type: Type of message
            subject: Brief subject line
            content: Full message content
            priority: Priority level
            context: Additional context data
            requires_response: Whether a response is expected
            parent_message_id: ID of message being replied to

        Returns:
            New AgentMessage instance
        """
        return AgentMessage(
            sender=self.role,
            recipient=recipient,
            message_type=message_type,
            subject=subject,
            content=content,
            priority=priority,
            context=context or {},
            requires_response=requires_response,
            parent_message_id=parent_message_id,
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.

        Returns:
            Dictionary with status information
        """
        return {
            "role": self.role.value,
            "name": self.name,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "run_count": self._run_count,
            "pending_messages": len(self.get_pending_messages()),
        }
