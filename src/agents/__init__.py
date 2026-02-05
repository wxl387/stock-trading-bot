"""
Multi-Agent Collaboration System

This package provides AI-powered agents that monitor and improve the trading bot:
- StockAnalystAgent: Monitors market performance, detects issues, suggests improvements
- DeveloperAgent: Receives suggestions, decides actions, implements changes automatically
- AgentOrchestrator: Coordinates agent schedules and message passing
"""

from .base_agent import AgentRole, MessagePriority, MessageType, AgentMessage, BaseAgent
from .message_queue import MessageQueue, get_message_queue
from .llm_client import LLMClient, get_llm_client
from .agent_notifier import AgentNotifier, get_agent_notifier
from .stock_analyst import StockAnalystAgent
from .developer_agent import DeveloperAgent
from .orchestrator import AgentOrchestrator

__all__ = [
    # Enums and data classes
    "AgentRole",
    "MessagePriority",
    "MessageType",
    "AgentMessage",
    # Base class
    "BaseAgent",
    # Infrastructure
    "MessageQueue",
    "get_message_queue",
    "LLMClient",
    "get_llm_client",
    "AgentNotifier",
    "get_agent_notifier",
    # Agents
    "StockAnalystAgent",
    "DeveloperAgent",
    # Orchestration
    "AgentOrchestrator",
]
