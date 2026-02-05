"""
Multi-Agent Trading System

This package provides AI-powered agents that monitor and manage the trading system:

4-Agent Architecture:
- MarketIntelligenceAgent: Information gathering, news, macro analysis, sector trends
- RiskGuardianAgent: Risk monitoring, protection, drawdown alerts, emergency actions
- PortfolioStrategistAgent: Stock selection, portfolio composition, rebalancing
- OperationsAgent: Execution, config changes, system health, retraining

Legacy (deprecated):
- StockAnalystAgent: Replaced by MarketIntelligence + PortfolioStrategist
- DeveloperAgent: Replaced by OperationsAgent

The AgentOrchestrator coordinates agent schedules and message passing using APScheduler.
"""

from .base_agent import AgentRole, MessagePriority, MessageType, AgentMessage, BaseAgent
from .message_queue import MessageQueue, get_message_queue
from .llm_client import LLMClient, get_llm_client
from .agent_notifier import AgentNotifier, get_agent_notifier

# New 4-agent system
from .market_intelligence import MarketIntelligenceAgent
from .risk_guardian import RiskGuardianAgent
from .portfolio_strategist import PortfolioStrategistAgent
from .operations_agent import OperationsAgent

# Legacy agents (kept for backward compatibility)
from .stock_analyst import StockAnalystAgent
from .developer_agent import DeveloperAgent

# Orchestration
from .orchestrator import AgentOrchestrator, get_orchestrator

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
    # New 4-agent system
    "MarketIntelligenceAgent",
    "RiskGuardianAgent",
    "PortfolioStrategistAgent",
    "OperationsAgent",
    # Legacy agents (deprecated)
    "StockAnalystAgent",
    "DeveloperAgent",
    # Orchestration
    "AgentOrchestrator",
    "get_orchestrator",
]
