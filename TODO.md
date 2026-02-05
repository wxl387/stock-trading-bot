# Trading Bot TODO

## Phase 25: 4-Agent Trading System (COMPLETE)

### Overview
Replaced 2-agent system with comprehensive 4-agent architecture.

### Phase 1: Infrastructure Updates
- [x] Update `AgentRole` enum with 4 new roles
- [x] Update `agent_notifier.py` with new Discord colors
- [x] Update `__init__.py` exports

### Phase 2: Market Intelligence Agent
- [x] Create `src/agents/market_intelligence.py`
- [x] Implement news scanning with SentimentFetcher
- [x] Implement earnings tracking
- [x] Implement macro analysis
- [x] Add LLM prompts for news/macro analysis

### Phase 3: Risk Guardian Agent
- [x] Create `src/agents/risk_guardian.py`
- [x] Implement risk metrics calculation
- [x] Implement drawdown monitoring
- [x] Implement correlation tracking
- [x] Implement emergency actions
- [x] Add LLM prompts for risk assessment

### Phase 4: Portfolio Strategist Agent
- [x] Create `src/agents/portfolio_strategist.py`
- [x] Migrate screening from stock_analyst.py
- [x] Implement performance attribution
- [x] Implement rebalancing logic
- [x] Add LLM prompts for portfolio decisions

### Phase 5: Operations Agent
- [x] Create `src/agents/operations_agent.py`
- [x] Migrate actions from developer_agent.py
- [x] Add execution quality monitoring
- [x] Add system health checks
- [x] Add LLM prompts for operations

### Phase 6: Orchestrator & Integration
- [x] Rewrite orchestrator for 4-agent system
- [x] Schedule all 16 agent jobs
- [x] Update configuration loading
- [x] Test full system verification

---

## Phase 24: Dynamic AI-Driven Stock Selection (COMPLETE)

### Phase 1: Fundamental Data Infrastructure
- [x] Create `src/data/fundamental_fetcher.py` - fetch financial data via yfinance
- [x] Create `src/data/stock_universe.py` - manage S&P 500/NASDAQ 100 constituents
- [x] Update `src/data/feature_engineer.py` - add fundamental features

### Phase 2: Stock Screening System
- [x] Create `src/screening/` directory
- [x] Create `src/screening/stock_screener.py` - multi-factor scoring
- [x] Create `src/screening/screening_strategies.py` - growth/value/momentum/balanced

### Phase 3: Symbol Management
- [x] Create `src/core/symbol_manager.py` - dynamic add/remove at runtime
- [x] Create `src/risk/market_timing.py` - market condition analysis
- [x] Update `config/trading_config.yaml` - add dynamic_symbols config

### Phase 4: Agent Enhancements
- [x] Update `src/agents/stock_analyst.py` - add screening methods
- [x] Update `src/agents/llm_client.py` - add screening prompts
- [x] Update `src/agents/developer_agent.py` - add symbol actions
- [x] Update `src/agents/orchestrator.py` - add screening schedules

### Phase 5: Trading Engine Integration
- [x] Update `src/core/trading_engine.py` - integrate SymbolManager
- [x] Test full system with dynamic symbols
- [x] Documentation and validation

---

## Phase 23: Multi-Agent System (COMPLETE)

### Phase 1: Core Infrastructure
- [x] Create `src/agents/` directory
- [x] Implement `base_agent.py` with AgentMessage dataclass
- [x] Implement `message_queue.py` SQLite queue
- [x] Add ANTHROPIC_API_KEY to .env.example

### Phase 2: LLM Integration
- [x] Implement `llm_client.py` Claude wrapper
- [x] Add performance analysis prompts
- [x] Add decision support prompts
- [x] Configure rate limiting

### Phase 3: Discord Integration
- [x] Implement `agent_notifier.py` Discord integration
- [x] Color-coded embeds per agent
- [x] Priority indicators
- [x] Conversation logging

### Phase 4: Stock Analyst Agent
- [x] Implement `stock_analyst.py`
- [x] Integrate with DataAggregator
- [x] Integrate with DegradationMonitor
- [x] Add performance analysis logic
- [x] Add health check task
- [x] Add degradation check task
- [x] Add daily review task

### Phase 5: Developer Agent
- [x] Implement `developer_agent.py`
- [x] Add config modification capabilities
- [x] Integrate with ScheduledRetrainer
- [x] Implement action cooldowns
- [x] Add trigger_retrain action
- [x] Add adjust_confidence action
- [x] Add adjust_position_size action
- [x] Add toggle_feature action

### Phase 6: Orchestration
- [x] Implement `orchestrator.py`
- [x] Configure APScheduler jobs
- [x] Integrate with TradingEngine
- [x] Add to trading_config.yaml

---

## Phase 22: Production Deployment (NEXT)

### Paper Trading Setup
- [ ] Switch mode from `simulated` to `paper`
- [ ] Connect to WebBull paper trading account
- [ ] Enable regime-aware optimization
- [ ] Set up automated daily trading cycle
- [ ] Configure notifications for all trades

### Paper Trading Monitoring (2-4 weeks)
- [ ] Track Sharpe ratio over rolling 2-week periods
- [ ] Compare actual vs estimated transaction costs
- [ ] Monitor portfolio drift before rebalancing
- [ ] Track regime transitions and adaptations
- [ ] Verify system reliability (no crashes)

### Small Live Capital (after paper trading success)
- [ ] Deploy with $5,000-$10,000 real capital
- [ ] Continue monitoring all metrics
- [ ] Scale to full capital after 1-2 months

---

## Future Enhancements

### Agent System Improvements
- [ ] Add more sophisticated LLM prompts
- [ ] Implement multi-turn conversations
- [ ] Add sentiment analysis integration
- [ ] Add market news monitoring
- [ ] Add risk-adjusted position sizing suggestions

### System Reliability
- [ ] Add Prometheus metrics export
- [ ] Add Grafana dashboards
- [ ] Add health check endpoints
- [ ] Add automated recovery from failures

### Advanced Features
- [ ] Options strategies support
- [ ] Pairs trading / statistical arbitrage
- [ ] Sector rotation strategies
- [ ] Tax-loss harvesting integration

---

## Quick Commands

```bash
# Start trading with agents enabled
python scripts/start_trading.py --simulated --interval 60 --ensemble

# Check agent status
python -c "from src.agents.orchestrator import get_orchestrator; print(get_orchestrator().get_status())"

# View agent conversation history
cat logs/agent_conversations.log

# Test message queue
python -c "from src.agents.message_queue import get_message_queue; print(get_message_queue().get_stats())"
```

---

*Last updated: 2026-02-04*
