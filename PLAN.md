# Stock Trading Bot - Project Plan

## Project Overview

An ML-powered stock trading bot with ensemble models, automated retraining, risk management, and market regime detection. Supports simulated, paper, and live trading modes.

---

## Completed Phases

### Phase 1-12: Core Infrastructure (Initial Commit)
- **Trading Engine**: Main orchestrator with trading cycles, position management
- **Brokers**: SimulatedBroker (offline), WebullBroker (paper/live)
- **ML Models**: XGBoost, LSTM, CNN, MLP, Transformer
- **Ensemble Model**: Combines XGBoost+LSTM+CNN with voting strategies
- **Data Pipeline**: Yahoo Finance fetcher, 1h caching, multi-symbol support
- **Feature Engineering**: 50+ technical indicators (SMA, EMA, MACD, RSI, Bollinger, ATR, etc.)
- **Risk Manager**: Position sizing, daily loss limits, stop-loss, take-profit
- **Backtester**: Vectorized backtesting with vectorbt, comprehensive metrics
- **Dashboard**: Streamlit app with portfolio, positions, trade history, analytics
- **Notifications**: Discord and Telegram integration
- **Sentiment Fetcher**: BlueSky/viaNexus API, Finnhub fallback
- **Macro Data**: FRED API integration (VIX, unemployment, treasury rates)

### Phase 13: FinBERT News Sentiment Analysis
- Local NLP using FinBERT transformer model
- Sentiment scoring for news articles
- Integration with feature engineering pipeline

### Phase 14: Walk-Forward Backtesting Framework
- Multi-window walk-forward validation (no look-ahead bias)
- Configurable train/test/step periods
- Out-of-sample performance tracking

### Phase 15: Walk-Forward Hyperparameter Optimization
- Optuna-based hyperparameter search
- 50+ trials per optimization run
- Multi-metric optimization (Sharpe, profit factor, accuracy)

### Phase 16: Market Regime Detection
- Bull/Bear/Choppy/Volatile regime classification
- VIX-based volatility detection
- ADX-based trend detection
- Regime-specific confidence and position sizing adjustments
- Integrated into backtester and optimizer

### Phase 17: Risk Management Controls
- Stop-loss: Fixed percentage and ATR-based
- Trailing stops: Lock in profits as price moves favorably
- Kelly Criterion: Half-Kelly position sizing based on win rate
- Circuit breaker: Halt trading after consecutive losses or daily loss limit
- Max drawdown protection with recovery mode

### Phase 18: Automated Retraining Pipeline
- **Degradation Monitor** (`src/ml/degradation_monitor.py`):
  - Periodic health checks on production models
  - Detection signals: accuracy drop, confidence collapse, Sharpe decline, win rate
  - Persists history to `models/monitoring_log.json`
- **Auto-Rollback** (`src/ml/auto_rollback.py`):
  - Grace period monitoring (default 5 days) for new deployments
  - Automatic rollback to previous version if degradation detected
  - Rollback history in `models/versions/registry.json`
- **Walk-Forward Validation**: Multi-metric gate before deployment
  - Weighted comparison: Sharpe (35%), profit factor (25%), accuracy (20%), max drawdown (20%)
- **Scheduled Retrainer**: APScheduler jobs for retraining, degradation checks, grace period checks
- **CLI Commands**:
  - `--check-degradation`: Run degradation check
  - `--walk-forward-validate`: Use walk-forward validation for deployment
  - `--rollback MODEL_TYPE`: Manual rollback
  - `--monitoring-status`: Show monitoring status

### Phase 19: Portfolio Optimization Module
- **PortfolioOptimizer** (`src/portfolio/portfolio_optimizer.py`):
  - 5 optimization methods: equal-weight, max Sharpe, risk-parity, minimum variance, mean-variance
  - Ledoit-Wolf covariance shrinkage for numerical stability
  - ML signal integration (tilt weights based on BUY/SELL signals)
  - Weight constraints (min/max per asset) with infeasible constraint detection
  - Automatic fallback to equal-weight if optimization fails
- **EfficientFrontier** (`src/portfolio/efficient_frontier.py`):
  - Calculate efficient frontier (100+ points)
  - Find tangency portfolio (maximum Sharpe ratio)
  - Find minimum variance portfolio
  - Plotly visualization support with capital market line
- **CorrelationAnalyzer** (`src/portfolio/correlation_analyzer.py`):
  - Correlation matrix calculation with Ledoit-Wolf shrinkage
  - Diversification ratio (weighted avg vol / portfolio vol)
  - Hierarchical clustering to identify correlated assets
  - Concentration risk detection and warnings
  - Correlation statistics (mean, median, max, min)
- **PortfolioRebalancer** (`src/portfolio/rebalancer.py`):
  - Threshold-based rebalancing (drift > 5%)
  - Calendar-based rebalancing (weekly/monthly/quarterly)
  - Combined triggers (both threshold AND calendar)
  - Minimal trade generation (only what's needed)
  - Trade filtering (skip trades < $100)
  - Rebalancing history tracking
- **Testing**: 34/34 tests passing, comprehensive manual test suite

### Phase 20: Portfolio Optimization Integration ✅ COMPLETE

(Full details above)

### Phase 21: Portfolio Optimization Enhancements ✅ COMPLETE
- **Efficient Frontier Visualization** (`src/portfolio/efficient_frontier.py` + dashboard):
  - Interactive Plotly chart showing efficient frontier
  - Marks tangency portfolio (Max Sharpe), minimum variance, and current portfolio
  - Capital market line overlay
  - Expandable weights display for tangency portfolio
  - Integrated into "Portfolio Optimization" tab in dashboard
- **Correlation Heatmap** (`src/portfolio/correlation_analyzer.py` + dashboard):
  - Interactive correlation matrix heatmap with color scale
  - Hierarchical clustering for asset grouping (3 clusters identified)
  - Correlation statistics (mean: 0.456, max: 0.582)
  - Diversification ratio calculation (1.35)
  - Concentration risk warnings (57.1% in Cloud/AI cluster)
- **Regime-Aware Portfolio Optimization** (`src/portfolio/portfolio_optimizer.py`):
  - New `optimize_regime_aware()` method
  - Automatic method selection based on market regime:
    - BULL → Max Sharpe (aggressive growth)
    - BEAR → Min Variance (defensive, max 20% position limit)
    - CHOPPY → Risk Parity (balanced risk)
    - VOLATILE → Min Variance (defensive, 15% cash buffer)
  - Integrated with RegimeDetector for automatic regime detection
  - Configuration setting: `regime_aware: true` (enabled by default)
- **Transaction Cost Modeling** (`src/portfolio/transaction_costs.py`):
  - New `TransactionCostModel` class with comprehensive cost estimation
  - Slippage estimation (10 basis points base, volume-adjusted)
  - Market impact modeling (square-root law)
  - Commission costs (zero for commission-free brokers)
  - Integration with PortfolioRebalancer for automatic cost calculation
  - Dashboard display of expected costs, turnover, and breakdown
- **Testing & Validation**:
  - **Efficient Frontier**: 22 points calculated, Sharpe 1.43
  - **Correlation Analysis**: 7x7 matrix, 3 clusters, diversification ratio 1.35
  - **Regime-Aware**: All 4 regimes tested, Sharpe 1.05-1.25
  - **Transaction Costs**: 0.01% for small rebalancing, 0.155% for complete rebalancing
  - **All Tests Passed**: 4/4 comprehensive tests
  - See `PHASE_21_VALIDATION.md` for full details

---

## Current State

### What Works End-to-End
1. **Simulated Trading**: Fetch data → Optimize portfolio → Generate signals → Execute trades → Monitor dashboard
2. **Backtesting**: Walk-forward validation with regime detection and risk management
3. **Model Training**: XGBoost, LSTM, CNN individually or as ensemble
4. **Scheduled Retraining**: Auto-deployment with degradation detection (Phase 18)
5. **Risk Management**: Stop-loss, trailing stops, position sizing, drawdown protection
6. **Portfolio Optimization**: Fully integrated with automatic rebalancing (Phases 19-20)
7. **Notifications**: Trade alerts via Discord

### Running Services
- **Trading Bot**: `python scripts/start_trading.py --simulated --interval 60 --ensemble`
- **Dashboard**: `streamlit run src/dashboard/app.py --server.headless true`

### Configuration
All settings in `config/trading_config.yaml`:
- Trading mode: `simulated` / `paper` / `live`
- Symbols: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- Risk limits: 10% max position, 5% daily loss limit, 80% max exposure
- Stop-loss: 3% fixed, ATR multiplier 2.0
- Trailing stop: 2% distance
- Model: ensemble with soft voting
- Retraining: Weekly schedule, auto-deploy if >1% improvement
- **Portfolio Optimization**: Enabled, max Sharpe method, 10% drift threshold, monthly rebalancing

### Phase 18 Features (Disabled by Default)
Enable in `config/trading_config.yaml` under `retraining:`:
```yaml
degradation_detection:
  enabled: true
  check_interval_hours: 12

auto_rollback:
  enabled: true
  grace_period_days: 5

walk_forward_validation:
  enabled: true
  n_windows: 3
```

### Phase 20 Features (Integrated and Active)
Portfolio optimization is fully integrated into the trading engine and enabled by default.

**Current Configuration** (`config/trading_config.yaml`):
```yaml
portfolio_optimization:
  enabled: true                     # ✅ Active
  method: "max_sharpe"              # Maximize Sharpe ratio
  lookback_days: 252                # 1 year of returns
  min_weight: 0.05                  # Min 5% per asset
  max_weight: 0.30                  # Max 30% per asset
  risk_free_rate: 0.05              # 5% annual

  incorporate_signals: true         # Tilt weights based on ML signals
  signal_tilt_strength: 0.15        # 15% max tilt

  rebalancing:
    enabled: true
    trigger_type: "combined"        # threshold AND calendar
    drift_threshold: 0.10           # Rebalance if drift > 10%
    frequency: "monthly"            # Monthly rebalancing
    min_trade_value: 200.0          # Min $200 per trade
    max_trades_per_rebalance: 8     # Max 8 trades
```

**How It Works:**
- Portfolio optimizer runs automatically during each trading cycle
- Calculates optimal weights using selected method (default: max Sharpe)
- Checks if rebalancing is needed (drift > 10% OR monthly)
- Executes rebalancing trades automatically
- Passes target weights to ML strategy for position sizing
- Logs optimization metrics (Sharpe, return, volatility)

**Dashboard:**
Navigate to "Portfolio Optimization" tab to view:
- Current vs target allocation
- Optimization settings
- Rebalancing status
- Documentation

**Testing:**
```bash
# Manual test of portfolio optimization module
python scripts/test_portfolio_optimization.py

# Integration test (Phase 20)
python scripts/test_phase20_integration.py

# Unit tests
pytest tests/test_portfolio_optimizer.py -v
```

---

## Immediate Next Steps (Phase 22 Options)

**Phase 20 & 21 Status**: ✅ COMPLETE AND THOROUGHLY VALIDATED
- Phase 20: Portfolio optimization integration validated across 1, 2, 5 year periods
- Phase 21: Advanced visualizations, regime-aware optimization, transaction cost modeling
- All tests passed (4/4), all features working
- Production-ready with LOW risk level

### Option 1: Deploy to Production (Recommended)
- Move from simulated to paper trading mode with real broker
- Monitor performance with real market data and regime-aware optimization
- Collect live trading metrics with transaction cost tracking
- Transition to live trading with real capital after validation period
- **Why**: Phases 20-21 thoroughly validated; ready for real-world deployment

### Option 2: Advanced Backtesting & Validation
- Rolling window efficient frontier analysis
- Regime transition backtesting (how strategy performs during regime changes)
- Transaction cost impact analysis on long-term performance
- Multi-period optimization (tactical + strategic)
- **Why**: Further validate regime-aware system before production deployment

### Option 2: Advanced Order Execution
- Limit orders, stop-limit, OCO (one-cancels-other)
- Smart order routing to minimize slippage
- Partial fill handling and retry logic
- Execution quality analytics
- **Why**: Only market orders supported; advanced execution reduces trading costs

### Option 3: Real-Time Data & Intraday Trading
- WebSocket integration for live price feeds
- Intraday signal generation (1m/5m/15m bars)
- Streaming feature updates
- **Why**: Currently daily data only; real-time enables intraday strategies

### Option 4: Production Monitoring & Observability
- Prometheus metrics export
- Strategy P&L time-series tracking
- Model prediction drift detection
- System health dashboard
- **Why**: Needed for reliable 24/7 operation

### Option 5: Multi-Strategy Orchestration
- Per-symbol strategy selection
- Strategy correlation tracking
- Adaptive confidence thresholds
- A/B testing framework
- **Why**: Single strategy for all symbols; multi-strategy improves adaptability

---

## Future Roadmap

### Near-Term (Phases 19-22)
- [x] Portfolio optimization with efficient frontier (Phase 19 - Complete)
- [x] Portfolio optimization integration into trading engine (Phase 20 - Complete & Validated)
- [ ] Production deployment (paper trading → live trading)
- [ ] Portfolio optimization enhancements (visualizations, regime-awareness)
- [ ] Advanced order types (limit, stop-limit, OCO)
- [ ] Real-time WebSocket data feeds
- [ ] Production monitoring with Prometheus/Grafana

### Medium-Term (Phases 23-26)
- [ ] Options strategies support
- [ ] Pairs trading / statistical arbitrage
- [ ] Sector rotation strategies
- [ ] Tax-loss harvesting integration

### Long-Term (Phases 27+)
- [ ] Multi-asset support (crypto, forex)
- [ ] Distributed execution across brokers
- [ ] Institutional-grade compliance framework
- [ ] Cloud deployment with auto-scaling

---

## Quick Start Guide

### 1. Install Dependencies
```bash
cd /path/to/stock_trading
python -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
```

### 2. Configure
Edit `config/trading_config.yaml` for your preferences.
Set environment variables in `.env`:
```
DISCORD_WEBHOOK_URL=your_webhook
WEBULL_EMAIL=your_email  # for paper/live trading
WEBULL_PASSWORD=your_password
```

### 3. Train Models (if needed)
```bash
python scripts/train_model.py --model xgboost
python scripts/train_lstm.py
python scripts/train_cnn.py
```

### 4. Run Backtest
```bash
python scripts/run_backtest.py --walk-forward --regime-aware
```

### 5. Start Trading Bot
```bash
# Simulated mode (no real money)
python scripts/start_trading.py --simulated --interval 60 --ensemble

# Paper trading (Webull paper account)
python scripts/start_trading.py --paper --interval 60 --ensemble
```

### 6. Start Dashboard
```bash
streamlit run src/dashboard/app.py --server.headless true
# Access at http://localhost:8501
```

### 7. Manual Retraining
```bash
python scripts/retrain_models.py --models all --deploy --walk-forward-validate
```

### 8. Check Model Health
```bash
python scripts/retrain_models.py --check-degradation
python scripts/retrain_models.py --monitoring-status
```

### 9. Test Portfolio Optimization (Phases 19-20)
```bash
# Manual test suite demonstrating all optimization features
python scripts/test_portfolio_optimization.py

# Unit tests (34 tests)
pytest tests/test_portfolio_optimizer.py -v

# Integration tests (Phase 20)
python scripts/test_phase20_integration.py
```

### 10. View Portfolio Optimization
```bash
# Start dashboard and navigate to "Portfolio Optimization" tab
streamlit run src/dashboard/app.py --server.headless true
# Navigate to: http://localhost:8501 → Portfolio Optimization
```

---

## Project Structure

```
stock_trading/
├── config/
│   ├── settings.py           # Environment and logging setup
│   └── trading_config.yaml   # Main configuration file
├── data/                     # Cached data and broker state
├── models/
│   ├── versions/             # Model version history
│   │   └── registry.json     # Production model registry
│   ├── trading_model.pkl     # Production XGBoost
│   ├── lstm_trading_model/   # Production LSTM
│   └── cnn_trading_model/    # Production CNN
├── scripts/
│   ├── start_trading.py      # Main entry point
│   ├── run_backtest.py       # Backtesting (with --optimize flag)
│   ├── retrain_models.py     # Retraining pipeline
│   ├── test_portfolio_optimization.py  # Portfolio optimization manual tests
│   ├── test_phase20_integration.py     # Phase 20 integration tests
│   └── train_*.py            # Individual model training
├── src/
│   ├── core/
│   │   └── trading_engine.py # Main orchestrator
│   ├── broker/
│   │   ├── simulated_broker.py
│   │   └── webull_broker.py
│   ├── strategy/
│   │   └── ml_strategy.py    # Signal generation
│   ├── risk/
│   │   ├── risk_manager.py   # Position sizing, limits
│   │   └── regime_detector.py # Market regime detection
│   ├── portfolio/            # Phase 19: Portfolio Optimization
│   │   ├── portfolio_optimizer.py    # Multi-method optimization
│   │   ├── efficient_frontier.py     # Mean-variance frontier
│   │   ├── correlation_analyzer.py   # Correlation & clustering
│   │   └── rebalancer.py             # Rebalancing triggers
│   ├── ml/
│   │   ├── models/           # XGBoost, LSTM, CNN, Ensemble
│   │   ├── training.py       # Model training
│   │   ├── retraining.py     # Retraining pipeline
│   │   ├── scheduled_retrainer.py  # APScheduler jobs
│   │   ├── degradation_monitor.py  # Model health checks
│   │   └── auto_rollback.py  # Rollback manager
│   ├── data/
│   │   ├── data_fetcher.py   # Yahoo Finance
│   │   ├── feature_engineer.py
│   │   ├── sentiment_fetcher.py
│   │   └── macro_fetcher.py
│   ├── backtest/
│   │   └── backtester.py
│   ├── dashboard/
│   │   └── app.py            # Streamlit dashboard
│   └── notifications/
│       ├── discord_notifier.py
│       └── telegram_notifier.py
└── tests/                    # Pytest test suite
    └── test_portfolio_optimizer.py   # 34 tests for portfolio module
```

---

## Notes for Continuation

When continuing development on another device:
1. Clone the repo and install dependencies (including `scipy`, `cvxpy`, and `pyarrow` for portfolio optimization)
2. Copy `.env` file with API keys (not in git)
3. Copy `data/simulated_broker_state.json` to preserve positions (optional)
4. Copy `models/` directory to preserve trained models (or retrain)
5. Run `python scripts/retrain_models.py --status` to verify model state
6. Start with `python scripts/start_trading.py --simulated` to test

**Phase 20 Complete:** Portfolio optimization is now fully integrated into the trading engine. The bot automatically:
- Calculates optimal portfolio weights using max Sharpe method
- Rebalances when drift exceeds 10% OR monthly
- Uses target weights for position sizing
- Logs optimization metrics

**Testing Phase 20:**
```bash
# Manual test of portfolio optimization module
python scripts/test_portfolio_optimization.py

# Unit tests (34 tests)
pytest tests/test_portfolio_optimizer.py -v

# Integration tests (Phase 20)
python scripts/test_phase20_integration.py
```

For Phase 21, recommended enhancements include:
- Full backtesting with portfolio optimization (walk-forward comparison)
- Advanced dashboard visualizations (efficient frontier, correlation heatmap)
- Regime-aware optimization (adapt to bull/bear/volatile markets)
- Rolling window analysis for validation

---

*Last updated: 2026-01-28*
*Current version: Phases 20-21 complete and validated*
*Phase 20: Portfolio Optimization Integration (validated)*
*Phase 21: Advanced visualizations, regime-aware optimization, transaction costs (validated)*
