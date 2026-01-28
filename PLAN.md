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

---

## Current State

### What Works End-to-End
1. **Simulated Trading**: Fetch data → Generate signals → Execute trades → Monitor dashboard
2. **Backtesting**: Walk-forward validation with regime detection and risk management
3. **Model Training**: XGBoost, LSTM, CNN individually or as ensemble
4. **Scheduled Retraining**: Auto-deployment with degradation detection (Phase 18)
5. **Risk Management**: Stop-loss, trailing stops, position sizing, drawdown protection
6. **Notifications**: Trade alerts via Discord

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

---

## Immediate Next Steps (Phase 19 Options)

### Option 1: Portfolio Optimization Module (Recommended)
- Mean-variance optimization (efficient frontier)
- Risk-parity and minimum-variance allocation
- Correlation-based diversification constraints
- Rebalancing triggers (threshold/calendar-based)
- **Why**: Currently equal allocation; portfolio optimization would significantly improve risk-adjusted returns

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
- [ ] Portfolio optimization with efficient frontier
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
│   ├── run_backtest.py       # Backtesting
│   ├── retrain_models.py     # Retraining pipeline
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
```

---

## Notes for Continuation

When continuing development on another device:
1. Clone the repo and install dependencies
2. Copy `.env` file with API keys (not in git)
3. Copy `data/simulated_broker_state.json` to preserve positions (optional)
4. Copy `models/` directory to preserve trained models (or retrain)
5. Run `python scripts/retrain_models.py --status` to verify model state
6. Start with `python scripts/start_trading.py --simulated` to test

For Phase 19, the recommended path is **Portfolio Optimization** - it provides the highest impact improvement to risk-adjusted returns without requiring external dependencies or real-time infrastructure.

---

*Last updated: 2026-01-23*
*Current version: Phase 18 complete*
