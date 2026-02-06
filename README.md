# Stock Trading Bot

An automated stock trading system with ML-driven signal generation, multi-agent collaboration, portfolio optimization, and adaptive risk management. Targets US equities via WeBull with support for paper and live trading modes.

## Key Features

- **ML Ensemble Predictions** -- XGBoost, LSTM, CNN with soft/weighted voting
- **4-Agent Architecture** -- Market Intelligence, Risk Guardian, Portfolio Strategist, Operations (Claude-powered)
- **Market Regime Detection** -- Bull/Bear/Choppy/Volatile classification with regime-specific parameters
- **Portfolio Optimization** -- Max Sharpe, Risk Parity, Min Variance with regime-aware adaptation
- **Dynamic Symbol Selection** -- Automated screening from S&P 500 / NASDAQ 100 universes
- **Tiered Risk Management** -- VIX-based sizing, ATR stops, multi-level take-profit, drawdown protection
- **FinBERT Sentiment Analysis** -- News sentiment scoring via Finnhub + local NLP
- **Notifications** -- Telegram, Discord, and email reporting
- **Streamlit Dashboard** -- Real-time monitoring on port 8501

---

## Architecture

### Multi-Agent System

Four specialized agents collaborate via a message queue, orchestrated centrally and powered by Claude:

| Agent | Role | Key Tasks |
|---|---|---|
| **Market Intelligence** | Information gathering | News scanning, earnings checks, macro analysis, VIX spike alerts |
| **Risk Guardian** | Risk monitoring & protection | Drawdown monitoring, correlation checks, emergency position reduction |
| **Portfolio Strategist** | Allocation & selection | Performance review, rebalancing, stock screening, portfolio optimization |
| **Operations** | Execution & system health | Message processing, execution quality, model degradation detection, auto-retrain |

### ML Pipeline

```
Market Data --> Feature Engineering (12 technical indicators + sentiment)
                    |
                    v
              +-----------+
              | XGBoost   |  weight: 1.0
              | LSTM      |  weight: 0.8
              | CNN       |  weight: 0.8
              +-----------+
                    |
                    v
            Soft Voting Ensemble --> Signal (BUY/SELL/HOLD) + Confidence
                    |
                    v
            Regime Filter --> Risk Manager --> Order Execution
```

### Regime Detection

Market regime is classified using VIX levels and ADX trend strength, with per-regime parameters for stop-loss, position sizing, and minimum confidence thresholds:

| Regime | VIX Condition | Position Size Mult | Stop Loss | Min Confidence |
|---|---|---|---|---|
| Bull | ADX > 25, not volatile | 1.0x | 5% | 0.50 |
| Bear | ADX > 25, bearish | 0.7x | 3% | 0.55 |
| Choppy | ADX < 20 | 0.5x | 2% | 0.58 |
| Volatile | VIX > 30 | 0.5x | 8% | 0.55 |

---

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd stock-trading-bot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys (WeBull, Finnhub, Anthropic, etc.)

# Train models
python scripts/train_model.py          # XGBoost (primary)
python scripts/train_lstm.py           # LSTM
python scripts/train_cnn.py            # CNN

# Run (paper trading, simulated broker)
python scripts/start_trading.py --simulated --interval 120 --ensemble

# Run (paper trading, WeBull)
python scripts/start_trading.py --interval 60 --ensemble

# Run (live trading -- requires confirmation prompt)
python scripts/start_trading.py --live --interval 60 --ensemble
```

### Available Scripts

| Script | Purpose |
|---|---|
| `start_trading.py` | Main entry point. Flags: `--simulated`, `--live`, `--ensemble`, `--interval`, `--capital`, `--force` |
| `train_model.py` | Train XGBoost model |
| `train_lstm.py` | Train LSTM model |
| `train_cnn.py` | Train CNN model |
| `train_mlp.py` | Train MLP model |
| `run_backtest.py` | Run backtesting suite |
| `run_wf_optimize.py` | Walk-forward optimization |
| `retrain_models.py` | Retrain all models (scheduled or manual) |
| `check_performance.py` | Check current portfolio performance |
| `scheduled_reporter.py` | Run scheduled performance reports |
| `auto_trading.sh` | Shell wrapper for automated execution |
| `run_bot_daemon.sh` | Daemonized bot runner |

---

## Configuration

All trading parameters live in `config/trading_config.yaml`. Key sections:

### Risk Management

```yaml
risk_management:
  max_position_pct: 0.10          # 10% max per position
  max_positions: 20
  max_daily_loss_pct: 0.05        # 5% daily loss limit
  max_total_exposure: 0.80        # 80% max invested
  max_sector_exposure: 0.30       # 30% max per sector
  pause_after_consecutive_losses: 3
```

### Stop Loss & Take Profit

```yaml
  stop_loss:
    default_type: "atr"           # fixed, atr, or trailing
    atr_multiplier: 2.0

  take_profit:
    levels:
      - [0.05, 0.33]             # 5% gain  -> sell 33%
      - [0.10, 0.50]             # 10% gain -> sell 50% of remaining
      - [0.15, 1.0]              # 15% gain -> sell everything
```

### VIX-Based Position Sizing

```yaml
  volatility_sizing:
    vix_thresholds:
      low: 15                     # Calm
      normal: 25                  # Normal
      high: 35                    # Elevated
      extreme: 50                 # Extreme
    size_multipliers:
      low: 1.2
      normal: 1.0
      high: 0.7
      extreme: 0.5
```

### ML Model

```yaml
ml_model:
  primary_model: "xgboost"       # or "ensemble"
  prediction_horizon: 5
  confidence_threshold: 0.55
  ensemble:
    voting_method: "soft"
    weights: { xgboost: 1.0, lstm: 0.8, cnn: 0.8 }
```

### Portfolio Optimization

```yaml
portfolio_optimization:
  method: "max_sharpe"            # max_sharpe, risk_parity, minimum_variance, equal_weight
  regime_aware: true              # Adapt method per market regime
  min_weight: 0.05
  max_weight: 0.30
  rebalancing:
    trigger_type: "threshold"
    drift_threshold: 0.08
    frequency: "weekly"
```

---

## Deployment

### Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f trading-bot

# Stop
docker-compose down
```

Volumes mount `data/`, `logs/`, and `models/` for persistence. Environment variables are loaded from `.env`.

### macOS launchd

```bash
# Install the launch agent
cp deploy/com.tradingbot.plist ~/Library/LaunchAgents/

# Load (start on network availability, auto-restart)
launchctl load ~/Library/LaunchAgents/com.tradingbot.plist

# Start manually
launchctl start com.tradingbot

# Stop
launchctl stop com.tradingbot

# Unload
launchctl unload ~/Library/LaunchAgents/com.tradingbot.plist
```

Logs go to `logs/launchd_stdout.log` and `logs/launchd_stderr.log`. The agent restarts automatically when network is available (`KeepAlive.NetworkState`).

---

## Testing

```bash
# Run full test suite
pytest

# Verbose with short tracebacks (default via pytest.ini)
pytest -v --tb=short

# Skip slow tests
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Single test file
pytest tests/test_risk_manager.py
```

### Test Coverage

| Test File | Coverage Area |
|---|---|
| `test_trading_engine.py` | Core trading loop, order execution |
| `test_risk_manager.py` | Position limits, stop loss, drawdown protection |
| `test_strategy.py` | ML strategy signal generation |
| `test_models.py` | XGBoost, LSTM, CNN model training/prediction |
| `test_portfolio_optimizer.py` | Portfolio optimization, rebalancing |
| `test_regime_detector.py` | Market regime classification |
| `test_backtester.py` | Backtesting engine |
| `test_broker.py` | Broker abstraction, simulated/WeBull |
| `test_data_fetcher.py` | Market data retrieval |
| `test_feature_engineer.py` | Technical indicator computation |
| `test_analytics.py` | Performance metrics, attribution |
| `test_metrics.py` | Sharpe, drawdown, win rate calculations |
| `test_degradation_monitor.py` | Model degradation detection |
| `test_integration_smoke.py` | End-to-end smoke tests |

---

## Project Structure

```
stock-trading-bot/
├── config/
│   ├── settings.py               # Environment config, logging setup
│   └── trading_config.yaml       # All trading parameters
├── src/
│   ├── agents/                   # Multi-agent system
│   │   ├── orchestrator.py       # Agent coordination
│   │   ├── market_intelligence.py
│   │   ├── risk_guardian.py
│   │   ├── portfolio_strategist.py
│   │   ├── operations_agent.py
│   │   ├── llm_client.py         # Claude API integration
│   │   ├── message_queue.py      # Inter-agent messaging
│   │   └── agent_notifier.py     # Discord notifications for agents
│   ├── analytics/                # Performance analytics & reporting
│   ├── backtest/                 # Walk-forward backtesting engine
│   ├── broker/                   # Broker abstraction (WeBull, simulated)
│   ├── core/                     # Trading engine, symbol manager
│   ├── dashboard/                # Streamlit dashboard
│   ├── data/                     # Data fetchers (market, sentiment, macro, news)
│   ├── ml/                       # ML models & training
│   │   ├── models/               # XGBoost, LSTM, CNN, MLP, Transformer, Ensemble
│   │   ├── finbert_analyzer.py   # FinBERT sentiment NLP
│   │   ├── degradation_monitor.py
│   │   ├── auto_rollback.py
│   │   └── walk_forward_optimizer.py
│   ├── notifications/            # Telegram, Discord, email
│   ├── portfolio/                # Portfolio optimization & rebalancing
│   ├── reporting/                # Scheduled performance reports
│   ├── risk/                     # Risk manager, regime detector, market timing
│   ├── screening/                # Stock screening strategies
│   └── strategy/                 # ML trading strategy
├── scripts/                      # Entry points & utilities
├── tests/                        # pytest test suite
├── deploy/
│   └── com.tradingbot.plist      # macOS launchd agent
├── models/                       # Trained model artifacts
├── data/                         # Market data cache, reports
├── logs/                         # Application logs (rotating, 10MB x 5)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pytest.ini
```

---

## Environment Variables

Configure in `.env` (copy from `.env.example`):

| Variable | Required | Description |
|---|---|---|
| `TRADING_MODE` | No | `paper` (default) or `live` |
| `WEBULL_EMAIL` | Yes* | WeBull account email |
| `WEBULL_PASSWORD` | Yes* | WeBull account password |
| `WEBULL_TRADE_PIN` | Yes* | WeBull 6-digit trade PIN |
| `WEBULL_DEVICE_ID` | No | WeBull device identifier |
| `FINNHUB_API_KEY` | Yes | Finnhub API key (news/sentiment data) |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key (multi-agent system) |
| `ALPHA_VANTAGE_API_KEY` | No | Alpha Vantage API key (alternative data source) |
| `FRED_API_KEY` | No | FRED API key (macroeconomic data) |
| `TELEGRAM_BOT_TOKEN` | No | Telegram bot token for notifications |
| `TELEGRAM_CHAT_ID` | No | Telegram chat ID for notifications |
| `DISCORD_WEBHOOK_URL` | No | Discord webhook URL for notifications |
| `BLUESKY_API_TOKEN` | No | BlueSky/MT Newswires API token |
| `SMTP_SERVER` | No | SMTP server for email reports (default: `smtp.gmail.com`) |
| `SMTP_PORT` | No | SMTP port (default: `587`) |
| `SMTP_USE_TLS` | No | Enable TLS for SMTP (default: `true`) |
| `SENDER_EMAIL` | No | Email sender address |
| `SENDER_PASSWORD` | No | Email sender password (Gmail app password) |
| `RECIPIENT_EMAILS` | No | Comma-separated recipient emails |
| `LOG_LEVEL` | No | Logging level (default: `INFO`) |

\* WeBull credentials not required when running with `--simulated` flag.
