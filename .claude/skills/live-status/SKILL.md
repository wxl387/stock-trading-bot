---
name: live-status
description: Query the trading bot's current status — positions, P&L, stop-losses, agent health, message queue depth
disable-model-invocation: true
allowed-tools: Bash(python *), Bash(.venv/bin/python *), Read, Grep, Glob
---

## Live Status Check

Query the running bot's state from local files and report a dashboard summary. Parse `$ARGUMENTS` for optional focus area.

### Argument Mapping

| User says | Focus |
|-----------|-------|
| (empty) | Full status report |
| `positions` or `pos` | Positions and P&L only |
| `risk` | Risk state and stop-losses |
| `agents` | Agent health and message queue |
| `trades` | Recent trade history |
| `performance` or `perf` | Run `scripts/check_performance.py` |

### Data Sources

Read these files to build the status report:

1. **Broker state**: `data/simulated_broker_state.json`
   - `cash`, `positions`, `trades`/`trade_history`, `initial_capital`

2. **Risk manager state**: `data/risk_manager_state.json`
   - `stop_losses`, `take_profits`, `daily_pnl`, `trading_paused`

3. **Agent messages DB**: `data/agent_messages.db`
   - Query total/unprocessed message counts:
   ```
   .venv/bin/python -c "
   import sqlite3
   conn = sqlite3.connect('data/agent_messages.db')
   total = conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
   unproc = conn.execute('SELECT COUNT(*) FROM messages WHERE processed=0').fetchone()[0]
   print(f'Messages: {total} total, {unproc} unprocessed')
   for row in conn.execute('SELECT recipient, COUNT(*) FROM messages WHERE processed=0 GROUP BY recipient'):
       print(f'  {row[0]}: {row[1]} unprocessed')
   conn.close()
   "
   ```

4. **Log file**: `logs/trading.log`
   - Read last 20 lines for recent activity and errors

5. **Config**: `config/trading_config.yaml`
   - Current mode (paper/live), confidence threshold, stop-loss settings

### Report Format

```
TRADING BOT STATUS
==================
Mode: PAPER | Confidence: 0.55 | Stop-loss: ATR 4x

PORTFOLIO
  Cash: $XX,XXX
  Positions: N open
  Total Value: $XX,XXX
  P&L: +$X,XXX (+X.X%)

POSITIONS
  SYMBOL  Shares  Entry    Current  P&L
  AAPL    10      $180.00  $185.00  +$50.00

STOP-LOSSES
  SYMBOL  Type     Trigger
  AAPL    ATR 4x   $165.00

RISK STATE
  Daily P&L: -$XXX
  Trading Paused: No
  Drawdown: X.X%

AGENT HEALTH
  Messages: XX total, X unprocessed
  Last activity: [from log]

RECENT ACTIVITY
  [last 5 log entries]
```

### Steps

1. Check if data files exist. If `data/simulated_broker_state.json` doesn't exist, report "Bot has not been started yet"
2. Read the relevant data files based on the focus argument
3. For `performance` focus, run: `.venv/bin/python scripts/check_performance.py`
4. Format and display the report
5. Flag any warnings: trading paused, high drawdown, many unprocessed messages, stale log (no activity in >1 hour)
