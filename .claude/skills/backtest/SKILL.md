---
name: backtest
description: Run a walk-forward backtest of the ML trading strategy with configurable parameters
disable-model-invocation: true
argument-hint: "[options: walk-forward, ensemble, stops, regime, quick]"
allowed-tools: Bash(python *), Bash(.venv/bin/python *), Read, Grep, Glob
---

## Walk-Forward Backtest

Run a backtest using `scripts/run_backtest.py`. Parse the arguments from `$ARGUMENTS` and map them to CLI flags.

### Argument Mapping

| User says | CLI flags |
|-----------|-----------|
| `walk-forward` or `wf` | `--walk-forward` |
| `ensemble` | `--ensemble` |
| `stops` | `--stop-loss --trailing-stop` |
| `regime` | `--use-regime` |
| `quick` | `--period 6mo --test-period 42` |
| `optimize` | `--optimize` |
| `kelly` | `--kelly` |
| `circuit-breaker` or `cb` | `--circuit-breaker` |
| `conf=X.XX` | `--confidence X.XX` |
| `symbols=AAPL,MSFT` | `--symbols AAPL,MSFT` |
| `capital=50000` | `--capital 50000` |
| `no-chart` | `--no-chart` |

If no arguments given, use the recommended defaults: `--walk-forward --ensemble --confidence 0.55`

### Steps

1. Build the CLI command from the parsed arguments
2. Show the user the exact command being run
3. Run: `.venv/bin/python scripts/run_backtest.py <flags>`
   - Use a timeout of 600 seconds (backtests can be slow)
   - The script outputs results to stdout
4. After completion, read `data/backtest_results.png` if it exists and show the equity curve
5. Summarize key metrics: total return, Sharpe ratio, win rate, number of trades, max drawdown, alpha vs SPY

### Important Notes

- Always use `.venv/bin/python` (system Python causes TF deadlocks)
- Walk-forward backtests retrain models in each window and take 5-15 minutes
- The default confidence threshold is 0.55 (optimal from previous analysis)
- Current best result: +76% return, 0.81 Sharpe, 63% win rate (walk-forward, no stops, 0.55 conf)
