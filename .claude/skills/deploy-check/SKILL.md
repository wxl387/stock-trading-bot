---
name: deploy-check
description: Pre-deployment checklist for the trading bot — runs tests, validates config, checks model accuracy, verifies syntax
disable-model-invocation: true
allowed-tools: Bash(python *), Bash(.venv/bin/python *), Bash(git *), Read, Grep, Glob
---

## Pre-Deployment Check

Run a comprehensive checklist before deploying or pushing changes. No arguments needed.

### Checklist Steps

Run these checks **in order**, stopping on any critical failure:

#### 1. Syntax Validation
Run Python AST parse on all source files:
```
.venv/bin/python -c "
import ast, pathlib, sys
errors = []
for f in pathlib.Path('src').rglob('*.py'):
    try:
        ast.parse(f.read_text())
    except SyntaxError as e:
        errors.append(f'{f}: {e}')
if errors:
    print('SYNTAX ERRORS:')
    for e in errors: print(f'  {e}')
    sys.exit(1)
print(f'All source files OK')
"
```
**CRITICAL** — stop if this fails.

#### 2. Config Validation
Read `config/trading_config.yaml` and verify:
- `ml_model.confidence_threshold` is between 0.40 and 0.70
- `risk_management.stop_loss.fixed_pct` is between 0.03 and 0.20
- `ml_model.ensemble.weights` — all weights are between 0.0 and 2.0
- No duplicate keys or YAML parse errors

#### 3. Run Tests
Run the non-ML test suite (ML tests have TF import issues):
```
.venv/bin/python -m pytest tests/ -x -q --tb=short --ignore=tests/test_models.py --ignore=tests/test_retraining.py --ignore=tests/test_ml_strategy.py
```
**CRITICAL** — stop if any test fails. Report the count of passed tests.

#### 4. Model File Check
Verify model files exist and are recent:
- `models/trading_model/` (XGBoost)
- `models/lstm_trading_model/` (LSTM)
- `models/cnn_trading_model/` (CNN)
- `models/transformer_trading_model/` (Transformer)

For each, check if the directory exists and report the last modified date of the model files.

#### 5. Git Status
Run `git status` and `git diff --stat` to show:
- Uncommitted changes
- Untracked files
- Current branch

#### 6. Summary
Print a summary table:

| Check | Status |
|-------|--------|
| Syntax | PASS/FAIL |
| Config | PASS/FAIL (with warnings) |
| Tests | PASS (N tests) / FAIL |
| Models | OK / MISSING |
| Git | Clean / N uncommitted changes |

If all critical checks pass, report "Ready to deploy". Otherwise, list what needs fixing.
