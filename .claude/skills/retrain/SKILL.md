---
name: retrain
description: Retrain ML trading models (xgboost, lstm, cnn, transformer, or all) with proper TF import ordering
disable-model-invocation: true
argument-hint: "[model: all|xgboost|lstm|cnn|transformer] [options: tune, deploy]"
allowed-tools: Bash(python *), Bash(.venv/bin/python *), Read, Grep, Glob
---

## Model Retraining

Retrain ML models using `scripts/retrain_models.py`. Parse the arguments from `$ARGUMENTS`.

### Argument Mapping

| User says | CLI flags |
|-----------|-----------|
| `all` (or no model specified) | `--models all` |
| `xgboost` or `xgb` | `--models xgboost` |
| `lstm` | `--models lstm` |
| `cnn` | `--models cnn` |
| `transformer` | `--models transformer` |
| `tune` | `--tune` |
| `deploy` | `--deploy` |
| `force-deploy` | `--force-deploy` |
| `epochs=N` | `--epochs N` |
| `trials=N` | `--tune-trials N` |
| `status` | `--status` |
| `versions` | `--list-versions` |

For individual model tuning: `tune-xgboost`, `tune-lstm`, `tune-cnn`, `tune-transformer` map to their respective flags.

### Steps

1. Parse arguments and build the CLI command
2. Show the user the exact command being run
3. **CRITICAL**: Run with `.venv/bin/python` only (system Python 3.9.6 causes TF mutex deadlocks)
4. Run: `.venv/bin/python scripts/retrain_models.py <flags>`
   - Timeout: 600 seconds for single model, 1800 seconds for all models or tuning
5. After completion, summarize:
   - Training accuracy and test accuracy for each model
   - Whether the model improved over production
   - Whether deployment happened (if --deploy was used)
6. If the user asked to retrain the transformer specifically, remind them that it currently has 49.8% accuracy and its ensemble weight is 0.3 (lowered to avoid diluting signal)

### Important Notes

- **TF import order bug**: pandas before TF causes model.fit() deadlock on macOS. The scripts already handle this, but always use `.venv/bin/python`.
- XGBoost requires `libomp` (already installed via Homebrew)
- LSTM/CNN training uses MPS GPU acceleration on Apple M4
- Transformer is undertrained (49.8% accuracy) — needs more data/tuning before increasing its ensemble weight
- Default deployment threshold is 1% accuracy improvement
