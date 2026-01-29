# Automated Performance Reporting - Setup Complete ✅

**Date**: 2026-01-28
**Status**: Ready to use

---

## 📋 What Was Set Up

### 1. Performance Report Generator
**File**: `src/reporting/performance_reporter.py`

Comprehensive report generation engine with:
- Portfolio value calculations with current prices
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Trade statistics (realized/unrealized P&L, profit factor)
- Benchmark comparison vs SPY (S&P 500)
- Risk assessment and recommendations
- Text and HTML report formats

### 2. Email Notification System
**File**: `src/notifications/email_notifier.py`

Email delivery system with:
- SMTP configuration (Gmail, Outlook, Yahoo, custom)
- Plain text and HTML email support
- File attachment support
- Trade notifications
- Alert notifications
- Performance report delivery

### 3. Scheduled Reporter Daemon
**File**: `scripts/scheduled_reporter.py`

Automated scheduling with:
- Daily, weekly, and monthly report schedules
- Configurable report times
- Email and Discord delivery
- Save reports to file
- Test mode for immediate reports

### 4. Manual Performance Checker
**File**: `scripts/check_performance.py`

Quick performance check tool with:
- Portfolio value and returns
- Current positions with P&L
- Trade statistics
- Benchmark comparison
- Console-friendly output

### 5. Configuration
**Files**: `config/trading_config.yaml`, `.env.example`

Configuration added for:
- Report schedules (daily/weekly/monthly)
- Delivery methods (email/Discord/file)
- Email SMTP settings
- Report timing preferences

---

## 🚀 Quick Start Guide

### Step 1: Configure Email (5 minutes)

1. **For Gmail users** (recommended):
   ```bash
   # 1. Enable 2FA at: https://myaccount.google.com/security
   # 2. Create App Password at: https://myaccount.google.com/apppasswords
   # 3. Copy the 16-character password
   ```

2. **Update .env file**:
   ```bash
   # Edit .env file (or create from .env.example)
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USE_TLS=true
   SENDER_EMAIL=your_email@gmail.com
   SENDER_PASSWORD=your_16_char_app_password
   RECIPIENT_EMAILS=recipient1@example.com,recipient2@example.com
   ```

### Step 2: Test Email Setup (2 minutes)

```bash
# Test email configuration
python << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.notifications.email_notifier import EmailNotifier

notifier = EmailNotifier()
if notifier.enabled:
    print("✅ Configured:", notifier.sender_email)
    notifier.send_alert("Test", "Email setup successful!", "info")
    print("✅ Check your inbox!")
else:
    print("❌ Check .env file settings")
EOF
```

### Step 3: Generate Test Report (1 minute)

```bash
# Generate and send a test daily report
python scripts/scheduled_reporter.py --test daily

# This will:
# ✅ Generate performance report
# ✅ Send via email (if configured)
# ✅ Send via Discord (if configured)
# ✅ Save to data/reports/ directory
```

### Step 4: Start Automated Reports (30 seconds)

```bash
# Run scheduled reporter in foreground (see logs)
python scripts/scheduled_reporter.py --daemon

# OR run in background (Linux/Mac)
nohup python scripts/scheduled_reporter.py --daemon > logs/reporter.log 2>&1 &

# Check it's running
ps aux | grep scheduled_reporter
```

---

## 📊 Report Schedules

### Default Schedule (Configured)

| Report Type | Frequency | Time | Enabled |
|-------------|-----------|------|---------|
| **Daily** | Every day | 5:00 PM | ✅ Yes |
| **Weekly** | Monday | 9:00 AM | ✅ Yes |
| **Monthly** | 1st of month | 9:00 AM | ❌ No |

### Customize Schedule

Edit `config/trading_config.yaml`:

```yaml
reporting:
  daily:
    enabled: true
    time: "17:00"        # Change time here

  weekly:
    enabled: true
    day: "monday"        # Change day here
    time: "09:00"        # Change time here

  monthly:
    enabled: false       # Enable monthly reports
    day: 1
    time: "09:00"
```

---

## 💡 Usage Examples

### Example 1: Quick Performance Check

```bash
# Console-only performance summary (no email)
python scripts/check_performance.py
```

**Output**:
```
======================================================================
                    TRADING BOT PERFORMANCE
======================================================================

📅 Report Date: 2026-01-28 15:30:00
💼 Initial Capital: $100,000.00
💰 Current Cash: $45,234.56
📊 Position Value: $58,765.44
💵 Total Portfolio Value: $104,000.00

----------------------------------------------------------------------
RETURNS
----------------------------------------------------------------------
Total Return:                    $4,000.00
Return %:                            4.00%
SPY (S&P 500) Return:                2.80%
Outperformance vs SPY:               1.20%

----------------------------------------------------------------------
POSITIONS
----------------------------------------------------------------------
Symbol     Shares      Entry     Current         Value        P&L
----------------------------------------------------------------------
AAPL          50     $175.50     $182.30     $9,115.00   $340.00
MSFT          30     $380.25     $385.70    $11,571.00   $163.50
...
```

### Example 2: Test Report Before Scheduling

```bash
# Test daily report (immediate)
python scripts/scheduled_reporter.py --test daily

# Test weekly report
python scripts/scheduled_reporter.py --test weekly

# Test monthly report
python scripts/scheduled_reporter.py --test monthly
```

### Example 3: Run Scheduled Reports

```bash
# Start in foreground (see all logs)
python scripts/scheduled_reporter.py --daemon

# Start in background (Linux/Mac)
nohup python scripts/scheduled_reporter.py --daemon > logs/reporter.log 2>&1 &

# View background logs
tail -f logs/reporter.log

# Stop background process
pkill -f scheduled_reporter.py
```

### Example 4: Manual Report Generation

```python
# Python script for custom reports
from pathlib import Path
from src.reporting import PerformanceReporter
from config.settings import DATA_DIR

# Create reporter
reporter = PerformanceReporter(DATA_DIR)

# Generate text report
text_report = reporter.generate_text_report('daily')
print(text_report)

# Generate HTML report
html_report = reporter.generate_html_report('daily')

# Save reports
reporter.save_report(text_report, 'my_report', format='txt')
reporter.save_report(html_report, 'my_report', format='html')
```

---

## 📁 File Locations

### Reports Saved To:
```
data/reports/
  ├── daily_report_20260128_170000.txt
  ├── daily_report_20260128_170000.html
  ├── weekly_report_20260127_090000.txt
  └── weekly_report_20260127_090000.html
```

### Configuration Files:
```
config/trading_config.yaml     # Report schedule configuration
.env                            # Email credentials (gitignored)
.env.example                    # Template for .env
```

### Source Code:
```
src/reporting/
  ├── __init__.py
  └── performance_reporter.py   # Report generation

src/notifications/
  ├── email_notifier.py         # Email delivery
  └── discord_notifier.py       # Discord delivery (existing)

scripts/
  ├── scheduled_reporter.py     # Scheduled daemon
  └── check_performance.py      # Manual checker
```

---

## ⚙️ Configuration Reference

### Email Settings (.env)

```bash
# Gmail (recommended)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECIPIENT_EMAILS=email1@example.com,email2@example.com

# Outlook
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587

# Yahoo
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587

# Custom SMTP
SMTP_SERVER=your.smtp.server
SMTP_PORT=587
```

### Report Settings (trading_config.yaml)

```yaml
reporting:
  # Daily reports
  daily:
    enabled: true                   # Enable/disable
    time: "17:00"                   # 24-hour format

  # Weekly reports
  weekly:
    enabled: true
    day: "monday"                   # Day name
    time: "09:00"

  # Monthly reports
  monthly:
    enabled: false
    day: 1                          # Day of month (1-31)
    time: "09:00"

  # Delivery methods
  send_email: true                  # Email delivery
  send_discord: true                # Discord delivery
  save_to_file: true                # Save to file
```

---

## 🔍 What's Included in Reports

### Portfolio Summary
- Initial capital
- Current cash balance
- Position value
- Total portfolio value
- Total return ($ and %)

### Performance Metrics
- Sharpe ratio (risk-adjusted returns)
- Max drawdown (largest decline)
- Win rate (% profitable trades)
- Profit factor (gross profit / gross loss)
- Realized P&L

### Benchmark Comparison
- SPY (S&P 500) return
- Outperformance vs benchmark
- Relative performance

### Trading Activity
- Total trades executed
- Closed positions
- Winning vs losing trades
- Average win/loss
- Best/worst trades

### Current Positions
- Symbol, shares, entry price
- Current price and value
- Unrealized P&L per position
- Total unrealized P&L

### Risk Assessment
- Risk level (LOW/MEDIUM/HIGH)
- Identified risk factors
- Warnings and alerts

### Recommendations
- Actionable suggestions
- Performance improvements
- Risk mitigation advice

---

## 🐛 Troubleshooting

### "Email notifier not configured"

**Fix**: Update `.env` with email settings:
```bash
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECIPIENT_EMAILS=recipient@example.com
```

### "Authentication failed"

**Fix**:
- Gmail: Use App Password, not regular password
- Enable 2FA first
- Generate app password at: https://myaccount.google.com/apppasswords

### "No broker state found"

**Fix**:
- Run trading bot first: `python scripts/start_trading.py --simulated`
- Check `data/simulated_broker_state.json` exists

### Reports not being sent

**Fix**:
```bash
# Check scheduled reporter is running
ps aux | grep scheduled_reporter

# View logs
tail -f logs/reporter.log

# Test email manually
python scripts/scheduled_reporter.py --test daily
```

---

## 📈 Sample Report Email

**Subject**: 📊 Daily Trading Bot Performance Report

**Body** (condensed):
```
======================================================================
           DAILY TRADING BOT PERFORMANCE REPORT
======================================================================

Portfolio Summary:
💼 Initial Capital:        $100,000.00
💵 Total Portfolio Value:  $104,000.00
📈 Total Return:           $4,000.00 (4.00%)

Performance Metrics:
📊 Sharpe Ratio:           1.25
📉 Max Drawdown:           5.20%
🎯 Win Rate:               62.5%
💰 Profit Factor:          1.85

Benchmark Comparison:
📈 SPY Return:             2.80%
🟢 Outperformance:         1.20%

Current Positions:
AAPL   50 shares @ $175.50  →  $182.30  =  +$340.00
MSFT   30 shares @ $380.25  →  $385.70  =  +$163.50
...

Risk Assessment: LOW
✅ No significant risk factors identified

Recommendations:
1. ✅ System performing well - continue monitoring
======================================================================
```

---

## 🎯 Best Practices

1. **Test First**
   - Always run `--test daily` before scheduling
   - Verify email delivery and formatting
   - Check spam folder initially

2. **Monitor Delivery**
   - Watch first few reports
   - Add sender to contacts
   - Whitelist email address

3. **Adjust Schedule**
   - Daily for active trading
   - Weekly for long-term strategies
   - Monthly for passive monitoring

4. **Review Regularly**
   - Focus on Sharpe, win rate, drawdown
   - Compare to benchmark
   - Act on recommendations

5. **Keep Archives**
   - Reports auto-saved to `data/reports/`
   - Archive important reports
   - Track long-term trends

---

## ✅ Verification Checklist

- [ ] Email configured in `.env`
- [ ] Test email sent successfully
- [ ] Test report generated
- [ ] Report schedule configured
- [ ] Scheduled reporter running
- [ ] First report received
- [ ] Reports readable and formatted correctly
- [ ] All metrics displaying correctly
- [ ] Satisfied with report content

---

## 📚 Documentation

**Setup Guide**: `AUTOMATED_REPORTING_SETUP.md`
- Detailed configuration instructions
- Email provider setup (Gmail, Outlook, Yahoo)
- Troubleshooting guide
- Advanced customization

**Quick Reference**: This file (`AUTOMATED_REPORTING_COMPLETE.md`)
- Quick start guide
- Usage examples
- Configuration reference

**API Documentation**:
- `src/reporting/performance_reporter.py` - Report generation
- `src/notifications/email_notifier.py` - Email delivery
- `scripts/scheduled_reporter.py` - Scheduling

---

## 🚀 Next Steps

### Immediate
1. Configure email in `.env`
2. Test: `python scripts/scheduled_reporter.py --test daily`
3. Start: `python scripts/scheduled_reporter.py --daemon`

### Optional Enhancements
- Add custom metrics to reports
- Create custom report templates
- Add Plotly charts to HTML reports
- Integrate with Slack/Teams
- Add SMS notifications
- Create PDF reports
- Add historical performance charts

---

## 💬 Support

**Issues**:
- Check `AUTOMATED_REPORTING_SETUP.md` for detailed troubleshooting
- Review logs: `tail -f logs/reporter.log`
- Test email: `python scripts/scheduled_reporter.py --test daily`

**Customization**:
- Edit `src/reporting/performance_reporter.py` for custom metrics
- Edit `scripts/scheduled_reporter.py` for custom schedules
- Edit `config/trading_config.yaml` for schedule changes

---

*Automated Reporting Setup Complete - 2026-01-28*
*All systems ready for deployment* ✅
