# Automated Performance Reporting - Setup Guide

Automated performance reports for your trading bot delivered via email and Discord on a schedule.

---

## 📋 Features

### Report Types
- **Daily Reports** - End-of-day performance summary (default: 5:00 PM)
- **Weekly Reports** - Weekly performance review (default: Monday 9:00 AM)
- **Monthly Reports** - Monthly performance analysis (default: 1st of month, 9:00 AM)

### Report Content
- 💼 Portfolio value and returns
- 📊 Performance metrics (Sharpe ratio, max drawdown, win rate)
- 📈 Benchmark comparison vs SPY (S&P 500)
- 🎯 Trading statistics (realized/unrealized P&L, profit factor)
- 📋 Current positions with P&L
- ⚠️  Risk assessment
- 💡 Recommendations

### Delivery Methods
- ✉️  **Email** - Full reports with text and HTML formatting
- 💬 **Discord** - Condensed reports via webhook
- 💾 **File** - Saved to `data/reports/` directory

---

## 🚀 Quick Start

### 1. Run a Test Report

Generate a test report immediately to see the output:

```bash
# Generate a test daily report (sent via email and Discord)
python scripts/scheduled_reporter.py --test daily

# Generate a test weekly report
python scripts/scheduled_reporter.py --test weekly

# Or use the manual performance check script
python scripts/check_performance.py
```

### 2. Start Automated Reporting

Run as a background daemon with scheduled reports:

```bash
# Run in foreground (see logs)
python scripts/scheduled_reporter.py --daemon

# Run in background (Linux/Mac)
nohup python scripts/scheduled_reporter.py --daemon > logs/reporter.log 2>&1 &

# Check if running
ps aux | grep scheduled_reporter
```

### 3. View Reports

Reports are saved to:
```
data/reports/
  ├── daily_report_20260128_170000.txt
  ├── daily_report_20260128_170000.html
  ├── weekly_report_20260127_090000.txt
  └── weekly_report_20260127_090000.html
```

---

## ⚙️ Configuration

### 1. Email Setup (Required for Email Reports)

#### For Gmail Users:

1. **Enable 2-Factor Authentication**
   - Go to: https://myaccount.google.com/security
   - Enable 2-Step Verification

2. **Create App Password**
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and your device
   - Copy the 16-character password

3. **Update .env File**

Create or edit `.env` file in project root:

```bash
# Email configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_16_char_app_password
RECIPIENT_EMAILS=recipient1@example.com,recipient2@example.com
```

#### For Other Email Providers:

**Outlook/Hotmail:**
```bash
SMTP_SERVER=smtp-mail.outlook.com
SMTP_PORT=587
```

**Yahoo:**
```bash
SMTP_SERVER=smtp.mail.yahoo.com
SMTP_PORT=587
```

**Custom SMTP:**
```bash
SMTP_SERVER=your.smtp.server
SMTP_PORT=587
SMTP_USE_TLS=true
```

### 2. Discord Setup (Optional)

Already configured if you have Discord notifications enabled:

```bash
# In .env file
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_url
```

### 3. Report Schedule Configuration

Edit `config/trading_config.yaml`:

```yaml
reporting:
  # Daily reports
  daily:
    enabled: true                   # Enable/disable daily reports
    time: "17:00"                   # 5:00 PM (after market close)

  # Weekly reports
  weekly:
    enabled: true                   # Enable/disable weekly reports
    day: "monday"                   # Day of week
    time: "09:00"                   # 9:00 AM

  # Monthly reports
  monthly:
    enabled: false                  # Enable/disable monthly reports
    day: 1                          # 1st of month
    time: "09:00"                   # 9:00 AM

  # Delivery methods
  send_email: true                  # Send via email
  send_discord: true                # Send via Discord
  save_to_file: true                # Save to file
```

---

## 🧪 Testing Email Configuration

Test your email setup:

```bash
# Test email notifier directly
python << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.notifications.email_notifier import EmailNotifier

notifier = EmailNotifier()

if notifier.enabled:
    print("✅ Email notifier configured")
    print(f"Sender: {notifier.sender_email}")
    print(f"Recipients: {', '.join(notifier.recipient_emails)}")

    # Send test email
    result = notifier.send_alert(
        title="Test Email",
        message="This is a test email from your trading bot.",
        severity="info"
    )

    if result:
        print("✅ Test email sent successfully!")
    else:
        print("❌ Failed to send test email")
else:
    print("❌ Email notifier not configured")
    print("Check your .env file settings")
EOF
```

---

## 📊 Sample Report Output

### Daily Report Example:

```
======================================================================
           DAILY TRADING BOT PERFORMANCE REPORT
======================================================================

📅 Report Date: 2026-01-28 17:00:00
📊 Report Type: Daily

----------------------------------------------------------------------
PORTFOLIO SUMMARY
----------------------------------------------------------------------
💼 Initial Capital:        $  100,000.00
💰 Current Cash:           $   45,234.56
📊 Position Value:         $   58,765.44
💵 Total Portfolio Value:  $  104,000.00
📈 Total Return:           $    4,000.00 (  4.00%)

----------------------------------------------------------------------
PERFORMANCE METRICS
----------------------------------------------------------------------
📊 Sharpe Ratio:                     1.25
📉 Max Drawdown:                    5.20% ($5,200.00)
🎯 Win Rate:                       62.5%
💰 Profit Factor:                   1.85
💵 Realized P&L:           $    2,500.00

----------------------------------------------------------------------
BENCHMARK COMPARISON (SPY)
----------------------------------------------------------------------
📈 SPY Return:                      2.80%
🟢 Outperformance:                  1.20%

----------------------------------------------------------------------
TRADING ACTIVITY
----------------------------------------------------------------------
📊 Total Trades:                      32
✅ Closed Positions:                  16
🟢 Winning Trades:                    10
🔴 Losing Trades:                      6
💰 Average Win:            $    450.00
💸 Average Loss:           $    250.00
🏆 Best Trade:             $    850.00
💔 Worst Trade:            $   -420.00

----------------------------------------------------------------------
CURRENT POSITIONS
----------------------------------------------------------------------
Symbol     Shares        Entry      Current          Value        P&L
----------------------------------------------------------------------
AAPL          50       $175.50      $182.30      $9,115.00  🟢$   340.00
MSFT          30       $380.25      $385.70     $11,571.00  🟢$   163.50
GOOGL         15       $142.80      $145.20      $2,178.00  🟢$    36.00
NVDA          45       $520.00      $515.30     $23,188.50  🔴$  -211.50
...
----------------------------------------------------------------------
Total Unrealized P&L:                             $    1,500.00

----------------------------------------------------------------------
RISK ASSESSMENT
----------------------------------------------------------------------
Risk Level: LOW
✅ No significant risk factors identified

----------------------------------------------------------------------
RECOMMENDATIONS
----------------------------------------------------------------------
1. ✅ System performing well - continue monitoring
======================================================================
```

---

## 🔧 Advanced Configuration

### Custom Report Schedules

You can add custom schedules using cron expressions. Edit `scripts/scheduled_reporter.py`:

```python
# Example: Report every 4 hours during market hours
scheduler.add_job(
    self.custom_report,
    CronTrigger(hour='9-16/4'),  # 9 AM, 1 PM
    id='custom_report',
    name='Custom Report'
)
```

### Email with Attachments

Modify `scripts/scheduled_reporter.py` to attach charts or CSV files:

```python
# Save report as file first
filepath = self.reporter.save_report(text_report, filename, format='txt')

# Send with attachment
self.email_notifier.send_email(
    subject="Performance Report",
    body=text_report,
    attachments=[filepath]
)
```

### Custom Report Format

Create custom report templates by modifying `src/reporting/performance_reporter.py`:

```python
def generate_custom_report(self):
    """Generate a custom report format."""
    state = self.load_broker_state()

    # Your custom logic here
    report = []
    report.append("Custom Report Header")
    # ... add your custom metrics

    return "\n".join(report)
```

---

## 🐛 Troubleshooting

### Email Not Sending

**Issue**: "Email notifier disabled: Missing configuration"

**Solution**: Check `.env` file has all required email settings:
```bash
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password
RECIPIENT_EMAILS=recipient@example.com
```

**Issue**: "Authentication failed"

**Solutions**:
- For Gmail: Use App Password, not regular password
- Enable "Less secure app access" (not recommended)
- Check 2FA is enabled
- Verify password is correct

**Issue**: "Connection refused"

**Solutions**:
- Check SMTP server address
- Verify port number (587 for TLS, 465 for SSL)
- Check firewall settings

### No Reports Being Generated

**Issue**: Scheduled reporter not running

**Solutions**:
```bash
# Check if process is running
ps aux | grep scheduled_reporter

# Check logs
tail -f logs/reporter.log

# Test report generation
python scripts/scheduled_reporter.py --test daily
```

**Issue**: Reports saved but not sent

**Solutions**:
- Check email configuration in `.env`
- Check Discord webhook URL
- Review logs for errors: `tail -f logs/trading.log`

### Reports Are Empty

**Issue**: "No broker state found"

**Solutions**:
- Ensure trading bot has run at least once
- Check `data/simulated_broker_state.json` exists
- Verify trading bot is configured correctly

---

## 📅 Recommended Schedule

### For Active Trading (Simulated/Paper):
```yaml
daily:
  enabled: true
  time: "17:00"      # After market close

weekly:
  enabled: true
  day: "monday"
  time: "09:00"      # Start of week

monthly:
  enabled: false      # May be redundant with weekly
```

### For Live Trading:
```yaml
daily:
  enabled: true
  time: "17:00"      # Daily review critical

weekly:
  enabled: true
  day: "sunday"      # Weekend analysis
  time: "10:00"

monthly:
  enabled: true      # Monthly performance review
  day: 1
  time: "09:00"
```

---

## 💡 Best Practices

1. **Start with Test Reports**
   - Run `--test daily` before enabling scheduled reports
   - Verify email delivery and formatting

2. **Monitor Report Delivery**
   - Check spam folder for first few reports
   - Add sender email to contacts
   - Whitelist email address

3. **Adjust Schedule as Needed**
   - Daily reports during active trading
   - Weekly for long-term strategies
   - Monthly for passive monitoring

4. **Review Report Content**
   - Focus on key metrics: Sharpe, win rate, drawdown
   - Compare to benchmark (SPY)
   - Act on recommendations

5. **Save Important Reports**
   - Reports auto-saved to `data/reports/`
   - Archive monthly reports for records
   - Track performance over time

---

## 🔗 Related Commands

```bash
# Quick performance check (no email)
python scripts/check_performance.py

# View broker state
cat data/simulated_broker_state.json | python -m json.tool

# View recent trades
grep "EXECUTED" logs/trading.log | tail -20

# Start trading bot with reports
python scripts/start_trading.py --simulated &
python scripts/scheduled_reporter.py --daemon &

# Stop all services
pkill -f start_trading.py
pkill -f scheduled_reporter.py
```

---

## 📚 File Reference

**New Files Created**:
- `src/reporting/performance_reporter.py` - Report generation engine
- `src/notifications/email_notifier.py` - Email delivery
- `scripts/scheduled_reporter.py` - Scheduling daemon
- `scripts/check_performance.py` - Manual performance check

**Modified Files**:
- `config/trading_config.yaml` - Added reporting section
- `.env.example` - Added email configuration template

**Output**:
- `data/reports/*.txt` - Text reports
- `data/reports/*.html` - HTML reports
- `logs/reporter.log` - Reporter logs (if running in background)

---

## ❓ FAQ

**Q: Can I get reports for paper trading only?**
A: Yes, reports work with simulated, paper, and live trading modes.

**Q: How much do Gmail API limits matter?**
A: Gmail allows 500 emails/day. Daily reports use 1-3 emails, well within limits.

**Q: Can I customize report content?**
A: Yes, edit `src/reporting/performance_reporter.py` to add custom metrics.

**Q: Can reports include charts?**
A: Currently text-based. Future: Add Plotly charts to HTML reports.

**Q: Can I send reports to Slack/Teams?**
A: Add custom notifiers following the email_notifier.py pattern.

**Q: What if I don't want email, only Discord?**
A: Set `send_email: false` in config, keep `send_discord: true`.

**Q: Can I test without actually sending emails?**
A: Run `python scripts/check_performance.py` for console output only.

---

## 🎯 Next Steps

1. ✅ Configure email in `.env` file
2. ✅ Test email with: `python scripts/scheduled_reporter.py --test daily`
3. ✅ Adjust schedule in `config/trading_config.yaml`
4. ✅ Start scheduled reporter: `python scripts/scheduled_reporter.py --daemon`
5. ✅ Monitor first few reports for delivery
6. ✅ Adjust format/schedule as needed

---

*Automated Reporting Setup Guide - 2026-01-28*
*Part of Phase 22+ enhancements*
