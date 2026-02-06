"""
Email Notifier - Send email notifications for trading events and reports.
"""
import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send email notifications for trading events and performance reports."""

    def __init__(self, config: dict = None):
        """
        Initialize email notifier.

        Args:
            config: Email configuration dict with keys:
                - smtp_server: SMTP server address
                - smtp_port: SMTP server port
                - sender_email: Sender email address
                - sender_password: Sender password or app password
                - recipient_emails: List of recipient email addresses
                - use_tls: Whether to use TLS (default: True)
        """
        # Load from config or environment variables
        if config:
            self.smtp_server = config.get('smtp_server')
            self.smtp_port = config.get('smtp_port', 587)
            self.sender_email = config.get('sender_email')
            self.sender_password = config.get('sender_password')
            self.recipient_emails = config.get('recipient_emails', [])
            self.use_tls = config.get('use_tls', True)
        else:
            # Load from environment
            self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
            self.sender_email = os.getenv('SENDER_EMAIL')
            self.sender_password = os.getenv('SENDER_PASSWORD')
            recipient_str = os.getenv('RECIPIENT_EMAILS', '')
            self.recipient_emails = [e.strip() for e in recipient_str.split(',') if e.strip()]
            self.use_tls = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'

        self.enabled = all([
            self.smtp_server,
            self.sender_email,
            self.sender_password,
            self.recipient_emails
        ])

        if not self.enabled:
            logger.warning("Email notifier disabled: Missing configuration")
            logger.info("Set SMTP_SERVER, SENDER_EMAIL, SENDER_PASSWORD, and RECIPIENT_EMAILS in .env")

    def send_email(
        self,
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        attachments: Optional[List[Path]] = None
    ) -> bool:
        """
        Send an email notification.

        Args:
            subject: Email subject
            body: Plain text email body
            html_body: Optional HTML email body
            attachments: Optional list of file paths to attach

        Returns:
            True if email sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Cannot send email: Email notifier not configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(self.recipient_emails)

            # Attach plain text body
            msg.attach(MIMEText(body, 'plain'))

            # Attach HTML body if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))

            # Attach files if provided
            if attachments:
                for filepath in attachments:
                    if not filepath.exists():
                        logger.warning(f"Attachment not found: {filepath}")
                        continue

                    with open(filepath, 'rb') as f:
                        attachment = MIMEApplication(f.read(), Name=filepath.name)
                        attachment['Content-Disposition'] = f'attachment; filename="{filepath.name}"'
                        msg.attach(attachment)

            # Connect to SMTP server and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=30) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully: {subject}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_performance_report(
        self,
        report_text: str,
        report_html: Optional[str] = None,
        report_type: str = "daily"
    ) -> bool:
        """
        Send a performance report via email.

        Args:
            report_text: Plain text report
            report_html: Optional HTML report
            report_type: Type of report (daily, weekly, monthly)

        Returns:
            True if sent successfully
        """
        subject = f"📊 {report_type.title()} Trading Bot Performance Report"

        # Use HTML if provided, otherwise plain text
        if report_html:
            return self.send_email(subject, report_text, html_body=report_html)
        else:
            return self.send_email(subject, report_text)

    def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "info"
    ) -> bool:
        """
        Send an alert email.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity (info, warning, error, critical)

        Returns:
            True if sent successfully
        """
        emoji_map = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨'
        }

        emoji = emoji_map.get(severity.lower(), 'ℹ️')
        subject = f"{emoji} Trading Bot Alert: {title}"

        body = f"""
Trading Bot Alert

Severity: {severity.upper()}
Title: {title}

Message:
{message}

---
Sent from your automated trading bot
        """

        return self.send_email(subject, body.strip())

    def send_trade_notification(
        self,
        action: str,
        symbol: str,
        shares: int,
        price: float,
        total_value: float,
        reason: str = ""
    ) -> bool:
        """
        Send a trade notification email.

        Args:
            action: Trade action (BUY/SELL)
            symbol: Stock symbol
            shares: Number of shares
            price: Execution price
            total_value: Total trade value
            reason: Optional reason for trade

        Returns:
            True if sent successfully
        """
        emoji = "🟢" if action == "BUY" else "🔴"
        subject = f"{emoji} Trade Executed: {action} {symbol}"

        body = f"""
Trade Executed

Action: {action}
Symbol: {symbol}
Shares: {shares}
Price: ${price:.2f}
Total Value: ${total_value:,.2f}
"""

        if reason:
            body += f"\nReason: {reason}"

        body += "\n\n---\nSent from your automated trading bot"

        return self.send_email(subject, body.strip())


# Example usage
if __name__ == "__main__":
    # Test email notifier
    notifier = EmailNotifier()

    if notifier.enabled:
        print("Email notifier configured successfully")
        print(f"Sender: {notifier.sender_email}")
        print(f"Recipients: {', '.join(notifier.recipient_emails)}")

        # Send test email
        test_result = notifier.send_alert(
            title="Email Notifier Test",
            message="This is a test email from your trading bot.",
            severity="info"
        )

        if test_result:
            print("✅ Test email sent successfully")
        else:
            print("❌ Failed to send test email")
    else:
        print("❌ Email notifier not configured")
        print("Set these environment variables in .env:")
        print("  SMTP_SERVER (e.g., smtp.gmail.com)")
        print("  SMTP_PORT (e.g., 587)")
        print("  SENDER_EMAIL")
        print("  SENDER_PASSWORD (use app password for Gmail)")
        print("  RECIPIENT_EMAILS (comma-separated)")
