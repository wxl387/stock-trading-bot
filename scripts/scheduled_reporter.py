#!/usr/bin/env python3
"""
Scheduled Performance Reporter - Automatically generate and send performance reports.
Runs as a background service with configurable schedules.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import argparse
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import setup_logging, DATA_DIR, Settings
from src.reporting import PerformanceReporter
from src.notifications.email_notifier import EmailNotifier
from src.notifications.discord_notifier import DiscordNotifier

logger = setup_logging()


class ScheduledReporter:
    """Automated performance report scheduler."""

    def __init__(self):
        """Initialize the scheduled reporter."""
        self.reporter = PerformanceReporter(DATA_DIR)
        self.email_notifier = EmailNotifier()
        self.discord_notifier = DiscordNotifier()

        # Load schedules from config
        config = Settings.load_trading_config()
        report_config = config.get('reporting', {})
        self.daily_enabled = report_config.get('daily', {}).get('enabled', True)
        self.daily_time = report_config.get('daily', {}).get('time', '17:00')  # 5 PM

        self.weekly_enabled = report_config.get('weekly', {}).get('enabled', True)
        self.weekly_day = report_config.get('weekly', {}).get('day', 'monday')
        self.weekly_time = report_config.get('weekly', {}).get('time', '09:00')  # 9 AM

        self.monthly_enabled = report_config.get('monthly', {}).get('enabled', False)
        self.monthly_day = report_config.get('monthly', {}).get('day', 1)
        self.monthly_time = report_config.get('monthly', {}).get('time', '09:00')  # 9 AM

        # Notification settings
        self.send_email = report_config.get('send_email', True)
        self.send_discord = report_config.get('send_discord', True)
        self.save_to_file = report_config.get('save_to_file', True)

        logger.info("Scheduled Reporter initialized")
        logger.info(f"Daily reports: {self.daily_enabled} (at {self.daily_time})")
        logger.info(f"Weekly reports: {self.weekly_enabled} (on {self.weekly_day} at {self.weekly_time})")
        logger.info(f"Monthly reports: {self.monthly_enabled} (on day {self.monthly_day} at {self.monthly_time})")

    def generate_and_send_report(self, report_type: str):
        """
        Generate and send a performance report.

        Args:
            report_type: Type of report (daily, weekly, monthly)
        """
        logger.info(f"Generating {report_type} performance report...")

        try:
            # Generate reports
            text_report = self.reporter.generate_text_report(report_type)
            html_report = self.reporter.generate_html_report(report_type)

            # Save to file if enabled
            if self.save_to_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{report_type}_report_{timestamp}"

                self.reporter.save_report(text_report, filename, format='txt')
                self.reporter.save_report(html_report, filename, format='html')

            # Send via email if enabled
            if self.send_email and self.email_notifier.enabled:
                logger.info(f"Sending {report_type} report via email...")
                self.email_notifier.send_performance_report(
                    report_text=text_report,
                    report_html=html_report,
                    report_type=report_type
                )

            # Send via Discord if enabled
            if self.send_discord and self.discord_notifier.webhook_url:
                logger.info(f"Sending {report_type} report via Discord...")

                # Create a condensed version for Discord
                lines = text_report.split('\n')
                # Take first 50 lines or until we hit the limit
                condensed_lines = lines[:50]
                if len(lines) > 50:
                    condensed_lines.append("...")
                    condensed_lines.append(f"(Full report: {len(lines)} lines)")

                condensed_report = '\n'.join(condensed_lines)

                # Discord has a 2000 character limit, so truncate if needed
                if len(condensed_report) > 1900:
                    condensed_report = condensed_report[:1900] + "\n...(truncated)"

                self.discord_notifier.send_message(
                    f"📊 **{report_type.title()} Performance Report**\n```\n{condensed_report}\n```"
                )

            logger.info(f"{report_type.title()} report generated and sent successfully")

        except Exception as e:
            logger.error(f"Error generating {report_type} report: {e}", exc_info=True)

            # Send error notification
            if self.email_notifier.enabled:
                self.email_notifier.send_alert(
                    title=f"Report Generation Failed ({report_type})",
                    message=f"Error generating {report_type} report: {str(e)}",
                    severity="error"
                )

    def daily_report(self):
        """Generate and send daily report."""
        self.generate_and_send_report('daily')

    def weekly_report(self):
        """Generate and send weekly report."""
        self.generate_and_send_report('weekly')

    def monthly_report(self):
        """Generate and send monthly report."""
        self.generate_and_send_report('monthly')

    def start(self):
        """Start the scheduled reporter."""
        scheduler = BlockingScheduler()

        # Schedule daily reports
        if self.daily_enabled:
            hour, minute = map(int, self.daily_time.split(':'))
            scheduler.add_job(
                self.daily_report,
                CronTrigger(hour=hour, minute=minute),
                id='daily_report',
                name='Daily Performance Report'
            )
            logger.info(f"Scheduled daily reports at {self.daily_time}")

        # Schedule weekly reports
        if self.weekly_enabled:
            hour, minute = map(int, self.weekly_time.split(':'))
            day_map = {
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            day_of_week = day_map.get(self.weekly_day.lower(), 0)

            scheduler.add_job(
                self.weekly_report,
                CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
                id='weekly_report',
                name='Weekly Performance Report'
            )
            logger.info(f"Scheduled weekly reports on {self.weekly_day} at {self.weekly_time}")

        # Schedule monthly reports
        if self.monthly_enabled:
            hour, minute = map(int, self.monthly_time.split(':'))
            scheduler.add_job(
                self.monthly_report,
                CronTrigger(day=self.monthly_day, hour=hour, minute=minute),
                id='monthly_report',
                name='Monthly Performance Report'
            )
            logger.info(f"Scheduled monthly reports on day {self.monthly_day} at {self.monthly_time}")

        # Print scheduled jobs
        logger.info("Scheduled jobs:")
        for job in scheduler.get_jobs():
            logger.info(f"  - {job.name}: {job.next_run_time}")

        logger.info("Scheduled reporter started. Press Ctrl+C to stop.")

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduled reporter stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Scheduled Performance Reporter")
    parser.add_argument(
        '--test',
        choices=['daily', 'weekly', 'monthly'],
        help='Generate a test report immediately'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as a daemon (scheduled reports)'
    )

    args = parser.parse_args()

    reporter = ScheduledReporter()

    if args.test:
        # Generate a test report immediately
        logger.info(f"Generating test {args.test} report...")
        reporter.generate_and_send_report(args.test)
        logger.info("Test report complete")

    elif args.daemon:
        # Run as daemon with scheduled reports
        reporter.start()

    else:
        # Default: show help
        parser.print_help()
        print("\nExamples:")
        print("  # Generate a test daily report immediately")
        print("  python scripts/scheduled_reporter.py --test daily")
        print("")
        print("  # Run as daemon with scheduled reports")
        print("  python scripts/scheduled_reporter.py --daemon")
        print("")
        print("Configuration is in config/trading_config.yaml under 'reporting' section")


if __name__ == "__main__":
    main()
