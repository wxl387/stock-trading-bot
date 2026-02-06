"""
Telegram notification module.
"""
import logging
import asyncio
from typing import Optional
from datetime import datetime

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

from config.settings import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Telegram notification handler.
    Sends alerts via Telegram bot.
    """

    def __init__(self):
        """Initialize Telegram notifier."""
        if not HAS_TELEGRAM:
            logger.warning("python-telegram-bot not installed. Telegram notifications disabled.")
            self.enabled = False
            return

        self.token = settings.TELEGRAM_BOT_TOKEN
        self.chat_id = settings.TELEGRAM_CHAT_ID

        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials not configured")
            self.enabled = False
            return

        self.bot = Bot(token=self.token)
        self.enabled = True
        logger.info("Telegram notifier initialized")

    async def send_async(self, message: str, parse_mode: str = "Markdown") -> bool:
        """
        Send message asynchronously.

        Args:
            message: Message text.
            parse_mode: Telegram parse mode.

        Returns:
            True if sent successfully.
        """
        if not self.enabled:
            return False

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send(self, message: str) -> bool:
        """
        Send message synchronously.

        Args:
            message: Message text.

        Returns:
            True if sent successfully.
        """
        if not self.enabled:
            return False

        try:
            asyncio.get_running_loop()
            # Already in an async context — can't use run_until_complete
            logger.warning("Cannot send sync Telegram message from async context")
            return False
        except RuntimeError:
            pass

        return asyncio.run(self.send_async(message))

    def notify_trade(
        self,
        action: str,
        symbol: str,
        shares: int,
        price: float,
        confidence: Optional[float] = None
    ) -> bool:
        """
        Send trade notification.

        Args:
            action: BUY or SELL.
            symbol: Stock symbol.
            shares: Number of shares.
            price: Trade price.
            confidence: ML confidence score.

        Returns:
            True if sent successfully.
        """
        emoji = "🟢" if action == "BUY" else "🔴"

        message = f"""
{emoji} *ORDER EXECUTED*

Symbol: `{symbol}`
Action: {action}
Quantity: {shares} shares
Price: ${price:.2f}
Total: ${shares * price:,.2f}
"""

        if confidence:
            message += f"\nML Confidence: {confidence:.1%}"

        message += f"\n_Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"

        return self.send(message)

    def notify_stop_loss(
        self,
        symbol: str,
        exit_price: float,
        loss_amount: float,
        loss_pct: float
    ) -> bool:
        """
        Send stop loss notification.

        Args:
            symbol: Stock symbol.
            exit_price: Exit price.
            loss_amount: Dollar loss.
            loss_pct: Percentage loss.

        Returns:
            True if sent successfully.
        """
        message = f"""
🔴 *STOP-LOSS TRIGGERED*

Symbol: `{symbol}`
Exit Price: ${exit_price:.2f}
Loss: ${loss_amount:,.2f} ({loss_pct:.1%})

Position closed automatically.
"""

        return self.send(message)

    def notify_take_profit(
        self,
        symbol: str,
        exit_price: float,
        quantity: int,
        gain_amount: float,
        gain_pct: float
    ) -> bool:
        """
        Send take-profit notification.

        Args:
            symbol: Stock symbol.
            exit_price: Exit price.
            quantity: Shares sold.
            gain_amount: Dollar gain.
            gain_pct: Percentage gain.

        Returns:
            True if sent successfully.
        """
        message = f"""
🎯 *TAKE-PROFIT TRIGGERED*

Symbol: `{symbol}`
Exit Price: ${exit_price:.2f}
Quantity: {quantity} shares
Gain: ${gain_amount:,.2f} ({gain_pct:.1%})

_Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""

        return self.send(message)

    def notify_daily_summary(
        self,
        portfolio_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        trades_count: int,
        win_rate: Optional[float] = None
    ) -> bool:
        """
        Send daily summary notification.

        Args:
            portfolio_value: Total portfolio value.
            daily_pnl: Daily P&L.
            daily_pnl_pct: Daily P&L percentage.
            trades_count: Number of trades today.
            win_rate: Win rate percentage.

        Returns:
            True if sent successfully.
        """
        emoji = "📈" if daily_pnl >= 0 else "📉"
        sign = "+" if daily_pnl >= 0 else ""

        message = f"""
{emoji} *DAILY SUMMARY* - {datetime.now().strftime('%Y-%m-%d')}

Portfolio Value: ${portfolio_value:,.2f}
Daily P&L: {sign}${daily_pnl:,.2f} ({sign}{daily_pnl_pct:.2%})

Trades Executed: {trades_count}
"""

        if win_rate is not None:
            message += f"Win Rate: {win_rate:.1%}"

        return self.send(message)

    def notify_risk_warning(self, warning: str, details: str = "") -> bool:
        """
        Send risk warning notification.

        Args:
            warning: Warning message.
            details: Additional details.

        Returns:
            True if sent successfully.
        """
        message = f"""
⚠️ *RISK WARNING*

{warning}
"""

        if details:
            message += f"\n{details}"

        message += f"\n_Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"

        return self.send(message)

    def notify_error(self, error: str) -> bool:
        """
        Send error notification.

        Args:
            error: Error message.

        Returns:
            True if sent successfully.
        """
        message = f"""
🔥 *SYSTEM ERROR*

{error}

_Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_
"""

        return self.send(message)
