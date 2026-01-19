"""
Notification manager that orchestrates all notification channels.
"""
import logging
from typing import Optional, Dict, List, Any

from config.settings import Settings
from src.notifications.discord_notifier import DiscordNotifier

logger = logging.getLogger(__name__)


class NotifierManager:
    """
    Central notification manager.
    Routes notifications to configured channels (Discord, Telegram, etc.)
    based on configuration settings.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize notification manager.

        Args:
            config: Optional config dict. If None, loads from trading_config.yaml.
        """
        if config is None:
            config = Settings.load_trading_config()

        self.config = config.get("notifications", {})
        self.enabled = self.config.get("enabled", True)
        self.notify_on = self.config.get("notify_on", {})

        # Initialize notifiers based on config
        self.discord: Optional[DiscordNotifier] = None
        self.telegram = None  # TelegramNotifier can be added later

        channels = self.config.get("channels", [])

        if "discord" in channels:
            self.discord = DiscordNotifier()
            if self.discord.enabled:
                logger.info("Discord notifications enabled")

        if "telegram" in channels:
            try:
                from src.notifications.telegram_notifier import TelegramNotifier
                self.telegram = TelegramNotifier()
                if self.telegram.enabled:
                    logger.info("Telegram notifications enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Telegram: {e}")

        # Check if any notifier is available
        self.has_active_notifier = (
            (self.discord and self.discord.enabled) or
            (self.telegram and getattr(self.telegram, 'enabled', False))
        )

        if not self.has_active_notifier:
            logger.warning("No notification channels configured or enabled")

    def _should_notify(self, event_type: str) -> bool:
        """Check if event type should trigger notification."""
        if not self.enabled:
            return False
        return self.notify_on.get(event_type, True)

    def notify_trade(
        self,
        action: str,
        symbol: str,
        shares: int,
        price: float,
        confidence: Optional[float] = None,
        total: Optional[float] = None
    ) -> bool:
        """
        Send trade notification.

        Args:
            action: BUY or SELL.
            symbol: Stock symbol.
            shares: Number of shares.
            price: Trade price.
            confidence: ML confidence score.
            total: Total trade value.

        Returns:
            True if sent to at least one channel.
        """
        if not self._should_notify("trade_executed"):
            return False

        success = False

        if self.discord and self.discord.enabled:
            success = self.discord.notify_trade(
                action=action,
                symbol=symbol,
                shares=shares,
                price=price,
                confidence=confidence,
                total=total
            ) or success

        if self.telegram and getattr(self.telegram, 'enabled', False):
            success = self.telegram.notify_trade(
                action=action,
                symbol=symbol,
                shares=shares,
                price=price,
                confidence=confidence
            ) or success

        return success

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
        if not self._should_notify("stop_loss_triggered"):
            return False

        success = False

        if self.discord and self.discord.enabled:
            success = self.discord.notify_stop_loss(
                symbol=symbol,
                exit_price=exit_price,
                loss_amount=loss_amount,
                loss_pct=loss_pct
            ) or success

        if self.telegram and getattr(self.telegram, 'enabled', False):
            success = self.telegram.notify_stop_loss(
                symbol=symbol,
                exit_price=exit_price,
                loss_amount=loss_amount,
                loss_pct=loss_pct
            ) or success

        return success

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
            quantity: Number of shares sold.
            gain_amount: Dollar gain.
            gain_pct: Percentage gain.

        Returns:
            True if sent successfully.
        """
        if not self._should_notify("take_profit_triggered"):
            return False

        success = False

        if self.discord and self.discord.enabled:
            success = self.discord.notify_take_profit(
                symbol=symbol,
                exit_price=exit_price,
                quantity=quantity,
                gain_amount=gain_amount,
                gain_pct=gain_pct
            ) or success

        if self.telegram and getattr(self.telegram, 'enabled', False):
            success = self.telegram.notify_take_profit(
                symbol=symbol,
                exit_price=exit_price,
                quantity=quantity,
                gain_amount=gain_amount,
                gain_pct=gain_pct
            ) or success

        return success

    def notify_daily_summary(
        self,
        portfolio_value: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        trades_count: int,
        win_rate: Optional[float] = None,
        positions_count: int = 0
    ) -> bool:
        """
        Send daily summary notification.

        Args:
            portfolio_value: Total portfolio value.
            daily_pnl: Daily P&L.
            daily_pnl_pct: Daily P&L percentage.
            trades_count: Number of trades today.
            win_rate: Win rate percentage.
            positions_count: Number of open positions.

        Returns:
            True if sent successfully.
        """
        if not self._should_notify("daily_summary"):
            return False

        success = False

        if self.discord and self.discord.enabled:
            success = self.discord.notify_daily_summary(
                portfolio_value=portfolio_value,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                trades_count=trades_count,
                win_rate=win_rate,
                positions_count=positions_count
            ) or success

        if self.telegram and getattr(self.telegram, 'enabled', False):
            success = self.telegram.notify_daily_summary(
                portfolio_value=portfolio_value,
                daily_pnl=daily_pnl,
                daily_pnl_pct=daily_pnl_pct,
                trades_count=trades_count,
                win_rate=win_rate
            ) or success

        return success

    def notify_risk_warning(
        self,
        warning_type: str,
        details: str = ""
    ) -> bool:
        """
        Send risk warning notification.

        Args:
            warning_type: Type of warning.
            details: Additional details.

        Returns:
            True if sent successfully.
        """
        if not self._should_notify("risk_warning"):
            return False

        success = False

        if self.discord and self.discord.enabled:
            success = self.discord.notify_risk_warning(
                warning_type=warning_type,
                details=details
            ) or success

        if self.telegram and getattr(self.telegram, 'enabled', False):
            success = self.telegram.notify_risk_warning(
                warning=warning_type,
                details=details
            ) or success

        return success

    def notify_retraining(
        self,
        status: str,
        models: Optional[Dict[str, Dict]] = None,
        deployments: Optional[Dict[str, bool]] = None,
        duration_seconds: Optional[float] = None
    ) -> bool:
        """
        Send retraining notification.

        Args:
            status: "started" or "completed".
            models: Dict of model results.
            deployments: Dict of deployment status.
            duration_seconds: Retraining duration.

        Returns:
            True if sent successfully.
        """
        # Check config - use a custom key for retraining
        if not self.enabled:
            return False

        success = False

        if self.discord and self.discord.enabled:
            success = self.discord.notify_retraining(
                status=status,
                models=models,
                deployments=deployments,
                duration_seconds=duration_seconds
            ) or success

        return success

    def notify_startup(
        self,
        mode: str,
        symbols: List[str],
        model_type: str = "ensemble"
    ) -> bool:
        """
        Send bot startup notification.

        Args:
            mode: Trading mode.
            symbols: List of symbols.
            model_type: Model type.

        Returns:
            True if sent successfully.
        """
        if not self.enabled:
            return False

        success = False

        if self.discord and self.discord.enabled:
            success = self.discord.notify_startup(
                mode=mode,
                symbols=symbols,
                model_type=model_type
            ) or success

        return success

    def notify_shutdown(self, reason: str = "Manual shutdown") -> bool:
        """
        Send bot shutdown notification.

        Args:
            reason: Shutdown reason.

        Returns:
            True if sent successfully.
        """
        if not self.enabled:
            return False

        success = False

        if self.discord and self.discord.enabled:
            success = self.discord.notify_shutdown(reason=reason) or success

        return success

    def notify_error(self, error: str, details: str = "") -> bool:
        """
        Send error notification.

        Args:
            error: Error message.
            details: Additional details.

        Returns:
            True if sent successfully.
        """
        if not self._should_notify("system_error"):
            return False

        success = False

        if self.discord and self.discord.enabled:
            success = self.discord.notify_error(
                error=error,
                details=details
            ) or success

        if self.telegram and getattr(self.telegram, 'enabled', False):
            success = self.telegram.notify_error(error=error) or success

        return success


# Singleton instance
_notifier: Optional[NotifierManager] = None


def get_notifier() -> NotifierManager:
    """Get singleton notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = NotifierManager()
    return _notifier
