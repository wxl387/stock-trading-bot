"""
Discord notification module using webhooks.
"""
import logging
import requests
from typing import Optional, Dict, List, Any
from datetime import datetime

from config.settings import settings

logger = logging.getLogger(__name__)

# Discord embed colors
COLOR_GREEN = 0x00FF00   # Success/Buy
COLOR_RED = 0xFF0000     # Error/Sell/Stop-loss
COLOR_ORANGE = 0xFFA500  # Warning
COLOR_BLUE = 0x0099FF    # Info
COLOR_PURPLE = 0x9B59B6  # Retraining


class DiscordNotifier:
    """
    Discord notification handler using webhooks.
    Sends alerts via Discord webhook with rich embeds.
    """

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL. If None, reads from settings.
        """
        self.webhook_url = webhook_url or getattr(settings, 'DISCORD_WEBHOOK_URL', None)

        if not self.webhook_url:
            logger.warning("Discord webhook URL not configured. Notifications disabled.")
            self.enabled = False
            return

        self.enabled = True
        logger.info("Discord notifier initialized")

    def _send_webhook(self, payload: Dict[str, Any]) -> bool:
        """
        Send payload to Discord webhook.

        Args:
            payload: Discord webhook payload.

        Returns:
            True if sent successfully.
        """
        if not self.enabled:
            return False

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to send Discord webhook: {e}")
            return False

    def _create_embed(
        self,
        title: str,
        description: str = "",
        color: int = COLOR_BLUE,
        fields: Optional[List[Dict]] = None,
        footer: Optional[str] = None
    ) -> Dict:
        """
        Create Discord embed object.

        Args:
            title: Embed title.
            description: Embed description.
            color: Embed color (hex).
            fields: List of field dicts with name, value, inline.
            footer: Footer text.

        Returns:
            Embed dict.
        """
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": datetime.utcnow().isoformat()
        }

        if fields:
            embed["fields"] = fields

        if footer:
            embed["footer"] = {"text": footer}

        return embed

    def send(self, content: str = "", embeds: Optional[List[Dict]] = None) -> bool:
        """
        Send message to Discord.

        Args:
            content: Plain text content.
            embeds: List of embed objects.

        Returns:
            True if sent successfully.
        """
        payload = {}
        if content:
            payload["content"] = content
        if embeds:
            payload["embeds"] = embeds

        return self._send_webhook(payload)

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
            total: Total value (optional, calculated if not provided).

        Returns:
            True if sent successfully.
        """
        is_buy = action.upper() == "BUY"
        color = COLOR_GREEN if is_buy else COLOR_RED
        emoji = ":green_circle:" if is_buy else ":red_circle:"
        total_value = total or (shares * price)

        fields = [
            {"name": "Symbol", "value": f"`{symbol}`", "inline": True},
            {"name": "Action", "value": action.upper(), "inline": True},
            {"name": "Shares", "value": str(shares), "inline": True},
            {"name": "Price", "value": f"${price:.2f}", "inline": True},
            {"name": "Total", "value": f"${total_value:,.2f}", "inline": True},
        ]

        if confidence:
            fields.append({
                "name": "ML Confidence",
                "value": f"{confidence:.1%}",
                "inline": True
            })

        embed = self._create_embed(
            title=f"{emoji} Order Executed",
            color=color,
            fields=fields
        )

        return self.send(embeds=[embed])

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
        fields = [
            {"name": "Symbol", "value": f"`{symbol}`", "inline": True},
            {"name": "Exit Price", "value": f"${exit_price:.2f}", "inline": True},
            {"name": "Loss", "value": f"${abs(loss_amount):,.2f} ({abs(loss_pct):.1%})", "inline": True},
        ]

        embed = self._create_embed(
            title=":octagonal_sign: Stop-Loss Triggered",
            description="Position closed automatically.",
            color=COLOR_RED,
            fields=fields
        )

        return self.send(embeds=[embed])

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
        fields = [
            {"name": "Symbol", "value": f"`{symbol}`", "inline": True},
            {"name": "Exit Price", "value": f"${exit_price:.2f}", "inline": True},
            {"name": "Shares Sold", "value": str(quantity), "inline": True},
            {"name": "Gain", "value": f"+${gain_amount:,.2f} (+{gain_pct:.1%})", "inline": True},
        ]

        embed = self._create_embed(
            title=":moneybag: Take-Profit Hit",
            description="Partial position closed at profit target.",
            color=COLOR_GREEN,
            fields=fields
        )

        return self.send(embeds=[embed])

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
        is_positive = daily_pnl >= 0
        emoji = ":chart_with_upwards_trend:" if is_positive else ":chart_with_downwards_trend:"
        color = COLOR_GREEN if is_positive else COLOR_RED
        sign = "+" if is_positive else ""

        fields = [
            {"name": "Portfolio Value", "value": f"${portfolio_value:,.2f}", "inline": True},
            {"name": "Daily P&L", "value": f"{sign}${daily_pnl:,.2f} ({sign}{daily_pnl_pct:.2%})", "inline": True},
            {"name": "Trades Today", "value": str(trades_count), "inline": True},
            {"name": "Open Positions", "value": str(positions_count), "inline": True},
        ]

        if win_rate is not None:
            fields.append({
                "name": "Win Rate",
                "value": f"{win_rate:.1%}",
                "inline": True
            })

        embed = self._create_embed(
            title=f"{emoji} Daily Summary - {datetime.now().strftime('%Y-%m-%d')}",
            color=color,
            fields=fields
        )

        return self.send(embeds=[embed])

    def notify_risk_warning(
        self,
        warning_type: str,
        details: str = ""
    ) -> bool:
        """
        Send risk warning notification.

        Args:
            warning_type: Type of warning (e.g., "Daily Loss Limit", "Max Positions").
            details: Additional details.

        Returns:
            True if sent successfully.
        """
        embed = self._create_embed(
            title=":warning: Risk Warning",
            description=warning_type,
            color=COLOR_ORANGE,
            fields=[{"name": "Details", "value": details, "inline": False}] if details else None
        )

        return self.send(embeds=[embed])

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
            models: Dict of model results with accuracy.
            deployments: Dict of model deployment status.
            duration_seconds: Retraining duration.

        Returns:
            True if sent successfully.
        """
        if status.lower() == "started":
            embed = self._create_embed(
                title=":gear: Model Retraining Started",
                description="Scheduled model retraining has begun.",
                color=COLOR_PURPLE
            )
        else:
            fields = []

            if models:
                for model_type, info in models.items():
                    accuracy = info.get("accuracy", 0)
                    deployed = deployments.get(model_type, False) if deployments else False
                    status_emoji = ":white_check_mark:" if deployed else ":x:"
                    fields.append({
                        "name": model_type.upper(),
                        "value": f"Accuracy: {accuracy:.1%}\nDeployed: {status_emoji}",
                        "inline": True
                    })

            if duration_seconds:
                fields.append({
                    "name": "Duration",
                    "value": f"{duration_seconds:.1f} seconds",
                    "inline": True
                })

            deployed_count = sum(deployments.values()) if deployments else 0
            total_count = len(models) if models else 0

            embed = self._create_embed(
                title=":white_check_mark: Model Retraining Completed",
                description=f"Deployed {deployed_count}/{total_count} models.",
                color=COLOR_PURPLE,
                fields=fields if fields else None
            )

        return self.send(embeds=[embed])

    def notify_startup(
        self,
        mode: str,
        symbols: List[str],
        model_type: str = "ensemble"
    ) -> bool:
        """
        Send bot startup notification.

        Args:
            mode: Trading mode (simulated, paper, live).
            symbols: List of trading symbols.
            model_type: Model being used.

        Returns:
            True if sent successfully.
        """
        fields = [
            {"name": "Mode", "value": mode.upper(), "inline": True},
            {"name": "Model", "value": model_type.upper(), "inline": True},
            {"name": "Symbols", "value": ", ".join(symbols[:10]) + ("..." if len(symbols) > 10 else ""), "inline": False},
        ]

        embed = self._create_embed(
            title=":rocket: Trading Bot Started",
            color=COLOR_BLUE,
            fields=fields
        )

        return self.send(embeds=[embed])

    def notify_shutdown(self, reason: str = "Manual shutdown") -> bool:
        """
        Send bot shutdown notification.

        Args:
            reason: Shutdown reason.

        Returns:
            True if sent successfully.
        """
        embed = self._create_embed(
            title=":stop_sign: Trading Bot Stopped",
            description=reason,
            color=COLOR_ORANGE
        )

        return self.send(embeds=[embed])

    def notify_error(self, error: str, details: str = "") -> bool:
        """
        Send error notification.

        Args:
            error: Error message.
            details: Stack trace or additional details.

        Returns:
            True if sent successfully.
        """
        fields = []
        if details:
            # Truncate long details
            truncated = details[:1000] + "..." if len(details) > 1000 else details
            fields.append({
                "name": "Details",
                "value": f"```{truncated}```",
                "inline": False
            })

        embed = self._create_embed(
            title=":fire: System Error",
            description=error,
            color=COLOR_RED,
            fields=fields if fields else None
        )

        return self.send(embeds=[embed])
