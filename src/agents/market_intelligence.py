"""
Market Intelligence Agent

Information gathering agent that monitors news, macro data, earnings, and sector trends.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .base_agent import (
    AgentMessage,
    AgentRole,
    BaseAgent,
    MessagePriority,
    MessageType,
)

logger = logging.getLogger(__name__)


class MarketIntelligenceAgent(BaseAgent):
    """
    Market Intelligence Agent - gathers and analyzes market information.

    Responsibilities:
    - Monitor breaking news for held symbols
    - Track earnings announcements and surprises
    - Analyze macro economic indicators
    - Detect sector rotations and trends
    - Alert on significant market events (VIX spikes, circuit breakers)

    Schedule:
    - News scan: Every 1 hour
    - Earnings check: Every 4 hours
    - Macro analysis: Every 6 hours
    - Sector analysis: Daily (6 AM)

    Outputs To:
    - Risk Guardian: Market alerts, volatility warnings
    - Portfolio Strategist: Sector recommendations, earnings alerts
    - Operations: System-wide market status
    """

    def __init__(
        self,
        config: Dict[str, Any],
        message_queue,
        notifier=None,
        llm_client=None,
    ):
        """
        Initialize Market Intelligence agent.

        Args:
            config: Agent configuration
            message_queue: Shared message queue
            notifier: Optional Discord notifier
            llm_client: Optional LLM client for intelligent analysis
        """
        super().__init__(
            role=AgentRole.MARKET_INTELLIGENCE,
            config=config,
            message_queue=message_queue,
            notifier=notifier,
            llm_client=llm_client,
        )

        # Load configuration
        mi_config = config.get("market_intelligence", {})
        self.news_scan_hours = mi_config.get("news_scan_hours", 1)
        self.earnings_check_hours = mi_config.get("earnings_check_hours", 4)
        self.macro_analysis_hours = mi_config.get("macro_analysis_hours", 6)
        self.sector_analysis_time = mi_config.get("sector_analysis_time", "06:00")
        self.news_sources = mi_config.get("news_sources", ["finnhub", "bluesky"])
        self.alert_on_vix_spike = mi_config.get("alert_on_vix_spike", 25)

        # Track last analysis times
        self._last_news_scan: Optional[datetime] = None
        self._last_earnings_check: Optional[datetime] = None
        self._last_macro_analysis: Optional[datetime] = None
        self._last_sector_analysis: Optional[datetime] = None

        # Lazy-loaded components
        self._sentiment_fetcher = None
        self._news_fetcher = None
        self._macro_fetcher = None
        self._fundamental_fetcher = None
        self._symbol_manager = None

        logger.info(
            f"Market Intelligence agent initialized: "
            f"news_scan={self.news_scan_hours}h, "
            f"earnings_check={self.earnings_check_hours}h, "
            f"macro_analysis={self.macro_analysis_hours}h"
        )

    @property
    def sentiment_fetcher(self):
        """Lazy load sentiment fetcher."""
        if self._sentiment_fetcher is None:
            try:
                from src.data.sentiment_fetcher import get_sentiment_fetcher
                self._sentiment_fetcher = get_sentiment_fetcher()
            except ImportError as e:
                logger.error(f"Failed to import SentimentFetcher: {e}")
        return self._sentiment_fetcher

    @property
    def news_fetcher(self):
        """Lazy load news fetcher."""
        if self._news_fetcher is None:
            try:
                from src.data.news_fetcher import NewsFetcher
                self._news_fetcher = NewsFetcher()
            except ImportError as e:
                logger.error(f"Failed to import NewsFetcher: {e}")
        return self._news_fetcher

    @property
    def macro_fetcher(self):
        """Lazy load macro data fetcher."""
        if self._macro_fetcher is None:
            try:
                from src.data.macro_fetcher import get_macro_fetcher
                self._macro_fetcher = get_macro_fetcher()
            except ImportError as e:
                logger.debug(f"MacroFetcher not available: {e}")
        return self._macro_fetcher

    @property
    def fundamental_fetcher(self):
        """Lazy load fundamental data fetcher."""
        if self._fundamental_fetcher is None:
            try:
                from src.data.fundamental_fetcher import get_fundamental_fetcher
                self._fundamental_fetcher = get_fundamental_fetcher()
            except ImportError as e:
                logger.debug(f"FundamentalFetcher not available: {e}")
        return self._fundamental_fetcher

    @property
    def symbol_manager(self):
        """Lazy load symbol manager."""
        if self._symbol_manager is None:
            try:
                from src.core.symbol_manager import get_symbol_manager
                from config.settings import Settings
                config = Settings.load_trading_config()
                self._symbol_manager = get_symbol_manager(config)
            except ImportError as e:
                logger.error(f"Failed to import SymbolManager: {e}")
        return self._symbol_manager

    def analyze(self) -> List[AgentMessage]:
        """
        Perform analysis and generate observations/suggestions.

        Returns:
            List of messages to send to other agents
        """
        # Default behavior: return empty list
        # Orchestrator calls specific methods for scheduled tasks
        return []

    def run_news_scan(self) -> List[AgentMessage]:
        """
        Scan news for portfolio symbols and detect significant events.

        Returns:
            List of messages with news alerts
        """
        self._last_news_scan = datetime.now()
        messages = []

        try:
            # Get current portfolio symbols
            portfolio_symbols = []
            if self.symbol_manager:
                portfolio_symbols = self.symbol_manager.get_active_symbols()

            if not portfolio_symbols:
                logger.info("No portfolio symbols to scan news for")
                return messages

            # Gather news for each symbol
            all_news = []
            significant_news = []

            for symbol in portfolio_symbols:
                try:
                    if self.news_fetcher:
                        news_items = self.news_fetcher.fetch_news(symbol, last=10)
                        for item in news_items:
                            item["symbol"] = symbol
                            all_news.append(item)

                            # Check for significant news (high sentiment magnitude)
                            sentiment = item.get("sentiment_score", 0)
                            if abs(sentiment) > 0.5:
                                significant_news.append(item)
                except Exception as e:
                    logger.warning(f"Failed to fetch news for {symbol}: {e}")

            # Check VIX level
            vix_alert = self._check_vix_spike()

            if significant_news or vix_alert:
                content = self._format_news_scan_report(
                    all_news, significant_news, vix_alert
                )

                # Use LLM for enhanced analysis
                if self.llm_client and self.llm_client.is_available():
                    llm_analysis = self.llm_client.analyze_news_impact(
                        significant_news, portfolio_symbols
                    )
                    if llm_analysis:
                        content += f"\n\n### AI Analysis\n{llm_analysis}"

                priority = MessagePriority.HIGH if vix_alert else MessagePriority.NORMAL

                # Alert Risk Guardian about significant news
                messages.append(self.create_message(
                    recipient=AgentRole.RISK_GUARDIAN,
                    message_type=MessageType.OBSERVATION,
                    subject=f"News Alert: {len(significant_news)} significant items",
                    content=content,
                    priority=priority,
                    context={
                        "news_count": len(all_news),
                        "significant_count": len(significant_news),
                        "vix_alert": vix_alert,
                        "symbols_scanned": portfolio_symbols,
                    },
                ))

                # Also alert Portfolio Strategist if there are significant earnings news
                earnings_news = [n for n in significant_news if "earnings" in n.get("headline", "").lower()]
                if earnings_news:
                    messages.append(self.create_message(
                        recipient=AgentRole.PORTFOLIO_STRATEGIST,
                        message_type=MessageType.OBSERVATION,
                        subject=f"Earnings News Alert: {len(earnings_news)} items",
                        content=self._format_earnings_news(earnings_news),
                        priority=MessagePriority.NORMAL,
                        context={"earnings_news": earnings_news},
                    ))

            logger.info(f"News scan complete: {len(all_news)} items, {len(significant_news)} significant")

        except Exception as e:
            logger.error(f"Error during news scan: {e}")

        return messages

    def run_earnings_check(self) -> List[AgentMessage]:
        """
        Check for upcoming and recent earnings announcements.

        Returns:
            List of messages with earnings alerts
        """
        self._last_earnings_check = datetime.now()
        messages = []

        if not self.fundamental_fetcher:
            logger.warning("Fundamental fetcher not available for earnings check")
            return messages

        try:
            portfolio_symbols = []
            if self.symbol_manager:
                portfolio_symbols = self.symbol_manager.get_active_symbols()

            if not portfolio_symbols:
                logger.info("No portfolio symbols to check earnings for")
                return messages

            upcoming_earnings = []
            recent_surprises = []

            for symbol in portfolio_symbols:
                try:
                    earnings_data = self.fundamental_fetcher.get_earnings_calendar(symbol)

                    # Check for upcoming earnings (next 7 days)
                    if earnings_data.get("next_earnings_date"):
                        next_date = earnings_data["next_earnings_date"]
                        days_until = (next_date - datetime.now().date()).days
                        if 0 <= days_until <= 7:
                            upcoming_earnings.append({
                                "symbol": symbol,
                                "date": next_date.isoformat(),
                                "days_until": days_until,
                            })

                    # Check for recent earnings surprises
                    if earnings_data.get("last_surprise"):
                        surprise = earnings_data["last_surprise"]
                        if abs(surprise) > 10:  # >10% surprise
                            recent_surprises.append({
                                "symbol": symbol,
                                "surprise_pct": surprise,
                                "reported_date": earnings_data.get("last_report_date"),
                            })
                except Exception as e:
                    logger.debug(f"Failed to get earnings for {symbol}: {e}")

            if upcoming_earnings or recent_surprises:
                content = self._format_earnings_report(upcoming_earnings, recent_surprises)

                # Notify Portfolio Strategist about earnings events
                messages.append(self.create_message(
                    recipient=AgentRole.PORTFOLIO_STRATEGIST,
                    message_type=MessageType.OBSERVATION,
                    subject=f"Earnings Alert: {len(upcoming_earnings)} upcoming, {len(recent_surprises)} surprises",
                    content=content,
                    priority=MessagePriority.HIGH if recent_surprises else MessagePriority.NORMAL,
                    context={
                        "upcoming_earnings": upcoming_earnings,
                        "recent_surprises": recent_surprises,
                    },
                ))

            logger.info(f"Earnings check complete: {len(upcoming_earnings)} upcoming, {len(recent_surprises)} surprises")

        except Exception as e:
            logger.error(f"Error during earnings check: {e}")

        return messages

    def run_macro_analysis(self) -> List[AgentMessage]:
        """
        Analyze macro economic indicators.

        Returns:
            List of messages with macro analysis
        """
        self._last_macro_analysis = datetime.now()
        messages = []

        try:
            macro_data = self._gather_macro_data()

            if not macro_data:
                logger.info("No macro data available for analysis")
                return messages

            # Analyze macro conditions
            analysis = self._analyze_macro_conditions(macro_data)

            content = self._format_macro_report(macro_data, analysis)

            # Use LLM for enhanced analysis
            if self.llm_client and self.llm_client.is_available():
                llm_analysis = self.llm_client.analyze_macro_environment(macro_data)
                if llm_analysis:
                    content += f"\n\n### AI Analysis\n{llm_analysis}"

            # Determine priority based on macro conditions
            priority = MessagePriority.NORMAL
            if analysis.get("risk_elevated"):
                priority = MessagePriority.HIGH

            # Send to Risk Guardian and Portfolio Strategist
            messages.append(self.create_message(
                recipient=AgentRole.RISK_GUARDIAN,
                message_type=MessageType.OBSERVATION,
                subject=f"Macro Analysis: {analysis.get('overall_outlook', 'Neutral')}",
                content=content,
                priority=priority,
                context={
                    "macro_data": macro_data,
                    "analysis": analysis,
                },
            ))

            messages.append(self.create_message(
                recipient=AgentRole.PORTFOLIO_STRATEGIST,
                message_type=MessageType.OBSERVATION,
                subject=f"Macro Environment Update: {analysis.get('overall_outlook', 'Neutral')}",
                content=content,
                priority=MessagePriority.NORMAL,
                context={
                    "macro_data": macro_data,
                    "analysis": analysis,
                },
            ))

            logger.info(f"Macro analysis complete: {analysis.get('overall_outlook', 'N/A')}")

        except Exception as e:
            logger.error(f"Error during macro analysis: {e}")

        return messages

    def run_sector_analysis(self) -> List[AgentMessage]:
        """
        Analyze sector performance and rotations.

        Returns:
            List of messages with sector analysis
        """
        self._last_sector_analysis = datetime.now()
        messages = []

        try:
            sector_data = self._gather_sector_data()

            if not sector_data:
                logger.info("No sector data available for analysis")
                return messages

            # Detect sector rotation
            rotation_signals = self._detect_sector_rotation(sector_data)

            content = self._format_sector_report(sector_data, rotation_signals)

            # Use LLM for enhanced analysis
            if self.llm_client and self.llm_client.is_available():
                llm_analysis = self.llm_client.detect_sector_rotation(sector_data)
                if llm_analysis:
                    content += f"\n\n### AI Analysis\n{llm_analysis}"

            # Notify Portfolio Strategist about sector trends
            messages.append(self.create_message(
                recipient=AgentRole.PORTFOLIO_STRATEGIST,
                message_type=MessageType.OBSERVATION,
                subject=f"Sector Analysis: {rotation_signals.get('trend', 'Mixed')}",
                content=content,
                priority=MessagePriority.NORMAL,
                context={
                    "sector_data": sector_data,
                    "rotation_signals": rotation_signals,
                },
            ))

            logger.info(f"Sector analysis complete")

        except Exception as e:
            logger.error(f"Error during sector analysis: {e}")

        return messages

    def _check_vix_spike(self) -> Optional[Dict[str, Any]]:
        """Check for VIX spike above threshold."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="5d")

            if not hist.empty:
                current_vix = hist["Close"].iloc[-1]
                prev_vix = hist["Close"].iloc[-2] if len(hist) > 1 else current_vix

                if current_vix > self.alert_on_vix_spike:
                    return {
                        "current": current_vix,
                        "previous": prev_vix,
                        "change": (current_vix - prev_vix) / prev_vix * 100 if prev_vix != 0 else 0,
                        "threshold": self.alert_on_vix_spike,
                    }
        except Exception as e:
            logger.debug(f"Failed to check VIX: {e}")

        return None

    def _gather_macro_data(self) -> Dict[str, Any]:
        """Gather macro economic data."""
        data = {}

        try:
            import yfinance as yf

            # Key macro indicators via ETFs/indices
            indicators = {
                "spy": "^GSPC",      # S&P 500
                "vix": "^VIX",       # Volatility
                "tlt": "TLT",        # Long-term treasuries
                "gold": "GLD",       # Gold
                "dxy": "DX-Y.NYB",   # Dollar index
            }

            for name, symbol in indicators.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1mo")
                    if not hist.empty:
                        data[name] = {
                            "current": hist["Close"].iloc[-1],
                            "change_1d": (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100 if len(hist) > 1 else 0,
                            "change_1w": (hist["Close"].iloc[-1] / hist["Close"].iloc[-5] - 1) * 100 if len(hist) > 5 else 0,
                            "change_1m": (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100,
                        }
                except Exception:
                    pass

            # Add macro fetcher data if available
            if self.macro_fetcher:
                try:
                    macro_indicators = self.macro_fetcher.get_indicators()
                    data.update(macro_indicators)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error gathering macro data: {e}")

        return data

    def _analyze_macro_conditions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze macro conditions and determine outlook."""
        analysis = {
            "overall_outlook": "Neutral",
            "risk_elevated": False,
            "signals": [],
        }

        # Check VIX level
        if "vix" in data:
            vix_level = data["vix"].get("current", 20)
            if vix_level > 30:
                analysis["risk_elevated"] = True
                analysis["signals"].append(f"High VIX ({vix_level:.1f})")
                analysis["overall_outlook"] = "Cautious"
            elif vix_level < 15:
                analysis["signals"].append(f"Low VIX ({vix_level:.1f}) - complacency risk")

        # Check market trend
        if "spy" in data:
            spy_1m = data["spy"].get("change_1m", 0)
            if spy_1m > 5:
                analysis["signals"].append(f"Strong market momentum (+{spy_1m:.1f}% 1M)")
                if analysis["overall_outlook"] == "Neutral":
                    analysis["overall_outlook"] = "Bullish"
            elif spy_1m < -5:
                analysis["signals"].append(f"Weak market ({spy_1m:.1f}% 1M)")
                analysis["overall_outlook"] = "Bearish"
                analysis["risk_elevated"] = True

        # Check bond market
        if "tlt" in data:
            tlt_1m = data["tlt"].get("change_1m", 0)
            if tlt_1m > 3:
                analysis["signals"].append("Bond rally - flight to safety")
            elif tlt_1m < -3:
                analysis["signals"].append("Bond selloff - rising rates")

        return analysis

    def _gather_sector_data(self) -> Dict[str, Any]:
        """Gather sector performance data."""
        data = {}

        try:
            import yfinance as yf

            # Sector ETFs
            sectors = {
                "Technology": "XLK",
                "Healthcare": "XLV",
                "Financials": "XLF",
                "Consumer Discretionary": "XLY",
                "Consumer Staples": "XLP",
                "Energy": "XLE",
                "Utilities": "XLU",
                "Real Estate": "XLRE",
                "Materials": "XLB",
                "Industrials": "XLI",
                "Communication": "XLC",
            }

            for sector, etf in sectors.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period="3mo")
                    if not hist.empty:
                        data[sector] = {
                            "etf": etf,
                            "current": hist["Close"].iloc[-1],
                            "change_1d": (hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) * 100 if len(hist) > 1 else 0,
                            "change_1w": (hist["Close"].iloc[-1] / hist["Close"].iloc[-5] - 1) * 100 if len(hist) > 5 else 0,
                            "change_1m": (hist["Close"].iloc[-1] / hist["Close"].iloc[-21] - 1) * 100 if len(hist) > 21 else 0,
                            "change_3m": (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100,
                        }
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Error gathering sector data: {e}")

        return data

    def _detect_sector_rotation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect sector rotation patterns."""
        signals = {
            "trend": "Mixed",
            "leaders": [],
            "laggards": [],
            "rotation_detected": False,
        }

        if not data:
            return signals

        # Sort by 1-month performance
        sorted_sectors = sorted(
            data.items(),
            key=lambda x: x[1].get("change_1m", 0),
            reverse=True
        )

        # Top 3 leaders and bottom 3 laggards
        signals["leaders"] = [
            {"sector": s[0], "return_1m": s[1].get("change_1m", 0)}
            for s in sorted_sectors[:3]
        ]
        signals["laggards"] = [
            {"sector": s[0], "return_1m": s[1].get("change_1m", 0)}
            for s in sorted_sectors[-3:]
        ]

        # Detect rotation patterns
        defensive = ["Utilities", "Consumer Staples", "Healthcare"]
        cyclical = ["Technology", "Consumer Discretionary", "Financials"]

        def_returns = [data[s].get("change_1m", 0) for s in defensive if s in data]
        cyc_returns = [data[s].get("change_1m", 0) for s in cyclical if s in data]

        if def_returns and cyc_returns:
            avg_def = sum(def_returns) / len(def_returns)
            avg_cyc = sum(cyc_returns) / len(cyc_returns)

            if avg_def > avg_cyc + 3:
                signals["trend"] = "Defensive rotation"
                signals["rotation_detected"] = True
            elif avg_cyc > avg_def + 3:
                signals["trend"] = "Risk-on rotation"
                signals["rotation_detected"] = True

        return signals

    def _format_news_scan_report(
        self,
        all_news: List[Dict],
        significant_news: List[Dict],
        vix_alert: Optional[Dict]
    ) -> str:
        """Format news scan report."""
        lines = [
            "## News Scan Report",
            f"**Scan Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Total Items:** {len(all_news)}",
            f"**Significant Items:** {len(significant_news)}",
            "",
        ]

        if vix_alert:
            lines.extend([
                "### :rotating_light: VIX Alert",
                f"- Current VIX: {vix_alert['current']:.1f}",
                f"- Change: {vix_alert['change']:+.1f}%",
                f"- Threshold: {vix_alert['threshold']}",
                "",
            ])

        if significant_news:
            lines.append("### Significant News")
            for item in significant_news[:10]:
                sentiment = item.get("sentiment_score", 0)
                emoji = ":chart_with_upwards_trend:" if sentiment > 0 else ":chart_with_downwards_trend:"
                lines.append(f"\n{emoji} **{item.get('symbol', 'N/A')}**")
                lines.append(f"   {item.get('headline', 'No headline')[:100]}")
                lines.append(f"   Sentiment: {sentiment:+.2f}")

        return "\n".join(lines)

    def _format_earnings_news(self, earnings_news: List[Dict]) -> str:
        """Format earnings-related news."""
        lines = [
            "## Earnings News Alert",
            f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        for item in earnings_news:
            lines.append(f"**{item.get('symbol', 'N/A')}**: {item.get('headline', '')[:100]}")

        return "\n".join(lines)

    def _format_earnings_report(
        self,
        upcoming: List[Dict],
        surprises: List[Dict]
    ) -> str:
        """Format earnings report."""
        lines = [
            "## Earnings Report",
            f"**Check Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        if upcoming:
            lines.append("### Upcoming Earnings (Next 7 Days)")
            for item in upcoming:
                lines.append(f"- **{item['symbol']}**: {item['date']} ({item['days_until']} days)")

        if surprises:
            lines.extend(["", "### Recent Earnings Surprises"])
            for item in surprises:
                emoji = ":chart_with_upwards_trend:" if item["surprise_pct"] > 0 else ":chart_with_downwards_trend:"
                lines.append(f"- {emoji} **{item['symbol']}**: {item['surprise_pct']:+.1f}% surprise")

        return "\n".join(lines)

    def _format_macro_report(
        self,
        data: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        """Format macro analysis report."""
        lines = [
            "## Macro Analysis Report",
            f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Overall Outlook:** {analysis.get('overall_outlook', 'N/A')}",
            "",
            "### Key Indicators",
        ]

        if "spy" in data:
            lines.append(f"- **S&P 500**: {data['spy']['current']:,.0f} ({data['spy']['change_1d']:+.1f}% 1D, {data['spy']['change_1m']:+.1f}% 1M)")

        if "vix" in data:
            lines.append(f"- **VIX**: {data['vix']['current']:.1f} ({data['vix']['change_1d']:+.1f}% 1D)")

        if "tlt" in data:
            lines.append(f"- **TLT (Bonds)**: ${data['tlt']['current']:.2f} ({data['tlt']['change_1m']:+.1f}% 1M)")

        if "gold" in data:
            lines.append(f"- **Gold (GLD)**: ${data['gold']['current']:.2f} ({data['gold']['change_1m']:+.1f}% 1M)")

        if analysis.get("signals"):
            lines.extend(["", "### Analysis Signals"])
            for signal in analysis["signals"]:
                lines.append(f"- {signal}")

        return "\n".join(lines)

    def _format_sector_report(
        self,
        data: Dict[str, Any],
        rotation: Dict[str, Any]
    ) -> str:
        """Format sector analysis report."""
        lines = [
            "## Sector Analysis Report",
            f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Trend:** {rotation.get('trend', 'Mixed')}",
            "",
        ]

        if rotation.get("leaders"):
            lines.append("### Sector Leaders (1M)")
            for leader in rotation["leaders"]:
                lines.append(f"- **{leader['sector']}**: {leader['return_1m']:+.1f}%")

        if rotation.get("laggards"):
            lines.extend(["", "### Sector Laggards (1M)"])
            for laggard in rotation["laggards"]:
                lines.append(f"- **{laggard['sector']}**: {laggard['return_1m']:+.1f}%")

        if data:
            lines.extend(["", "### All Sectors (1M Performance)"])
            sorted_sectors = sorted(data.items(), key=lambda x: x[1].get("change_1m", 0), reverse=True)
            for sector, metrics in sorted_sectors:
                lines.append(f"- {sector}: {metrics.get('change_1m', 0):+.1f}%")

        return "\n".join(lines)

    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming messages from other agents.

        Args:
            message: Incoming message

        Returns:
            Optional response message
        """
        logger.info(f"Processing message: {message.subject}")

        # Handle queries for market data
        if message.message_type == MessageType.QUERY:
            return self._handle_query(message)

        # Acknowledge requests
        if message.message_type == MessageType.ACTION:
            return self.create_message(
                recipient=message.sender,
                message_type=MessageType.ACKNOWLEDGMENT,
                subject=f"Acknowledged: {message.subject}",
                content="Request received and will be processed.",
                priority=MessagePriority.LOW,
                parent_message_id=message.id,
            )

        return None

    def _handle_query(self, message: AgentMessage) -> AgentMessage:
        """Handle queries from other agents."""
        query_type = message.context.get("query_type", "general")

        if query_type == "market_conditions":
            macro_data = self._gather_macro_data()
            analysis = self._analyze_macro_conditions(macro_data)
            content = self._format_macro_report(macro_data, analysis)
        elif query_type == "sector_performance":
            sector_data = self._gather_sector_data()
            rotation = self._detect_sector_rotation(sector_data)
            content = self._format_sector_report(sector_data, rotation)
        elif query_type == "vix_status":
            vix_alert = self._check_vix_spike()
            if vix_alert:
                content = f"VIX Alert: {vix_alert['current']:.1f} (change: {vix_alert['change']:+.1f}%)"
            else:
                content = "VIX within normal range"
        else:
            content = f"Unknown query type: {query_type}"

        return self.create_message(
            recipient=message.sender,
            message_type=MessageType.RESPONSE,
            subject=f"Re: {message.subject}",
            content=content,
            priority=MessagePriority.NORMAL,
            parent_message_id=message.id,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get agent status including last check times."""
        status = super().get_status()
        status.update({
            "last_news_scan": self._last_news_scan.isoformat() if self._last_news_scan else None,
            "last_earnings_check": self._last_earnings_check.isoformat() if self._last_earnings_check else None,
            "last_macro_analysis": self._last_macro_analysis.isoformat() if self._last_macro_analysis else None,
            "last_sector_analysis": self._last_sector_analysis.isoformat() if self._last_sector_analysis else None,
            "config": {
                "news_scan_hours": self.news_scan_hours,
                "earnings_check_hours": self.earnings_check_hours,
                "macro_analysis_hours": self.macro_analysis_hours,
                "alert_on_vix_spike": self.alert_on_vix_spike,
            },
        })
        return status
