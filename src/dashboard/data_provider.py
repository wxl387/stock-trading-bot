"""
Data provider for the trading dashboard.
Reads broker state and provides formatted data for display.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

from config.settings import DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

# File paths
BROKER_STATE_FILE = DATA_DIR / "simulated_broker_state.json"
PORTFOLIO_HISTORY_FILE = DATA_DIR / "portfolio_history.json"
MODEL_REGISTRY_FILE = MODELS_DIR / "versions" / "registry.json"
RETRAINING_LOG_FILE = MODELS_DIR / "retraining_log.json"


class DashboardDataProvider:
    """
    Provides data for the trading dashboard.
    Reads from broker state file without interfering with running bot.
    """

    def __init__(self):
        """Initialize data provider."""
        self._broker_state: Optional[Dict] = None
        self._portfolio_history: Optional[List[Dict]] = None
        self._last_load_time: Optional[datetime] = None

    def refresh(self) -> None:
        """Force refresh data from files."""
        self._broker_state = None
        self._portfolio_history = None
        self._load_broker_state()
        self._load_portfolio_history()

    def _load_broker_state(self) -> Dict:
        """Load broker state from JSON file."""
        if self._broker_state is not None:
            return self._broker_state

        try:
            if BROKER_STATE_FILE.exists():
                with open(BROKER_STATE_FILE, "r") as f:
                    self._broker_state = json.load(f)
                self._last_load_time = datetime.now()
            else:
                self._broker_state = self._get_empty_state()
        except Exception as e:
            logger.error(f"Error loading broker state: {e}")
            self._broker_state = self._get_empty_state()

        return self._broker_state

    def _load_portfolio_history(self) -> List[Dict]:
        """Load portfolio history from JSON file."""
        if self._portfolio_history is not None:
            return self._portfolio_history

        try:
            if PORTFOLIO_HISTORY_FILE.exists():
                with open(PORTFOLIO_HISTORY_FILE, "r") as f:
                    data = json.load(f)
                    self._portfolio_history = data.get("history", [])
            else:
                self._portfolio_history = []
        except Exception as e:
            logger.error(f"Error loading portfolio history: {e}")
            self._portfolio_history = []

        return self._portfolio_history

    def _get_empty_state(self) -> Dict:
        """Return empty broker state structure."""
        return {
            "account_id": "N/A",
            "initial_capital": 100000,
            "cash": 100000,
            "positions": {},
            "realized_pnl": 0,
            "trades": []
        }

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """
        Get portfolio metrics for dashboard display.

        Returns:
            Dict with portfolio_value, cash, total_pnl, total_pnl_pct, etc.
        """
        state = self._load_broker_state()

        initial_capital = state.get("initial_capital", 100000)
        cash = state.get("cash", 0)
        positions = state.get("positions", {})
        realized_pnl = state.get("realized_pnl", 0)

        # Calculate positions value
        positions_value = sum(
            pos.get("quantity", 0) * pos.get("last_price", pos.get("avg_cost", 0))
            for pos in positions.values()
        )

        # Calculate unrealized P&L
        unrealized_pnl = sum(
            (pos.get("last_price", pos.get("avg_cost", 0)) - pos.get("avg_cost", 0)) * pos.get("quantity", 0)
            for pos in positions.values()
        )

        portfolio_value = cash + positions_value
        total_pnl = portfolio_value - initial_capital
        total_pnl_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0

        # Calculate trade metrics
        trades = state.get("trades", [])
        win_rate = self._calculate_win_rate(trades)

        return {
            "portfolio_value": portfolio_value,
            "cash": cash,
            "positions_value": positions_value,
            "initial_capital": initial_capital,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "num_positions": len(positions),
            "win_rate": win_rate,
            "total_trades": len(trades),
            "last_updated": self._last_load_time.isoformat() if self._last_load_time else "N/A"
        }

    def get_positions(self) -> pd.DataFrame:
        """
        Get current positions as DataFrame.

        Returns:
            DataFrame with position details.
        """
        state = self._load_broker_state()
        positions = state.get("positions", {})

        if not positions:
            return pd.DataFrame(columns=[
                "Symbol", "Shares", "Avg Cost", "Current Price",
                "Market Value", "P&L ($)", "P&L (%)"
            ])

        rows = []
        for symbol, pos in positions.items():
            quantity = pos.get("quantity", 0)
            avg_cost = pos.get("avg_cost", 0)
            current_price = pos.get("last_price", avg_cost)
            market_value = quantity * current_price
            pnl_dollar = (current_price - avg_cost) * quantity
            pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0

            rows.append({
                "Symbol": symbol,
                "Shares": quantity,
                "Avg Cost": avg_cost,
                "Current Price": current_price,
                "Market Value": market_value,
                "P&L ($)": pnl_dollar,
                "P&L (%)": pnl_pct
            })

        df = pd.DataFrame(rows)
        return df.sort_values("Market Value", ascending=False)

    def get_trade_history(self, limit: int = 100) -> pd.DataFrame:
        """
        Get trade history as DataFrame.

        Args:
            limit: Maximum number of trades to return.

        Returns:
            DataFrame with trade history.
        """
        state = self._load_broker_state()
        trades = state.get("trades", [])

        if not trades:
            return pd.DataFrame(columns=[
                "Date", "Symbol", "Action", "Shares", "Price", "Total"
            ])

        rows = []
        for trade in trades[-limit:]:
            timestamp = trade.get("timestamp", "")
            if timestamp:
                try:
                    date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date = timestamp
            else:
                date = "N/A"

            rows.append({
                "Date": date,
                "Symbol": trade.get("symbol", "N/A"),
                "Action": trade.get("side", "N/A"),
                "Shares": trade.get("quantity", 0),
                "Price": trade.get("price", 0),
                "Total": trade.get("quantity", 0) * trade.get("price", 0)
            })

        df = pd.DataFrame(rows)
        return df.iloc[::-1]  # Most recent first

    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get portfolio value history for charting.

        Returns:
            DataFrame with date and portfolio_value columns.
        """
        history = self._load_portfolio_history()

        if not history:
            # Return demo data if no history exists
            return self._generate_demo_history()

        rows = []
        for entry in history:
            rows.append({
                "Date": pd.to_datetime(entry.get("date")),
                "Portfolio Value": entry.get("portfolio_value", 0),
                "Cash": entry.get("cash", 0),
                "Total P&L": entry.get("total_pnl", 0)
            })

        return pd.DataFrame(rows)

    def _generate_demo_history(self) -> pd.DataFrame:
        """Generate demo history based on current state."""
        state = self._load_broker_state()
        current_value = state.get("cash", 100000)

        # Add positions value
        for pos in state.get("positions", {}).values():
            current_value += pos.get("quantity", 0) * pos.get("last_price", pos.get("avg_cost", 0))

        initial = state.get("initial_capital", 100000)

        # Generate 30 days of history leading to current value
        dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
        values = []
        for i, date in enumerate(dates):
            # Interpolate from initial to current with some noise
            progress = i / (len(dates) - 1)
            value = initial + (current_value - initial) * progress
            # Add small random variation
            import random
            variation = random.uniform(-0.005, 0.005) * value
            values.append(value + variation)

        return pd.DataFrame({
            "Date": dates,
            "Portfolio Value": values,
            "Cash": [state.get("cash", 100000)] * len(dates),
            "Total P&L": [v - initial for v in values]
        })

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0

        # Group trades by symbol to pair buys and sells
        # This is a simplified calculation
        buy_prices = {}
        wins = 0
        total_completed = 0

        for trade in trades:
            symbol = trade.get("symbol")
            side = trade.get("side", "").upper()
            price = trade.get("price", 0)

            if side == "BUY":
                if symbol not in buy_prices:
                    buy_prices[symbol] = []
                buy_prices[symbol].append(price)
            elif side == "SELL" and symbol in buy_prices and buy_prices[symbol]:
                avg_buy = sum(buy_prices[symbol]) / len(buy_prices[symbol])
                if price > avg_buy:
                    wins += 1
                total_completed += 1
                buy_prices[symbol] = []

        return (wins / total_completed * 100) if total_completed > 0 else 0.0

    def get_allocation_data(self) -> pd.DataFrame:
        """
        Get position allocation data for pie chart.

        Returns:
            DataFrame with Symbol and Allocation columns.
        """
        positions_df = self.get_positions()

        if positions_df.empty:
            return pd.DataFrame(columns=["Symbol", "Allocation"])

        total_value = positions_df["Market Value"].sum()
        if total_value == 0:
            return pd.DataFrame(columns=["Symbol", "Allocation"])

        positions_df["Allocation"] = positions_df["Market Value"] / total_value * 100

        return positions_df[["Symbol", "Allocation", "Market Value"]]

    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model status and retraining info.

        Returns:
            Dict with model versions, accuracy, and retraining schedule.
        """
        status = {
            "production_models": {},
            "last_retrain": None,
            "next_retrain": None,
            "retraining_enabled": False,
            "recent_retrains": []
        }

        # Load model registry
        try:
            if MODEL_REGISTRY_FILE.exists():
                with open(MODEL_REGISTRY_FILE, "r") as f:
                    registry = json.load(f)

                production = registry.get("production", {})
                for model_type, info in production.items():
                    status["production_models"][model_type] = {
                        "version": info.get("version", "N/A"),
                        "deployed_at": info.get("deployed_at", "N/A"),
                        "accuracy": info.get("metrics", {}).get("accuracy", 0) * 100
                    }
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")

        # Load retraining log
        try:
            if RETRAINING_LOG_FILE.exists():
                with open(RETRAINING_LOG_FILE, "r") as f:
                    log = json.load(f)

                history = log.get("history", [])
                if history:
                    # Most recent retrain
                    latest = history[-1]
                    status["last_retrain"] = latest.get("completed_at", latest.get("started_at"))

                    # Recent retrains (last 5)
                    status["recent_retrains"] = history[-5:][::-1]
        except Exception as e:
            logger.error(f"Error loading retraining log: {e}")

        # Load retraining config
        try:
            from config.settings import Settings
            config = Settings.load_trading_config()
            retraining_config = config.get("retraining", {})
            status["retraining_enabled"] = retraining_config.get("enabled", False)
            status["schedule"] = retraining_config.get("schedule", "weekly")
            status["day_of_week"] = retraining_config.get("day_of_week", "sun")
            status["hour"] = retraining_config.get("hour", 2)
        except Exception as e:
            logger.error(f"Error loading retraining config: {e}")

        return status

    def save_portfolio_snapshot(self) -> None:
        """Save current portfolio state to history file."""
        metrics = self.get_portfolio_metrics()
        state = self._load_broker_state()

        snapshot = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "portfolio_value": metrics["portfolio_value"],
            "cash": metrics["cash"],
            "positions_value": metrics["positions_value"],
            "total_pnl": metrics["total_pnl"],
            "realized_pnl": metrics["realized_pnl"],
            "unrealized_pnl": metrics["unrealized_pnl"],
            "num_positions": metrics["num_positions"]
        }

        # Load existing history
        history = self._load_portfolio_history()

        # Check if we already have an entry for today
        today = snapshot["date"]
        history = [h for h in history if h.get("date") != today]

        # Add new snapshot
        history.append(snapshot)

        # Keep last 365 days
        history = history[-365:]

        # Save to file and update in-memory cache
        try:
            with open(PORTFOLIO_HISTORY_FILE, "w") as f:
                json.dump({"history": history}, f, indent=2)
            self._portfolio_history = history
            logger.info(f"Saved portfolio snapshot for {today}")
        except Exception as e:
            logger.error(f"Error saving portfolio history: {e}")


# Singleton instance
_data_provider: Optional[DashboardDataProvider] = None


def get_data_provider() -> DashboardDataProvider:
    """Get singleton data provider instance."""
    global _data_provider
    if _data_provider is None:
        _data_provider = DashboardDataProvider()
    return _data_provider
