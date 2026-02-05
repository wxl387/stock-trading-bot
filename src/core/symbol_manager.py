"""
Symbol Manager

Manages dynamic addition and removal of symbols at runtime.
Tracks symbol performance, enforces constraints, and handles cooldown periods.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import threading

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

# Module-level singleton
_symbol_manager: Optional["SymbolManager"] = None
_lock = threading.Lock()


def get_symbol_manager(config: Optional[Dict] = None) -> "SymbolManager":
    """Get or create the singleton SymbolManager instance."""
    global _symbol_manager
    with _lock:
        if _symbol_manager is None:
            _symbol_manager = SymbolManager(config or {})
        return _symbol_manager


@dataclass
class SymbolEntry:
    """Tracking entry for a managed symbol."""
    symbol: str
    added_date: datetime
    add_reason: str = ""
    add_score: float = 0.0
    sector: str = ""

    # Performance tracking
    entry_price: float = 0.0
    current_price: float = 0.0
    total_return: float = 0.0
    days_held: int = 0

    # Status
    is_active: bool = True
    removal_date: Optional[datetime] = None
    removal_reason: str = ""

    # Cooldown tracking
    cooldown_until: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "added_date": self.added_date.isoformat(),
            "add_reason": self.add_reason,
            "add_score": self.add_score,
            "sector": self.sector,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "total_return": self.total_return,
            "days_held": self.days_held,
            "is_active": self.is_active,
            "removal_date": self.removal_date.isoformat() if self.removal_date else None,
            "removal_reason": self.removal_reason,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolEntry":
        return cls(
            symbol=data["symbol"],
            added_date=datetime.fromisoformat(data["added_date"]),
            add_reason=data.get("add_reason", ""),
            add_score=data.get("add_score", 0.0),
            sector=data.get("sector", ""),
            entry_price=data.get("entry_price", 0.0),
            current_price=data.get("current_price", 0.0),
            total_return=data.get("total_return", 0.0),
            days_held=data.get("days_held", 0),
            is_active=data.get("is_active", True),
            removal_date=datetime.fromisoformat(data["removal_date"]) if data.get("removal_date") else None,
            removal_reason=data.get("removal_reason", ""),
            cooldown_until=datetime.fromisoformat(data["cooldown_until"]) if data.get("cooldown_until") else None,
        )


@dataclass
class SymbolReviewResult:
    """Result of a symbol review."""
    timestamp: datetime = field(default_factory=datetime.now)
    symbols_added: List[str] = field(default_factory=list)
    symbols_removed: List[str] = field(default_factory=list)
    underperformers: List[str] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    sector_exposure: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbols_added": self.symbols_added,
            "symbols_removed": self.symbols_removed,
            "underperformers": self.underperformers,
            "recommendations": self.recommendations,
            "sector_exposure": self.sector_exposure,
        }


class SymbolManager:
    """
    Manages dynamic addition and removal of trading symbols.

    Features:
    - Add/remove symbols at runtime
    - Track symbol performance
    - Enforce constraints (min/max symbols, sector exposure)
    - Cooldown periods for removed symbols
    - Persist state across restarts
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SymbolManager.

        Args:
            config: Configuration dictionary with dynamic_symbols settings
        """
        self.config = config
        ds_config = config.get("dynamic_symbols", {})

        # Constraints
        self.min_symbols = ds_config.get("constraints", {}).get("min_symbols", 5)
        self.max_symbols = ds_config.get("constraints", {}).get("max_symbols", 20)
        self.max_sector_exposure = ds_config.get("constraints", {}).get("max_sector_exposure", 0.30)
        self.max_single_stock = ds_config.get("constraints", {}).get("max_single_stock", 0.15)

        # Cooldown
        self.cooldown_days = ds_config.get("cooldown_days", 30)

        # Entry/Exit thresholds
        entry_config = ds_config.get("entry", {})
        self.min_entry_score = entry_config.get("min_score", 70)
        self.require_technical_confirmation = entry_config.get("require_technical_confirmation", True)

        exit_config = ds_config.get("exit", {})
        self.underperformance_threshold = exit_config.get("underperformance_threshold", -0.10)
        self.max_holding_days = exit_config.get("max_holding_days", 90)
        self.loss_threshold = exit_config.get("loss_threshold", -0.15)

        # State file
        self.state_file = DATA_DIR / "symbol_manager_state.json"

        # Active symbols tracking
        self._symbols: Dict[str, SymbolEntry] = {}
        self._history: List[SymbolEntry] = []  # Removed symbols

        # Load existing state
        self._load_state()

        logger.info(f"SymbolManager initialized: {len(self.get_active_symbols())} active symbols, "
                   f"constraints: {self.min_symbols}-{self.max_symbols} symbols")

    def add_symbol(
        self,
        symbol: str,
        reason: str,
        score: float,
        sector: str = "",
        entry_price: float = 0.0
    ) -> bool:
        """
        Add a symbol to the active trading universe.

        Args:
            symbol: Stock ticker symbol
            reason: Reason for adding
            score: Screening score (0-100)
            sector: Stock sector
            entry_price: Price at time of addition

        Returns:
            True if added successfully, False otherwise
        """
        # Check if already active
        if symbol in self._symbols and self._symbols[symbol].is_active:
            logger.warning(f"Symbol {symbol} is already active")
            return False

        # Check maximum symbols constraint
        active_count = len(self.get_active_symbols())
        if active_count >= self.max_symbols:
            logger.warning(f"Cannot add {symbol}: at max symbols ({self.max_symbols})")
            return False

        # Check minimum score
        if score < self.min_entry_score:
            logger.warning(f"Cannot add {symbol}: score {score:.1f} below minimum {self.min_entry_score}")
            return False

        # Check cooldown
        if not self._check_cooldown(symbol):
            cooldown_entry = self._get_cooldown_entry(symbol)
            if cooldown_entry:
                logger.warning(f"Cannot add {symbol}: on cooldown until {cooldown_entry.cooldown_until}")
            return False

        # Check sector exposure
        if sector and not self._check_sector_exposure(sector):
            logger.warning(f"Cannot add {symbol}: sector {sector} at max exposure ({self.max_sector_exposure:.0%})")
            return False

        # Create entry
        entry = SymbolEntry(
            symbol=symbol,
            added_date=datetime.now(),
            add_reason=reason,
            add_score=score,
            sector=sector,
            entry_price=entry_price,
            current_price=entry_price,
        )

        self._symbols[symbol] = entry
        self._save_state()

        logger.info(f"Added symbol {symbol} (score={score:.1f}, reason={reason})")
        return True

    def remove_symbol(
        self,
        symbol: str,
        reason: str,
        apply_cooldown: bool = True
    ) -> bool:
        """
        Remove a symbol from the active trading universe.

        Args:
            symbol: Stock ticker symbol
            reason: Reason for removal
            apply_cooldown: Whether to apply cooldown before re-adding

        Returns:
            True if removed successfully, False otherwise
        """
        if symbol not in self._symbols or not self._symbols[symbol].is_active:
            logger.warning(f"Symbol {symbol} is not active")
            return False

        # Check minimum symbols constraint
        active_count = len(self.get_active_symbols())
        if active_count <= self.min_symbols:
            logger.warning(f"Cannot remove {symbol}: at min symbols ({self.min_symbols})")
            return False

        # Update entry
        entry = self._symbols[symbol]
        entry.is_active = False
        entry.removal_date = datetime.now()
        entry.removal_reason = reason
        entry.days_held = (datetime.now() - entry.added_date).days

        if apply_cooldown:
            entry.cooldown_until = datetime.now() + timedelta(days=self.cooldown_days)

        # Move to history
        self._history.append(entry)

        self._save_state()

        logger.info(f"Removed symbol {symbol} (reason={reason}, held {entry.days_held} days)")
        return True

    def get_active_symbols(self) -> List[str]:
        """Get list of currently active symbols."""
        return [s for s, e in self._symbols.items() if e.is_active]

    def get_symbol_entry(self, symbol: str) -> Optional[SymbolEntry]:
        """Get entry for a specific symbol."""
        return self._symbols.get(symbol)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for all symbols.

        Args:
            prices: Dictionary mapping symbol to current price
        """
        for symbol, price in prices.items():
            if symbol in self._symbols:
                entry = self._symbols[symbol]
                entry.current_price = price
                if entry.entry_price > 0:
                    entry.total_return = (price - entry.entry_price) / entry.entry_price
                entry.days_held = (datetime.now() - entry.added_date).days

        self._save_state()

    def review_symbols(
        self,
        benchmark_return: float = 0.0
    ) -> SymbolReviewResult:
        """
        Review all active symbols for potential removal.

        Args:
            benchmark_return: Benchmark return to compare against (e.g., SPY return)

        Returns:
            SymbolReviewResult with underperformers and recommendations
        """
        result = SymbolReviewResult()

        active_symbols = self.get_active_symbols()

        for symbol in active_symbols:
            entry = self._symbols[symbol]

            # Check for underperformance vs benchmark
            relative_return = entry.total_return - benchmark_return
            if relative_return < self.underperformance_threshold:
                result.underperformers.append(symbol)
                result.recommendations.append({
                    "symbol": symbol,
                    "action": "consider_removal",
                    "reason": f"Underperforming benchmark by {relative_return:.1%}",
                    "return": entry.total_return,
                    "days_held": entry.days_held,
                })

            # Check for loss threshold
            if entry.total_return < self.loss_threshold:
                result.recommendations.append({
                    "symbol": symbol,
                    "action": "remove",
                    "reason": f"Loss exceeds threshold ({entry.total_return:.1%} < {self.loss_threshold:.1%})",
                    "return": entry.total_return,
                    "days_held": entry.days_held,
                })

            # Check for max holding days
            if entry.days_held > self.max_holding_days:
                result.recommendations.append({
                    "symbol": symbol,
                    "action": "consider_removal",
                    "reason": f"Held for {entry.days_held} days (max: {self.max_holding_days})",
                    "return": entry.total_return,
                    "days_held": entry.days_held,
                })

        # Calculate sector exposure
        result.sector_exposure = self.get_sector_exposure()

        return result

    def get_sector_exposure(self) -> Dict[str, float]:
        """
        Calculate current sector exposure.

        Returns:
            Dictionary mapping sector to exposure percentage
        """
        active_symbols = self.get_active_symbols()
        if not active_symbols:
            return {}

        sector_counts: Dict[str, int] = {}
        for symbol in active_symbols:
            entry = self._symbols.get(symbol)
            if entry and entry.sector:
                sector_counts[entry.sector] = sector_counts.get(entry.sector, 0) + 1

        total = len(active_symbols)
        return {sector: count / total for sector, count in sector_counts.items()}

    def can_add_to_sector(self, sector: str) -> bool:
        """Check if a symbol can be added to a specific sector."""
        return self._check_sector_exposure(sector)

    def get_cooldown_symbols(self) -> List[str]:
        """Get symbols currently on cooldown."""
        now = datetime.now()
        return [
            e.symbol for e in self._history
            if e.cooldown_until and e.cooldown_until > now
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        active_symbols = self.get_active_symbols()
        returns = [self._symbols[s].total_return for s in active_symbols if self._symbols[s].total_return != 0]

        return {
            "active_count": len(active_symbols),
            "total_tracked": len(self._symbols),
            "history_count": len(self._history),
            "cooldown_count": len(self.get_cooldown_symbols()),
            "avg_return": sum(returns) / len(returns) if returns else 0.0,
            "avg_days_held": sum(self._symbols[s].days_held for s in active_symbols) / len(active_symbols) if active_symbols else 0,
            "sector_exposure": self.get_sector_exposure(),
        }

    def _check_cooldown(self, symbol: str) -> bool:
        """Check if symbol is past cooldown period."""
        # Check in history for cooldown
        for entry in self._history:
            if entry.symbol == symbol and entry.cooldown_until:
                if datetime.now() < entry.cooldown_until:
                    return False
        return True

    def _get_cooldown_entry(self, symbol: str) -> Optional[SymbolEntry]:
        """Get the cooldown entry for a symbol if it exists."""
        for entry in self._history:
            if entry.symbol == symbol and entry.cooldown_until:
                if datetime.now() < entry.cooldown_until:
                    return entry
        return None

    def _check_sector_exposure(self, sector: str) -> bool:
        """Check if adding to a sector would exceed max exposure."""
        if not sector:
            return True

        current_exposure = self.get_sector_exposure()
        sector_exposure = current_exposure.get(sector, 0.0)

        # Calculate what exposure would be after adding
        active_count = len(self.get_active_symbols())
        new_exposure = (sector_exposure * active_count + 1) / (active_count + 1)

        return new_exposure <= self.max_sector_exposure

    def _save_state(self) -> None:
        """Save state to file."""
        try:
            state = {
                "symbols": {s: e.to_dict() for s, e in self._symbols.items()},
                "history": [e.to_dict() for e in self._history[-100:]],  # Keep last 100
                "timestamp": datetime.now().isoformat(),
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _load_state(self) -> None:
        """Load state from file."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Load symbols
            for symbol, data in state.get("symbols", {}).items():
                self._symbols[symbol] = SymbolEntry.from_dict(data)

            # Load history
            for data in state.get("history", []):
                self._history.append(SymbolEntry.from_dict(data))

            logger.info(f"Loaded state: {len(self._symbols)} symbols, {len(self._history)} history entries")

        except Exception as e:
            logger.error(f"Error loading state: {e}")

    def initialize_with_symbols(
        self,
        symbols: List[str],
        reason: str = "initial",
        sectors: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Initialize with a list of symbols (for migration from static to dynamic).

        Args:
            symbols: List of symbols to add
            reason: Reason for adding
            sectors: Optional mapping of symbol to sector

        Returns:
            Number of symbols successfully added
        """
        added = 0
        sectors = sectors or {}

        for symbol in symbols:
            # Skip if already exists
            if symbol in self._symbols:
                continue

            entry = SymbolEntry(
                symbol=symbol,
                added_date=datetime.now(),
                add_reason=reason,
                add_score=100.0,  # Initial symbols get max score
                sector=sectors.get(symbol, ""),
            )

            self._symbols[symbol] = entry
            added += 1

        self._save_state()
        logger.info(f"Initialized with {added} symbols")
        return added

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "enabled": self.config.get("dynamic_symbols", {}).get("enabled", False),
            "active_symbols": self.get_active_symbols(),
            "active_count": len(self.get_active_symbols()),
            "min_symbols": self.min_symbols,
            "max_symbols": self.max_symbols,
            "cooldown_days": self.cooldown_days,
            "statistics": self.get_statistics(),
        }
