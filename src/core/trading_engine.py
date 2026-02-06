"""
Main trading engine that orchestrates all components.
"""
import logging
import signal
import time
from datetime import datetime, time as dt_time
from typing import List, Dict, Optional
import schedule

from config.settings import settings, Settings
from src.broker.base_broker import BaseBroker, OrderSide, OrderStatus, OrderType
from src.broker.simulated_broker import SimulatedBroker
from src.strategy.ml_strategy import MLStrategy, SignalType
from src.risk.risk_manager import RiskManager, StopLossType
from src.risk.regime_detector import RegimeDetector, get_regime_detector, MarketRegime
from src.ml.scheduled_retrainer import ScheduledRetrainer, get_scheduled_retrainer
from src.notifications.notifier_manager import NotifierManager, get_notifier
from src.portfolio.portfolio_optimizer import PortfolioOptimizer, OptimizationMethod
from src.portfolio.rebalancer import PortfolioRebalancer, RebalanceTrigger

# Optional import for AgentOrchestrator
try:
    from src.agents.orchestrator import AgentOrchestrator, get_orchestrator
    AGENTS_AVAILABLE = True
except ImportError:
    AgentOrchestrator = None
    get_orchestrator = None
    AGENTS_AVAILABLE = False

# Optional import for SymbolManager
try:
    from src.core.symbol_manager import SymbolManager, get_symbol_manager
    SYMBOL_MANAGER_AVAILABLE = True
except ImportError:
    SymbolManager = None
    get_symbol_manager = None
    SYMBOL_MANAGER_AVAILABLE = False

# Optional import for WebullBroker (requires webull package)
try:
    from src.broker.webull_broker import WebullBroker
    WEBULL_AVAILABLE = True
except ImportError:
    WebullBroker = None
    WEBULL_AVAILABLE = False

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Main trading engine that orchestrates broker, strategy, and risk management.
    """

    def __init__(
        self,
        symbols: List[str],
        paper_trading: bool = True,
        simulated: bool = False,
        initial_capital: float = 100000.0,
        ignore_market_hours: bool = False,
        use_ensemble: bool = False
    ):
        """
        Initialize TradingEngine.

        Args:
            symbols: List of stock symbols to trade.
            paper_trading: Whether to use paper trading mode (WeBull).
            simulated: Whether to use simulated broker (no credentials needed).
            initial_capital: Initial capital for simulated broker.
            ignore_market_hours: Whether to ignore market hours check (for testing).
            use_ensemble: Whether to use ensemble model (XGBoost + LSTM + CNN).
        """
        self.symbols = symbols
        self.paper_trading = paper_trading
        self.simulated = simulated
        self.initial_capital = initial_capital
        self.ignore_market_hours = ignore_market_hours
        self.use_ensemble = use_ensemble
        self.is_running = False

        # Load config
        self.config = Settings.load_trading_config()

        # Initialize components
        self.broker: Optional[BaseBroker] = None
        ml_config = self.config.get("ml_model", {})
        self.strategy = MLStrategy(
            confidence_threshold=ml_config.get("confidence_threshold", 0.6),
            min_confidence_sell=ml_config.get("min_confidence_sell", 0.55),
        )
        self.risk_manager = RiskManager(
            max_position_pct=self.config.get("risk_management", {}).get("max_position_pct", 0.10),
            max_daily_loss_pct=self.config.get("risk_management", {}).get("max_daily_loss_pct", 0.05),
            max_total_exposure=self.config.get("risk_management", {}).get("max_total_exposure", 0.80)
        )

        # Market hours (Eastern Time)
        self.market_open = dt_time(9, 30)
        self.market_close = dt_time(16, 0)

        # Initialize scheduled retrainer
        self.retrainer: Optional[ScheduledRetrainer] = None
        retraining_config = self.config.get("retraining", {})
        if retraining_config.get("enabled", False):
            self.retrainer = get_scheduled_retrainer(self.config)
            logger.info("Scheduled retraining enabled")

        # Initialize notifier
        self.notifier: Optional[NotifierManager] = None
        if self.config.get("notifications", {}).get("enabled", False):
            self.notifier = get_notifier()
            logger.info("Notifications enabled")

        # Initialize regime detector
        self.regime_detector: Optional[RegimeDetector] = None
        if self.config.get("risk_management", {}).get("regime_detection", {}).get("enabled", True):
            self.regime_detector = get_regime_detector()
            logger.info("Market regime detection enabled")

        # Initialize portfolio optimizer and rebalancer
        self.portfolio_optimizer: Optional[PortfolioOptimizer] = None
        self.portfolio_rebalancer: Optional[PortfolioRebalancer] = None
        portfolio_config = self.config.get("portfolio_optimization", {})
        if portfolio_config.get("enabled", False):
            # Initialize optimizer
            self.portfolio_optimizer = PortfolioOptimizer(
                lookback_days=portfolio_config.get("lookback_days", 252),
                min_weight=portfolio_config.get("min_weight", 0.0),
                max_weight=portfolio_config.get("max_weight", 0.30),
                risk_free_rate=portfolio_config.get("risk_free_rate", 0.05)
            )
            logger.info(f"Portfolio optimizer enabled: method={portfolio_config.get('method', 'max_sharpe')}")

            # Initialize rebalancer if enabled
            rebalancing_config = portfolio_config.get("rebalancing", {})
            if rebalancing_config.get("enabled", True):
                trigger_type = rebalancing_config.get("trigger_type", "combined")
                self.portfolio_rebalancer = PortfolioRebalancer(
                    drift_threshold=rebalancing_config.get("drift_threshold", 0.10),
                    calendar_frequency=rebalancing_config.get("frequency", "monthly"),
                    trigger_type=RebalanceTrigger[trigger_type.upper()],
                    min_trade_value=rebalancing_config.get("min_trade_value", 200.0),
                    max_trades_per_rebalance=rebalancing_config.get("max_trades_per_rebalance", 8)
                )
                logger.info(f"Portfolio rebalancing enabled: trigger={trigger_type}, drift={rebalancing_config.get('drift_threshold', 0.10)}")

        # Initialize agent orchestrator for multi-agent collaboration
        self.agent_orchestrator: Optional[AgentOrchestrator] = None
        if AGENTS_AVAILABLE and self.config.get("agents", {}).get("enabled", False):
            self.agent_orchestrator = get_orchestrator(self.config)
            logger.info("Agent orchestrator enabled")

        # Initialize symbol manager for dynamic symbol selection
        self.symbol_manager: Optional[SymbolManager] = None
        dynamic_symbols_config = self.config.get("dynamic_symbols", {})
        if SYMBOL_MANAGER_AVAILABLE and dynamic_symbols_config.get("enabled", False):
            self.symbol_manager = get_symbol_manager(self.config)
            # Ensure base config symbols are always in the symbol manager
            current_symbols = set(self.symbol_manager.get_active_symbols())
            missing_symbols = [s for s in symbols if s not in current_symbols]
            if missing_symbols:
                # Fetch sector info for missing symbols
                sectors = self._fetch_sectors_for_symbols(missing_symbols)
                self.symbol_manager.initialize_with_symbols(
                    symbols=missing_symbols,
                    reason="base_config_symbols",
                    sectors=sectors
                )
            logger.info(f"Symbol manager enabled: {len(self.symbol_manager.get_active_symbols())} active symbols")

        logger.info(f"Trading Engine initialized with {len(symbols)} symbols")
        if simulated:
            logger.info(f"Mode: SIMULATED (offline, ${initial_capital:,.2f} capital)")
        else:
            logger.info(f"Mode: {'Paper Trading' if paper_trading else 'LIVE TRADING'}")

    def start(self) -> None:
        """Start the trading engine."""
        logger.info("Starting Trading Engine...")

        # Initialize broker based on mode
        if self.simulated:
            self.broker = SimulatedBroker(initial_capital=self.initial_capital)
        else:
            if not WEBULL_AVAILABLE:
                logger.error("WebullBroker not available. Install 'webull' package or use --simulated mode.")
                raise ImportError("webull package not installed. Use --simulated mode instead.")
            self.broker = WebullBroker(paper_trading=self.paper_trading)

        # Connect to broker
        if not self.broker.connect():
            logger.error("Failed to connect to broker")
            raise ConnectionError("Broker connection failed")

        # Load ML model
        try:
            if self.use_ensemble:
                loaded = self.strategy.load_ensemble(
                    xgboost_name="trading_model",
                    lstm_name="lstm_trading_model",
                    cnn_name="cnn_trading_model"
                )
                logger.info(f"Loaded ensemble with models: {loaded}")
            else:
                self.strategy.load_model("trading_model")
        except FileNotFoundError:
            logger.warning("No trained model found. Please train a model first.")
            self.model_loaded = False
        else:
            self.model_loaded = True

        # Reset daily limits
        account = self.broker.get_account_info()
        self.risk_manager.reset_daily_limits(account.portfolio_value)

        self.is_running = True

        # Start scheduled retrainer if enabled
        if self.retrainer:
            self.retrainer.start()
            status = self.retrainer.get_status()
            logger.info(f"Scheduled retraining active. Next run: {status.get('next_run', 'N/A')}")

        # Start agent orchestrator if enabled
        if self.agent_orchestrator:
            self.agent_orchestrator.start()
            logger.info("Agent orchestrator started for multi-agent collaboration")

        logger.info("Trading Engine started successfully")

        # Send startup notification
        if self.notifier:
            mode = "simulated" if self.simulated else ("paper" if self.paper_trading else "live")
            model_type = "ensemble" if self.use_ensemble else "xgboost"
            self.notifier.notify_startup(
                mode=mode,
                symbols=self.symbols,
                model_type=model_type
            )

    def stop(self) -> None:
        """Stop the trading engine."""
        logger.info("Stopping Trading Engine...")
        self.is_running = False

        # Stop scheduled retrainer
        if self.retrainer:
            self.retrainer.stop()

        # Stop agent orchestrator
        if self.agent_orchestrator:
            self.agent_orchestrator.stop()
            logger.info("Agent orchestrator stopped")

        if self.broker:
            self.broker.disconnect()

        # Send shutdown notification
        if self.notifier:
            self.notifier.notify_shutdown(reason="Trading engine stopped")

        logger.info("Trading Engine stopped")

    def run_trading_cycle(self) -> Dict:
        """
        Run a single trading cycle.

        Returns:
            Dictionary with cycle results.
        """
        if not self.is_running:
            return {"status": "not_running"}

        if not self._is_market_open():
            return {"status": "market_closed"}

        cycle_results = {
            "timestamp": datetime.now(),
            "signals": {},
            "trades": [],
            "errors": []
        }

        try:
            # Get current account state
            account = self.broker.get_account_info()
            positions = self.broker.get_positions()
            current_positions = {p.symbol: p.quantity for p in positions}
            position_market_values = {p.symbol: p.market_value for p in positions}

            # Check if agents have halted trading
            if self.agent_orchestrator and self.agent_orchestrator.is_trading_halted():
                halt_reason = self.agent_orchestrator.get_halt_reason()
                logger.warning(f"Trading halted by agents: {halt_reason}")
                cycle_results["status"] = "blocked"
                cycle_results["block_reason"] = f"Agent halt: {halt_reason}"
                return cycle_results

            # Check if trading is allowed
            risk_check = self.risk_manager.check_can_trade()
            if not risk_check.approved:
                logger.warning(f"Trading blocked: {risk_check.reason}")
                cycle_results["status"] = "blocked"
                cycle_results["block_reason"] = risk_check.reason
                return cycle_results

            # Check drawdown protection
            drawdown_check = self.risk_manager.check_drawdown(account.portfolio_value)
            if not drawdown_check.approved:
                logger.warning(f"Trading blocked by drawdown: {drawdown_check.reason}")
                cycle_results["status"] = "blocked"
                cycle_results["block_reason"] = drawdown_check.reason

                # Send risk warning notification
                if self.notifier:
                    self.notifier.notify_risk_warning(
                        warning_type="Max Drawdown Breached",
                        details=drawdown_check.reason
                    )
                return cycle_results

            # Detect current market regime
            current_regime = None
            if self.regime_detector:
                current_regime = self.regime_detector.detect_regime()
                cycle_results["regime"] = current_regime.value if hasattr(current_regime, 'value') else str(current_regime)

            # Get current prices and check stop losses
            quotes = self.broker.get_quotes(list(current_positions.keys()))
            current_prices = {s: q.last for s, q in quotes.items()}

            # Check stop losses
            triggered_stops = self.risk_manager.check_stop_losses(current_prices)
            for symbol in triggered_stops:
                self._execute_stop_loss(symbol)
                cycle_results["trades"].append({
                    "type": "STOP_LOSS",
                    "symbol": symbol
                })

            # Check take-profits
            triggered_tps = self.risk_manager.check_take_profits(current_prices)
            for symbol, quantity, target_price in triggered_tps:
                self._execute_take_profit(symbol, quantity, target_price)
                cycle_results["trades"].append({
                    "type": "TAKE_PROFIT",
                    "symbol": symbol,
                    "quantity": quantity,
                    "target_price": target_price
                })

            # Update trailing stops
            for symbol in current_positions:
                if symbol in current_prices:
                    self.risk_manager.update_trailing_stop(symbol, current_prices[symbol])

            # Update symbol prices in symbol manager
            self._update_symbol_prices()

            # Get active trading symbols (from symbol manager if enabled)
            trading_symbols = self.get_active_symbols()

            # Portfolio optimization and rebalancing
            target_weights = None
            if self.portfolio_optimizer:
                # Get target portfolio weights
                target_weights = self._get_target_portfolio_weights()

                # Check if rebalancing is needed
                if target_weights and self.portfolio_rebalancer:
                    # Convert positions list to dict for rebalancer
                    positions_dict = {p.symbol: p for p in positions}
                    rebalance_signal = self.portfolio_rebalancer.check_rebalance_needed(
                        current_positions=positions_dict,
                        target_weights=target_weights,
                        portfolio_value=account.portfolio_value,
                        current_prices=current_prices
                    )

                    if rebalance_signal.should_rebalance:
                        logger.info(f"Rebalancing triggered: {rebalance_signal.reason}")
                        logger.info(f"  Drift: {rebalance_signal.drift_pct:.1%}, "
                                   f"Trades needed: {len(rebalance_signal.trades_needed)}")

                        # Execute rebalancing trades
                        for trade in rebalance_signal.trades_needed:
                            try:
                                # Handle both dict and object trades
                                t_action = trade.get('action') if isinstance(trade, dict) else trade.action
                                t_symbol = trade.get('symbol') if isinstance(trade, dict) else trade.symbol
                                t_shares = trade.get('shares') if isinstance(trade, dict) else trade.shares
                                t_price = trade.get('price', 0.0) if isinstance(trade, dict) else trade.price

                                if t_action == "BUY":
                                    order = self.broker.place_order(
                                        symbol=t_symbol,
                                        side=OrderSide.BUY,
                                        quantity=t_shares,
                                        order_type=OrderType.MARKET
                                    )
                                else:  # SELL
                                    order = self.broker.place_order(
                                        symbol=t_symbol,
                                        side=OrderSide.SELL,
                                        quantity=abs(t_shares),
                                        order_type=OrderType.MARKET
                                    )

                                if order:
                                    logger.info(f"Rebalance trade: {t_action} {t_shares} {t_symbol} @ ${t_price:.2f}")
                                    cycle_results["trades"].append({
                                        "type": "REBALANCE",
                                        "action": t_action,
                                        "symbol": t_symbol,
                                        "shares": abs(t_shares),
                                        "price": t_price
                                    })

                                    # Send notification
                                    if self.notifier:
                                        self.notifier.notify_trade(
                                            symbol=t_symbol,
                                            action=t_action,
                                            quantity=abs(t_shares),
                                            price=t_price,
                                            reason="Portfolio Rebalancing"
                                        )

                            except Exception as e:
                                logger.error(f"Rebalance trade error for {t_symbol}: {e}")
                                cycle_results["errors"].append(f"Rebalance {t_symbol}: {str(e)}")

            # Generate trading signals (use active symbols from symbol manager if available)
            if not getattr(self, 'model_loaded', False):
                logger.warning("Skipping signal generation — no ML model loaded")
                recommendations = []
            else:
                recommendations = self.strategy.get_trade_recommendations(
                    symbols=trading_symbols,
                    portfolio_value=account.portfolio_value,
                    current_positions=current_positions,
                    risk_manager=self.risk_manager,
                    target_weights=target_weights  # Pass target weights to strategy
                )

            # Execute trades
            for rec in recommendations:
                try:
                    trade_result = self._execute_trade(rec, account.portfolio_value, current_positions, position_market_values)
                    if trade_result:
                        cycle_results["trades"].append(trade_result)
                except Exception as e:
                    logger.error(f"Trade execution error: {e}")
                    cycle_results["errors"].append(str(e))

            cycle_results["status"] = "success"

        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
            cycle_results["status"] = "error"
            cycle_results["errors"].append(str(e))

            # Send error notification
            if self.notifier:
                self.notifier.notify_error(
                    error=f"Trading cycle error: {str(e)}"
                )

        return cycle_results

    def _execute_trade(
        self,
        recommendation: Dict,
        portfolio_value: float,
        current_positions: Dict[str, int],
        position_market_values: Optional[Dict[str, float]] = None
    ) -> Optional[Dict]:
        """
        Execute a trade recommendation.

        Args:
            recommendation: Trade recommendation dict.
            portfolio_value: Current portfolio value.
            current_positions: Current positions (symbol -> quantity).
            position_market_values: Current position market values (symbol -> $value).

        Returns:
            Trade result dict or None.
        """
        symbol = recommendation.get("symbol")
        action = recommendation.get("action")
        shares = recommendation.get("shares")
        price = recommendation.get("price")

        if not all([symbol, action, shares, price]):
            logger.warning(f"Malformed recommendation, skipping: {recommendation}")
            return None

        # Defense-in-depth: check agent halt before each trade
        if self.agent_orchestrator and self.agent_orchestrator.is_trading_halted():
            logger.warning(f"Trade for {symbol} blocked: agents have halted trading")
            return None

        # Risk check for buys
        if action == "BUY":
            risk_check = self.risk_manager.check_position(
                symbol=symbol,
                quantity=shares,
                price=price,
                portfolio_value=portfolio_value,
                current_positions=position_market_values or {s: qty * price for s, qty in current_positions.items()}
            )

            if not risk_check.approved:
                logger.warning(f"Trade rejected by risk manager: {risk_check.reason}")
                if risk_check.adjusted_quantity and risk_check.adjusted_quantity > 0:
                    shares = risk_check.adjusted_quantity
                    logger.info(f"Adjusted quantity to {shares} shares")
                else:
                    return None

        # Execute order
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL

        order = self.broker.place_order(
            symbol=symbol,
            side=side,
            quantity=shares,
            order_type=OrderType.MARKET
        )

        # Check if order was actually filled (not rejected)
        if order.is_filled():
            logger.info(f"Executed {action} order for {shares} {symbol}")
        elif order.status == OrderStatus.PARTIALLY_FILLED and order.filled_quantity > 0:
            logger.warning(
                f"Partial fill for {symbol}: {order.filled_quantity}/{shares} shares filled"
            )
            shares = order.filled_quantity
        else:
            logger.warning(f"Order not filled for {shares} {symbol}: status={order.status.value}")
            return None

        # Send trade notification
        if self.notifier:
            confidence = recommendation.get("confidence")
            self.notifier.notify_trade(
                action=action,
                symbol=symbol,
                shares=shares,
                price=price,
                confidence=confidence
            )

        # Set stop loss for new positions
        if action == "BUY" and "stop_loss" in recommendation:
            self.risk_manager.set_stop_loss(
                symbol=symbol,
                entry_price=price,
                stop_type=StopLossType.TRAILING,
                stop_price=recommendation["stop_loss"]
            )

        # Set take-profit for new positions
        if action == "BUY":
            tp_config = self.config.get("risk_management", {}).get("take_profit", {})
            if tp_config.get("enabled", True):
                tp_levels = tp_config.get("levels", [(0.05, 0.33), (0.10, 0.50), (0.15, 1.0)])
                self.risk_manager.set_take_profit(
                    symbol=symbol,
                    entry_price=price,
                    quantity=shares,
                    tp_levels=tp_levels
                )

        return {
            "order_id": order.order_id,
            "action": action,
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "timestamp": datetime.now()
        }

    def _execute_stop_loss(self, symbol: str) -> None:
        """Execute stop loss for a position."""
        position = self.broker.get_position(symbol)
        if position and position.quantity > 0:
            order = self.broker.place_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=position.quantity,
                order_type=OrderType.MARKET,
            )
            self.risk_manager.remove_stop_loss(symbol)
            self.risk_manager.remove_take_profit(symbol)

            # Calculate P&L from actual fill price
            entry_price = position.avg_cost
            fill_price = getattr(order, "filled_price", None)
            if not fill_price:
                fill_price = getattr(position, "current_price", None) or entry_price
            pnl = (fill_price - entry_price) * position.quantity
            self.risk_manager.update_pnl(pnl, is_loss=(pnl < 0))

            logger.warning(f"Stop loss executed for {symbol}. Fill: ${fill_price:.2f}, P&L: ${pnl:.2f}")

            # Send stop-loss notification
            if self.notifier:
                loss_pct = (pnl / (position.quantity * entry_price)) if entry_price > 0 else 0
                self.notifier.notify_stop_loss(
                    symbol=symbol,
                    exit_price=fill_price,
                    loss_amount=pnl,
                    loss_pct=loss_pct
                )

    def _execute_take_profit(self, symbol: str, quantity: int, target_price: float) -> None:
        """Execute take-profit partial sale."""
        position = self.broker.get_position(symbol)
        if not position or position.quantity <= 0:
            return

        # Ensure we don't sell more than we have
        sell_qty = min(quantity, position.quantity)
        if sell_qty <= 0:
            return

        order = self.broker.sell(symbol, sell_qty)
        if order is None or (hasattr(order, 'status') and order.status.value == "rejected"):
            logger.warning(f"Take-profit sell FAILED for {symbol}: order rejected or None")
            return

        # Calculate P&L for partial sale
        entry_price = position.avg_cost
        fill = getattr(order, "filled_price", None) or target_price
        pnl = (fill - entry_price) * sell_qty
        self.risk_manager.update_pnl(pnl, is_loss=(pnl < 0))

        logger.info(f"Take-profit executed for {symbol}: sold {sell_qty} shares. P&L: ${pnl:.2f}")

        # Send take-profit notification
        if self.notifier:
            gain_pct = (target_price - entry_price) / entry_price if entry_price > 0 else 0
            self.notifier.notify_take_profit(
                symbol=symbol,
                exit_price=target_price,
                quantity=sell_qty,
                gain_amount=pnl,
                gain_pct=gain_pct
            )

        # Remove stop-loss if fully exited
        if position.quantity - sell_qty <= 0:
            self.risk_manager.remove_stop_loss(symbol)

    def _get_target_portfolio_weights(self, signals: Optional[Dict] = None) -> Optional[Dict[str, float]]:
        """
        Get target portfolio weights from optimizer.

        Args:
            signals: Optional ML trading signals for signal-based tilting

        Returns:
            Dictionary mapping symbols to target weights, or None if optimizer disabled
        """
        if not self.portfolio_optimizer:
            return None

        try:
            portfolio_config = self.config.get("portfolio_optimization", {})
            use_regime_aware = portfolio_config.get("regime_aware", False)

            # Check if regime-aware optimization is enabled
            if use_regime_aware and self.regime_detector:
                # Detect current market regime
                regime_status = self.regime_detector.get_status()
                current_regime = regime_status.get("current_regime")

                if current_regime:
                    logger.info(f"Using regime-aware portfolio optimization: {current_regime}")

                    # Run regime-aware optimization
                    weights = self.portfolio_optimizer.optimize_regime_aware(
                        symbols=self.symbols,
                        market_regime=current_regime,
                        signals=signals if portfolio_config.get("incorporate_signals", True) else None
                    )

                    if weights and weights.weights:
                        regime_str = current_regime.value if hasattr(current_regime, 'value') else str(current_regime)
                        logger.info(f"Regime-aware optimization ({regime_str}): "
                                   f"method={weights.method.value}, "
                                   f"Sharpe={weights.sharpe_ratio:.3f}, "
                                   f"return={weights.expected_return:.2%}")
                        return weights.weights
                    else:
                        logger.warning("Regime-aware optimization returned no weights, falling back to standard")
                else:
                    logger.warning("Could not detect market regime, using standard optimization")

            # Standard optimization (no regime awareness)
            method_str = portfolio_config.get("method", "max_sharpe")

            # Map string to OptimizationMethod enum
            method_map = {
                "equal_weight": OptimizationMethod.EQUAL_WEIGHT,
                "max_sharpe": OptimizationMethod.MAX_SHARPE,
                "risk_parity": OptimizationMethod.RISK_PARITY,
                "minimum_variance": OptimizationMethod.MINIMUM_VARIANCE,
                "mean_variance": OptimizationMethod.MEAN_VARIANCE
            }
            method = method_map.get(method_str, OptimizationMethod.MAX_SHARPE)

            # Run optimization
            weights = self.portfolio_optimizer.optimize(
                symbols=self.symbols,
                method=method,
                signals=signals if portfolio_config.get("incorporate_signals", True) else None
            )

            if weights and weights.weights:
                logger.info(f"Portfolio optimization: method={method_str}, "
                           f"Sharpe={weights.sharpe_ratio:.3f}, "
                           f"return={weights.expected_return:.2%}")
                return weights.weights
            else:
                logger.warning("Portfolio optimization returned no weights")
                return None

        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return None

    def _is_market_open(self) -> bool:
        """Check if market is currently open."""
        if self.ignore_market_hours:
            return True

        now = datetime.now()

        # Check if weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False

        # Check market hours
        current_time = now.time()
        return self.market_open <= current_time <= self.market_close

    def _fetch_sectors_for_symbols(self, symbols: List[str]) -> Dict[str, str]:
        """Fetch sector information for symbols."""
        sectors = {}
        try:
            from src.data.fundamental_fetcher import get_fundamental_fetcher
            fetcher = get_fundamental_fetcher()
            for symbol in symbols:
                try:
                    data = fetcher.fetch_fundamentals(symbol)
                    if data.sector:
                        sectors[symbol] = data.sector
                except Exception:
                    pass
        except ImportError:
            pass
        return sectors

    def get_active_symbols(self) -> List[str]:
        """
        Get the current list of active trading symbols.

        If symbol manager is enabled, returns dynamically managed symbols.
        Otherwise, returns the static symbols list.
        """
        if self.symbol_manager:
            return self.symbol_manager.get_active_symbols()
        return self.symbols

    def _update_symbol_prices(self) -> None:
        """Update symbol manager with current prices."""
        if not self.symbol_manager or not self.broker:
            return

        try:
            active_symbols = self.symbol_manager.get_active_symbols()
            if not active_symbols:
                return

            quotes = self.broker.get_quotes(active_symbols)
            prices = {s: q.last for s, q in quotes.items() if q.last > 0}
            self.symbol_manager.update_prices(prices)
        except Exception as e:
            logger.warning(f"Failed to update symbol prices: {e}")

    def get_status(self) -> Dict:
        """
        Get current engine status.

        Returns:
            Status dictionary.
        """
        status = {
            "is_running": self.is_running,
            "mode": "paper" if self.paper_trading else "live",
            "market_open": self._is_market_open(),
            "symbols": self.symbols
        }

        if self.broker and self.broker.is_connected():
            try:
                account = self.broker.get_account_info()
                positions = self.broker.get_positions()

                status["account"] = {
                    "portfolio_value": account.portfolio_value,
                    "cash": account.cash,
                    "buying_power": account.buying_power,
                    "positions_count": len(positions)
                }

                status["risk"] = self.risk_manager.get_risk_summary(account.portfolio_value)

            except Exception as e:
                status["error"] = str(e)

        # Add retraining status
        if self.retrainer:
            status["retraining"] = self.retrainer.get_status()

        # Add regime detection status
        if self.regime_detector:
            status["regime"] = self.regime_detector.get_status()

        # Add symbol manager status
        if self.symbol_manager:
            status["symbol_manager"] = self.symbol_manager.get_status()

        return status

    def get_retraining_status(self) -> Optional[Dict]:
        """Get scheduled retraining status."""
        if self.retrainer:
            return self.retrainer.get_status()
        return None

    def trigger_retrain(self) -> Optional[Dict]:
        """Manually trigger model retraining."""
        if self.retrainer:
            return self.retrainer.trigger_retrain_now()
        return None


def run_bot(
    symbols: List[str],
    paper_trading: bool = True,
    simulated: bool = False,
    initial_capital: float = 100000.0,
    interval_seconds: int = 60,
    ignore_market_hours: bool = False,
    use_ensemble: bool = False
):
    """
    Run the trading bot.

    Args:
        symbols: List of stock symbols to trade.
        paper_trading: Whether to use paper trading (WeBull).
        simulated: Whether to use simulated broker (no credentials).
        initial_capital: Initial capital for simulated mode.
        interval_seconds: Seconds between trading cycles.
        ignore_market_hours: Whether to ignore market hours (for testing).
        use_ensemble: Whether to use ensemble model (XGBoost + LSTM + CNN).
    """
    engine = TradingEngine(
        symbols=symbols,
        paper_trading=paper_trading,
        simulated=simulated,
        initial_capital=initial_capital,
        ignore_market_hours=ignore_market_hours,
        use_ensemble=use_ensemble
    )

    # Handle SIGTERM (Docker/systemd) the same as SIGINT (Ctrl+C)
    def _handle_sigterm(signum, frame):
        logger.info("Received SIGTERM — initiating graceful shutdown")
        engine.is_running = False

    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        engine.start()

        logger.info(f"Bot running. Checking every {interval_seconds} seconds.")

        while engine.is_running:
            result = engine.run_trading_cycle()
            logger.debug(f"Cycle result: {result['status']}")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")

    finally:
        engine.stop()
