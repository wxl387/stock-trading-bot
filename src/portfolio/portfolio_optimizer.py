"""
Portfolio optimization with multiple allocation strategies.

Implements mean-variance optimization, risk-parity, efficient frontier,
and signal-aware weighting.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.data.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    EQUAL_WEIGHT = "equal_weight"
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAX_SHARPE = "max_sharpe"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass
class PortfolioWeights:
    """Optimized portfolio weights."""
    weights: Dict[str, float]
    method: OptimizationMethod
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "weights": self.weights,
            "method": self.method.value,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class PortfolioOptimizer:
    """
    Portfolio optimizer implementing multiple allocation strategies.

    Methods:
    - Equal-weight (baseline)
    - Mean-variance optimization (efficient frontier)
    - Risk-parity allocation (equal risk contribution)
    - Minimum-variance portfolio
    - Maximum Sharpe ratio
    - Maximum diversification ratio
    """

    def __init__(
        self,
        lookback_days: int = 252,
        min_weight: float = 0.0,
        max_weight: float = 0.25,
        target_return: Optional[float] = None,
        risk_free_rate: float = 0.05,
        correlation_threshold: float = 0.8,
        use_shrinkage: bool = True,
        cache_ttl_hours: int = 4
    ):
        """
        Initialize portfolio optimizer with constraints.

        Args:
            lookback_days: Days of historical returns for optimization
            min_weight: Minimum weight per asset (0-1)
            max_weight: Maximum weight per asset (0-1)
            target_return: Target return for mean-variance optimization
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            correlation_threshold: Threshold for correlation warnings
            use_shrinkage: Use Ledoit-Wolf covariance shrinkage
            cache_ttl_hours: Hours to cache correlation matrices
        """
        self.lookback_days = lookback_days
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_return = target_return
        self.risk_free_rate = risk_free_rate
        self.correlation_threshold = correlation_threshold
        self.use_shrinkage = use_shrinkage
        self.cache_ttl_hours = cache_ttl_hours

        self.data_fetcher = DataFetcher()
        self._cache: Dict = {}
        self._cache_timestamp: Optional[datetime] = None

        logger.info(f"Initialized PortfolioOptimizer: lookback={lookback_days}d, "
                   f"weights=[{min_weight:.2f}, {max_weight:.2f}], rf={risk_free_rate}")

    def optimize(
        self,
        symbols: List[str],
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        signals: Optional[Dict[str, 'TradingSignal']] = None,
        constraints: Optional[Dict] = None
    ) -> PortfolioWeights:
        """
        Optimize portfolio allocation.

        Args:
            symbols: List of symbols to optimize
            method: Optimization method to use
            signals: Optional ML signals to incorporate (signal strength as weights)
            constraints: Additional constraints (sector limits, etc.)

        Returns:
            PortfolioWeights with optimized allocation
        """
        try:
            # Fetch historical returns
            returns = self._fetch_returns(symbols)

            if returns.empty or len(returns) < 20:
                logger.warning(f"Insufficient data for optimization ({len(returns)} days). Using equal-weight.")
                return self._equal_weight_fallback(symbols)

            # Select optimization method
            if method == OptimizationMethod.EQUAL_WEIGHT:
                weights_dict = self.equal_weight_optimize(symbols)
            elif method == OptimizationMethod.MAX_SHARPE:
                weights_dict = self.max_sharpe_optimize(returns)
            elif method == OptimizationMethod.RISK_PARITY:
                weights_dict = self.risk_parity_optimize(returns)
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                weights_dict = self.minimum_variance_optimize(returns)
            elif method == OptimizationMethod.MEAN_VARIANCE:
                weights_dict = self.mean_variance_optimize(returns)
            else:
                logger.warning(f"Unknown method {method}. Using equal-weight.")
                weights_dict = self.equal_weight_optimize(symbols)

            # Apply signal tilts if provided
            if signals and len(signals) > 0:
                weights_dict = self._apply_signal_tilts(weights_dict, signals)

            # Calculate portfolio metrics
            expected_return, expected_vol = self._calculate_portfolio_metrics(
                weights_dict, returns
            )
            sharpe_ratio = (expected_return - self.risk_free_rate) / expected_vol if expected_vol > 0 else 0.0

            # Check correlation constraints
            corr_warnings = self._check_correlation_constraints(weights_dict, returns)

            portfolio_weights = PortfolioWeights(
                weights=weights_dict,
                method=method,
                expected_return=expected_return,
                expected_volatility=expected_vol,
                sharpe_ratio=sharpe_ratio,
                timestamp=datetime.now(),
                metadata={
                    "lookback_days": self.lookback_days,
                    "n_assets": len(weights_dict),
                    "correlation_warnings": corr_warnings,
                    "signals_incorporated": signals is not None
                }
            )

            logger.info(f"Optimized portfolio ({method.value}): "
                       f"return={expected_return:.3f}, vol={expected_vol:.3f}, "
                       f"sharpe={sharpe_ratio:.3f}")

            return portfolio_weights

        except Exception as e:
            logger.error(f"Optimization failed: {e}. Using equal-weight fallback.")
            return self._equal_weight_fallback(symbols)

    def optimize_regime_aware(
        self,
        symbols: List[str],
        market_regime: 'MarketRegime',
        signals: Optional[Dict[str, 'TradingSignal']] = None,
        constraints: Optional[Dict] = None
    ) -> PortfolioWeights:
        """
        Regime-aware portfolio optimization: automatically select method based on market regime.

        Optimization strategy by regime:
        - BULL: Max Sharpe (aggressive growth)
        - BEAR: Minimum Variance (defensive, capital preservation)
        - CHOPPY: Risk Parity (balanced risk across assets)
        - VOLATILE: Minimum Variance with cash buffer (defensive)

        Args:
            symbols: List of symbols to optimize
            market_regime: Current market regime (can be MarketRegime enum or string)
            signals: Optional ML signals to incorporate
            constraints: Additional constraints

        Returns:
            PortfolioWeights with regime-appropriate allocation
        """
        try:
            from src.risk.regime_detector import MarketRegime

            # Convert string to enum if needed
            if isinstance(market_regime, str):
                market_regime = MarketRegime(market_regime)

            # Select optimization method based on regime
            regime_methods = {
                MarketRegime.BULL: OptimizationMethod.MAX_SHARPE,
                MarketRegime.BEAR: OptimizationMethod.MINIMUM_VARIANCE,
                MarketRegime.CHOPPY: OptimizationMethod.RISK_PARITY,
                MarketRegime.VOLATILE: OptimizationMethod.MINIMUM_VARIANCE
            }

            method = regime_methods.get(market_regime, OptimizationMethod.MAX_SHARPE)

            logger.info(f"Regime-aware optimization: {market_regime.value} → {method.value}")

            # Optimize with selected method
            portfolio_weights = self.optimize(
                symbols=symbols,
                method=method,
                signals=signals,
                constraints=constraints
            )

            # Apply regime-specific adjustments
            if market_regime == MarketRegime.VOLATILE:
                # In volatile markets, add cash buffer by reducing all weights
                cash_buffer = 0.15  # 15% cash allocation
                adjusted_weights = {
                    symbol: weight * (1.0 - cash_buffer)
                    for symbol, weight in portfolio_weights.weights.items()
                }
                portfolio_weights.weights = adjusted_weights
                portfolio_weights.metadata['cash_buffer'] = cash_buffer
                portfolio_weights.metadata['regime_adjustment'] = 'volatile_cash_buffer'

                logger.info(f"Applied {cash_buffer:.1%} cash buffer for volatile regime")

            elif market_regime == MarketRegime.BEAR:
                # In bear markets, reduce concentration (lower max weights)
                max_position = 0.20  # Max 20% per position in bear market
                adjusted_weights = {}
                total_excess = 0.0

                for symbol, weight in portfolio_weights.weights.items():
                    if weight > max_position:
                        adjusted_weights[symbol] = max_position
                        total_excess += (weight - max_position)
                    else:
                        adjusted_weights[symbol] = weight

                # Redistribute excess weight to underweight positions
                if total_excess > 0:
                    underweight_symbols = [
                        s for s, w in adjusted_weights.items()
                        if w < max_position
                    ]
                    if underweight_symbols:
                        redistribution = total_excess / len(underweight_symbols)
                        for symbol in underweight_symbols:
                            adjusted_weights[symbol] = min(
                                max_position,
                                adjusted_weights[symbol] + redistribution
                            )

                portfolio_weights.weights = adjusted_weights
                portfolio_weights.metadata['max_position_override'] = max_position
                portfolio_weights.metadata['regime_adjustment'] = 'bear_concentration_limit'

                logger.info(f"Applied {max_position:.1%} max position limit for bear regime")

            # Add regime information to metadata
            portfolio_weights.metadata['market_regime'] = market_regime.value
            portfolio_weights.metadata['regime_aware'] = True

            return portfolio_weights

        except Exception as e:
            logger.error(f"Regime-aware optimization failed: {e}. Using equal-weight fallback.")
            return self._equal_weight_fallback(symbols)

    def equal_weight_optimize(self, symbols: List[str]) -> Dict[str, float]:
        """
        Equal-weight allocation (baseline).

        Args:
            symbols: List of symbols

        Returns:
            Dict mapping symbol to weight
        """
        n = len(symbols)
        weight = 1.0 / n
        return {symbol: weight for symbol in symbols}

    def max_sharpe_optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Maximum Sharpe ratio optimization.

        Args:
            returns: DataFrame of historical returns (symbols as columns)

        Returns:
            Dict mapping symbol to weight
        """
        n_assets = len(returns.columns)

        # Calculate mean returns and covariance
        mean_returns = returns.mean() * 252  # Annualize
        cov_matrix = self._calculate_covariance(returns) * 252  # Annualize

        # Objective: Minimize negative Sharpe ratio
        def neg_sharpe(weights):
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 999

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds: [min_weight, max_weight] for each asset
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess: equal weight
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Max Sharpe optimization did not converge: {result.message}")

        weights = result.x
        weights_dict = dict(zip(returns.columns, weights))

        # Ensure constraints are satisfied
        weights_dict = self._enforce_constraints(weights_dict)

        return weights_dict

    def risk_parity_optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Risk-parity allocation: equal risk contribution from each asset.

        Args:
            returns: DataFrame of historical returns

        Returns:
            Dict mapping symbol to weight
        """
        n_assets = len(returns.columns)
        cov_matrix = self._calculate_covariance(returns)

        # Objective: Minimize variance of risk contributions
        def risk_parity_objective(weights):
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol if port_vol > 0 else weights

            # Minimize variance of risk contributions
            target_contrib = 1.0 / n_assets
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess: inverse volatility
        vols = returns.std()
        init_weights = (1.0 / vols) / (1.0 / vols).sum()
        init_weights = init_weights.values

        # Optimize
        result = minimize(
            risk_parity_objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Risk parity optimization did not converge: {result.message}")

        weights = result.x
        weights_dict = dict(zip(returns.columns, weights))

        # Ensure constraints are satisfied
        weights_dict = self._enforce_constraints(weights_dict)

        return weights_dict

    def minimum_variance_optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Minimum-variance portfolio: minimize overall portfolio volatility.

        Args:
            returns: DataFrame of historical returns

        Returns:
            Dict mapping symbol to weight
        """
        n_assets = len(returns.columns)
        cov_matrix = self._calculate_covariance(returns)

        # Objective: Minimize variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess: equal weight
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Minimum variance optimization did not converge: {result.message}")

        weights = result.x
        weights_dict = dict(zip(returns.columns, weights))

        # Ensure constraints are satisfied
        weights_dict = self._enforce_constraints(weights_dict)

        return weights_dict

    def mean_variance_optimize(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Markowitz mean-variance optimization.
        Maximize return for given risk or minimize risk for given return.

        Args:
            returns: DataFrame of historical returns
            target_return: Target annual return (if None, uses self.target_return)

        Returns:
            Dict mapping symbol to weight
        """
        if target_return is None and self.target_return is None:
            # No target return specified, use max Sharpe instead
            logger.info("No target return specified. Using max Sharpe optimization.")
            return self.max_sharpe_optimize(returns)

        target = target_return or self.target_return
        n_assets = len(returns.columns)

        mean_returns = returns.mean() * 252  # Annualize
        cov_matrix = self._calculate_covariance(returns) * 252  # Annualize

        # Objective: Minimize variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints: weights sum to 1, return = target
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target}
        ]
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))

        # Initial guess: equal weight
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Mean-variance optimization did not converge: {result.message}. "
                         f"Falling back to max Sharpe.")
            return self.max_sharpe_optimize(returns)

        weights = result.x
        weights_dict = dict(zip(returns.columns, weights))

        # Ensure constraints are satisfied
        weights_dict = self._enforce_constraints(weights_dict)

        return weights_dict

    def _fetch_returns(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch historical returns for optimization.

        Args:
            symbols: List of symbols

        Returns:
            DataFrame with returns (symbols as columns)
        """
        try:
            # Fetch historical data for each symbol
            returns_data = {}

            for symbol in symbols:
                try:
                    df = self.data_fetcher.fetch_historical(symbol, period=f"{self.lookback_days + 30}d")
                    if df is not None and not df.empty:
                        # Check for both uppercase and lowercase column names
                        close_col = 'Close' if 'Close' in df.columns else 'close'
                        if close_col in df.columns:
                            returns = df[close_col].pct_change().dropna()
                            returns_data[symbol] = returns
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")

            if not returns_data:
                return pd.DataFrame()

            # Combine into single DataFrame
            returns_df = pd.DataFrame(returns_data)

            # Align dates (use intersection)
            returns_df = returns_df.dropna()

            # Limit to lookback_days
            if len(returns_df) > self.lookback_days:
                returns_df = returns_df.tail(self.lookback_days)

            logger.debug(f"Fetched returns for {len(returns_df.columns)} symbols, "
                        f"{len(returns_df)} days")

            return returns_df

        except Exception as e:
            logger.error(f"Error fetching returns: {e}")
            return pd.DataFrame()

    def _calculate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate covariance matrix with optional shrinkage.

        Args:
            returns: DataFrame of returns

        Returns:
            Covariance matrix
        """
        if self.use_shrinkage:
            # Use Ledoit-Wolf shrinkage for more stable estimates
            lw = LedoitWolf()
            cov_matrix, _ = lw.fit(returns).covariance_, None
            return cov_matrix
        else:
            return returns.cov().values

    def _calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> Tuple[float, float]:
        """
        Calculate expected return and volatility for portfolio.

        Args:
            weights: Dict mapping symbol to weight
            returns: DataFrame of returns

        Returns:
            (expected_return, expected_volatility) as annual percentages
        """
        # Ensure symbols match
        symbols = [s for s in weights.keys() if s in returns.columns]
        if not symbols:
            return 0.0, 0.0

        weights_array = np.array([weights[s] for s in symbols])
        returns_subset = returns[symbols]

        # Annual return
        mean_returns = returns_subset.mean() * 252
        expected_return = np.dot(weights_array, mean_returns)

        # Annual volatility
        cov_matrix = self._calculate_covariance(returns_subset) * 252
        expected_vol = np.sqrt(np.dot(weights_array, np.dot(cov_matrix, weights_array)))

        return float(expected_return), float(expected_vol)

    def _apply_signal_tilts(
        self,
        base_weights: Dict[str, float],
        signals: Dict[str, 'TradingSignal'],
        tilt_strength: float = 0.2
    ) -> Dict[str, float]:
        """
        Tilt optimized weights based on ML signal confidence.

        Args:
            base_weights: Base weights from optimization
            signals: Trading signals with confidence
            tilt_strength: Maximum tilt as fraction (0.2 = 20%)

        Returns:
            Tilted weights (normalized to sum to 1)
        """
        tilted_weights = base_weights.copy()

        for symbol, signal in signals.items():
            if symbol not in base_weights:
                continue

            base_weight = base_weights[symbol]

            # Tilt based on confidence and signal direction
            # BUY signals increase weight, SELL signals decrease weight
            if hasattr(signal, 'signal') and hasattr(signal, 'confidence'):
                signal_type = signal.signal.value if hasattr(signal.signal, 'value') else str(signal.signal)

                if signal_type == "BUY":
                    tilt = tilt_strength * signal.confidence
                    tilted_weights[symbol] = base_weight * (1 + tilt)
                elif signal_type == "SELL":
                    tilt = tilt_strength * signal.confidence
                    tilted_weights[symbol] = base_weight * (1 - tilt)

        # Normalize to sum to 1
        total = sum(tilted_weights.values())
        if total > 0:
            tilted_weights = {k: v / total for k, v in tilted_weights.items()}

        # Ensure constraints still met
        tilted_weights = self._enforce_constraints(tilted_weights)

        logger.debug(f"Applied signal tilts: {len(signals)} signals processed")

        return tilted_weights

    def _enforce_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Enforce min/max weight constraints and normalize.

        Args:
            weights: Weights to constrain

        Returns:
            Constrained weights
        """
        n = len(weights)

        # Check if constraints are feasible
        # If max_weight * n < 1, we can't satisfy both constraints
        if self.max_weight * n < 1.0:
            logger.warning(f"Infeasible constraints: max_weight={self.max_weight}, "
                          f"n_assets={n}. Relaxing max_weight to {1.0/n:.3f}")
            # Use equal weight as it's the fairest solution
            return {k: 1.0 / n for k in weights.keys()}

        # First, clip to min/max
        clipped = {k: np.clip(v, self.min_weight, self.max_weight)
                   for k, v in weights.items()}

        # Normalize to sum to 1
        total = sum(clipped.values())
        if total > 0:
            normalized = {k: v / total for k, v in clipped.items()}
        else:
            # If all weights are 0, use equal weight
            normalized = {k: 1.0 / n for k in weights.keys()}

        # After normalization, check if any weight exceeds max_weight
        # If so, we need to iteratively adjust
        max_iterations = 10
        for iteration in range(max_iterations):
            exceeds_max = {k: v for k, v in normalized.items() if v > self.max_weight}
            if not exceeds_max:
                break

            # Clip weights that exceed max
            for k in exceeds_max:
                normalized[k] = self.max_weight

            # Redistribute the excess among weights that aren't at max
            total_clipped = sum(self.max_weight for _ in exceeds_max)
            remaining_keys = [k for k in normalized if k not in exceeds_max]

            if remaining_keys:
                excess = 1.0 - total_clipped

                # Check if excess can be distributed without exceeding max_weight
                equal_share = excess / len(remaining_keys)
                if equal_share <= self.max_weight:
                    # Distribute proportionally
                    total_remaining = sum(normalized[k] for k in remaining_keys)
                    if total_remaining > 0:
                        for k in remaining_keys:
                            normalized[k] = (normalized[k] / total_remaining) * excess
                    else:
                        # Equal distribution
                        for k in remaining_keys:
                            normalized[k] = equal_share
                else:
                    # All remaining would also exceed max, need to relax constraints
                    logger.warning("Cannot satisfy max_weight constraint. Using proportional allocation.")
                    for k in weights.keys():
                        normalized[k] = 1.0 / n
                    break
            else:
                # All weights are at max but don't sum to 1
                # Relax the constraint proportionally
                logger.warning("All weights at max constraint. Using proportional allocation.")
                for k in weights.keys():
                    normalized[k] = 1.0 / n
                break

        return normalized

    def _check_correlation_constraints(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> List[str]:
        """
        Check if weights violate correlation constraints.

        Args:
            weights: Portfolio weights
            returns: Historical returns

        Returns:
            List of warning messages
        """
        warnings = []

        try:
            # Calculate correlation matrix
            symbols = [s for s in weights.keys() if s in returns.columns]
            if len(symbols) < 2:
                return warnings

            corr_matrix = returns[symbols].corr()

            # Check for high correlation pairs with significant weights
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i >= j:
                        continue

                    corr = corr_matrix.loc[sym1, sym2]
                    combined_weight = weights[sym1] + weights[sym2]

                    if abs(corr) > self.correlation_threshold and combined_weight > 0.3:
                        warnings.append(
                            f"{sym1}-{sym2}: high correlation ({corr:.2f}), "
                            f"combined weight {combined_weight:.1%}"
                        )

            if warnings:
                logger.warning(f"Correlation warnings: {len(warnings)} pairs")

        except Exception as e:
            logger.warning(f"Error checking correlations: {e}")

        return warnings

    def _equal_weight_fallback(self, symbols: List[str]) -> PortfolioWeights:
        """
        Fallback to equal-weight allocation when optimization fails.

        Args:
            symbols: List of symbols

        Returns:
            PortfolioWeights with equal weights
        """
        weights = self.equal_weight_optimize(symbols)

        return PortfolioWeights(
            weights=weights,
            method=OptimizationMethod.EQUAL_WEIGHT,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            timestamp=datetime.now(),
            metadata={"fallback": True}
        )
