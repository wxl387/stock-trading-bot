"""
Efficient frontier calculation and visualization.

Implements Markowitz mean-variance optimization for efficient frontier.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


class EfficientFrontier:
    """
    Efficient frontier calculation and visualization.

    Calculates the efficient frontier using mean-variance optimization
    and identifies key portfolios (tangency, minimum variance).
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        use_shrinkage: bool = True
    ):
        """
        Initialize efficient frontier calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            use_shrinkage: Use Ledoit-Wolf covariance shrinkage
        """
        self.risk_free_rate = risk_free_rate
        self.use_shrinkage = use_shrinkage

        logger.info(f"Initialized EfficientFrontier: rf={risk_free_rate}")

    def calculate_frontier(
        self,
        returns: pd.DataFrame,
        num_points: int = 100,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier points.

        Args:
            returns: DataFrame of historical returns (symbols as columns)
            num_points: Number of frontier points to calculate
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset

        Returns:
            DataFrame with columns: expected_return, volatility, sharpe, weights
        """
        try:
            n_assets = len(returns.columns)

            # Calculate mean returns and covariance
            mean_returns = returns.mean() * 252  # Annualize
            cov_matrix = self._calculate_covariance(returns) * 252

            # Find minimum and maximum possible returns
            min_return, max_return = self._get_return_range(
                returns, min_weight, max_weight
            )

            # Generate target returns
            target_returns = np.linspace(min_return, max_return, num_points)

            frontier_data = []

            for target_return in target_returns:
                try:
                    # Optimize for this target return
                    weights, port_vol = self._optimize_for_return(
                        mean_returns,
                        cov_matrix,
                        target_return,
                        min_weight,
                        max_weight
                    )

                    if weights is not None:
                        sharpe = (target_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

                        frontier_data.append({
                            'expected_return': target_return,
                            'volatility': port_vol,
                            'sharpe': sharpe,
                            'weights': dict(zip(returns.columns, weights))
                        })

                except Exception as e:
                    logger.debug(f"Failed to optimize for return {target_return:.3f}: {e}")
                    continue

            if not frontier_data:
                logger.warning("No frontier points calculated")
                return pd.DataFrame()

            frontier_df = pd.DataFrame(frontier_data)
            logger.info(f"Calculated {len(frontier_df)} efficient frontier points")

            return frontier_df

        except Exception as e:
            logger.error(f"Error calculating efficient frontier: {e}")
            return pd.DataFrame()

    def find_tangency_portfolio(
        self,
        returns: pd.DataFrame,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Dict:
        """
        Find tangency portfolio (max Sharpe ratio).

        Args:
            returns: DataFrame of historical returns
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset

        Returns:
            Dict with keys: weights, expected_return, volatility, sharpe
        """
        try:
            n_assets = len(returns.columns)

            mean_returns = returns.mean() * 252
            cov_matrix = self._calculate_covariance(returns) * 252

            # Objective: Minimize negative Sharpe ratio
            def neg_sharpe(weights):
                port_return = np.dot(weights, mean_returns)
                port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                return -(port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 999

            # Constraints
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

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
                logger.warning(f"Tangency portfolio optimization failed: {result.message}")
                return {}

            weights = result.x
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

            tangency = {
                'weights': dict(zip(returns.columns, weights)),
                'expected_return': float(port_return),
                'volatility': float(port_vol),
                'sharpe': float(sharpe)
            }

            logger.info(f"Tangency portfolio: return={port_return:.3f}, "
                       f"vol={port_vol:.3f}, sharpe={sharpe:.3f}")

            return tangency

        except Exception as e:
            logger.error(f"Error finding tangency portfolio: {e}")
            return {}

    def find_minimum_variance_portfolio(
        self,
        returns: pd.DataFrame,
        min_weight: float = 0.0,
        max_weight: float = 1.0
    ) -> Dict:
        """
        Find global minimum variance portfolio.

        Args:
            returns: DataFrame of historical returns
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset

        Returns:
            Dict with keys: weights, expected_return, volatility, sharpe
        """
        try:
            n_assets = len(returns.columns)

            mean_returns = returns.mean() * 252
            cov_matrix = self._calculate_covariance(returns) * 252

            # Objective: Minimize variance
            def portfolio_variance(weights):
                return np.dot(weights, np.dot(cov_matrix, weights))

            # Constraints
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

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
                logger.warning(f"Minimum variance optimization failed: {result.message}")
                return {}

            weights = result.x
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

            min_var = {
                'weights': dict(zip(returns.columns, weights)),
                'expected_return': float(port_return),
                'volatility': float(port_vol),
                'sharpe': float(sharpe)
            }

            logger.info(f"Minimum variance portfolio: return={port_return:.3f}, "
                       f"vol={port_vol:.3f}")

            return min_var

        except Exception as e:
            logger.error(f"Error finding minimum variance portfolio: {e}")
            return {}

    def plot_frontier(
        self,
        frontier_df: pd.DataFrame,
        current_portfolio: Optional[Dict] = None,
        tangency_portfolio: Optional[Dict] = None,
        min_var_portfolio: Optional[Dict] = None
    ):
        """
        Plot efficient frontier with key portfolios marked.

        Args:
            frontier_df: DataFrame from calculate_frontier()
            current_portfolio: Optional current portfolio to mark
            tangency_portfolio: Optional tangency portfolio to mark
            min_var_portfolio: Optional min variance portfolio to mark

        Returns:
            Plotly figure object
        """
        try:
            import plotly.graph_objects as go

            fig = go.Figure()

            # Plot efficient frontier
            fig.add_trace(go.Scatter(
                x=frontier_df['volatility'],
                y=frontier_df['expected_return'],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='blue', width=2)
            ))

            # Mark tangency portfolio
            if tangency_portfolio:
                fig.add_trace(go.Scatter(
                    x=[tangency_portfolio['volatility']],
                    y=[tangency_portfolio['expected_return']],
                    mode='markers',
                    name='Tangency (Max Sharpe)',
                    marker=dict(size=12, color='green', symbol='star')
                ))

            # Mark minimum variance portfolio
            if min_var_portfolio:
                fig.add_trace(go.Scatter(
                    x=[min_var_portfolio['volatility']],
                    y=[min_var_portfolio['expected_return']],
                    mode='markers',
                    name='Minimum Variance',
                    marker=dict(size=12, color='orange', symbol='diamond')
                ))

            # Mark current portfolio
            if current_portfolio:
                fig.add_trace(go.Scatter(
                    x=[current_portfolio.get('volatility', 0)],
                    y=[current_portfolio.get('expected_return', 0)],
                    mode='markers',
                    name='Current Portfolio',
                    marker=dict(size=12, color='red', symbol='circle')
                ))

            # Add capital market line (if tangency portfolio exists)
            if tangency_portfolio:
                # Line from risk-free rate to tangency portfolio and beyond
                x_cml = np.linspace(0, frontier_df['volatility'].max() * 1.2, 100)
                tangency_sharpe = tangency_portfolio['sharpe']
                y_cml = self.risk_free_rate + tangency_sharpe * x_cml

                fig.add_trace(go.Scatter(
                    x=x_cml,
                    y=y_cml,
                    mode='lines',
                    name='Capital Market Line',
                    line=dict(color='gray', dash='dash', width=1)
                ))

            fig.update_layout(
                title='Efficient Frontier',
                xaxis_title='Volatility (Annual)',
                yaxis_title='Expected Return (Annual)',
                hovermode='closest',
                showlegend=True,
                template='plotly_white'
            )

            return fig

        except ImportError:
            logger.warning("Plotly not available. Install with: pip install plotly")
            return None
        except Exception as e:
            logger.error(f"Error plotting frontier: {e}")
            return None

    def _calculate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Calculate covariance matrix with optional shrinkage."""
        if self.use_shrinkage:
            lw = LedoitWolf()
            cov_matrix, _ = lw.fit(returns).covariance_, None
            return cov_matrix
        else:
            return returns.cov().values

    def _get_return_range(
        self,
        returns: pd.DataFrame,
        min_weight: float,
        max_weight: float
    ) -> Tuple[float, float]:
        """
        Get feasible return range for efficient frontier.

        Args:
            returns: Historical returns
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset

        Returns:
            (min_return, max_return)
        """
        mean_returns = returns.mean() * 252

        # Minimum return: all weight on lowest-return asset (within constraints)
        sorted_returns = sorted(mean_returns)
        min_return = sorted_returns[0]

        # Maximum return: all weight on highest-return asset (within constraints)
        max_return = sorted_returns[-1]

        # Add some padding
        range_padding = (max_return - min_return) * 0.1
        min_return -= range_padding
        max_return += range_padding

        return min_return, max_return

    def _optimize_for_return(
        self,
        mean_returns: pd.Series,
        cov_matrix: np.ndarray,
        target_return: float,
        min_weight: float,
        max_weight: float
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Optimize portfolio for specific target return.

        Args:
            mean_returns: Annual mean returns
            cov_matrix: Annual covariance matrix
            target_return: Target annual return
            min_weight: Minimum weight per asset
            max_weight: Maximum weight per asset

        Returns:
            (weights, volatility) or (None, 0) if optimization fails
        """
        n_assets = len(mean_returns)

        # Objective: Minimize variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints: weights sum to 1, return = target
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
        ]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        # Initial guess: equal weight
        init_weights = np.array([1.0 / n_assets] * n_assets)

        # Optimize
        result = minimize(
            portfolio_variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'disp': False}
        )

        if result.success:
            weights = result.x
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return weights, float(port_vol)
        else:
            return None, 0.0
