"""
Correlation analysis and diversification metrics for portfolio construction.

Analyzes correlation structure, identifies correlated asset clusters,
and calculates diversification metrics.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import LedoitWolf

from src.data.data_fetcher import DataFetcher

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Analyzes correlation structure of portfolio holdings.

    Features:
    - Correlation matrix calculation with shrinkage
    - Diversification ratio
    - Cluster analysis for correlated assets
    - Concentration risk warnings
    """

    def __init__(
        self,
        lookback_days: int = 252,
        correlation_threshold: float = 0.8,
        use_shrinkage: bool = True
    ):
        """
        Initialize correlation analyzer.

        Args:
            lookback_days: Days of historical data for correlation
            correlation_threshold: Threshold for high correlation warnings
            use_shrinkage: Use Ledoit-Wolf covariance shrinkage
        """
        self.lookback_days = lookback_days
        self.correlation_threshold = correlation_threshold
        self.use_shrinkage = use_shrinkage

        self.data_fetcher = DataFetcher()

        logger.info(f"Initialized CorrelationAnalyzer: lookback={lookback_days}d, "
                   f"threshold={correlation_threshold}")

    def calculate_correlation_matrix(
        self,
        symbols: List[str],
        returns: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for symbols.

        Args:
            symbols: List of symbols
            returns: Optional pre-computed returns DataFrame

        Returns:
            Correlation matrix (DataFrame)
        """
        try:
            if returns is None:
                returns = self._fetch_returns(symbols)

            if returns.empty:
                logger.warning("No returns data available for correlation calculation")
                return pd.DataFrame()

            # Calculate correlation
            if self.use_shrinkage:
                # Use Ledoit-Wolf shrinkage for more stable estimates
                lw = LedoitWolf()
                cov_matrix = lw.fit(returns).covariance_

                # Convert covariance to correlation
                std_devs = np.sqrt(np.diag(cov_matrix))
                corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

                # Create DataFrame
                corr_df = pd.DataFrame(
                    corr_matrix,
                    index=returns.columns,
                    columns=returns.columns
                )
            else:
                corr_df = returns.corr()

            logger.debug(f"Calculated correlation matrix for {len(symbols)} symbols")

            return corr_df

        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()

    def calculate_diversification_ratio(
        self,
        weights: Dict[str, float],
        returns: Optional[pd.DataFrame] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        volatilities: Optional[pd.Series] = None
    ) -> float:
        """
        Calculate diversification ratio: weighted avg volatility / portfolio volatility.
        Higher ratio = more diversified portfolio.

        Args:
            weights: Portfolio weights
            returns: Optional returns DataFrame
            correlation_matrix: Optional pre-computed correlation matrix
            volatilities: Optional pre-computed volatilities

        Returns:
            Diversification ratio (higher is better)
        """
        try:
            symbols = list(weights.keys())

            # Get returns if not provided
            if returns is None:
                returns = self._fetch_returns(symbols)
                if returns.empty:
                    return 0.0

            # Calculate volatilities
            if volatilities is None:
                volatilities = returns.std() * np.sqrt(252)  # Annualize

            # Calculate correlation matrix
            if correlation_matrix is None:
                correlation_matrix = self.calculate_correlation_matrix(
                    symbols, returns
                )
                if correlation_matrix.empty:
                    return 0.0

            # Ensure symbols match
            symbols = [s for s in symbols if s in volatilities.index and s in correlation_matrix.index]
            if not symbols:
                return 0.0

            # Convert to arrays
            weights_array = np.array([weights[s] for s in symbols])
            vols_array = volatilities[symbols].values
            corr_array = correlation_matrix.loc[symbols, symbols].values

            # Weighted average volatility
            weighted_avg_vol = np.dot(weights_array, vols_array)

            # Portfolio volatility (using correlation matrix)
            # σ_p = sqrt(w' * Σ * w) where Σ = D * C * D
            # D is diagonal matrix of volatilities, C is correlation matrix
            cov_matrix = np.outer(vols_array, vols_array) * corr_array
            portfolio_vol = np.sqrt(np.dot(weights_array, np.dot(cov_matrix, weights_array)))

            # Diversification ratio
            if portfolio_vol > 0:
                div_ratio = weighted_avg_vol / portfolio_vol
            else:
                div_ratio = 1.0

            logger.debug(f"Diversification ratio: {div_ratio:.3f} "
                        f"(weighted_vol={weighted_avg_vol:.3f}, "
                        f"portfolio_vol={portfolio_vol:.3f})")

            return float(div_ratio)

        except Exception as e:
            logger.error(f"Error calculating diversification ratio: {e}")
            return 0.0

    def find_correlated_clusters(
        self,
        correlation_matrix: pd.DataFrame,
        n_clusters: Optional[int] = None
    ) -> Dict[int, List[str]]:
        """
        Find clusters of highly correlated assets using hierarchical clustering.

        Args:
            correlation_matrix: Correlation matrix
            n_clusters: Number of clusters (if None, auto-determine)

        Returns:
            Dict mapping cluster_id to list of symbols
        """
        try:
            if correlation_matrix.empty or len(correlation_matrix) < 2:
                return {}

            # Convert correlation to distance (1 - |correlation|)
            distance_matrix = 1 - np.abs(correlation_matrix.values)

            # Auto-determine number of clusters if not specified
            if n_clusters is None:
                # Use elbow method: aim for 2-4 clusters for typical portfolios
                n_symbols = len(correlation_matrix)
                n_clusters = max(2, min(4, n_symbols // 2))

            # Perform hierarchical clustering
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )

            labels = clustering.fit_predict(distance_matrix)

            # Group symbols by cluster
            clusters = {}
            for i, symbol in enumerate(correlation_matrix.index):
                cluster_id = int(labels[i])
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(symbol)

            logger.info(f"Found {len(clusters)} correlation clusters")
            for cluster_id, symbols in clusters.items():
                logger.debug(f"  Cluster {cluster_id}: {symbols}")

            return clusters

        except Exception as e:
            logger.error(f"Error finding correlated clusters: {e}")
            return {}

    def check_concentration_risk(
        self,
        weights: Dict[str, float],
        correlation_matrix: pd.DataFrame,
        max_correlated_exposure: float = 0.4
    ) -> Dict:
        """
        Check for over-concentration in correlated assets.

        Args:
            weights: Portfolio weights
            correlation_matrix: Correlation matrix
            max_correlated_exposure: Max allowed exposure to correlated assets

        Returns:
            Dict with warnings and metrics
        """
        try:
            result = {
                'has_concentration_risk': False,
                'warnings': [],
                'clusters': {},
                'max_cluster_weight': 0.0
            }

            if correlation_matrix.empty or len(weights) < 2:
                return result

            # Find correlated clusters
            clusters = self.find_correlated_clusters(correlation_matrix)
            result['clusters'] = clusters

            # Check each cluster's total weight
            for cluster_id, symbols in clusters.items():
                # Calculate total weight in this cluster
                cluster_weight = sum(weights.get(s, 0.0) for s in symbols)

                result['max_cluster_weight'] = max(
                    result['max_cluster_weight'],
                    cluster_weight
                )

                # Check if cluster weight exceeds threshold
                if cluster_weight > max_correlated_exposure:
                    result['has_concentration_risk'] = True
                    result['warnings'].append(
                        f"Cluster {cluster_id} ({', '.join(symbols)}): "
                        f"{cluster_weight:.1%} exposure exceeds "
                        f"{max_correlated_exposure:.1%} limit"
                    )

            # Check for high-correlation pairs with significant combined weight
            symbols = list(weights.keys())
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i >= j:
                        continue

                    if sym1 not in correlation_matrix.index or sym2 not in correlation_matrix.columns:
                        continue

                    corr = correlation_matrix.loc[sym1, sym2]
                    combined_weight = weights[sym1] + weights[sym2]

                    if abs(corr) > self.correlation_threshold and combined_weight > 0.3:
                        result['has_concentration_risk'] = True
                        result['warnings'].append(
                            f"High correlation pair {sym1}-{sym2}: "
                            f"corr={corr:.2f}, combined weight={combined_weight:.1%}"
                        )

            if result['warnings']:
                logger.warning(f"Concentration risk detected: {len(result['warnings'])} warnings")

            return result

        except Exception as e:
            logger.error(f"Error checking concentration risk: {e}")
            return {'has_concentration_risk': False, 'warnings': [], 'clusters': {}}

    def get_correlation_stats(
        self,
        correlation_matrix: pd.DataFrame
    ) -> Dict:
        """
        Calculate correlation statistics.

        Args:
            correlation_matrix: Correlation matrix

        Returns:
            Dict with correlation statistics
        """
        try:
            if correlation_matrix.empty:
                return {}

            # Extract upper triangle (excluding diagonal)
            n = len(correlation_matrix)
            upper_triangle = []

            for i in range(n):
                for j in range(i + 1, n):
                    upper_triangle.append(correlation_matrix.iloc[i, j])

            correlations = np.array(upper_triangle)

            stats = {
                'mean_correlation': float(np.mean(correlations)),
                'median_correlation': float(np.median(correlations)),
                'max_correlation': float(np.max(correlations)),
                'min_correlation': float(np.min(correlations)),
                'std_correlation': float(np.std(correlations)),
                'n_high_correlation_pairs': int(np.sum(np.abs(correlations) > self.correlation_threshold)),
                'n_total_pairs': len(correlations)
            }

            logger.debug(f"Correlation stats: mean={stats['mean_correlation']:.3f}, "
                        f"max={stats['max_correlation']:.3f}, "
                        f"high_corr_pairs={stats['n_high_correlation_pairs']}")

            return stats

        except Exception as e:
            logger.error(f"Error calculating correlation stats: {e}")
            return {}

    def _fetch_returns(self, symbols: List[str]) -> pd.DataFrame:
        """
        Fetch historical returns for symbols.

        Args:
            symbols: List of symbols

        Returns:
            DataFrame with returns (symbols as columns)
        """
        try:
            returns_data = {}

            for symbol in symbols:
                try:
                    df = self.data_fetcher.fetch_historical(
                        symbol,
                        period=f"{self.lookback_days + 30}d"
                    )
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
