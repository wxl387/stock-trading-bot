"""
Feature engineering module for technical indicators and ML features.
"""
import logging
from typing import List, Optional
import pandas as pd
import numpy as np

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for stock data.
    Adds technical indicators and derived features for ML models.
    """

    def __init__(self):
        """Initialize FeatureEngineer."""
        if not HAS_TA:
            logger.warning("'ta' library not installed. Some indicators may not be available.")

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all available features to the DataFrame.

        Args:
            df: DataFrame with OHLCV data.

        Returns:
            DataFrame with added features.
        """
        df = df.copy()

        # Price features
        df = self.add_price_features(df)

        # Technical indicators
        df = self.add_trend_indicators(df)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_volume_indicators(df)

        # Statistical features
        df = self.add_statistical_features(df)

        # Time features
        df = self.add_time_features(df)

        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        df = df.copy()

        # Returns
        df["returns_1d"] = df["close"].pct_change(1, fill_method=None)
        df["returns_5d"] = df["close"].pct_change(5, fill_method=None)
        df["returns_20d"] = df["close"].pct_change(20, fill_method=None)

        # Log returns
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Price ratios
        df["high_low_ratio"] = df["high"] / df["low"]
        df["close_open_ratio"] = df["close"] / df["open"]

        # Gap analysis
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["gap_up"] = (df["gap"] > 0.01).astype(int)
        df["gap_down"] = (df["gap"] < -0.01).astype(int)

        # True range
        df["true_range"] = self._calculate_true_range(df)

        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        df = df.copy()

        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()

        # Exponential Moving Averages
        for period in [12, 26, 50]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Price position relative to MAs
        df["price_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
        df["price_vs_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

        # MA crossovers
        df["sma_cross_20_50"] = (df["sma_20"] > df["sma_50"]).astype(int)

        if HAS_TA:
            # ADX - Average Directional Index
            adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"])
            df["adx"] = adx.adx()
            df["adx_pos"] = adx.adx_pos()
            df["adx_neg"] = adx.adx_neg()

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        df = df.copy()

        # RSI
        df["rsi_14"] = self._calculate_rsi(df["close"], period=14)
        df["rsi_7"] = self._calculate_rsi(df["close"], period=7)

        # RSI zones
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)

        if HAS_TA:
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"])
            df["stoch_k"] = stoch.stoch()
            df["stoch_d"] = stoch.stoch_signal()

            # Williams %R
            df["williams_r"] = ta.momentum.WilliamsRIndicator(
                df["high"], df["low"], df["close"]
            ).williams_r()

            # CCI - Commodity Channel Index
            df["cci"] = ta.trend.CCIIndicator(
                df["high"], df["low"], df["close"]
            ).cci()

            # ROC - Rate of Change
            df["roc"] = ta.momentum.ROCIndicator(df["close"]).roc()

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        df = df.copy()

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_width"] = (bb_range / df["bb_middle"]).replace([np.inf, -np.inf], np.nan)
        df["bb_position"] = ((df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan))

        # ATR - Average True Range
        df["atr_14"] = self._calculate_atr(df, period=14)
        df["atr_pct"] = (df["atr_14"] / df["close"].replace(0, np.nan))

        # Historical Volatility
        df["volatility_20d"] = df["log_returns"].rolling(window=20).std() * np.sqrt(252)

        if HAS_TA:
            # Keltner Channel
            keltner = ta.volatility.KeltnerChannel(df["high"], df["low"], df["close"])
            df["keltner_high"] = keltner.keltner_channel_hband()
            df["keltner_low"] = keltner.keltner_channel_lband()

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        df = df.copy()

        # Volume moving averages
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = (df["volume"] / df["volume_sma_20"].replace(0, np.nan))

        # On-Balance Volume
        df["obv"] = self._calculate_obv(df)

        # Volume Price Trend
        df["vpt"] = (df["close"].pct_change(fill_method=None) * df["volume"]).cumsum()

        if HAS_TA:
            # VWAP (approximate daily)
            df["vwap"] = (
                (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() /
                df["volume"].cumsum()
            )

            # Money Flow Index
            mfi = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"])
            df["mfi"] = mfi.money_flow_index()

            # Chaikin Money Flow
            cmf = ta.volume.ChaikinMoneyFlowIndicator(
                df["high"], df["low"], df["close"], df["volume"]
            )
            df["cmf"] = cmf.chaikin_money_flow()

        return df

    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        df = df.copy()

        # Rolling statistics
        for window in [5, 20]:
            df[f"returns_mean_{window}d"] = df["returns_1d"].rolling(window=window).mean()
            df[f"returns_std_{window}d"] = df["returns_1d"].rolling(window=window).std()
            df[f"returns_skew_{window}d"] = df["returns_1d"].rolling(window=window).skew()
            df[f"returns_kurt_{window}d"] = df["returns_1d"].rolling(window=window).kurt()

        # Z-scores
        df["close_zscore_20d"] = (
            (df["close"] - df["close"].rolling(window=20).mean()) /
            df["close"].rolling(window=20).std()
        )

        # Autocorrelation
        df["returns_autocorr_5d"] = df["returns_1d"].rolling(window=20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 6 else np.nan, raw=False
        )

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            df["day_of_week"] = df.index.dayofweek
            df["month"] = df.index.month
            df["quarter"] = df.index.quarter
            df["is_month_start"] = df.index.is_month_start.astype(int)
            df["is_month_end"] = df.index.is_month_end.astype(int)
            df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
            df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

        return df

    def create_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Create prediction labels for ML training.

        Args:
            df: DataFrame with price data.
            horizon: Number of days ahead to predict.
            threshold: Return threshold for classification.

        Returns:
            DataFrame with added label columns.
        """
        df = df.copy()

        # Future returns
        df["future_returns"] = df["close"].shift(-horizon) / df["close"] - 1

        # Binary classification: up or down
        df["label_binary"] = (df["future_returns"] > 0).astype(int)

        # Three-class classification: up, down, neutral
        df["label_3class"] = 1  # neutral
        df.loc[df["future_returns"] > threshold, "label_3class"] = 2  # up
        df.loc[df["future_returns"] < -threshold, "label_3class"] = 0  # down

        return df

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(100.0)  # RSI=100 when loss=0 (pure uptrend)

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        tr = self._calculate_true_range(df)
        return tr.rolling(window=period).mean()

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        if len(df) == 0:
            return obv
        obv.iloc[0] = 0

        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Add sentiment features from news analysis (FinBERT NLP).

        Features added:
        - sentiment_score: Daily avg sentiment (-1 to +1)
        - sentiment_ma5: 5-day rolling average
        - sentiment_volatility: 10-day rolling std
        - article_count: Daily article volume
        - sentiment_momentum: Change in sentiment (1-day diff)
        - sentiment_dispersion: Std of individual article scores
        - positive_ratio: % of positive articles
        - news_intensity: Article count relative to 20-day avg

        Args:
            df: DataFrame with price data
            symbol: Stock symbol for sentiment lookup
            use_cache: Whether to use cached sentiment data

        Returns:
            DataFrame with added sentiment features
        """
        df = df.copy()

        neutral_features = {
            "sentiment_score": 0.0,
            "sentiment_ma5": 0.0,
            "sentiment_volatility": 0.0,
            "article_count": 0,
            "sentiment_momentum": 0.0,
            "sentiment_dispersion": 0.0,
            "positive_ratio": 0.0,
            "news_intensity": 0.0,
        }

        if symbol is None:
            for feat, val in neutral_features.items():
                df[feat] = val
            return df

        try:
            from src.data.sentiment_fetcher import get_sentiment_fetcher

            fetcher = get_sentiment_fetcher()
            df = fetcher.get_sentiment_features(symbol, df, use_cache=use_cache)

            # Ensure all expected columns exist
            for feat, val in neutral_features.items():
                if feat not in df.columns:
                    df[feat] = val

        except Exception as e:
            logger.warning(f"Could not add sentiment features: {e}")
            for feat, val in neutral_features.items():
                df[feat] = val

        return df

    def add_macro_features(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Add macroeconomic indicators as features.

        Args:
            df: DataFrame with price data
            indicators: List of macro indicators to include
            use_cache: Whether to use cached macro data

        Returns:
            DataFrame with added macro features
        """
        df = df.copy()

        if indicators is None:
            indicators = ["vix", "unemployment", "cpi", "treasury_10y"]

        try:
            from src.data.macro_fetcher import get_macro_fetcher

            fetcher = get_macro_fetcher()
            df = fetcher.get_macro_features(df, indicators=indicators, use_cache=use_cache)

        except Exception as e:
            logger.warning(f"Could not add macro features: {e}")
            for ind in indicators:
                df[ind] = 0.0
                df[f'{ind}_ma20'] = 0.0

        return df

    def add_all_features_extended(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        include_sentiment: bool = False,
        include_macro: bool = False,
        include_cross_asset: bool = False,
        include_interactions: bool = False,
        include_lagged: bool = False,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Add all features including optional sentiment, macro, cross-asset, interactions, and lagged.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for sentiment/cross-asset lookup
            include_sentiment: Whether to include sentiment features
            include_macro: Whether to include macroeconomic features
            include_cross_asset: Whether to include cross-asset features (SPY/QQQ correlation)
            include_interactions: Whether to include feature interaction terms
            include_lagged: Whether to include lagged indicator features
            use_cache: Whether to use cached data

        Returns:
            DataFrame with all features added
        """
        # Add standard features
        df = self.add_all_features(df)

        # Add sentiment features if requested
        if include_sentiment:
            df = self.add_sentiment_features(df, symbol=symbol, use_cache=use_cache)

        # Add macro features if requested
        if include_macro:
            df = self.add_macro_features(df, use_cache=use_cache)

        # Add cross-asset features if requested
        if include_cross_asset:
            df = self.add_cross_asset_features(df, symbol=symbol, use_cache=use_cache)

        # Add interaction features if requested
        if include_interactions:
            df = self.add_interaction_features(df)

        # Add lagged features if requested (do this last, after all other features)
        if include_lagged:
            df = self.add_lagged_features(df, lag_days=[1, 5])

        return df

    def add_cross_asset_features(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Add cross-asset features including sector correlation and relative strength.

        Features added:
        - relative_strength_spy: Stock return vs SPY return (20-day rolling)
        - relative_strength_qqq: Stock return vs QQQ return (20-day rolling)
        - market_beta: Rolling beta vs SPY (60-day)
        - correlation_spy: Rolling correlation with SPY (20-day)
        - correlation_qqq: Rolling correlation with QQQ (20-day)
        """
        df = df.copy()

        try:
            from src.data.data_fetcher import DataFetcher
            fetcher = DataFetcher()

            # Get date range from df
            if isinstance(df.index, pd.DatetimeIndex):
                start_date = df.index.min().strftime('%Y-%m-%d')
                end_date = df.index.max().strftime('%Y-%m-%d')
            else:
                start_date = None
                end_date = None

            # Calculate stock returns
            stock_returns = df['close'].pct_change()

            # Fetch and process SPY data
            try:
                spy_df = fetcher.fetch_historical(
                    symbol='SPY',
                    start_date=start_date,
                    end_date=end_date,
                    period='1y',
                    use_cache=use_cache
                )
                if spy_df is not None and not spy_df.empty:
                    spy_returns = spy_df['close'].pct_change()
                    aligned_spy = spy_returns.reindex(df.index).fillna(0)

                    # Relative strength vs SPY
                    df['relative_strength_spy'] = (
                        stock_returns.rolling(20).mean() - aligned_spy.rolling(20).mean()
                    )

                    # Rolling correlation with SPY
                    df['correlation_spy'] = stock_returns.rolling(20).corr(aligned_spy)

                    # Rolling beta (covariance / variance)
                    cov = stock_returns.rolling(60).cov(aligned_spy)
                    var = aligned_spy.rolling(60).var()
                    df['market_beta'] = cov / var.replace(0, np.nan)
                else:
                    df['relative_strength_spy'] = 0.0
                    df['correlation_spy'] = 0.0
                    df['market_beta'] = 1.0
            except Exception as e:
                logger.warning(f"Could not fetch SPY data: {e}")
                df['relative_strength_spy'] = 0.0
                df['correlation_spy'] = 0.0
                df['market_beta'] = 1.0

            # Fetch and process QQQ data
            try:
                qqq_df = fetcher.fetch_historical(
                    symbol='QQQ',
                    start_date=start_date,
                    end_date=end_date,
                    period='1y',
                    use_cache=use_cache
                )
                if qqq_df is not None and not qqq_df.empty:
                    qqq_returns = qqq_df['close'].pct_change()
                    aligned_qqq = qqq_returns.reindex(df.index).fillna(0)

                    df['relative_strength_qqq'] = (
                        stock_returns.rolling(20).mean() - aligned_qqq.rolling(20).mean()
                    )
                    df['correlation_qqq'] = stock_returns.rolling(20).corr(aligned_qqq)
                else:
                    df['relative_strength_qqq'] = 0.0
                    df['correlation_qqq'] = 0.0
            except Exception as e:
                logger.warning(f"Could not fetch QQQ data: {e}")
                df['relative_strength_qqq'] = 0.0
                df['correlation_qqq'] = 0.0

            # Fill NaN values
            cross_asset_cols = [
                'relative_strength_spy', 'relative_strength_qqq',
                'correlation_spy', 'correlation_qqq', 'market_beta'
            ]
            for col in cross_asset_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

        except Exception as e:
            logger.warning(f"Could not add cross-asset features: {e}")
            df['relative_strength_spy'] = 0.0
            df['relative_strength_qqq'] = 0.0
            df['correlation_spy'] = 0.0
            df['correlation_qqq'] = 0.0
            df['market_beta'] = 1.0

        return df

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add feature interaction terms that capture complex relationships.

        Features added:
        - rsi_momentum: RSI * momentum (captures RSI in trending markets)
        - rsi_volume: RSI * volume_ratio (volume confirmation of RSI)
        - macd_volatility: MACD * ATR_pct (trend strength in volatility context)
        - trend_macd: SMA_cross * MACD_histogram
        - price_volume: price_vs_sma20 * volume_ratio
        - bb_volume: BB_position * volume_ratio
        - returns_vol_adjusted: returns / volatility (risk-adjusted)
        """
        df = df.copy()

        # RSI interactions
        if 'rsi_14' in df.columns:
            # Normalize RSI to [-1, 1] range for interactions
            rsi_normalized = (df['rsi_14'] - 50) / 50

            if 'returns_5d' in df.columns:
                df['rsi_momentum'] = rsi_normalized * df['returns_5d']

            if 'volume_ratio' in df.columns:
                df['rsi_volume'] = rsi_normalized * df['volume_ratio']

        # MACD interactions
        if 'macd_histogram' in df.columns:
            if 'atr_pct' in df.columns:
                df['macd_volatility'] = df['macd_histogram'] * df['atr_pct'] * 100

            if 'sma_cross_20_50' in df.columns:
                df['trend_macd'] = df['sma_cross_20_50'] * df['macd_histogram']

        # Price-Volume interactions
        if 'price_vs_sma20' in df.columns and 'volume_ratio' in df.columns:
            df['price_volume'] = df['price_vs_sma20'] * df['volume_ratio']

        # Bollinger Band position with volume
        if 'bb_position' in df.columns and 'volume_ratio' in df.columns:
            df['bb_volume'] = df['bb_position'] * df['volume_ratio']

        # Volatility regime interaction
        if 'volatility_20d' in df.columns and 'returns_1d' in df.columns:
            # Returns normalized by volatility (risk-adjusted returns)
            vol_daily = df['volatility_20d'].replace(0, 0.01) / np.sqrt(252)
            df['returns_vol_adjusted'] = df['returns_1d'] / vol_daily

        return df

    def add_lagged_features(
        self,
        df: pd.DataFrame,
        lag_days: List[int] = None
    ) -> pd.DataFrame:
        """
        Add lagged versions of key indicators.

        Args:
            df: DataFrame with features
            lag_days: List of lag periods (e.g., [1, 5] for yesterday and 5 days ago)

        Features added (for each lag):
        - rsi_14_lag{n}: Previous day's RSI
        - macd_lag{n}: Previous day's MACD
        - returns_1d_lag{n}: Previous day's returns
        - volume_ratio_lag{n}: Previous day's volume ratio
        - bb_position_lag{n}: Previous day's BB position
        - atr_pct_lag{n}: Previous day's ATR percentage

        Plus:
        - prior_week_return: 5-day return from 5 days ago
        - prior_week_positive: Was prior week positive (binary)
        - rsi_change_1d, rsi_change_5d: RSI momentum
        - macd_change_1d: MACD momentum
        """
        df = df.copy()

        if lag_days is None:
            lag_days = [1, 5]

        # Key indicators to lag
        indicators_to_lag = [
            'rsi_14', 'macd', 'macd_histogram', 'returns_1d',
            'volume_ratio', 'bb_position', 'atr_pct'
        ]

        for lag in lag_days:
            for indicator in indicators_to_lag:
                if indicator in df.columns:
                    df[f'{indicator}_lag{lag}'] = df[indicator].shift(lag)

        # Add weekly pattern features
        if 'close' in df.columns:
            # Returns from 5-10 days ago (prior week)
            df['prior_week_return'] = df['close'].pct_change(5).shift(5)

            # Was prior week positive?
            df['prior_week_positive'] = (df['prior_week_return'] > 0).astype(int)

        if 'volatility_20d' in df.columns:
            df['volatility_lag5'] = df['volatility_20d'].shift(5)

        # Rate of change features (momentum of indicators)
        if 'rsi_14' in df.columns:
            df['rsi_change_1d'] = df['rsi_14'].diff(1)
            df['rsi_change_5d'] = df['rsi_14'].diff(5)

        if 'macd' in df.columns:
            df['macd_change_1d'] = df['macd'].diff(1)

        return df

    def add_fundamental_features(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Add fundamental features from financial data.

        Features added:
        - pe_ratio: Price-to-Earnings ratio
        - pe_percentile: P/E percentile vs market (requires market_pe_values)
        - peg_ratio: Price/Earnings to Growth ratio
        - price_to_book: Price-to-Book ratio
        - debt_to_equity: Debt-to-Equity ratio
        - profit_margin: Profit margin
        - roe: Return on Equity
        - roa: Return on Assets
        - revenue_growth_yoy: Year-over-year revenue growth
        - earnings_growth_yoy: Year-over-year earnings growth
        - days_to_earnings: Days until next earnings report
        - near_earnings: Binary flag (1 if earnings within 14 days)

        Args:
            df: DataFrame with price data
            symbol: Stock symbol for fundamental lookup
            use_cache: Whether to use cached fundamental data

        Returns:
            DataFrame with added fundamental features
        """
        df = df.copy()

        # Default neutral values
        neutral_features = {
            "pe_ratio": 0.0,
            "pe_percentile": 50.0,
            "peg_ratio": 0.0,
            "price_to_book": 0.0,
            "debt_to_equity": 0.0,
            "profit_margin": 0.0,
            "roe": 0.0,
            "roa": 0.0,
            "revenue_growth_yoy": 0.0,
            "earnings_growth_yoy": 0.0,
            "days_to_earnings": 90.0,
            "near_earnings": 0,
        }

        if symbol is None:
            for feat, val in neutral_features.items():
                df[feat] = val
            return df

        try:
            from src.data.fundamental_fetcher import get_fundamental_fetcher

            fetcher = get_fundamental_fetcher()
            fundamentals = fetcher.fetch_fundamentals(symbol, use_cache=use_cache)

            # Add features as constant columns (fundamentals are point-in-time)
            df["pe_ratio"] = fundamentals.pe_ratio or 0.0
            df["pe_percentile"] = 50.0  # Would need market comparison
            df["peg_ratio"] = fundamentals.peg_ratio or 0.0
            df["price_to_book"] = fundamentals.price_to_book or 0.0
            df["debt_to_equity"] = fundamentals.debt_to_equity or 0.0
            df["profit_margin"] = fundamentals.profit_margin or 0.0
            df["roe"] = fundamentals.roe or 0.0
            df["roa"] = fundamentals.roa or 0.0
            df["revenue_growth_yoy"] = fundamentals.revenue_growth_yoy or 0.0
            df["earnings_growth_yoy"] = fundamentals.earnings_growth_yoy or 0.0

            # Days to earnings
            if fundamentals.days_to_earnings is not None:
                df["days_to_earnings"] = float(fundamentals.days_to_earnings)
                df["near_earnings"] = int(fundamentals.days_to_earnings <= 14)
            else:
                df["days_to_earnings"] = 90.0
                df["near_earnings"] = 0

            # Ensure all expected columns exist
            for feat, val in neutral_features.items():
                if feat not in df.columns:
                    df[feat] = val

        except Exception as e:
            logger.warning(f"Could not add fundamental features for {symbol}: {e}")
            for feat, val in neutral_features.items():
                df[feat] = val

        return df

    def add_all_features_with_fundamentals(
        self,
        df: pd.DataFrame,
        symbol: Optional[str] = None,
        include_sentiment: bool = False,
        include_macro: bool = False,
        include_fundamentals: bool = True,
        include_cross_asset: bool = False,
        include_interactions: bool = False,
        include_lagged: bool = False,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Add all features including fundamentals.

        This is an enhanced version of add_all_features_extended that includes
        fundamental data from financial statements.

        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for data lookup
            include_sentiment: Whether to include sentiment features
            include_macro: Whether to include macroeconomic features
            include_fundamentals: Whether to include fundamental features
            include_cross_asset: Whether to include cross-asset features
            include_interactions: Whether to include feature interactions
            include_lagged: Whether to include lagged features
            use_cache: Whether to use cached data

        Returns:
            DataFrame with all features added
        """
        # Add all standard features first
        df = self.add_all_features_extended(
            df=df,
            symbol=symbol,
            include_sentiment=include_sentiment,
            include_macro=include_macro,
            include_cross_asset=include_cross_asset,
            include_interactions=include_interactions,
            include_lagged=include_lagged,
            use_cache=use_cache
        )

        # Add fundamental features if requested
        if include_fundamentals:
            df = self.add_fundamental_features(df, symbol=symbol, use_cache=use_cache)

        return df

    def get_feature_names(self, include_sentiment: bool = False, include_macro: bool = False) -> List[str]:
        """
        Get list of feature names that will be created.

        Args:
            include_sentiment: Include sentiment feature names
            include_macro: Include macro feature names

        Returns:
            List of feature column names.
        """
        features = [
            # Price features
            "returns_1d", "returns_5d", "returns_20d", "log_returns",
            "high_low_ratio", "close_open_ratio", "gap", "true_range",

            # Trend indicators
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_12", "ema_26", "ema_50",
            "macd", "macd_signal", "macd_histogram",
            "price_vs_sma20", "price_vs_sma50", "sma_cross_20_50",
            "adx", "adx_pos", "adx_neg",

            # Momentum indicators
            "rsi_14", "rsi_7", "rsi_oversold", "rsi_overbought",
            "stoch_k", "stoch_d", "williams_r", "cci", "roc",

            # Volatility indicators
            "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
            "atr_14", "atr_pct", "volatility_20d",

            # Volume indicators
            "volume_sma_20", "volume_ratio", "obv", "vpt", "vwap", "mfi", "cmf",

            # Statistical features
            "returns_mean_5d", "returns_std_5d",
            "returns_mean_20d", "returns_std_20d",
            "close_zscore_20d",

            # Time features
            "day_of_week", "month", "quarter"
        ]

        if include_sentiment:
            features.extend([
                "sentiment_score", "sentiment_ma5", "sentiment_volatility",
                "article_count", "sentiment_momentum", "sentiment_dispersion",
                "positive_ratio", "news_intensity"
            ])

        if include_macro:
            features.extend([
                "vix", "vix_ma20",
                "unemployment", "unemployment_ma20",
                "cpi", "cpi_ma20",
                "treasury_10y", "treasury_10y_ma20"
            ])

        return features
