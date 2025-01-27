import numpy as np
import pandas as pd
import pandas_ta as ta


from core.features.feature_base import FeatureBase, FeatureConfig

class MovingConfig(FeatureConfig):
    name: str = "moving"
    ema_short: int = 10
    sma_long: int = 50

class Moving(FeatureBase[MovingConfig]):
    def calculate(self, candles):
        # Extract the configuration parameters
        ema_short = self.config.ema_short
        sma_long = self.config.sma_long

        # Ensure the candles has the required 'close' column
        if 'close' not in candles.columns:
            raise ValueError("Data handler does not contain 'close' column required for trend calculation.")

        # Calculate the trend score for each row
        candles = candles.copy()
        short_avg, long_avg  = self.calculatedAverages(candles['close'], ema_short, sma_long)
        candles['ema_short'] = short_avg
        candles['sma_long'] = long_avg
        return candles


    def calculatedAverages(self, series: pd.Series, short_window: int, long_window: int):

        # Calculating EMA and SMA using pandas_ta
        short_avg = ta.ema(series, length=10)  # 10-period EMA
        long_avg = ta.sma(series, length=50)  # 50-period SMA

        return short_avg, long_avg


