import numpy as np
import pandas as pd
import pandas_ta as ta


from core.features.feature_base import FeatureBase, FeatureConfig

class VolMomConfig(FeatureConfig):
    name: str = "volmom"
    roc_length: int = 14
    bbands_length: int = 20
    bbands_std: int = 2

class VolMom(FeatureBase[VolMomConfig]):
    def calculate(self, candles):
        # Extract the configuration parameters

        roc_length = self.config.roc_length
        bbands_length = self.config.bbands_length
        bbands_std = self.config.bbands_std
        
        # Volatility-Adjusted Momentum Calculation
        # Rate of Change (ROC)
        candles['roc'] = ta.roc(candles['close'], length=roc_length)  # 14-period ROC

        # Bollinger Bands
        # bbands = ta.bbands(candles['close'], length=bbands_std, std=bbands_length)  # 20-period Bollinger Bands with 2 std deviation
        bbands = ta.bbands(candles['close'], length=bbands_length, std=bbands_std)  # 20-period Bollinger Bands with 2 std deviation

        candles['bb_upper'] = bbands['BBU_20_2.0']
        candles['bb_middle'] = bbands['BBM_20_2.0']
        candles['bb_lower'] = bbands['BBL_20_2.0']

        return candles