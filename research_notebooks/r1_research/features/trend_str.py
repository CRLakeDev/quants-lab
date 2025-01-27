import pandas_ta as ta

from core.features.feature_base import FeatureBase, FeatureConfig

class TrendStrConfig(FeatureConfig):
    name: str = "trend_str"
    rsi_length: int = 14
    adx_period: int = 14

class TrendStr(FeatureBase[TrendStrConfig]):
    def calculate(self, candles):
        rsi_lentgh = self.config.rsi_length
        # Relative Strength Index (RSI)
        candles['rsi'] = ta.rsi(candles['close'], length=rsi_lentgh)  

        adx_period = self.config.adx_period
        # Average Directional Index (ADX) and +DI/-DI
        adx = ta.adx(candles['high'], candles['low'], candles['close'], length=adx_period) 
        candles['adx'] = adx['ADX_14']
        candles['+di'] = adx['DMP_14']
        candles['-di'] = adx['DMN_14']

        return candles


