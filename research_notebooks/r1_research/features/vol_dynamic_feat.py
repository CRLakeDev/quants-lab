import pandas_ta as ta

from core.features.feature_base import FeatureBase, FeatureConfig


class VolDynConfig(FeatureConfig):
    name: str = "voldyn"
    spike_zscore: int = 2


class VolDyn(FeatureBase[VolDynConfig]):
    def calculate(self, candles):
        spike_zscore = self.config.spike_zscore

        candles["obv"] = ta.obv(candles["close"], candles["volume"])

        # Calculate z-score to identify anomalous volume spikes
        candles["vol_zscore"] = (candles["volume"] - candles["volume"].mean()) / candles["volume"].std()

        # Flag significant volume spikes (z-score > 2)
        candles["vol_spike"] = candles["vol_zscore"] > spike_zscore

        return candles
