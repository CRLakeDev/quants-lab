from typing import List

from pydantic import Field, validator
import pandas_ta as ta  # noqa: F401

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers.directional_trading_controller_base import (
    DirectionalTradingControllerBase,
    DirectionalTradingControllerConfigBase,
)

from features.moving import MovingConfig, Moving
from features.trend_str import TrendStrConfig, TrendStr
from features.vol_dynamic_feat import VolDynConfig, VolDyn
from features.volmom import VolMomConfig, VolMom

class R1ControllerConfig(DirectionalTradingControllerConfigBase):
    controller_name = 'r1_controller'
    candles_config: List[CandlesConfig] = []
    candles_connector: str = Field(
        default=None)
    candles_trading_pair: str = Field(
        default=None)
    interval: str = Field(
        default="1h",
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter the candle interval (e.g., 1m, 5m, 1h, 1d): ",
            prompt_on_new=False))
    ema_short: int = Field(
        default=10,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter EMA Short: ",
            prompt_on_new=True)) 
    sma_long: int = Field(
        default=50,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter SMA Long: ",
            prompt_on_new=True)) 
    roc_length: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter ROC length: ",
            prompt_on_new=True)) 
    bbands_length: int = Field(
        default=20,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter BBands length: ",
            prompt_on_new=True)) 
    bbands_std: int = Field(
        default=2,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter BBands std: ",
            prompt_on_new=True)) 
    rsi_length: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter RSI length: ",
            prompt_on_new=True)) 
    adx_period: int = Field(
        default=14,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter ADX period: ",
            prompt_on_new=True)) 
    spike_zscore: int = Field(
        default=2,
        client_data=ClientFieldData(
            prompt=lambda mi: "Enter Z Score spike: ",
            prompt_on_new=True)) 
                    
    @validator("candles_connector", pre=True, always=True)
    def set_candles_connector(cls, v, values):
        if v is None or v == "":
            return values.get("connector_name")
        return v

    @validator("candles_trading_pair", pre=True, always=True)
    def set_candles_trading_pair(cls, v, values):
        if v is None or v == "":
            return values.get("trading_pair")
        return v

class R1Controller(DirectionalTradingControllerBase):
    def __init__(self, config: R1ControllerConfig, *args, **kwargs):
        self.config = config
        self.max_records = max(config.spike_zscore, config.adx_period, config.bbands_length, config.bbands_std,
                               config.sma_long, config.ema_short, config.roc_length, config.rsi_length)
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [CandlesConfig(
                connector=config.candles_connector,
                trading_pair=config.candles_trading_pair,
                interval=config.interval,
                max_records=self.max_records
            )]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(connector_name=self.config.candles_connector,
                                                      trading_pair=self.config.candles_trading_pair,
                                                      interval=self.config.interval,
                                                      max_records=self.max_records)
        # Add indicators
        df = self.calculate_techs(df)
        
        signalconfig = {}
        signalconfig["rsi_oversold"] = 30
        signalconfig["rsi_overbought"] = 70
        signalconfig["adx_strong_trend"] = 25
        signalconfig["adx_weak_trend"] = 20
        signalconfig["whale_trade_zscore"] = 2
        
        df = self.calculateSignal(df, signalconfig)

        # Update processed data
        self.processed_data["signal"] = df["signal"].iloc[-1]
        self.processed_data["features"] = df

    def calculate_techs(self, df):
        ema_short = self.config.ema_short
        sma_long = self.config.sma_long
        df['ema_short'] = ta.ema(df['close'], length=ema_short)
        df['sma_long'] = ta.sma(df['close'], length=sma_long)

        roc_length = self.config.roc_length
        bbands_length = self.config.bbands_length
        bbands_std = self.config.bbands_std
        # Volatility-Adjusted Momentum Calculation
        # Rate of Change (ROC)
        df['roc'] = ta.roc(df['close'], length=roc_length)

        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=bbands_length, std=bbands_std)

        df['bb_upper'] = bbands['BBU_20_2.0']
        # df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']

        rsi_lentgh = self.config.rsi_length
        # Relative Strength Index (RSI)
        df['rsi'] = ta.rsi(df['close'], length=rsi_lentgh)  

        adx_period = self.config.adx_period
        # Average Directional Index (ADX) and +DI/-DI
        adx = ta.adx(df['high'], df['low'], df['close'], length=adx_period) 
        df['adx'] = adx['ADX_14']
        df['+di'] = adx['DMP_14']
        df['-di'] = adx['DMN_14']
        
        spike_zscore = self.config.spike_zscore

        df["obv"] = ta.obv(df["close"], df["volume"])
        # Calculate z-score to identify anomalous volume spikes
        df["vol_zscore"] = (df["volume"] - df["volume"].mean()) / df["volume"].std()
        # Flag significant volume spikes (z-score > 2)
        df["vol_spike"] = df["vol_zscore"] > spike_zscore
        
        return df

    @staticmethod
    def calculateSignal(candles, signalconfig):
        # Price crosses above the EMA: Bullish signal (go long).
        mov_av_long_signal = candles["ema_short"] > candles["sma_long"]

        # Price crosses below the EMA: Bearish signal (go short).
        mov_av_short_signal = candles["ema_short"] < candles["sma_long"]

        # ROC > 0: Uptrend gaining momentum.
        roc_long_signal = candles["roc"] > 0
        # ROC < 0: Downtrend gaining momentum.
        roc_short_signal = candles["roc"] < 0
        # ROC crossing a high threshold (e.g., 5%) signals strong momentum bursts, useful for entering.

        # Overbought or breakout confirmation (short signal)
        bb_long_signal = candles["close"] > candles["bb_upper"]
        # Oversold or trend exhaustion (long signal)
        bb_short_signal = candles["close"] < candles["bb_lower"]

        # # Identify volatility squeezes (Narrow bands)
        # candles['band_width'] = candles['upper_band'] - candles['lower_band']
        # candles['volatility_squeeze'] = (candles['band_width'] / candles['band_width'].rolling(window=20).mean()) < 0.8

        # RSI < 30: Oversold (consider entering long positions).
        rsi_long_signal = candles["rsi"] < signalconfig["rsi_oversold"]
        # RSI > 70: Overbought (consider taking profit or going short).
        rsi_short_signal = candles["rsi"] > signalconfig["rsi_oversold"]

        # ADX > 25: Strong trend.
        adx_long_signal = (candles["adx"] > signalconfig["adx_strong_trend"]) & (candles["+di"] > candles["-di"])
        # ADX < 20: Weak or ranging market.
        adx_short_signal = (candles["adx"] < signalconfig["adx_weak_trend"]) & (candles["-di"] > candles["+di"])
        # Use +DI and -DI for direction confirmation:
        # +DI > -DI: Uptrend.
        # -DI > +DI: Downtrend.

        # Volume Confirmation
        avg_volume = candles["volume"].rolling(window=21).mean()
        strong_volume = candles["volume"] > 2 * avg_volume  # Strong breakout/breakdown volume confirmation
        weak_volume = candles["volume"] < avg_volume  # Low conviction for trend moves

        # On-Balance Volume (OBV)
        obv_change = candles["obv"].diff()
        obv_trend = (obv_change > 0).astype(int) - (obv_change < 0).astype(int)  # +1 for up, -1 for down

        # OBV divergence signals
        price_diff = candles["close"].diff()
        obv_divergence = ((price_diff > 0) & (obv_change <= 0)) | (  # Price up, OBV flat/down
            (price_diff < 0) & (obv_change >= 0)
        )  # Price down, OBV flat/up

        # Whale Trades or Volume Spikes
        whale_trade = candles["vol_zscore"] > signalconfig["whale_trade_zscore"]  # Significant volume spike based on Z-score

        # long_signal = mov_av_long_signal & roc_long_signal  & adx_long_signal & strong_volume & (obv_trend > 0)
        # short_signal = mov_av_short_signal & roc_short_signal & adx_short_signal & adx_short_signal & weak_volume  & (obv_trend < 0)
        long_signal = mov_av_long_signal & roc_long_signal 
        short_signal = mov_av_short_signal & roc_short_signal 
        
        candles["signal"] = 0
        candles.loc[long_signal, "signal"] = 1
        candles.loc[short_signal, "signal"] = -1
            
        return candles