"""
Technical Indicators Module for Trading Bot
Implements various technical analysis indicators including:
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- And more...
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Class for calculating various technical indicators"""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with OHLCV data

        Args:
            data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.data = data.copy()
        self.close = data['Close']
        self.high = data['High']
        self.low = data['Low']
        self.open = data['Open']
        self.volume = data['Volume'] if 'Volume' in data.columns else None

    def sma(self, period: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return self.close.rolling(window=period).mean()

    def ema(self, period: int = 20) -> pd.Series:
        """Exponential Moving Average"""
        return self.close.ewm(span=period).mean()

    def rsi(self, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = self.close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Moving Average Convergence Divergence"""
        ema_fast = self.ema(fast)
        ema_slow = self.ema(slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def bollinger_bands(self, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = self.sma(period)
        std = self.close.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

    def stochastic_oscillator(self, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        low_min = self.low.rolling(window=k_period).min()
        high_max = self.high.rolling(window=k_period).max()

        k_percent = 100 * ((self.close - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()

        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }

    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = self.high - self.low
        high_close = np.abs(self.high - self.close.shift())
        low_close = np.abs(self.low - self.close.shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range.rolling(window=period).mean()

    def williams_r(self, period: int = 14) -> pd.Series:
        """Williams %R"""
        high_max = self.high.rolling(window=period).max()
        low_min = self.low.rolling(window=period).min()

        williams_r = -100 * ((high_max - self.close) / (high_max - low_min))
        return williams_r

    def cci(self, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (self.high + self.low + self.close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )

        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci

    def obv(self) -> pd.Series:
        """On-Balance Volume"""
        if self.volume is None:
            return pd.Series(index=self.close.index)

        obv = pd.Series(index=self.close.index, dtype=float)
        obv.iloc[0] = self.volume.iloc[0]

        for i in range(1, len(self.close)):
            if self.close.iloc[i] > self.close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + self.volume.iloc[i]
            elif self.close.iloc[i] < self.close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - self.volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def fibonacci_retracement(self, high_price: float = None, low_price: float = None) -> Dict[str, float]:
        """Fibonacci Retracement Levels"""
        if high_price is None:
            high_price = self.high.max()
        if low_price is None:
            low_price = self.low.min()

        diff = high_price - low_price

        levels = {
            '0%': high_price,
            '23.6%': high_price - 0.236 * diff,
            '38.2%': high_price - 0.382 * diff,
            '50%': high_price - 0.5 * diff,
            '61.8%': high_price - 0.618 * diff,
            '78.6%': high_price - 0.786 * diff,
            '100%': low_price
        }

        return levels

    def pivot_points(self) -> Dict[str, float]:
        """Pivot Points (using last day's data)"""
        if len(self.data) < 1:
            return {}

        last_high = self.high.iloc[-1]
        last_low = self.low.iloc[-1]
        last_close = self.close.iloc[-1]

        pivot = (last_high + last_low + last_close) / 3

        r1 = 2 * pivot - last_low
        s1 = 2 * pivot - last_high
        r2 = pivot + (last_high - last_low)
        s2 = pivot - (last_high - last_low)
        r3 = last_high + 2 * (pivot - last_low)
        s3 = last_low - 2 * (last_high - pivot)

        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }

    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all indicators and return as DataFrame"""
        result_df = self.data.copy()

        # Moving Averages
        result_df['SMA_10'] = self.sma(10)
        result_df['SMA_20'] = self.sma(20)
        result_df['SMA_50'] = self.sma(50)
        result_df['EMA_10'] = self.ema(10)
        result_df['EMA_20'] = self.ema(20)
        result_df['EMA_50'] = self.ema(50)

        # RSI
        result_df['RSI'] = self.rsi(14)

        # MACD
        macd_data = self.macd()
        result_df['MACD'] = macd_data['macd']
        result_df['MACD_Signal'] = macd_data['signal']
        result_df['MACD_Histogram'] = macd_data['histogram']

        # Bollinger Bands
        bb_data = self.bollinger_bands()
        result_df['BB_Upper'] = bb_data['upper']
        result_df['BB_Middle'] = bb_data['middle']
        result_df['BB_Lower'] = bb_data['lower']

        # Stochastic
        stoch_data = self.stochastic_oscillator()
        result_df['Stoch_K'] = stoch_data['k_percent']
        result_df['Stoch_D'] = stoch_data['d_percent']

        # Other indicators
        result_df['ATR'] = self.atr()
        result_df['Williams_R'] = self.williams_r()
        result_df['CCI'] = self.cci()

        if self.volume is not None:
            result_df['OBV'] = self.obv()

        return result_df

    def get_trading_signals(self) -> Dict[str, str]:
        """
        Generate basic trading signals based on indicators

        Returns:
            Dictionary with signal types and their current status
        """
        signals = {}

        if len(self.data) < 50:
            return {'error': 'Insufficient data for signal generation'}

        # RSI signals
        current_rsi = self.rsi().iloc[-1]
        if current_rsi > 70:
            signals['RSI'] = 'SELL'
        elif current_rsi < 30:
            signals['RSI'] = 'BUY'
        else:
            signals['RSI'] = 'HOLD'

        # MACD signals
        macd_data = self.macd()
        if macd_data['macd'].iloc[-1] > macd_data['signal'].iloc[-1]:
            signals['MACD'] = 'BUY'
        else:
            signals['MACD'] = 'SELL'

        # Moving Average signals
        sma_20 = self.sma(20).iloc[-1]
        sma_50 = self.sma(50).iloc[-1]
        current_price = self.close.iloc[-1]

        if current_price > sma_20 > sma_50:
            signals['MA_Trend'] = 'BUY'
        elif current_price < sma_20 < sma_50:
            signals['MA_Trend'] = 'SELL'
        else:
            signals['MA_Trend'] = 'HOLD'

        # Bollinger Bands signals
        bb_data = self.bollinger_bands()
        if current_price > bb_data['upper'].iloc[-1]:
            signals['Bollinger'] = 'SELL'
        elif current_price < bb_data['lower'].iloc[-1]:
            signals['Bollinger'] = 'BUY'
        else:
            signals['Bollinger'] = 'HOLD'

        return signals


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # Generate synthetic OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)

    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': close_prices + np.random.randn(len(dates)) * 0.2,
        'High': close_prices + np.abs(np.random.randn(len(dates)) * 0.5),
        'Low': close_prices - np.abs(np.random.randn(len(dates)) * 0.5),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    })

    # Test technical indicators
    indicators = TechnicalIndicators(sample_data)

    print("RSI (last 5 values):")
    print(indicators.rsi().tail())

    print("\nMACD signals:")
    macd_data = indicators.macd()
    print(f"MACD: {macd_data['macd'].iloc[-1]:.4f}")
    print(f"Signal: {macd_data['signal'].iloc[-1]:.4f}")

    print("\nTrading Signals:")
    signals = indicators.get_trading_signals()
    for signal_type, signal in signals.items():
        print(f"{signal_type}: {signal}")
