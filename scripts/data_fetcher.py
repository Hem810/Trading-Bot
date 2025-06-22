"""
Data Fetcher Module for Trading Bot
Handles fetching stock data from various APIs including Yahoo Finance, Alpha Vantage
Supports both REST API and WebSocket connections for real-time data
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import asyncio
import websockets
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    """Main class for fetching stock market data from multiple sources"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.websocket_connection = None

    def fetch_yahoo_finance_data(self, symbol: str, period: str = "1y", 
                                interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')

        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period, interval=interval)

            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()

            # Add symbol column
            data['Symbol'] = symbol
            data.reset_index(inplace=True)

            logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def fetch_multiple_symbols(self, symbols: List[str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols

        Args:
            symbols: List of stock symbols
            period: Data period

        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        data_dict = {}
        for symbol in symbols:
            data_dict[symbol] = self.fetch_yahoo_finance_data(symbol, period)
            time.sleep(0.1)  # Avoid rate limiting

        return data_dict

    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        Get current real-time price for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Current price or None if error
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            logger.error(f"Error getting real-time price for {symbol}: {str(e)}")
            return None

    def fetch_alpha_vantage_data(self, symbol: str, function: str = "TIME_SERIES_DAILY") -> pd.DataFrame:
        """
        Fetch data from Alpha Vantage API

        Args:
            symbol: Stock symbol
            function: Alpha Vantage function name

        Returns:
            DataFrame with stock data
        """
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided")
            return pd.DataFrame()

        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': function,
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'full'
            }

            response = requests.get(url, params=params)
            data = response.json()

            # Extract time series data
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df.sort_index(inplace=True)
                df['Symbol'] = symbol

                logger.info(f"Successfully fetched {len(df)} records from Alpha Vantage for {symbol}")
                return df
            else:
                logger.warning(f"No time series data found for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            return pd.DataFrame()

    async def websocket_data_stream(self, symbol: str, callback_func):
        """
        WebSocket connection for real-time data streaming
        This is a simplified example - you would need to connect to actual WebSocket APIs

        Args:
            symbol: Stock symbol
            callback_func: Function to call with new data
        """
        try:
            # Simulate real-time data streaming
            while True:
                # In a real implementation, you would connect to actual WebSocket APIs
                # like Polygon.io, Alpaca, or other providers
                current_price = self.get_real_time_price(symbol)
                if current_price:
                    data = {
                        'symbol': symbol,
                        'price': current_price,
                        'timestamp': datetime.now().isoformat()
                    }
                    await callback_func(data)

                await asyncio.sleep(1)  # Update every second

        except Exception as e:
            logger.error(f"WebSocket error for {symbol}: {str(e)}")

    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get detailed stock information

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract key information
            stock_info = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0)
            }

            return stock_info

        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {str(e)}")
            return {}

    def save_data_to_csv(self, data: pd.DataFrame, filename: str, directory: str = "data"):
        """
        Save DataFrame to CSV file

        Args:
            data: DataFrame to save
            filename: Name of the file
            directory: Directory to save the file
        """
        try:
            filepath = f"{directory}/{filename}"
            data.to_csv(filepath, index=False)
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")

    def load_data_from_csv(self, filename: str, directory: str = "data") -> pd.DataFrame:
        """
        Load DataFrame from CSV file

        Args:
            filename: Name of the file
            directory: Directory containing the file

        Returns:
            DataFrame with loaded data
        """
        try:
            filepath = f"{directory}/{filename}"
            data = pd.read_csv(filepath)
            logger.info(f"Data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    # Initialize data fetcher
    fetcher = DataFetcher()

    # Test fetching data
    symbol = "AAPL"
    data = fetcher.fetch_yahoo_finance_data(symbol, period="1mo")

    if not data.empty:
        print(f"Fetched {len(data)} records for {symbol}")
        print(data.head())

        # Save data
        fetcher.save_data_to_csv(data, f"{symbol}_data.csv")

        # Get stock info
        info = fetcher.get_stock_info(symbol)
        print(f"Stock info: {info}")

        # Get real-time price
        current_price = fetcher.get_real_time_price(symbol)
        print(f"Current price: {current_price}")
