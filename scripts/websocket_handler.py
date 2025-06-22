"""
WebSocket Handler Module for Trading Bot
Handles real-time data streaming using WebSockets
Provides a unified interface for different WebSocket connections
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Dict, List, Callable, Optional
import threading
import queue
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketHandler:
    """Main WebSocket handler for real-time data streaming"""

    def __init__(self):
        self.connections = {}
        self.data_queue = queue.Queue()
        self.is_running = False
        self.callbacks = {}

    async def connect_to_data_stream(self, symbol: str, callback: Callable):
        """
        Connect to a real-time data stream for a specific symbol

        Args:
            symbol: Stock symbol to stream
            callback: Function to call when new data arrives
        """
        self.callbacks[symbol] = callback

        # Start data simulation (in a real implementation, connect to actual WebSocket APIs)
        await self._simulate_real_time_data(symbol)

    async def _simulate_real_time_data(self, symbol: str):
        """
        Simulate real-time data streaming
        In production, this would connect to actual WebSocket APIs like:
        - Polygon.io WebSocket
        - Alpha Vantage WebSocket
        - Binance WebSocket
        - etc.
        """
        from scripts.data_fetcher import DataFetcher

        fetcher = DataFetcher()

        while self.is_running:
            try:
                # Get current price (simulated real-time)
                current_price = fetcher.get_real_time_price(symbol)

                if current_price is not None:
                    data = {
                        'symbol': symbol,
                        'price': current_price,
                        'timestamp': datetime.now().isoformat(),
                        'volume': int(1000000 + (time.time() % 1000000)),  # Simulated volume
                        'type': 'price_update'
                    }

                    # Call the callback function
                    if symbol in self.callbacks:
                        await self.callbacks[symbol](data)

                    # Add to queue for other consumers
                    self.data_queue.put(data)

                await asyncio.sleep(1)  # Update every second

            except Exception as e:
                logger.error(f"Error in real-time data stream for {symbol}: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    def start_streaming(self, symbols: List[str], callback: Callable):
        """
        Start streaming data for multiple symbols

        Args:
            symbols: List of stock symbols
            callback: Callback function for data updates
        """
        self.is_running = True

        async def stream_multiple():
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(
                    self.connect_to_data_stream(symbol, callback)
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

        # Run in a separate thread to avoid blocking
        def run_async():
            asyncio.run(stream_multiple())

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()

        return thread

    def stop_streaming(self):
        """Stop all streaming connections"""
        self.is_running = False
        logger.info("Stopped all WebSocket connections")

    def get_latest_data(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Get the latest data from the queue

        Args:
            timeout: Timeout in seconds

        Returns:
            Latest data dictionary or None
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class StreamlitWebSocketIntegration:
    """
    Integration layer for WebSocket with Streamlit
    Handles real-time updates in Streamlit dashboard
    """

    def __init__(self):
        self.websocket_handler = WebSocketHandler()
        self.latest_data = {}
        self.data_history = {}

    async def data_callback(self, data: Dict):
        """
        Callback function for WebSocket data updates

        Args:
            data: Real-time data dictionary
        """
        symbol = data.get('symbol')
        if symbol:
            self.latest_data[symbol] = data

            # Maintain history
            if symbol not in self.data_history:
                self.data_history[symbol] = []

            self.data_history[symbol].append(data)

            # Keep only last 100 data points
            if len(self.data_history[symbol]) > 100:
                self.data_history[symbol] = self.data_history[symbol][-100:]

    def start_real_time_updates(self, symbols: List[str]):
        """
        Start real-time updates for Streamlit dashboard

        Args:
            symbols: List of symbols to track
        """
        self.websocket_handler.start_streaming(symbols, self.data_callback)

    def get_real_time_data(self, symbol: str) -> Optional[Dict]:
        """
        Get latest real-time data for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Latest data or None
        """
        return self.latest_data.get(symbol)

    def get_price_history(self, symbol: str) -> List[Dict]:
        """
        Get price history for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            List of historical data points
        """
        return self.data_history.get(symbol, [])

    def stop_updates(self):
        """Stop real-time updates"""
        self.websocket_handler.stop_streaming()


class TradingBotWebSocket:
    """
    WebSocket integration specifically for trading bot operations
    """

    def __init__(self, trading_bot_callback: Callable):
        self.websocket_handler = WebSocketHandler()
        self.trading_bot_callback = trading_bot_callback
        self.price_alerts = {}
        self.trading_signals = {}

    async def trading_data_callback(self, data: Dict):
        """
        Callback for trading-related data processing

        Args:
            data: Real-time market data
        """
        symbol = data.get('symbol')
        price = data.get('price')

        if symbol and price:
            # Check price alerts
            await self._check_price_alerts(symbol, price)

            # Generate trading signals
            await self._generate_trading_signals(symbol, data)

            # Call the main trading bot callback
            await self.trading_bot_callback(data)

    async def _check_price_alerts(self, symbol: str, price: float):
        """
        Check if any price alerts are triggered

        Args:
            symbol: Stock symbol
            price: Current price
        """
        if symbol in self.price_alerts:
            alerts = self.price_alerts[symbol]

            for alert in alerts:
                if alert['type'] == 'above' and price >= alert['price']:
                    logger.info(f"Price alert triggered: {symbol} above {alert['price']}")
                    alert['triggered'] = True
                elif alert['type'] == 'below' and price <= alert['price']:
                    logger.info(f"Price alert triggered: {symbol} below {alert['price']}")
                    alert['triggered'] = True

    async def _generate_trading_signals(self, symbol: str, data: Dict):
        """
        Generate trading signals based on real-time data

        Args:
            symbol: Stock symbol
            data: Real-time data
        """
        # Simple moving average crossover signal (example)
        # In practice, you would use more sophisticated indicators

        price = data.get('price')
        timestamp = data.get('timestamp')

        if symbol not in self.trading_signals:
            self.trading_signals[symbol] = {
                'prices': [],
                'signals': []
            }

        # Keep track of recent prices
        self.trading_signals[symbol]['prices'].append(price)

        # Keep only last 20 prices for moving average
        if len(self.trading_signals[symbol]['prices']) > 20:
            self.trading_signals[symbol]['prices'] = self.trading_signals[symbol]['prices'][-20:]

        # Generate signal if we have enough data
        if len(self.trading_signals[symbol]['prices']) >= 10:
            prices = self.trading_signals[symbol]['prices']
            sma_5 = sum(prices[-5:]) / 5
            sma_10 = sum(prices[-10:]) / 10

            if sma_5 > sma_10:
                signal = 'BUY'
            elif sma_5 < sma_10:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            signal_data = {
                'symbol': symbol,
                'signal': signal,
                'price': price,
                'timestamp': timestamp,
                'sma_5': sma_5,
                'sma_10': sma_10
            }

            self.trading_signals[symbol]['signals'].append(signal_data)

            # Keep only last 50 signals
            if len(self.trading_signals[symbol]['signals']) > 50:
                self.trading_signals[symbol]['signals'] = self.trading_signals[symbol]['signals'][-50:]

    def add_price_alert(self, symbol: str, price: float, alert_type: str = 'above'):
        """
        Add a price alert

        Args:
            symbol: Stock symbol
            price: Alert price
            alert_type: 'above' or 'below'
        """
        if symbol not in self.price_alerts:
            self.price_alerts[symbol] = []

        alert = {
            'price': price,
            'type': alert_type,
            'triggered': False,
            'created_at': datetime.now().isoformat()
        }

        self.price_alerts[symbol].append(alert)
        logger.info(f"Added price alert for {symbol}: {alert_type} {price}")

    def get_trading_signals(self, symbol: str) -> List[Dict]:
        """
        Get recent trading signals for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            List of recent signals
        """
        return self.trading_signals.get(symbol, {}).get('signals', [])

    def start_trading_stream(self, symbols: List[str]):
        """
        Start WebSocket stream for trading

        Args:
            symbols: List of symbols to monitor
        """
        self.websocket_handler.start_streaming(symbols, self.trading_data_callback)

    def stop_trading_stream(self):
        """Stop trading WebSocket stream"""
        self.websocket_handler.stop_streaming()


# Example usage and testing
if __name__ == "__main__":
    async def example_callback(data):
        print(f"Received data: {data}")

    # Test WebSocket handler
    async def test_websocket():
        handler = WebSocketHandler()
        handler.is_running = True

        print("Starting WebSocket test...")
        await handler.connect_to_data_stream("AAPL", example_callback)

    # Run test
    print("Testing WebSocket functionality...")
    # asyncio.run(test_websocket())  # Uncomment to test
    print("WebSocket module created successfully!")
