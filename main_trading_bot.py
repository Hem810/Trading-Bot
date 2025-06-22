"""
Main Trading Bot Module
Integrates all components: data fetching, technical analysis, prediction models, and WebSocket streaming
Provides a unified interface for the trading bot operations
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import json
import os

# Import custom modules
from scripts.data_fetcher import DataFetcher
from scripts.technical_indicators import TechnicalIndicators
from models.prediction_models import StockPredictor
from scripts.websocket_handler import TradingBotWebSocket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingBot:
    """
    Main Trading Bot Class
    Orchestrates all trading operations including data fetching, analysis, and predictions
    """

    def __init__(self, symbols: List[str], config: Dict = None):
        """
        Initialize Trading Bot

        Args:
            symbols: List of stock symbols to trade
            config: Configuration dictionary
        """
        self.symbols = symbols
        self.config = config or self._default_config()

        # Initialize components
        self.data_fetcher = DataFetcher(api_key=self.config.get('alpha_vantage_api_key'))
        self.websocket_handler = TradingBotWebSocket(self._handle_real_time_data)

        # Data storage
        self.historical_data = {}
        self.real_time_data = {}
        self.predictions = {}
        self.trading_signals = {}
        self.positions = {}

        # Performance tracking
        self.trade_history = []
        self.portfolio_value = self.config.get('initial_capital', 10000)
        self.performance_metrics = {}

        logger.info(f"Trading Bot initialized for symbols: {symbols}")

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'initial_capital': 10000,
            'risk_per_trade': 0.02,  # 2% risk per trade
            'stop_loss_pct': 0.05,   # 5% stop loss
            'take_profit_pct': 0.10, # 10% take profit
            'prediction_models': ['linear_regression', 'random_forest', 'lstm'],
            'technical_indicators': ['rsi', 'macd', 'bollinger_bands'],
            'data_update_interval': 60,  # seconds
            'save_data': True,
            'log_trades': True
        }

    def fetch_historical_data(self, period: str = "1y") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all symbols

        Args:
            period: Data period to fetch

        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        logger.info("Fetching historical data...")

        for symbol in self.symbols:
            try:
                data = self.data_fetcher.fetch_yahoo_finance_data(symbol, period)
                if not data.empty:
                    self.historical_data[symbol] = data
                    logger.info(f"Fetched {len(data)} records for {symbol}")

                    # Save data if configured
                    if self.config.get('save_data'):
                        self.data_fetcher.save_data_to_csv(
                            data, f"{symbol}_historical.csv"
                        )
                else:
                    logger.warning(f"No data fetched for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")

        return self.historical_data

    def calculate_technical_indicators(self, symbol: str) -> Dict:
        """
        Calculate technical indicators for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with indicator values
        """
        if symbol not in self.historical_data:
            logger.warning(f"No historical data available for {symbol}")
            return {}

        try:
            data = self.historical_data[symbol]
            indicators = TechnicalIndicators(data)

            # Calculate all indicators
            indicator_data = indicators.calculate_all_indicators()

            # Get trading signals
            signals = indicators.get_trading_signals()

            return {
                'data': indicator_data,
                'signals': signals,
                'pivot_points': indicators.pivot_points(),
                'fibonacci_levels': indicators.fibonacci_retracement()
            }

        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
            return {}

    def generate_predictions(self, symbol: str) -> Dict:
        """
        Generate price predictions using various models

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with prediction results
        """
        if symbol not in self.historical_data:
            logger.warning(f"No historical data available for {symbol}")
            return {}

        try:
            data = self.historical_data[symbol]
            predictor = StockPredictor(data)

            # Run selected prediction models
            results = {}
            model_methods = {
                'linear_regression': predictor.linear_regression_prediction,
                'ridge_regression': predictor.ridge_regression_prediction,
                'random_forest': predictor.random_forest_prediction,
                'svr': predictor.svr_prediction,
                'arima': predictor.arima_prediction,
                'lstm': predictor.lstm_prediction,
                'polynomial_regression': predictor.polynomial_regression_prediction
            }

            for model_name in self.config['prediction_models']:
                if model_name in model_methods:
                    logger.info(f"Running {model_name} prediction for {symbol}")
                    results[model_name] = model_methods[model_name]()

            # Compare all models
            comparison = predictor.compare_all_models()
            results['comparison'] = comparison

            self.predictions[symbol] = results
            return results

        except Exception as e:
            logger.error(f"Error generating predictions for {symbol}: {str(e)}")
            return {}

    def analyze_symbol(self, symbol: str) -> Dict:
        """
        Complete analysis of a symbol (indicators + predictions)

        Args:
            symbol: Stock symbol

        Returns:
            Complete analysis results
        """
        logger.info(f"Analyzing {symbol}...")

        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'technical_indicators': self.calculate_technical_indicators(symbol),
            'predictions': self.generate_predictions(symbol),
            'current_price': None,
            'recommendation': None
        }

        # Get current price
        current_price = self.data_fetcher.get_real_time_price(symbol)
        analysis['current_price'] = current_price

        # Generate overall recommendation
        analysis['recommendation'] = self._generate_recommendation(symbol, analysis)

        return analysis

    def _generate_recommendation(self, symbol: str, analysis: Dict) -> Dict:
        """
        Generate trading recommendation based on analysis

        Args:
            symbol: Stock symbol
            analysis: Analysis results

        Returns:
            Trading recommendation
        """
        recommendation = {
            'action': 'HOLD',
            'confidence': 0.0,
            'reasons': [],
            'target_price': None,
            'stop_loss': None
        }

        try:
            # Technical indicator signals
            technical_signals = analysis.get('technical_indicators', {}).get('signals', {})

            # Count buy/sell signals
            buy_signals = sum(1 for signal in technical_signals.values() if signal == 'BUY')
            sell_signals = sum(1 for signal in technical_signals.values() if signal == 'SELL')
            total_signals = len(technical_signals)

            if total_signals > 0:
                buy_ratio = buy_signals / total_signals
                sell_ratio = sell_signals / total_signals

                if buy_ratio > 0.6:
                    recommendation['action'] = 'BUY'
                    recommendation['confidence'] = buy_ratio
                    recommendation['reasons'].append(f"Technical indicators: {buy_signals}/{total_signals} BUY signals")
                elif sell_ratio > 0.6:
                    recommendation['action'] = 'SELL'
                    recommendation['confidence'] = sell_ratio
                    recommendation['reasons'].append(f"Technical indicators: {sell_signals}/{total_signals} SELL signals")

            # Prediction model consensus
            predictions = analysis.get('predictions', {})
            if predictions and 'comparison' in predictions:
                comparison_df = predictions['comparison']

                # Get next predictions from different models
                next_predictions = []
                for _, row in comparison_df.iterrows():
                    if isinstance(row['Next_Prediction'], (int, float)):
                        next_predictions.append(row['Next_Prediction'])

                if next_predictions and analysis['current_price']:
                    avg_prediction = np.mean(next_predictions)
                    current_price = analysis['current_price']

                    price_change_pct = (avg_prediction - current_price) / current_price * 100

                    if price_change_pct > 2:  # More than 2% increase predicted
                        if recommendation['action'] == 'HOLD':
                            recommendation['action'] = 'BUY'
                            recommendation['confidence'] = min(price_change_pct / 10, 0.8)
                        recommendation['reasons'].append(f"ML models predict {price_change_pct:.1f}% price increase")
                        recommendation['target_price'] = avg_prediction
                    elif price_change_pct < -2:  # More than 2% decrease predicted
                        if recommendation['action'] == 'HOLD':
                            recommendation['action'] = 'SELL'
                            recommendation['confidence'] = min(abs(price_change_pct) / 10, 0.8)
                        recommendation['reasons'].append(f"ML models predict {price_change_pct:.1f}% price decrease")

            # Set stop loss and target based on current price
            if analysis['current_price']:
                current_price = analysis['current_price']
                if recommendation['action'] == 'BUY':
                    recommendation['stop_loss'] = current_price * (1 - self.config['stop_loss_pct'])
                    if not recommendation['target_price']:
                        recommendation['target_price'] = current_price * (1 + self.config['take_profit_pct'])
                elif recommendation['action'] == 'SELL':
                    recommendation['stop_loss'] = current_price * (1 + self.config['stop_loss_pct'])
                    if not recommendation['target_price']:
                        recommendation['target_price'] = current_price * (1 - self.config['take_profit_pct'])

        except Exception as e:
            logger.error(f"Error generating recommendation for {symbol}: {str(e)}")
            recommendation['reasons'].append(f"Error in analysis: {str(e)}")

        return recommendation

    def analyze_all_symbols(self) -> Dict[str, Dict]:
        """
        Analyze all symbols

        Returns:
            Dictionary with analysis for each symbol
        """
        all_analysis = {}

        for symbol in self.symbols:
            all_analysis[symbol] = self.analyze_symbol(symbol)

        return all_analysis

    async def _handle_real_time_data(self, data: Dict):
        """
        Handle real-time data updates from WebSocket

        Args:
            data: Real-time market data
        """
        symbol = data.get('symbol')
        if symbol:
            self.real_time_data[symbol] = data

            # Check for trading opportunities
            await self._check_trading_opportunities(symbol, data)

    async def _check_trading_opportunities(self, symbol: str, data: Dict):
        """
        Check for trading opportunities based on real-time data

        Args:
            symbol: Stock symbol
            data: Real-time market data
        """
        # This is where you would implement your real-time trading logic
        # For now, just log the data
        price = data.get('price')
        timestamp = data.get('timestamp')

        logger.info(f"Real-time update: {symbol} = ${price:.2f} at {timestamp}")

        # You could add logic here to:
        # - Check if price hits stop-loss or take-profit levels
        # - Monitor for technical indicator crossovers
        # - Execute trades automatically based on signals

    def start_real_time_monitoring(self):
        """Start real-time monitoring of all symbols"""
        logger.info("Starting real-time monitoring...")
        self.websocket_handler.start_trading_stream(self.symbols)

    def stop_real_time_monitoring(self):
        """Stop real-time monitoring"""
        logger.info("Stopping real-time monitoring...")
        self.websocket_handler.stop_trading_stream()

    def get_portfolio_summary(self) -> Dict:
        """
        Get portfolio summary and performance metrics

        Returns:
            Portfolio summary dictionary
        """
        summary = {
            'total_value': self.portfolio_value,
            'positions': self.positions,
            'trade_history': self.trade_history,
            'performance_metrics': self.performance_metrics,
            'active_symbols': len(self.symbols),
            'last_update': datetime.now().isoformat()
        }

        return summary

    def save_analysis_to_file(self, analysis: Dict, filename: str = None):
        """
        Save analysis results to JSON file

        Args:
            analysis: Analysis results
            filename: Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.json"

        filepath = f"logs/{filename}"

        try:
            # Convert numpy types to native Python types for JSON serialization
            analysis_copy = self._convert_for_json(analysis)

            with open(filepath, 'w') as f:
                json.dump(analysis_copy, f, indent=2, default=str)

            logger.info(f"Analysis saved to {filepath}")

        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")

    def _convert_for_json(self, obj):
        """Convert object for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    symbols = ['AAPL', 'GOOGL', 'MSFT']

    config = {
        'initial_capital': 10000,
        'prediction_models': ['linear_regression', 'random_forest'],
        'save_data': True
    }

    # Initialize trading bot
    bot = TradingBot(symbols, config)

    # Fetch historical data
    print("Fetching historical data...")
    historical_data = bot.fetch_historical_data(period="6mo")

    if historical_data:
        print(f"Successfully fetched data for {len(historical_data)} symbols")

        # Analyze first symbol
        first_symbol = symbols[0]
        print(f"\nAnalyzing {first_symbol}...")
        analysis = bot.analyze_symbol(first_symbol)

        if analysis:
            print(f"Analysis completed for {first_symbol}")
            print(f"Current price: ${analysis.get('current_price', 'N/A')}")

            recommendation = analysis.get('recommendation', {})
            print(f"Recommendation: {recommendation.get('action', 'N/A')}")
            print(f"Confidence: {recommendation.get('confidence', 0):.2f}")

            # Save analysis
            bot.save_analysis_to_file({first_symbol: analysis})

    print("\nTrading bot setup complete!")
