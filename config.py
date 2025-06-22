"""
Configuration Management for Trading Bot
Centralized configuration for all bot components
"""

import os
from typing import Dict, List

class Config:
    """Configuration class for trading bot"""

    # API Keys (set as environment variables)
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')

    # Trading Configuration
    INITIAL_CAPITAL = 10000
    RISK_PER_TRADE = 0.02  # 2%
    STOP_LOSS_PCT = 0.05   # 5%
    TAKE_PROFIT_PCT = 0.10 # 10%

    # Prediction models to use
    PREDICTION_MODELS = [
        'linear_regression',
        'random_forest',
        'lstm',
        'arima'
    ]

    # Technical indicators
    TECHNICAL_INDICATORS = [
        'sma',
        'ema',
        'rsi',
        'macd',
        'bollinger_bands',
        'stochastic'
    ]

    # Data settings
    DATA_UPDATE_INTERVAL = 60  # seconds
    HISTORICAL_PERIOD = "1y"   # 1 year
    SAVE_DATA = True
    LOG_TRADES = True

    # WebSocket settings
    WEBSOCKET_TIMEOUT = 30
    RECONNECT_ATTEMPTS = 3

    # File paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"

    # Streamlit settings
    DASHBOARD_PORT = 8501
    AUTO_REFRESH_INTERVAL = 5  # seconds

    @classmethod
    def get_config(cls, symbol: str = None) -> Dict:
        """Get configuration dictionary"""
        config = {
            'initial_capital': cls.INITIAL_CAPITAL,
            'risk_per_trade': cls.RISK_PER_TRADE,
            'stop_loss_pct': cls.STOP_LOSS_PCT,
            'take_profit_pct': cls.TAKE_PROFIT_PCT,
            'prediction_models': cls.PREDICTION_MODELS,
            'technical_indicators': cls.TECHNICAL_INDICATORS,
            'data_update_interval': cls.DATA_UPDATE_INTERVAL,
            'save_data': cls.SAVE_DATA,
            'log_trades': cls.LOG_TRADES,
            'alpha_vantage_api_key': cls.ALPHA_VANTAGE_API_KEY,
            'polygon_api_key': cls.POLYGON_API_KEY
        }

        return config

    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        if not cls.ALPHA_VANTAGE_API_KEY:
            issues.append("Alpha Vantage API key not set")

        if cls.INITIAL_CAPITAL <= 0:
            issues.append("Initial capital must be positive")

        if not (0 < cls.RISK_PER_TRADE < 1):
            issues.append("Risk per trade must be between 0 and 1")

        if not cls.DEFAULT_SYMBOLS:
            issues.append("No default symbols configured")

        return issues

# Development/Testing configuration
class DevelopmentConfig(Config):
    INITIAL_CAPITAL = 1000
    SAVE_DATA = False
    LOG_TRADES = False
    DEFAULT_SYMBOLS = ['AAPL', 'MSFT']

# Production configuration
class ProductionConfig(Config):
    INITIAL_CAPITAL = 10000
    SAVE_DATA = True
    LOG_TRADES = True
    DEFAULT_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META']

# Get configuration based on environment
def get_config():
    """Get configuration based on environment variable"""
    env = os.getenv('TRADING_BOT_ENV', 'development')

    if env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()
