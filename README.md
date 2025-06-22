# 🤖 AI Trading Bot with Streamlit Dashboard

A comprehensive trading bot system that fetches stock data, predicts prices using various statistical and machine learning methods, and provides a real-time dashboard interface.

## 🚀 Features

### Data Fetching & Real-time Updates
- **Yahoo Finance Integration**: Fetch historical and real-time stock data
- **Alpha Vantage Support**: Professional-grade financial data API
- **WebSocket Streaming**: Real-time price updates with unified WebSocket handling
- **Multiple Symbol Support**: Monitor multiple stocks simultaneously

### Prediction Models
- **Linear Regression**: Basic statistical approach
- **Ridge Regression**: Regularized linear model
- **Random Forest**: Ensemble learning method
- **Support Vector Regression (SVR)**: Non-linear regression
- **ARIMA**: Time series forecasting
- **LSTM Neural Networks**: Deep learning for sequential data
- **Polynomial Regression**: Non-linear polynomial fitting

### Technical Analysis
- **Moving Averages**: SMA, EMA with multiple periods
- **RSI**: Relative Strength Index for momentum
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility indicators
- **Stochastic Oscillator**: Momentum indicator
- **Williams %R**: Momentum indicator
- **ATR**: Average True Range for volatility
- **CCI**: Commodity Channel Index
- **OBV**: On-Balance Volume
- **Fibonacci Retracements**: Support/resistance levels
- **Pivot Points**: Trading levels

### Interactive Dashboard
- **Real-time Monitoring**: Live price updates and charts
- **Technical Analysis Visualization**: Interactive charts with indicators
- **Model Performance Comparison**: Compare prediction accuracy
- **Trading Signals**: Buy/Sell/Hold recommendations
- **Portfolio Management**: Track positions and performance

## 📁 Project Structure

```
trading-bot/
├── data/                    # Data storage directory
├── models/                  # ML models and predictions
│   └── prediction_models.py # All prediction algorithms
├── utils/                   # Utility modules
│   ├── data_fetcher.py     # Data fetching from APIs
│   ├── technical_indicators.py # Technical analysis
│   └── websocket_handler.py # Real-time data streaming
├── dashboard/              # Streamlit dashboard
│   └── streamlit_app.py   # Main dashboard application
├── logs/                  # Logging directory
├── main_trading_bot.py    # Main bot orchestrator
├── config.py             # Configuration management
├── run_dashboard.py      # Startup script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🛠️ Installation & Setup

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd trading-bot

# Or download and extract the files
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Optional: Set API Keys (for enhanced features)
```bash
# For Alpha Vantage (optional)
export ALPHA_VANTAGE_API_KEY="your_api_key_here"

# For Polygon.io (optional)
export POLYGON_API_KEY="your_api_key_here"
```

### 4. Run the Dashboard
```bash
python run_dashboard.py
```

Or directly with Streamlit:
```bash
streamlit run dashboard/streamlit_app.py
```

## 💻 Usage

### Quick Start
1. **Launch Dashboard**: Run `python run_dashboard.py`
2. **Access Interface**: Open http://localhost:8501 in your browser
3. **Select Stocks**: Choose symbols from the sidebar (e.g., AAPL, GOOGL, MSFT)
4. **Configure Settings**: Adjust prediction models and risk parameters
5. **Initialize Bot**: Click "Initialize/Update Bot" in the sidebar
6. **Run Analysis**: Use "Run Analysis for All Symbols" button
7. **Explore Results**: Navigate through different tabs to view results

### Dashboard Tabs

#### 📊 Overview
- Portfolio summary with buy/sell/hold signals
- Aggregated recommendations across all symbols
- Confidence metrics and current prices

#### 📈 Technical Analysis
- Interactive candlestick charts with technical indicators
- Real-time technical signals (RSI, MACD, Bollinger Bands)
- Support and resistance levels

#### 🤖 Predictions
- Compare performance of different ML models
- View prediction accuracy metrics (RMSE, MAE, R², MAPE)
- Next-day price predictions from each model

#### ⚡ Real-time
- Live price monitoring with WebSocket updates
- Real-time price movement charts
- Volume and market data streaming

#### 💼 Portfolio
- Current positions and trade history
- Performance metrics and P&L tracking
- Risk management statistics

### Programmatic Usage

```python
from main_trading_bot import TradingBot
from config import Config

# Initialize bot
symbols = ['AAPL', 'GOOGL', 'MSFT']
config = Config.get_config()
bot = TradingBot(symbols, config)

# Fetch data and analyze
bot.fetch_historical_data()
analysis = bot.analyze_symbol('AAPL')

# Get predictions
predictions = bot.generate_predictions('AAPL')

# Start real-time monitoring
bot.start_real_time_monitoring()
```

## ⚙️ Configuration

### Environment Variables
```bash
# Optional API keys for enhanced data
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here

# Environment setting
TRADING_BOT_ENV=development  # or 'production'
```

### Config Parameters
- **Initial Capital**: Starting portfolio value
- **Risk per Trade**: Percentage of capital risked per trade
- **Stop Loss**: Automatic loss-cutting percentage
- **Take Profit**: Automatic profit-taking percentage
- **Prediction Models**: Which ML models to use
- **Update Intervals**: How often to fetch new data

## 🔧 Available Prediction Methods

### Statistical Methods
1. **Linear Regression**: Basic linear relationship modeling
2. **Ridge Regression**: L2 regularized linear regression
3. **Polynomial Regression**: Non-linear polynomial fitting
4. **ARIMA**: Autoregressive Integrated Moving Average

### Machine Learning Methods
1. **Random Forest**: Ensemble of decision trees
2. **Support Vector Regression**: Non-linear kernel-based regression
3. **LSTM Networks**: Deep learning for time series

### Model Evaluation Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## 📡 WebSocket Integration

The system uses a unified WebSocket handler that:
- Provides real-time price updates
- Supports multiple concurrent connections
- Handles connection failures gracefully
- Integrates seamlessly with Streamlit for live updates

### WebSocket Features
- **Real-time Price Streaming**: Live market data
- **Connection Management**: Automatic reconnection
- **Data Queuing**: Buffered data for reliability
- **Callback System**: Event-driven data processing

## 🚨 Risk Disclaimer

**⚠️ IMPORTANT: This is for educational and research purposes only.**

- This bot is designed for learning and experimentation
- **NOT** intended for live trading without proper risk management
- Historical performance does not guarantee future results
- Always do your own research before making financial decisions
- Consider consulting with financial professionals

## 🔧 Troubleshooting

### Common Issues

1. **Module Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Fetching Issues**
   - Check internet connection
   - Verify symbol names are correct
   - Some data sources may have rate limits

3. **WebSocket Connection Issues**
   - Firewall may block connections
   - Try restarting the dashboard
   - Check network connectivity

4. **Memory Issues with Large Datasets**
   - Reduce the number of symbols
   - Use shorter time periods
   - Close other applications

### Debug Mode
```bash
# Run with debug logging
TRADING_BOT_ENV=development python run_dashboard.py
```

## 📈 Performance Optimization

### For Better Performance:
1. **Limit Symbols**: Start with 3-5 symbols
2. **Reduce Models**: Use fewer prediction models initially
3. **Shorter Periods**: Use 6 months instead of 1 year for faster analysis
4. **Disable Real-time**: Turn off WebSocket updates when not needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is for educational purposes. Please ensure you comply with all applicable financial regulations and terms of service for data providers.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments and documentation
3. Search for similar issues online
4. Create an issue with detailed error information

---

**Happy Trading! 📈🤖**

*Remember: Past performance is not indicative of future results. Trade responsibly.*