"""
Streamlit Dashboard for Trading Bot
Provides a web-based interface for monitoring trading operations, predictions, and real-time data
Features:
- Real-time price monitoring
- Technical indicator visualization
- Prediction model comparison
- Trading signals dashboard
- Portfolio performance tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import asyncio
import threading
from datetime import datetime, timedelta
import json

# Import custom modules
import sys
import os
sys.path.append('.')

from main_trading_bot import TradingBot
from scripts.websocket_handler import StreamlitWebSocketIntegration
from models.prediction_models import StockPredictor

# Configure Streamlit page
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .signal-buy {
        color: #00ff00;
        font-weight: bold;
    }
    .signal-sell {
        color: #ff0000;
        font-weight: bold;
    }
    .signal-hold {
        color: #ffa500;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trading_bot' not in st.session_state:
    st.session_state.trading_bot = None
if 'websocket_integration' not in st.session_state:
    st.session_state.websocket_integration = StreamlitWebSocketIntegration()
if 'real_time_enabled' not in st.session_state:
    st.session_state.real_time_enabled = False
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}

def initialize_trading_bot(symbols, config):
    """Initialize the trading bot"""
    try:
        bot = TradingBot(symbols, config)
        bot.fetch_historical_data()
        return bot
    except Exception as e:
        st.error(f"Error initializing trading bot: {str(e)}")
        return None

def create_price_chart(data, symbol, indicators=None):
    """Create interactive price chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} Price & Indicators', 'RSI', 'MACD')
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    if indicators is not None:
        # Add moving averages
        if 'SMA_20' in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators['SMA_20'],
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

        if 'EMA_20' in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators['EMA_20'],
                    name='EMA 20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )

        # Bollinger Bands
        if all(col in indicators.columns for col in ['BB_Upper', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)'
                ),
                row=1, col=1
            )

        # RSI
        if 'RSI' in indicators.columns:
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators['RSI'],
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        if all(col in indicators.columns for col in ['MACD', 'MACD_Signal']):
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators['MACD'],
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=indicators.index,
                    y=indicators['MACD_Signal'],
                    name='MACD Signal',
                    line=dict(color='red')
                ),
                row=3, col=1
            )

            if 'MACD_Histogram' in indicators.columns:
                fig.add_trace(
                    go.Bar(
                        x=indicators.index,
                        y=indicators['MACD_Histogram'],
                        name='MACD Histogram',
                        marker_color='green'
                    ),
                    row=3, col=1
                )

    fig.update_layout(
        height=800,
        title=f'{symbol} Technical Analysis',
        xaxis_rangeslider_visible=False
    )

    return fig

def create_prediction_comparison_chart(predictions_df):
    """Create prediction comparison chart"""
    if predictions_df.empty:
        return None

    fig = go.Figure()

    # Create bar chart for different metrics
    metrics = ['RMSE', 'MAE', 'R2', 'MAPE']

    for metric in metrics:
        if metric in predictions_df.columns:
            # Filter out error values
            valid_data = predictions_df[predictions_df[metric] != 'Error']
            if not valid_data.empty:
                fig.add_trace(
                    go.Bar(
                        x=valid_data['Model'],
                        y=pd.to_numeric(valid_data[metric], errors='coerce'),
                        name=metric
                    )
                )

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Metric Value",
        barmode='group',
        height=400
    )

    return fig

def display_real_time_data(symbol):
    """Display real-time data for a symbol"""
    if st.session_state.real_time_enabled:
        real_time_data = st.session_state.websocket_integration.get_real_time_data(symbol)

        if real_time_data:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"${real_time_data['price']:.2f}")

            with col2:
                st.metric("Volume", f"{real_time_data.get('volume', 'N/A'):,}")

            with col3:
                st.metric("Last Update", real_time_data['timestamp'][-8:])

            with col4:
                # Calculate price change (mock for demo)
                change = np.random.uniform(-0.02, 0.02) * real_time_data['price']
                st.metric("Change", f"${change:.2f}", f"{change/real_time_data['price']*100:.2f}%")

            # Price history chart
            price_history = st.session_state.websocket_integration.get_price_history(symbol)
            if price_history:
                prices_df = pd.DataFrame(price_history)
                prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])

                fig = px.line(
                    prices_df, 
                    x='timestamp', 
                    y='price',
                    title=f'{symbol} Real-time Price Movement'
                )
                st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard function"""
    st.markdown('<h1 class="main-header">🤖 Trading Bot Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Symbol selection
    default_symbols = ['AAPL', 'GOOGL']
    selected_symbols = st.sidebar.multiselect(
        "Select Stocks to Analyze",
        options=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'],
        default=default_symbols[:3]
    )

    # Configuration options
    st.sidebar.subheader("Bot Configuration")
    initial_capital = st.sidebar.number_input("Initial Capital ($)", value=10000, min_value=1000)
    risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2) / 100

    prediction_models = st.sidebar.multiselect(
        "Prediction Models",
        options=['linear_regression', 'random_forest', 'lstm', 'arima', 'svr'],
        default=['linear_regression']
    )

    # Real-time toggle
    real_time_toggle = st.sidebar.toggle("Enable Real-time Data", value=False)

    if real_time_toggle != st.session_state.real_time_enabled:
        st.session_state.real_time_enabled = real_time_toggle
        if real_time_toggle:
            st.session_state.websocket_integration.start_real_time_updates(selected_symbols)
        else:
            st.session_state.websocket_integration.stop_updates()

    # Initialize or update trading bot
    if st.sidebar.button("Initialize/Update Bot") or st.session_state.trading_bot is None:
        if selected_symbols:
            config = {
                'initial_capital': initial_capital,
                'risk_per_trade': risk_per_trade,
                'prediction_models': prediction_models,
                'save_data': True
            }

            with st.spinner("Initializing trading bot..."):
                st.session_state.trading_bot = initialize_trading_bot(selected_symbols, config)

            if st.session_state.trading_bot:
                st.success("Trading bot initialized successfully!")
            else:
                st.error("Failed to initialize trading bot")
        else:
            st.warning("Please select at least one symbol")

    # Main dashboard content
    if st.session_state.trading_bot and selected_symbols:

        # Tab navigation
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "📈 Technical Analysis", 
            "🤖 Predictions", 
            "⚡ Real-time", 
        ])

        with tab1:
            st.header("Portfolio Overview")

            # Run analysis for all symbols
            if st.button("Run Analysis for All Symbols"):
                with st.spinner("Analyzing all symbols..."):
                    st.session_state.analysis_data = st.session_state.trading_bot.analyze_all_symbols()
                st.success("Analysis completed!")

            # Display analysis results
            if st.session_state.analysis_data:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    buy_signals = sum(1 for analysis in st.session_state.analysis_data.values() 
                                    if analysis.get('recommendation', {}).get('action') == 'BUY')
                    st.metric("Buy Signals", buy_signals)

                with col2:
                    sell_signals = sum(1 for analysis in st.session_state.analysis_data.values() 
                                     if analysis.get('recommendation', {}).get('action') == 'SELL')
                    st.metric("Sell Signals", sell_signals)

                with col3:
                    hold_signals = sum(1 for analysis in st.session_state.analysis_data.values() 
                                     if analysis.get('recommendation', {}).get('action') == 'HOLD')
                    st.metric("Hold Signals", hold_signals)

                with col4:
                    avg_confidence = np.mean([
                        analysis.get('recommendation', {}).get('confidence', 0)
                        for analysis in st.session_state.analysis_data.values()
                    ])
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")

                # Recommendations table
                st.subheader("Trading Recommendations")
                recommendations = []
                for symbol, analysis in st.session_state.analysis_data.items():
                    rec = analysis.get('recommendation', {})
                    recommendations.append({
                        'Symbol': symbol,
                        'Action': rec.get('action', 'N/A'),
                        'Confidence': rec.get('confidence', 0),
                        'Current Price': f"${analysis.get('current_price', 0):.2f}",
                        'Target Price': f"${rec.get('target_price', 0):.2f}" if rec.get('target_price') else 'N/A',
                        'Stop Loss': f"${rec.get('stop_loss', 0):.2f}" if rec.get('stop_loss') else 'N/A'
                    })

                recommendations_df = pd.DataFrame(recommendations)
                st.dataframe(recommendations_df, use_container_width=True)

        with tab2:
            st.header("Technical Analysis")

            # Symbol selector for detailed analysis
            selected_symbol = st.selectbox("Select symbol for detailed analysis", selected_symbols)

            if selected_symbol in st.session_state.analysis_data:
                analysis = st.session_state.analysis_data[selected_symbol]
                technical_data = analysis.get('technical_indicators', {})

                if technical_data and 'data' in technical_data:
                    # Create and display chart
                    historical_data = st.session_state.trading_bot.historical_data.get(selected_symbol)
                    if historical_data is not None:
                        chart = create_price_chart(
                            historical_data, 
                            selected_symbol, 
                            technical_data['data']
                        )
                        st.plotly_chart(chart, use_container_width=True)

                    # Technical signals
                    st.subheader("Technical Signals")
                    signals = technical_data.get('signals', {})

                    signal_cols = st.columns(len(signals))
                    for i, (indicator, signal) in enumerate(signals.items()):
                        with signal_cols[i]:
                            color_class = f"signal-{signal.lower()}"
                            st.markdown(f'<div class="{color_class}">{indicator}: {signal}</div>', 
                                      unsafe_allow_html=True)

                else:
                    st.info("Run analysis to see technical indicators")
            else:
                st.info("Please run analysis first")

        with tab3:
            st.header("Predictions")

            selected_symbol_pred = st.selectbox("Select symbol for predictions", selected_symbols, key="pred_symbol")

            if selected_symbol_pred in st.session_state.analysis_data:
                analysis = st.session_state.analysis_data[selected_symbol_pred]
                predictions = analysis.get('predictions', {})

                if predictions and 'comparison' in predictions:
                    # Model comparison chart
                    comparison_df = predictions['comparison']
                    comparison_chart = create_prediction_comparison_chart(comparison_df)
                    if comparison_chart:
                        st.plotly_chart(comparison_chart, use_container_width=True)

                    # Predictions table
                    st.subheader("Model Predictions")
                    st.dataframe(comparison_df, use_container_width=True)

                    # Individual model details
                    st.subheader("Detailed Model Results")
                    model_tabs = st.tabs(prediction_models)

                    for i, model in enumerate(prediction_models):
                        with model_tabs[i]:
                            if model in predictions and 'error' not in predictions[model]:
                                model_result = predictions[model]

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Metrics:**")
                                    for metric, value in model_result['metrics'].items():
                                        st.write(f"{metric}: {value:.4f}")

                                with col2:
                                    st.write("**Next Prediction:**")
                                    next_pred = model_result.get('next_prediction')
                                    if next_pred:
                                        current_price = analysis.get('current_price', 0)
                                        change_pct = ((next_pred - current_price) / current_price * 100) if current_price else 0
                                        st.write(f"${next_pred:.2f}")
                                        st.write(f"Change: {change_pct:+.2f}%")

                                # Feature importance for tree-based models
                                if 'feature_importance' in model_result:
                                    st.write("**Feature Importance:**")
                                    importance_df = pd.DataFrame(
                                        list(model_result['feature_importance'].items()),
                                        columns=['Feature', 'Importance']
                                    ).sort_values('Importance', ascending=False).head(10)

                                    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
                                    st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error(f"Error in {model} model")
                else:
                    st.info("Run analysis to see predictions")
            else:
                st.info("Please run analysis first")

        with tab4:
            st.header("Real-time Monitoring")

            if st.session_state.real_time_enabled:
                # Auto-refresh mechanism
                placeholder = st.empty()

                # Display real-time data for each symbol
                for symbol in selected_symbols:
                    with placeholder.container():
                        st.subheader(f"{symbol} Real-time Data")
                        display_real_time_data(symbol)
                        st.divider()

                # Auto-refresh every 5 seconds
                time.sleep(5)
                st.rerun()
            else:
                st.info("Enable real-time data in the sidebar to see live updates")


if __name__ == "__main__":
    main()
