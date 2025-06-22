"""
Prediction Models Module for Trading Bot
Implements various statistical and machine learning methods for stock price prediction:
- Linear Regression
- ARIMA (AutoRegressive Integrated Moving Average)
- LSTM (Long Short-Term Memory)
- Random Forest
- Support Vector Regression (SVR)
- Ridge Regression
- Polynomial Regression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """Main class for stock price prediction using various models"""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with stock data

        Args:
            data: DataFrame with stock data including 'Close' prices
        """
        self.data = data.copy()
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        self.metrics = {}

    def prepare_features(self, lookback_window: int = 5) -> tuple:
        """
        Prepare features for machine learning models

        Args:
            lookback_window: Number of previous days to use as features

        Returns:
            Tuple of (X, y) for training
        """
        # Use technical indicators as features
        from scripts.technical_indicators import TechnicalIndicators

        indicators = TechnicalIndicators(self.data)
        feature_data = indicators.calculate_all_indicators()

        # Select relevant features
        feature_columns = [
            'Close', 'Volume', 'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
            'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
            'Stoch_K', 'Stoch_D', 'ATR', 'Williams_R'
        ]

        # Remove columns that don't exist
        available_columns = [col for col in feature_columns if col in feature_data.columns]
        feature_data = feature_data[available_columns].dropna()

        if len(feature_data) < lookback_window + 1:
            raise ValueError("Insufficient data for feature preparation")

        # Create sequences for time series prediction
        X, y = [], []
        for i in range(lookback_window, len(feature_data)):
            X.append(feature_data.iloc[i-lookback_window:i].values.flatten())
            y.append(feature_data['Close'].iloc[i])

        return np.array(X), np.array(y)

    def linear_regression_prediction(self, test_size: float = 0.2) -> dict:
        """Linear Regression prediction"""
        try:
            X, y = self.prepare_features()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)

            # Store results
            self.models['linear_regression'] = model
            self.scalers['linear_regression'] = scaler
            self.predictions['linear_regression'] = y_pred

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            self.metrics['linear_regression'] = metrics

            return {
                'model': 'Linear Regression',
                'predictions': y_pred,
                'actual': y_test,
                'metrics': metrics,
                'next_prediction': self._predict_next_value('linear_regression')
            }

        except Exception as e:
            return {'error': f"Linear Regression error: {str(e)}"}

    def ridge_regression_prediction(self, alpha: float = 1.0, test_size: float = 0.2) -> dict:
        """Ridge Regression prediction"""
        try:
            X, y = self.prepare_features()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = Ridge(alpha=alpha)
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)

            # Store results
            self.models['ridge_regression'] = model
            self.scalers['ridge_regression'] = scaler
            self.predictions['ridge_regression'] = y_pred

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            self.metrics['ridge_regression'] = metrics

            return {
                'model': 'Ridge Regression',
                'predictions': y_pred,
                'actual': y_test,
                'metrics': metrics,
                'next_prediction': self._predict_next_value('ridge_regression')
            }

        except Exception as e:
            return {'error': f"Ridge Regression error: {str(e)}"}

    def random_forest_prediction(self, n_estimators: int = 100, test_size: float = 0.2) -> dict:
        """Random Forest prediction"""
        try:
            X, y = self.prepare_features()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

            # Train model
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Store results
            self.models['random_forest'] = model
            self.predictions['random_forest'] = y_pred

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            self.metrics['random_forest'] = metrics

            return {
                'model': 'Random Forest',
                'predictions': y_pred,
                'actual': y_test,
                'metrics': metrics,
                'next_prediction': self._predict_next_value('random_forest'),
                'feature_importance': dict(zip(
                    [f'feature_{i}' for i in range(X.shape[1])],
                    model.feature_importances_
                ))
            }

        except Exception as e:
            return {'error': f"Random Forest error: {str(e)}"}

    def svr_prediction(self, kernel: str = 'rbf', test_size: float = 0.2) -> dict:
        """Support Vector Regression prediction"""
        try:
            X, y = self.prepare_features()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = SVR(kernel=kernel)
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)

            # Store results
            self.models['svr'] = model
            self.scalers['svr'] = scaler
            self.predictions['svr'] = y_pred

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            self.metrics['svr'] = metrics

            return {
                'model': 'Support Vector Regression',
                'predictions': y_pred,
                'actual': y_test,
                'metrics': metrics,
                'next_prediction': self._predict_next_value('svr')
            }

        except Exception as e:
            return {'error': f"SVR error: {str(e)}"}

    def arima_prediction(self, order: tuple = (5, 1, 0), forecast_steps: int = 5) -> dict:
        """ARIMA prediction with robust error handling"""
        result_template = {
            'model': 'ARIMA',
            'predictions': [],
            'actual': [],
            'metrics': {'RMSE': float('nan'), 'MAE': float('nan'), 'R2': float('nan'), 'MAPE': float('nan')},
            'future_forecast': [],
            'next_prediction': None,
            'error': None
        }
        
        try:
            # Use only closing prices for ARIMA
            prices = self.data['Close'].dropna()

            if len(prices) < 50:
                raise ValueError("Insufficient data for ARIMA model (minimum 50 observations)")

            # Split data
            train_size = int(len(prices) * 0.8)
            train_data = prices[:train_size]
            test_data = prices[train_size:]

            # Fit ARIMA model
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()

            # Make predictions
            forecast = fitted_model.forecast(steps=len(test_data))
            future_forecast = fitted_model.forecast(steps=forecast_steps)

            # Store results
            self.models['arima'] = fitted_model
            self.predictions['arima'] = forecast

            # Calculate metrics
            metrics = self._calculate_metrics(test_data.values, forecast.values)
            self.metrics['arima'] = metrics

            # Return successful result
            return {
                'model': 'ARIMA',
                'predictions': forecast.tolist(),
                'actual': test_data.values.tolist(),
                'metrics': metrics,
                'future_forecast': future_forecast.tolist(),
                'next_prediction': future_forecast.iloc[0] if not future_forecast.empty else None
            }

        except Exception as e:
            # Return structured error with NaN metrics
            error_msg = f"ARIMA error: {str(e)}"
            result_template['error'] = error_msg
            return result_template


    def lstm_prediction(self, sequence_length: int = 60, epochs: int = 50, test_size: float = 0.2) -> dict:
            """LSTM Neural Network prediction"""
            try:
                # Prepare data for LSTM
                prices = self.data['Close'].values.reshape(-1, 1)

                # Scale data
                scaler = StandardScaler()
                scaled_prices = scaler.fit_transform(prices)

                # Create sequences
                X, y = [], []
                for i in range(sequence_length, len(scaled_prices)):
                    X.append(scaled_prices[i-sequence_length:i, 0])
                    y.append(scaled_prices[i, 0])

                X, y = np.array(X), np.array(y)
                X = X.reshape((X.shape[0], X.shape[1], 1))

                # Split data
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                # Build LSTM model
                model = Sequential([
                    LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                    Dropout(0.2),
                    LSTM(50, return_sequences=False),
                    Dropout(0.2),
                    Dense(25),
                    Dense(1)
                ])

                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train model
                model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)

                # Predictions
                y_pred_scaled = model.predict(X_test)

                # Inverse transform predictions
                y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

                # Store results
                self.models['lstm'] = model
                self.scalers['lstm'] = scaler
                self.predictions['lstm'] = y_pred

                # Calculate metrics
                metrics = self._calculate_metrics(y_test_actual, y_pred)
                self.metrics['lstm'] = metrics

                # Predict next value
                last_sequence = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
                next_pred_scaled = model.predict(last_sequence)
                next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]

                return {
                    'model': 'LSTM',
                    'predictions': y_pred,
                    'actual': y_test_actual,
                    'metrics': metrics,
                    'next_prediction': next_pred
                }

            except Exception as e:
                return {'error': f"LSTM error: {str(e)}"}

    def polynomial_regression_prediction(self, degree: int = 2, test_size: float = 0.2) -> dict:
        """Polynomial Regression prediction"""
        try:
            X, y = self.prepare_features()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree)
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.transform(X_test)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_poly)
            X_test_scaled = scaler.transform(X_test_poly)

            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            # Predictions
            y_pred = model.predict(X_test_scaled)

            # Store results
            self.models['polynomial_regression'] = model
            self.scalers['polynomial_regression'] = scaler
            self.models['poly_features'] = poly_features
            self.predictions['polynomial_regression'] = y_pred

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            self.metrics['polynomial_regression'] = metrics

            return {
                'model': 'Polynomial Regression',
                'predictions': y_pred,
                'actual': y_test,
                'metrics': metrics,
                'next_prediction': self._predict_next_value('polynomial_regression')
            }

        except Exception as e:
            return {'error': f"Polynomial Regression error: {str(e)}"}

    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> dict:
        """Calculate evaluation metrics"""
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }

    def _predict_next_value(self, model_name: str) -> float:
        """Predict the next value using a trained model"""
        try:
            if model_name == 'arima':
                return self.models[model_name].forecast(steps=1)[0]

            elif model_name == 'lstm':
                sequence_length = 60
                prices = self.data['Close'].values.reshape(-1, 1)
                scaler = self.scalers[model_name]
                scaled_prices = scaler.transform(prices)
                last_sequence = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
                next_pred_scaled = self.models[model_name].predict(last_sequence)
                return scaler.inverse_transform(next_pred_scaled)[0, 0]

            else:
                # For sklearn models
                X, _ = self.prepare_features()
                last_features = X[-1:] # Get last sample

                if model_name == 'polynomial_regression':
                    last_features = self.models['poly_features'].transform(last_features)

                if model_name in self.scalers:
                    last_features = self.scalers[model_name].transform(last_features)

                return self.models[model_name].predict(last_features)[0]

        except Exception as e:
            print(f"Error predicting next value for {model_name}: {str(e)}")
            return None

    def compare_all_models(self) -> pd.DataFrame:
        """Compare all model performances"""
        results = []

        # Define model functions
        model_functions = {
            'Linear Regression': self.linear_regression_prediction,
            'Ridge Regression': self.ridge_regression_prediction,
            'Random Forest': self.random_forest_prediction,
            'SVR': self.svr_prediction,
            'ARIMA': self.arima_prediction,
            'LSTM': self.lstm_prediction,
            'Polynomial Regression': self.polynomial_regression_prediction
        }

        for model_name, model_func in model_functions.items():
            try:
                result = model_func()
                if 'error' not in result:
                    results.append({
                        'Model': model_name,
                        'RMSE': result['metrics']['RMSE'],
                        'MAE': result['metrics']['MAE'],
                        'R2': result['metrics']['R2'],
                        'MAPE': result['metrics']['MAPE'],
                        'Next_Prediction': result.get('next_prediction', 'N/A')
                    })
                else:
                    results.append({
                        'Model': model_name,
                        'RMSE': 'Error',
                        'MAE': 'Error',
                        'R2': 'Error',
                        'MAPE': 'Error',
                        'Next_Prediction': 'Error'
                    })
            except Exception as e:
                results.append({
                    'Model': model_name,
                    'RMSE': f'Error: {str(e)}',
                    'MAE': 'Error',
                    'R2': 'Error',
                    'MAPE': 'Error',
                    'Next_Prediction': 'Error'
                })

        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # Generate synthetic stock data
    close_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)

    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': close_prices + np.random.randn(len(dates)) * 0.2,
        'High': close_prices + np.abs(np.random.randn(len(dates)) * 0.5),
        'Low': close_prices - np.abs(np.random.randn(len(dates)) * 0.5),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    })

    # Test prediction models
    predictor = StockPredictor(sample_data)

    print("Testing Linear Regression...")
    lr_result = predictor.linear_regression_prediction()
    if 'error' not in lr_result:
        print(f"RMSE: {lr_result['metrics']['RMSE']:.4f}")
        print(f"Next prediction: {lr_result['next_prediction']:.2f}")

    print("\nComparing all models...")
    comparison = predictor.compare_all_models()
    print(comparison)
