"""
Trading Model for EA_SCALPER_XAUUSD Library
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from datetime import datetime

class TradingModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.features = []
        
    def prepare_features(self, data):
        """Prepare features for machine learning model"""
        # Technical indicators as features
        features = pd.DataFrame()
        
        # Price-based features
        features['open'] = data['open']
        features['high'] = data['high']
        features['low'] = data['low']
        features['close'] = data['close']
        features['volume'] = data['volume'] if 'volume' in data.columns else 0
        
        # Moving averages
        features['ma_5'] = data['close'].rolling(window=5).mean()
        features['ma_10'] = data['close'].rolling(window=10).mean()
        features['ma_20'] = data['close'].rolling(window=20).mean()
        features['ma_50'] = data['close'].rolling(window=50).mean()
        
        # Price changes
        features['price_change'] = data['close'].pct_change()
        features['price_change_5'] = data['close'].pct_change(periods=5)
        
        # Volatility
        features['volatility'] = data['close'].rolling(window=20).std()
        
        # RSI
        features['rsi'] = self.calculate_rsi(data['close'])
        
        # MACD
        macd, macd_signal = self.calculate_macd(data['close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        
        # Bollinger Bands
        bb_upper, bb_lower = self.calculate_bollinger_bands(data['close'])
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Lagged features
        for i in range(1, 6):
            features[f'close_lag_{i}'] = data['close'].shift(i)
            features[f'volume_lag_{i}'] = features['volume'].shift(i) if 'volume' in features.columns else 0
            
        # Drop rows with NaN values
        features = features.dropna()
        
        self.features = features.columns.tolist()
        return features
        
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
        
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        return upper_band, lower_band
        
    def prepare_labels(self, data, forecast_period=1):
        """Prepare labels for classification (buy/sell/hold)"""
        # Simple approach: if price goes up in next period, it's a buy signal
        future_prices = data['close'].shift(-forecast_period)
        price_change = (future_prices - data['close']) / data['close']
        
        # Label based on price change threshold
        threshold = 0.001  # 0.1% threshold
        labels = pd.Series(['hold'] * len(data))
        labels[price_change > threshold] = 'buy'
        labels[price_change < -threshold] = 'sell'
        
        return labels
        
    def train(self, data):
        """Train the machine learning model"""
        # Prepare features and labels
        features = self.prepare_features(data)
        labels = self.prepare_labels(data)
        
        # Align features and labels (remove NaN rows)
        min_len = min(len(features), len(labels))
        features = features.iloc[:min_len]
        labels = labels.iloc[:min_len]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
        
    def predict(self, data):
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        features = self.prepare_features(data)
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        return predictions, probabilities
        
    def save_model(self, file_path):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'model': self.model,
            'features': self.features,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, file_path)
        print(f"Model saved to {file_path}")
        
    def load_model(self, file_path):
        """Load a trained model from disk"""
        model_data = joblib.load(file_path)
        self.model = model_data['model']
        self.features = model_data['features']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {file_path}")

# Example usage
if __name__ == "__main__":
    # Example of how to use the TradingModel class
    model = TradingModel()
    
    # Load your data here
    # data = pd.read_csv("your_market_data.csv")
    # model.train(data)
    # model.save_model("trading_model.pkl")
    
    # To load and use a saved model:
    # model.load_model("trading_model.pkl")
    # predictions, probabilities = model.predict(new_data)
    pass