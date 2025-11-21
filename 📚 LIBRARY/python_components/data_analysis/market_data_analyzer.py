"""
Market Data Analyzer for EA_SCALPER_XAUUSD Library
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json

class MarketDataAnalyzer:
    def __init__(self):
        self.data = None
        self.symbol = ""
        
    def load_data(self, file_path):
        """Load market data from file"""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                self.data = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported file format")
                
            print(f"Loaded {len(self.data)} records from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_indicators(self):
        """Calculate technical indicators"""
        if self.data is None:
            print("No data loaded")
            return
            
        # Calculate simple moving averages
        self.data['MA_20'] = self.data['close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['close'].rolling(window=50).mean()
        
        # Calculate RSI
        self.data['RSI'] = self.calculate_rsi(self.data['close'])
        
        # Calculate MACD
        self.data['MACD'], self.data['MACD_signal'] = self.calculate_macd(self.data['close'])
        
        print("Technical indicators calculated")
        
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
        
    def identify_patterns(self):
        """Identify market patterns"""
        if self.data is None:
            print("No data loaded")
            return
            
        # Simple pattern recognition
        self.data['bullish_engulfing'] = self.identify_bullish_engulfing()
        self.data['bearish_engulfing'] = self.identify_bearish_engulfing()
        
        print("Market patterns identified")
        
    def identify_bullish_engulfing(self):
        """Identify bullish engulfing patterns"""
        patterns = []
        for i in range(1, len(self.data)):
            # Previous candle is bearish and current is bullish
            # Current candle completely engulfs previous
            if (self.data['close'].iloc[i-1] < self.data['open'].iloc[i-1] and  # Previous bearish
                self.data['close'].iloc[i] > self.data['open'].iloc[i] and      # Current bullish
                self.data['open'].iloc[i] < self.data['close'].iloc[i-1] and    # Current opens below previous close
                self.data['close'].iloc[i] > self.data['open'].iloc[i-1]):      # Current closes above previous open
                patterns.append(True)
            else:
                patterns.append(False)
        patterns.insert(0, False)  # No pattern for first candle
        return patterns
        
    def identify_bearish_engulfing(self):
        """Identify bearish engulfing patterns"""
        patterns = []
        for i in range(1, len(self.data)):
            # Previous candle is bullish and current is bearish
            # Current candle completely engulfs previous
            if (self.data['close'].iloc[i-1] > self.data['open'].iloc[i-1] and  # Previous bullish
                self.data['close'].iloc[i] < self.data['open'].iloc[i] and      # Current bearish
                self.data['open'].iloc[i] > self.data['close'].iloc[i-1] and    # Current opens above previous close
                self.data['close'].iloc[i] < self.data['open'].iloc[i-1]):      # Current closes below previous open
                patterns.append(True)
            else:
                patterns.append(False)
        patterns.insert(0, False)  # No pattern for first candle
        return patterns
        
    def generate_report(self):
        """Generate analysis report"""
        if self.data is None:
            print("No data loaded")
            return
            
        report = {
            "symbol": self.symbol,
            "analysis_date": datetime.now().isoformat(),
            "total_records": len(self.data),
            "price_stats": {
                "open": {
                    "min": float(self.data['open'].min()),
                    "max": float(self.data['open'].max()),
                    "avg": float(self.data['open'].mean())
                },
                "high": {
                    "min": float(self.data['high'].min()),
                    "max": float(self.data['high'].max()),
                    "avg": float(self.data['high'].mean())
                },
                "low": {
                    "min": float(self.data['low'].min()),
                    "max": float(self.data['low'].max()),
                    "avg": float(self.data['low'].mean())
                },
                "close": {
                    "min": float(self.data['close'].min()),
                    "max": float(self.data['close'].max()),
                    "avg": float(self.data['close'].mean())
                }
            },
            "indicator_stats": {
                "rsi": {
                    "min": float(self.data['RSI'].min()),
                    "max": float(self.data['RSI'].max()),
                    "avg": float(self.data['RSI'].mean())
                },
                "ma_20": {
                    "min": float(self.data['MA_20'].min()),
                    "max": float(self.data['MA_20'].max()),
                    "avg": float(self.data['MA_20'].mean())
                },
                "ma_50": {
                    "min": float(self.data['MA_50'].min()),
                    "max": float(self.data['MA_50'].max()),
                    "avg": float(self.data['MA_50'].mean())
                }
            },
            "pattern_counts": {
                "bullish_engulfing": int(self.data['bullish_engulfing'].sum()),
                "bearish_engulfing": int(self.data['bearish_engulfing'].sum())
            }
        }
        
        return report
        
    def save_report(self, file_path):
        """Save analysis report to file"""
        report = self.generate_report()
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {file_path}")

# Example usage
if __name__ == "__main__":
    analyzer = MarketDataAnalyzer()
    # analyzer.load_data("market_data.csv")
    # analyzer.calculate_indicators()
    # analyzer.identify_patterns()
    # analyzer.save_report("analysis_report.json")