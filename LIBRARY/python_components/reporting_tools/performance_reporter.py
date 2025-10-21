"""
Performance Reporter for EA_SCALPER_XAUUSD Library
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

class PerformanceReporter:
    def __init__(self):
        self.trades = []
        self.account_data = []
        self.report_data = {}
        
    def add_trade(self, trade_data):
        """Add a trade to the performance records"""
        required_fields = ['timestamp', 'symbol', 'direction', 'lots', 'entry_price', 'exit_price', 'profit']
        for field in required_fields:
            if field not in trade_data:
                raise ValueError(f"Missing required field: {field}")
                
        self.trades.append(trade_data)
        
    def add_account_data(self, account_data):
        """Add account data point"""
        required_fields = ['timestamp', 'balance', 'equity', 'margin', 'free_margin']
        for field in required_fields:
            if field not in account_data:
                raise ValueError(f"Missing required field: {field}")
                
        self.account_data.append(account_data)
        
    def calculate_trade_statistics(self):
        """Calculate trade statistics"""
        if not self.trades:
            return {}
            
        df = pd.DataFrame(self.trades)
        
        # Basic statistics
        total_trades = len(df)
        winning_trades = len(df[df['profit'] > 0])
        losing_trades = len(df[df['profit'] < 0])
        
        # Profit statistics
        total_profit = df['profit'].sum()
        avg_profit = df['profit'].mean()
        max_profit = df['profit'].max()
        max_loss = df['profit'].min()
        
        # Win rate
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit factor
        gross_profits = df[df['profit'] > 0]['profit'].sum()
        gross_losses = abs(df[df['profit'] < 0]['profit'].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Average win/loss
        avg_win = df[df['profit'] > 0]['profit'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['profit'] < 0]['profit'].mean() if losing_trades > 0 else 0
        
        # Risk/reward ratio
        risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Consecutive wins/losses
        df['win'] = df['profit'] > 0
        df['group'] = (df['win'] != df['win'].shift()).cumsum()
        consecutive = df.groupby('group')['win'].agg(['count', 'first'])
        max_consecutive_wins = consecutive[consecutive['first'] == True]['count'].max() if not consecutive[consecutive['first'] == True].empty else 0
        max_consecutive_losses = consecutive[consecutive['first'] == False]['count'].max() if not consecutive[consecutive['first'] == False].empty else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'average_profit': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'risk_reward_ratio': risk_reward,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
        
    def calculate_account_statistics(self):
        """Calculate account statistics"""
        if not self.account_data:
            return {}
            
        df = pd.DataFrame(self.account_data)
        
        # Basic account statistics
        starting_balance = df['balance'].iloc[0] if len(df) > 0 else 0
        ending_balance = df['balance'].iloc[-1] if len(df) > 0 else 0
        peak_balance = df['balance'].max() if len(df) > 0 else 0
        max_drawdown = self.calculate_max_drawdown(df['balance']) if len(df) > 0 else 0
        
        # Equity statistics
        max_equity = df['equity'].max() if len(df) > 0 else 0
        min_equity = df['equity'].min() if len(df) > 0 else 0
        
        # Return calculations
        total_return = ((ending_balance - starting_balance) / starting_balance * 100) if starting_balance > 0 else 0
        
        return {
            'starting_balance': starting_balance,
            'ending_balance': ending_balance,
            'peak_balance': peak_balance,
            'max_drawdown': max_drawdown,
            'max_equity': max_equity,
            'min_equity': min_equity,
            'total_return_percent': total_return
        }
        
    def calculate_max_drawdown(self, balance_series):
        """Calculate maximum drawdown"""
        peak = balance_series.expanding().max()
        drawdown = (balance_series - peak) / peak * 100
        return drawdown.min()
        
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        trade_stats = self.calculate_trade_statistics()
        account_stats = self.calculate_account_statistics()
        
        self.report_data = {
            'report_generated': datetime.now().isoformat(),
            'trade_statistics': trade_stats,
            'account_statistics': account_stats
        }
        
        return self.report_data
        
    def save_report(self, file_path, format='json'):
        """Save performance report to file"""
        if not self.report_data:
            self.generate_performance_report()
            
        if format.lower() == 'json':
            with open(file_path, 'w') as f:
                json.dump(self.report_data, f, indent=2)
        elif format.lower() == 'csv':
            # Convert to DataFrame and save as CSV
            df_trades = pd.DataFrame(self.trades)
            df_account = pd.DataFrame(self.account_data)
            
            # Save to separate CSV files
            base_name = os.path.splitext(file_path)[0]
            df_trades.to_csv(f"{base_name}_trades.csv", index=False)
            df_account.to_csv(f"{base_name}_account.csv", index=False)
            
            # Save summary report
            summary_df = pd.DataFrame([self.report_data])
            summary_df.to_csv(f"{base_name}_summary.csv", index=False)
        else:
            raise ValueError("Unsupported format. Use 'json' or 'csv'")
            
        print(f"Performance report saved to {file_path}")
        
    def load_trades_from_file(self, file_path):
        """Load trades from file"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                self.trades = df.to_dict('records')
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    self.trades = json.load(f)
            else:
                raise ValueError("Unsupported file format")
            print(f"Loaded {len(self.trades)} trades from {file_path}")
        except Exception as e:
            print(f"Error loading trades: {e}")
            
    def load_account_data_from_file(self, file_path):
        """Load account data from file"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                self.account_data = df.to_dict('records')
            elif file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    self.account_data = json.load(f)
            else:
                raise ValueError("Unsupported file format")
            print(f"Loaded {len(self.account_data)} account data points from {file_path}")
        except Exception as e:
            print(f"Error loading account data: {e}")
            
    def plot_equity_curve(self, file_path=None):
        """Plot equity curve"""
        if not self.account_data:
            print("No account data available for plotting")
            return
            
        df = pd.DataFrame(self.account_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['equity'], label='Equity', linewidth=1)
        plt.plot(df['timestamp'], df['balance'], label='Balance', linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Equity Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if file_path:
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Equity curve saved to {file_path}")
        else:
            plt.show()
            
    def plot_drawdown_curve(self, file_path=None):
        """Plot drawdown curve"""
        if not self.account_data:
            print("No account data available for plotting")
            return
            
        df = pd.DataFrame(self.account_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate drawdown
        df['peak'] = df['balance'].expanding().max()
        df['drawdown'] = (df['balance'] - df['peak']) / df['peak'] * 100
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['drawdown'], label='Drawdown %', color='red', linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Drawdown (%)')
        plt.title('Drawdown Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if file_path:
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Drawdown curve saved to {file_path}")
        else:
            plt.show()

# Example usage
if __name__ == "__main__":
    # Example of how to use the PerformanceReporter class
    reporter = PerformanceReporter()
    
    # Add some sample trades
    # reporter.add_trade({
    #     'timestamp': '2023-01-01T10:00:00',
    #     'symbol': 'XAUUSD',
    #     'direction': 'BUY',
    #     'lots': 0.01,
    #     'entry_price': 1800.0,
    #     'exit_price': 1805.0,
    #     'profit': 50.0
    # })
    
    # Generate and save report
    # report = reporter.generate_performance_report()
    # reporter.save_report("performance_report.json")
    pass