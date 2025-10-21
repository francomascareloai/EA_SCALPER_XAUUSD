"""
Conversion Helper for EA_SCALPER_XAUUSD Library
"""
import re
import os
import json
from datetime import datetime

class MQLToPineConverter:
    def __init__(self):
        self.conversion_rules = {
            # Function mappings
            'iMA': 'ta.sma',
            'iEMA': 'ta.ema',
            'iRSI': 'ta.rsi',
            'iMACD': 'ta.macd',
            'iStochastic': 'ta.stoch',
            'iBands': 'ta.bb',
            'iATR': 'ta.atr',
            'iADX': 'ta.adx',
            
            # Variable type mappings
            'double': 'float',
            'int': 'int',
            'bool': 'bool',
            'string': 'string',
            
            # Control structure mappings
            'for': 'for',
            'while': 'while',
            'if': 'if',
            'else': 'else',
            
            # Common function mappings
            'Alert': 'alert',
            'Print': 'print',
            'Comment': 'label.new',
            'PlotIndexSetDouble': 'plot'
        }
        
    def convert_mql_to_pine(self, mql_code):
        """Convert MQL4/5 code to Pine Script"""
        pine_code = mql_code
        
        # Add Pine Script header
        pine_header = "// Converted from MQL4/5 to Pine Script\n"
        pine_header += "// Conversion date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n"
        
        # Add version directive
        if "@version=5" not in pine_code:
            pine_header += "//@version=5\n"
            
        # Determine if indicator or strategy
        if "int start()" in pine_code or "void OnTick()" in pine_code:
            pine_header += "strategy(\"Converted Strategy\", overlay=true)\n\n"
        else:
            pine_header += "indicator(\"Converted Indicator\", overlay=true)\n\n"
            
        # Replace function calls
        for mql_func, pine_func in self.conversion_rules.items():
            pine_code = re.sub(r'\b' + mql_func + r'\b', pine_func, pine_code)
            
        # Replace variable declarations
        pine_code = re.sub(r'extern\s+(\w+)\s+(\w+)', r'\2 = input.\1', pine_code)
        pine_code = re.sub(r'(\w+)\s+(\w+)\[\]', r'var \1[] \2 = array.new_\1()', pine_code)
        
        # Replace array indexing
        pine_code = re.sub(r'(\w+)\[(\d+)\]', r'\1[\2]', pine_code)
        
        # Replace control structures
        pine_code = re.sub(r'for\s*\((.*?);(.*?);(.*?)\)', r'for [\1 to \2 by \3]', pine_code)
        
        # Add header
        pine_code = pine_header + pine_code
        
        return pine_code
        
    def extract_indicators_from_mql(self, mql_code):
        """Extract indicator information from MQL code"""
        indicators = []
        
        # Find indicator function calls
        indicator_pattern = r'i(\w+)\([^)]*\)'
        matches = re.findall(indicator_pattern, mql_code)
        
        for match in matches:
            indicators.append({
                'type': match,
                'parameters': self.extract_parameters(mql_code, match)
            })
            
        return indicators
        
    def extract_parameters(self, mql_code, indicator_type):
        """Extract parameters for a specific indicator"""
        pattern = rf'i{indicator_type}\(([^)]*)\)'
        match = re.search(pattern, mql_code)
        
        if match:
            params = match.group(1).split(',')
            return [param.strip() for param in params]
        return []
        
    def generate_conversion_report(self, original_file, converted_file):
        """Generate a conversion report"""
        report = {
            'conversion_date': datetime.now().isoformat(),
            'original_file': original_file,
            'converted_file': converted_file,
            'status': 'success',
            'notes': [
                'Conversion completed successfully',
                'Manual review recommended',
                'Test the converted script before using in production'
            ]
        }
        
        return report
        
    def save_conversion_report(self, report, file_path):
        """Save conversion report to file"""
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Conversion report saved to {file_path}")

# Example usage
def convert_file(mql_file_path, output_dir):
    """Convert a single MQL file to Pine Script"""
    try:
        # Read MQL file
        with open(mql_file_path, 'r', encoding='utf-8') as f:
            mql_content = f.read()
            
        # Convert to Pine Script
        converter = MQLToPineConverter()
        pine_content = converter.convert_mql_to_pine(mql_content)
        
        # Save converted file
        filename = os.path.basename(mql_file_path)
        name, _ = os.path.splitext(filename)
        pine_filename = f"{name}.pine"
        pine_file_path = os.path.join(output_dir, pine_filename)
        
        with open(pine_file_path, 'w', encoding='utf-8') as f:
            f.write(pine_content)
            
        # Generate and save report
        report = converter.generate_conversion_report(mql_file_path, pine_file_path)
        report_file_path = os.path.join(output_dir, f"{name}_conversion_report.json")
        converter.save_conversion_report(report, report_file_path)
        
        print(f"Successfully converted {mql_file_path} to {pine_file_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {mql_file_path}: {e}")
        return False

if __name__ == "__main__":
    # Example usage
    # convert_file("example.mq4", "./converted/")
    pass