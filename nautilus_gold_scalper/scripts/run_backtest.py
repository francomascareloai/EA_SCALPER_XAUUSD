"""
NautilusTrader Tick-Based Backtest for Gold Scalper.

Uses real XAUUSD tick data (25M+ records) with NautilusTrader native engine.
Ticks are fed as QuoteTicks and aggregated to M5 bars for strategy consumption.
"""
import sys
from decimal import Decimal
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nautilus_trader.backtest.engine import BacktestEngine as NautilusEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, RiskEngineConfig
from nautilus_trader.model.identifiers import TraderId, Venue, InstrumentId, Symbol
from nautilus_trader.model.currencies import USD, XAU
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.data import QuoteTick, Bar, BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType, AggregationSource
from nautilus_trader.model.instruments import CurrencyPair

from src.strategies.gold_scalper_strategy import GoldScalperStrategy, GoldScalperConfig


def find_tick_file() -> Path:
    """Locate a tick file under Python_Agent_Hub/ml_pipeline/data."""
    root = Path(__file__).parent.parent.parent / "Python_Agent_Hub" / "ml_pipeline" / "data"
    candidates = list(root.glob("**/*tick*.parquet")) + list(root.glob("**/*ticks*.parquet"))
    if not candidates:
        candidates = list(root.glob("**/*tick*.csv")) + list(root.glob("**/*ticks*.csv"))
    if not candidates:
        raise FileNotFoundError(f"Nenhum arquivo de ticks encontrado em {root}")
    # choose the largest (most complete)
    return max(candidates, key=lambda p: p.stat().st_size)


def create_xauusd_instrument(venue: Venue) -> CurrencyPair:
    """Create XAUUSD instrument for backtesting."""
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol("XAU/USD"), venue),
        raw_symbol=Symbol("XAUUSD"),
        base_currency=XAU,
        quote_currency=USD,
        price_precision=2,
        size_precision=2,
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.01"),
        lot_size=Quantity.from_str("1"),
        max_quantity=Quantity.from_str("100"),
        min_quantity=Quantity.from_str("0.01"),
        max_price=Price.from_str("10000.00"),
        min_price=Price.from_str("100.00"),
        margin_init=Decimal("0.05"),
        margin_maint=Decimal("0.03"),
        maker_fee=Decimal("0.0001"),
        taker_fee=Decimal("0.0002"),
        ts_event=0,
        ts_init=0,
    )


def load_tick_data(
    filepath: str,
    start_date: str = None,
    end_date: str = None,
    sample_rate: int = 1,
) -> pd.DataFrame:
    """Load tick data from parquet file."""
    print(f"Loading tick data from {filepath}...")
    df = pd.read_parquet(filepath)
    
    if df['datetime'].dt.tz is None:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    else:
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')

    # Basic validation: no NaN, monotonic increasing timestamps
    if df['datetime'].isna().any():
        raise ValueError("Tick data contains NaN datetimes")
    if not df['datetime'].is_monotonic_increasing:
        df = df.sort_values('datetime')
        if not df['datetime'].is_monotonic_increasing:
            raise ValueError("Tick data timestamps are not monotonic even after sort")
    if df[['bid', 'ask']].isna().any().any():
        raise ValueError("Tick data contains NaN bid/ask values")
    
    if start_date:
        df = df[df['datetime'] >= start_date]
    if end_date:
        df = df[df['datetime'] <= end_date]
    
    if sample_rate > 1:
        df = df.iloc[::sample_rate]
    
    print(f"Loaded {len(df):,} ticks from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    return df


def create_quote_ticks(df: pd.DataFrame, instrument: CurrencyPair, slippage_ticks: int = 0) -> list:
    """Convert DataFrame to QuoteTick objects."""
    print("Converting to QuoteTick objects...")
    
    slip_value = float(instrument.price_increment) * max(0, slippage_ticks)
    ticks = []
    for idx, row in df.iterrows():
        ts_ns = int(row['datetime'].timestamp() * 1e9)
        bid_px = row['bid'] - slip_value
        ask_px = row['ask'] + slip_value
        
        tick = QuoteTick(
            instrument_id=instrument.id,
            bid_price=Price.from_str(f"{bid_px:.2f}"),
            ask_price=Price.from_str(f"{ask_px:.2f}"),
            bid_size=Quantity.from_str("1.00"),
            ask_size=Quantity.from_str("1.00"),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        ticks.append(tick)
        
        if len(ticks) % 500000 == 0:
            print(f"  Processed {len(ticks):,} ticks...")
    
    print(f"Created {len(ticks):,} QuoteTick objects")
    return ticks


def aggregate_ticks_to_bars(df: pd.DataFrame, bar_type: BarType, interval_minutes: int = 5) -> list:
    """Pre-aggregate ticks into M5 bars."""
    print(f"Aggregating ticks into {interval_minutes}-minute bars...")
    
    # Calculate mid price
    df = df.copy()
    df['mid'] = (df['bid'] + df['ask']) / 2
    
    # Resample to bars
    df.set_index('datetime', inplace=True)
    ohlcv = df['mid'].resample(f'{interval_minutes}min').ohlc()
    ohlcv['volume'] = df['mid'].resample(f'{interval_minutes}min').count()
    
    # Drop NaN rows
    ohlcv = ohlcv.dropna()
    
    print(f"Created {len(ohlcv):,} bars")
    
    bars = []
    for timestamp, row in ohlcv.iterrows():
        ts_ns = int(timestamp.timestamp() * 1e9)
        
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{row['open']:.2f}"),
            high=Price.from_str(f"{row['high']:.2f}"),
            low=Price.from_str(f"{row['low']:.2f}"),
            close=Price.from_str(f"{row['close']:.2f}"),
            volume=Quantity.from_str(f"{row['volume']:.2f}"),  # Match instrument precision
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        bars.append(bar)
    
    print(f"Created {len(bars):,} Bar objects")
    return bars


class BacktestRunner:
    """NautilusTrader-based tick backtest runner."""
    
    def __init__(
        self,
        initial_balance: float = 100_000.0,
        log_level: str = "WARNING",
        slippage_ticks: int = 2,
        commission_per_contract: float = 2.5,
    ):
        self.initial_balance = initial_balance
        self.log_level = log_level
        self.engine = None
        self.venue = Venue("SIM")
        self.instrument = None
        self.slippage_ticks = slippage_ticks
        self.commission_per_contract = commission_per_contract
    
    def run(
        self,
        start_date: str = "2024-10-01",
        end_date: str = "2024-12-31",
        sample_rate: int = 1,
        use_session_filter: bool = True,
        use_regime_filter: bool = True,
        use_mtf: bool = False,
        use_footprint: bool = True,
        prop_firm_enabled: bool = True,
        use_news_filter: bool = True,
        execution_threshold: int = 65,
        debug_mode: bool = False,
    ):
        """Run tick-based backtest with NautilusTrader."""
        
        print("=" * 60)
        print("NAUTILUS TRADER TICK-BASED BACKTEST")
        print("=" * 60)
        print(f"Period: {start_date} to {end_date}")
        print(f"Sample Rate: 1/{sample_rate} ticks")
        print(f"Filters: session={use_session_filter}, regime={use_regime_filter}, footprint={use_footprint}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        
        # Configure engine
        engine_config = BacktestEngineConfig(
            trader_id=TraderId("GOLD-TICK-001"),
            logging=LoggingConfig(log_level=self.log_level),
            risk_engine=RiskEngineConfig(bypass=False),
        )
        
        self.engine = NautilusEngine(config=engine_config)
        
        # Add venue
        self.engine.add_venue(
            venue=self.venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[Money(self.initial_balance, USD)],
            default_leverage=Decimal("20"),
        )
        
        # Create instrument
        self.instrument = create_xauusd_instrument(self.venue)
        self.engine.add_instrument(self.instrument)
        
        print(f"Instrument: {self.instrument.id}")
        
        # Load tick data - use parquet file directly
        tick_path = Path(__file__).parent.parent.parent / "data" / "ticks" / "xauusd_2020_2024_stride20.parquet"
        if not tick_path.exists():
            raise FileNotFoundError(f"Tick data not found at {tick_path}")
        df = load_tick_data(str(tick_path), start_date, end_date, sample_rate)
        
        # Create bar type (EXTERNAL since we pre-aggregate)
        bar_type = BarType(
            instrument_id=self.instrument.id,
            bar_spec=BarSpecification(
                step=5,
                aggregation=BarAggregation.MINUTE,
                price_type=PriceType.MID,
            ),
            aggregation_source=AggregationSource.EXTERNAL,
        )
        print(f"Bar type: {bar_type}")
        
        # Convert ticks to QuoteTicks for spread-aware execution
        quote_ticks = create_quote_ticks(df, self.instrument, slippage_ticks=self.slippage_ticks)

        # Pre-aggregate ticks to M5 bars
        bars = aggregate_ticks_to_bars(df, bar_type, interval_minutes=5)
        
        # Add data to engine
        self.engine.add_data(quote_ticks)
        self.engine.add_data(bars)
        print(f"Added {len(quote_ticks):,} ticks and {len(bars):,} bars to engine")
        
        # Configure strategy
        strategy_config = GoldScalperConfig(
            strategy_id="GOLD-TICK-001",
            instrument_id=self.instrument.id,
            ltf_bar_type=bar_type,
            
            # Thresholds
            execution_threshold=execution_threshold,
            
            # Filters
            use_session_filter=use_session_filter,
            use_regime_filter=use_regime_filter,
            use_mtf=use_mtf,
            use_footprint=use_footprint,
            prop_firm_enabled=prop_firm_enabled,
            use_news_filter=use_news_filter,
            account_balance=self.initial_balance,
            debug_mode=debug_mode,
            
            # Disable HTF requirement (no H1 bars from tick data)
            require_htf_align=False,
        )
        
        strategy = GoldScalperStrategy(config=strategy_config)
        self.engine.add_strategy(strategy)
        
        print(f"Strategy: {strategy_config.strategy_id}")
        
        # Run
        print("\n" + "=" * 60)
        print("RUNNING TICK BACKTEST...")
        print("=" * 60 + "\n")
        
        self.engine.run()
        
        # Print results
        self._print_results()
        
        self.engine.dispose()
        
        print("\n" + "=" * 60)
        print("TICK BACKTEST COMPLETE")
        print("=" * 60)
    
    def _print_results(self):
        """Print backtest results."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        try:
            fills = self.engine.trader.generate_order_fills_report()
            print(f"\nOrder Fills: {len(fills)}")
            if len(fills) > 0:
                print(fills.to_string())
        except Exception as e:
            print(f"Fills report error: {e}")
        
        try:
            positions = self.engine.trader.generate_positions_report()
            print(f"\nPositions Report:")
            if len(positions) > 0:
                print(positions.to_string())
            else:
                print("  No positions")
        except Exception as e:
            print(f"Positions report error: {e}")
        
        try:
            account = self.engine.trader.generate_account_report(self.venue)
            
            # Calculate summary
            final_balance = float(account['total'].iloc[-1]) if len(account) > 0 else 100000
            fills_count = len(fills) if 'fills' in locals() and fills is not None else 0
            total_commissions = fills_count * self.commission_per_contract
            total_pnl = final_balance - 100000 - total_commissions
            
            print(f"\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Final Balance: ${final_balance:,.2f}")
            print(f"Total PnL (net commissions): ${total_pnl:,.2f} ({total_pnl/1000:.2f}%)")
            if total_commissions > 0:
                print(f"Commissions: ${total_commissions:,.2f} ({fills_count} fills @ {self.commission_per_contract} each)")
            
            # Win rate from positions
            if 'positions' in dir() and len(positions) > 0 and 'realized_pnl' in positions.columns:
                pnls = positions['realized_pnl'].apply(
                    lambda x: float(str(x).replace(' USD', '')) if pd.notna(x) else 0
                )
                wins = (pnls > 0).sum()
                losses = (pnls < 0).sum()
                total = wins + losses
                print(f"Trades: {total} (W:{wins} L:{losses})")
                if total > 0:
                    print(f"Win Rate: {wins/total*100:.1f}%")
                    print(f"Avg PnL/trade: ${pnls.mean():.2f}")
            
            print(f"\nAccount Report:")
            print(account.to_string())
        except Exception as e:
            print(f"Account report error: {e}")
        
        # Strategy stats
        try:
            for strategy in self.engine.trader.strategies():
                if hasattr(strategy, 'stats'):
                    print(f"\nStrategy Stats ({strategy.id}):")
                    for k, v in strategy.stats.items():
                        print(f"  {k}: {v}")
        except Exception:
            pass


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run XAUUSD backtest')
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2024-03-31', help='End date')
    parser.add_argument('--threshold', type=int, default=70, help='Execution threshold')
    parser.add_argument('--sample', type=int, default=1, help='Tick sample rate (1 = all ticks)')
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--no-news', action='store_true', help='Disable news filter')
    args = parser.parse_args()
    
    runner = BacktestRunner(
        initial_balance=100_000.0,
        log_level="ERROR" if args.sweep else "INFO",
    )
    
    if args.sweep:
        # Parameter sweep mode
        import json
        from datetime import datetime
        
        thresholds = [60, 65, 70, 75]
        results = []
        
        for thresh in thresholds:
            print(f"\n>>> Testing threshold={thresh}...")
            try:
                runner = BacktestRunner(initial_balance=100_000.0, log_level="ERROR")
                runner.run(
                    start_date=args.start,
                    end_date=args.end,
                    sample_rate=args.sample,
                    use_session_filter=True,
                    use_regime_filter=True,
                    use_mtf=True,
                    use_footprint=True,
                    prop_firm_enabled=True,
                    use_news_filter=not args.no_news,
                    execution_threshold=thresh,
                    debug_mode=False,
                )
                
                # Get results
                account = runner.engine.trader.generate_account_report(runner.venue)
                final = float(account['total'].iloc[-1]) if len(account) > 0 else 100000
                pnl = final - 100000
                
                fills = runner.engine.trader.generate_order_fills_report()
                trades = len(fills) // 2
                
                results.append({
                    'threshold': thresh,
                    'pnl': pnl,
                    'trades': trades,
                    'final_balance': final,
                })
                print(f"    PnL: ${pnl:,.2f}, Trades: {trades}")
                runner.engine.dispose()
            except Exception as e:
                print(f"    ERROR: {e}")
                results.append({'threshold': thresh, 'error': str(e)})
        
        print("\n" + "="*60)
        print("PARAMETER SWEEP RESULTS")
        print("="*60)
        for r in results:
            if 'error' not in r:
                print(f"Threshold {r['threshold']}: PnL=${r['pnl']:,.2f}, Trades={r['trades']}")
    else:
        # Single run mode
        runner.run(
            start_date=args.start,
            end_date=args.end,
            sample_rate=args.sample,
            use_session_filter=True,
            use_regime_filter=True,
            use_mtf=True,  # HTF/MTF derived from aggregated bars
            use_footprint=True,
            prop_firm_enabled=True,
            use_news_filter=not args.no_news,
            execution_threshold=args.threshold,
            debug_mode=True,
        )


if __name__ == "__main__":
    main()
