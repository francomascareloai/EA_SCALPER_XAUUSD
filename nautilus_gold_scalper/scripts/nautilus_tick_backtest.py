"""
NautilusTrader Tick-Based Backtest for Gold Scalper Strategy.

Uses 25M+ tick data for realistic execution simulation.
"""
import sys
from decimal import Decimal
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig, LoggingConfig, RiskEngineConfig
from nautilus_trader.model.identifiers import TraderId, Venue, InstrumentId, Symbol
from nautilus_trader.model.currencies import USD, XAU
from nautilus_trader.model.enums import AccountType, OmsType, AggressorSide
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.data import QuoteTick, BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType, AggregationSource
from nautilus_trader.model.instruments import CurrencyPair

from src.strategies.gold_scalper_strategy import GoldScalperStrategy, GoldScalperConfig


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
    sample_rate: int = 1,  # Take every Nth tick
) -> pd.DataFrame:
    """Load tick data from parquet file."""
    print(f"Loading tick data from {filepath}...")
    df = pd.read_parquet(filepath)
    
    # Ensure datetime is UTC
    if df['datetime'].dt.tz is None:
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    else:
        df['datetime'] = df['datetime'].dt.tz_convert('UTC')
    
    # Filter by date
    if start_date:
        df = df[df['datetime'] >= start_date]
    if end_date:
        df = df[df['datetime'] <= end_date]
    
    # Sample if needed (for faster testing)
    if sample_rate > 1:
        df = df.iloc[::sample_rate]
    
    print(f"Loaded {len(df):,} ticks from {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    return df


def create_quote_ticks(df: pd.DataFrame, instrument: CurrencyPair) -> list:
    """Convert DataFrame to QuoteTick objects."""
    print("Converting to QuoteTick objects...")
    
    ticks = []
    for idx, row in df.iterrows():
        ts_ns = int(row['datetime'].timestamp() * 1e9)
        
        tick = QuoteTick(
            instrument_id=instrument.id,
            bid_price=Price.from_str(f"{row['bid']:.2f}"),
            ask_price=Price.from_str(f"{row['ask']:.2f}"),
            bid_size=Quantity.from_str("1.00"),  # Match instrument precision (2 decimals)
            ask_size=Quantity.from_str("1.00"),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        ticks.append(tick)
        
        if len(ticks) % 100000 == 0:
            print(f"  Processed {len(ticks):,} ticks...")
    
    print(f"Created {len(ticks):,} QuoteTick objects")
    return ticks


def run_tick_backtest(
    start_date: str = "2024-10-01",
    end_date: str = "2024-10-31",  # 1 month for testing
    initial_balance: float = 100_000.0,
    sample_rate: int = 10,  # Sample every 10th tick for speed
    log_level: str = "WARNING",
):
    """Run backtest using tick data."""
    
    print("=" * 60)
    print("NAUTILUS TRADER TICK-BASED BACKTEST")
    print("=" * 60)
    
    # Configure engine
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("GOLD-TICK-001"),
        logging=LoggingConfig(log_level=log_level),
        risk_engine=RiskEngineConfig(bypass=True),
    )
    
    engine = BacktestEngine(config=engine_config)
    
    # Add venue
    SIM = Venue("SIM")
    engine.add_venue(
        venue=SIM,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=USD,
        starting_balances=[Money(initial_balance, USD)],
        default_leverage=Decimal("20"),
    )
    
    # Create instrument
    xauusd = create_xauusd_instrument(SIM)
    engine.add_instrument(xauusd)
    
    print(f"Instrument: {xauusd.id}")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    
    # Load tick data
    tick_path = Path(__file__).parent.parent.parent / "data/ticks/xauusd_2020_2024_stride20.parquet"
    df = load_tick_data(str(tick_path), start_date, end_date, sample_rate)
    
    # Convert to QuoteTicks
    ticks = create_quote_ticks(df, xauusd)
    
    # Add tick data to engine
    engine.add_data(ticks)
    print(f"Added {len(ticks):,} ticks to engine")
    
    # Create bar type for internal aggregation from ticks
    bar_type = BarType(
        instrument_id=xauusd.id,
        bar_spec=BarSpecification(
            step=5,
            aggregation=BarAggregation.MINUTE,
            price_type=PriceType.MID,  # Use mid price from bid/ask
        ),
        aggregation_source=AggregationSource.INTERNAL,  # Aggregate ticks internally
    )
    print(f"Bar type: {bar_type} (aggregated from ticks)")
    
    # Configure strategy
    strategy_config = GoldScalperConfig(
        strategy_id="GOLD-TICK-001",
        instrument_id=xauusd.id,
        ltf_bar_type=bar_type,  # Subscribe to bars aggregated from ticks
        
        # Thresholds
        execution_threshold=50,
        
        # Disable filters for initial testing
        use_session_filter=False,
        use_regime_filter=False,
        use_mtf=False,
        use_footprint=False,
        prop_firm_enabled=False,
        account_balance=initial_balance,
        debug_mode=True,
    )
    
    strategy = GoldScalperStrategy(config=strategy_config)
    engine.add_strategy(strategy)
    
    print(f"\nStrategy: {strategy_config.strategy_id}")
    print(f"Sample Rate: 1/{sample_rate} ticks")
    
    # Run
    print("\n" + "=" * 60)
    print("RUNNING TICK BACKTEST...")
    print("=" * 60 + "\n")
    
    engine.run()
    
    # Results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    try:
        fills = engine.trader.generate_order_fills_report()
        print(f"\nOrder Fills: {len(fills)}")
        if len(fills) > 0:
            print(fills)
    except Exception as e:
        print(f"Fills error: {e}")
    
    try:
        positions = engine.trader.generate_positions_report()
        print(f"\nPositions:")
        print(positions)
    except Exception as e:
        print(f"Positions error: {e}")
    
    try:
        account = engine.trader.generate_account_report(SIM)
        print(f"\nAccount:")
        print(account)
    except Exception as e:
        print(f"Account error: {e}")
    
    engine.dispose()
    
    print("\n" + "=" * 60)
    print("TICK BACKTEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_tick_backtest(
        start_date="2024-10-01",
        end_date="2024-10-07",  # 1 week for quick test
        initial_balance=100_000.0,
        sample_rate=20,  # Every 20th tick
        log_level="INFO",
    )
