"""
Test Apex Trading Compliance - Verify all P0 blockers are implemented.

Tests:
1. Time constraints (4:59 PM ET)
2. Consistency rule (30% max daily profit)
3. Circuit breaker integration
4. Trailing DD calculation
5. Account termination on breach
"""
import pytest
from datetime import datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk import (
    PropFirmManager,
    PropFirmLimits,
    AccountTerminatedException,
    CircuitBreaker,
)
from src.risk.time_constraint_manager import TimeConstraintManager
from src.risk.consistency_tracker import ConsistencyTracker


class TestApexCompliance:
    """Test suite for Apex Trading compliance."""
    
    def test_time_constraint_4_59_pm_et(self):
        """Test that TimeConstraintManager blocks trading at 4:59 PM ET."""
        # Create mock strategy
        class MockConfig:
            instrument_id = "XAUUSD"
        
        class MockCache:
            def positions_open(self):
                return []
        
        class MockStrategy:
            def __init__(self):
                self._is_trading_allowed = True
                self.log = MockLogger()
                self.config = MockConfig()
                self.cache = MockCache()
            
            def close_all_positions(self, instrument_id):
                pass
        
        class MockLogger:
            def error(self, msg):
                print(f"ERROR: {msg}")
            def warning(self, msg):
                print(f"WARNING: {msg}")
        
        strategy = MockStrategy()
        
        # Initialize TimeConstraintManager with 4:59 PM ET cutoff
        time_mgr = TimeConstraintManager(
            strategy=strategy,
            cutoff=time(16, 59),  # 4:59 PM
            warning=time(16, 0),
            urgent=time(16, 30),
            emergency=time(16, 55),
            allow_overnight=False,
        )
        
        # Test: Before cutoff (4:30 PM ET)
        et_tz = ZoneInfo("America/New_York")
        dt_before = datetime(2025, 12, 7, 16, 30, 0, tzinfo=et_tz)
        ts_before_ns = int(dt_before.timestamp() * 1e9)
        
        assert time_mgr.check(ts_before_ns) == True, "Trading should be allowed before 4:59 PM ET"
        
        # Test: At cutoff (4:59 PM ET)
        dt_cutoff = datetime(2025, 12, 7, 16, 59, 0, tzinfo=et_tz)
        ts_cutoff_ns = int(dt_cutoff.timestamp() * 1e9)
        
        result = time_mgr.check(ts_cutoff_ns)
        assert result == False, "Trading should be BLOCKED at 4:59 PM ET"
        assert strategy._is_trading_allowed == False, "Strategy should be blocked after cutoff"
    
    def test_consistency_rule_30_percent(self):
        """Test that ConsistencyTracker blocks trading at 30% daily profit limit."""
        tracker = ConsistencyTracker(initial_balance=100_000.0)
        
        # Simulate: $5k total profit over 20 days
        for _ in range(20):
            tracker.update_profit(250.0, datetime.now(ZoneInfo("America/New_York")))
            tracker.reset_daily()  # Reset for next day
        
        # Total profit: $5k
        assert tracker.total_profit == Decimal("5000.0")
        
        # Day 21: Add $1.2k profit (24% of $5k - below 25% buffer)
        tracker.update_profit(1200.0, datetime.now(ZoneInfo("America/New_York")))
        assert tracker.can_trade(), "Trading allowed at 24% daily profit"
        
        # Add another $100 (total $1.3k = 26% of $5k - above 25% buffer)
        tracker.update_profit(100.0, datetime.now(ZoneInfo("America/New_York")))
        assert not tracker.can_trade(), "Trading should be BLOCKED above 25% buffer"
    
    def test_circuit_breaker_integration(self):
        """Test that CircuitBreaker escalates correctly on consecutive losses."""
        cb = CircuitBreaker(daily_loss_limit=0.05, total_loss_limit=0.10)
        cb.update_equity(100_000.0)
        
        # Test: 3 consecutive losses → Level 1
        for _ in range(3):
            cb.register_trade_result(pnl=-100.0, is_win=False)
            cb.update_equity(cb._state.current_equity - 100.0)
        
        assert cb.get_level().value >= 1, "Should escalate to Level 1 after 3 losses"
        assert not cb.can_trade(), "Should block trading in Level 1 cooldown"
    
    def test_trailing_dd_calculation(self):
        """Test that PropFirmManager calculates trailing DD correctly."""
        limits = PropFirmLimits(
            account_size=100_000.0,
            daily_loss_limit=5_000.0,  # 5%
            trailing_drawdown=5_000.0,  # 5% Apex trailing DD
        )
        prop_mgr = PropFirmManager(limits=limits)
        prop_mgr.initialize(100_000.0)
        
        # Test: Equity increases → HWM updates
        prop_mgr.update_equity(105_000.0)
        assert prop_mgr._high_water == 105_000.0, "HWM should update when equity increases"
        
        # Test: Equity drops to $100k (5k loss from HWM = 4.76% DD - below 5%)
        prop_mgr.update_equity(100_000.0)
        state = prop_mgr.get_state()
        assert state.trailing_dd_current == 5_000.0, "Trailing DD should be $5k"
        assert state.is_trading_allowed == True, "Trading allowed at 4.76% DD"
        
        # Test: Equity drops to $99.75k (5.25k loss from HWM = 5% DD - AT LIMIT)
        prop_mgr.update_equity(99_750.0)
        state = prop_mgr.get_state()
        assert state.trailing_dd_current == 5_250.0, "Trailing DD should be $5.25k"
        assert state.is_hard_breached == True, "Should breach at 5% DD"
    
    def test_account_termination_on_breach(self):
        """Test that PropFirmManager raises AccountTerminatedException on breach."""
        limits = PropFirmLimits(
            account_size=100_000.0,
            trailing_drawdown=5_000.0,  # 5% Apex trailing DD
        )
        prop_mgr = PropFirmManager(limits=limits)
        prop_mgr.initialize(100_000.0)
        
        # Update equity to $95k (5k loss = 5% DD)
        prop_mgr.update_equity(95_000.0)
        
        # Test: can_trade() should trigger _hard_stop() which raises exception
        with pytest.raises(AccountTerminatedException):
            prop_mgr.can_trade()
    
    def test_config_values_loaded(self):
        """Test that config values are loaded correctly from YAML."""
        # This test verifies that GoldScalperConfig has correct default values
        from src.strategies.gold_scalper_strategy import GoldScalperConfig
        
        config = GoldScalperConfig()
        
        # Verify Apex-specific values
        assert config.flatten_time_et == "16:59", "Cutoff time should be 4:59 PM ET"
        assert config.allow_overnight == False, "Overnight should be disabled for Apex"
        assert config.consistency_cap_pct == 30.0, "Consistency limit should be 30%"
        assert config.cb_level_1_losses == 3, "Circuit breaker L1 should trigger at 3 losses"
        assert config.cb_level_5_dd == 4.5, "Circuit breaker L5 should trigger at 4.5% DD"


if __name__ == "__main__":
    # Run basic tests without pytest
    test = TestApexCompliance()
    
    print("Testing Time Constraint (4:59 PM ET)...")
    test.test_time_constraint_4_59_pm_et()
    print("PASS\n")
    
    print("Testing Consistency Rule (30%)...")
    test.test_consistency_rule_30_percent()
    print("PASS\n")
    
    print("Testing Circuit Breaker Integration...")
    test.test_circuit_breaker_integration()
    print("PASS\n")
    
    print("Testing Trailing DD Calculation...")
    test.test_trailing_dd_calculation()
    print("PASS\n")
    
    print("Testing Account Termination...")
    test.test_account_termination_on_breach()
    print("PASS\n")
    
    print("Testing Config Values...")
    test.test_config_values_loaded()
    print("PASS\n")
    
    print("=" * 60)
    print("ALL APEX COMPLIANCE TESTS PASSED")
    print("=" * 60)
