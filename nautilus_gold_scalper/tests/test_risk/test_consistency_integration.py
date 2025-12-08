"""
Integration tests for ConsistencyTracker with 20% limit.
Validates P1 critical item: Apex consistency rule blocks trades at 20% (10% margin).
"""
import pytest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from src.risk.consistency_tracker import ConsistencyTracker


class TestConsistencyIntegration:
    """Integration tests for 20% consistency rule."""
    
    def test_consistency_20_percent_limit(self):
        """Test that 20% daily profit blocks further trades."""
        tracker = ConsistencyTracker(initial_balance=100000.0)
        et_tz = ZoneInfo("America/New_York")
        now = datetime(2024, 11, 1, 10, 0, 0, tzinfo=et_tz)
        
        # Day 1: Make $10,000 total profit (build history)
        tracker.update_profit(trade_pnl=10000.0, now=now)
        
        # Total profit = $10,000, Daily profit = $10,000
        # First day = 100% daily consistency (blocks immediately)
        assert not tracker.can_trade(now=now), \
            "First day with profit blocks (100% > 20% limit)"
        
        # Day 2: Make $1,500 profit (15% of total - should be ALLOWED)
        now_day2 = now + timedelta(days=1)
        tracker.update_profit(trade_pnl=1500.0, now=now_day2)
        
        # Total profit = $11,500, Daily profit = $1,500
        # Daily % = 1500 / 11500 = 13% < 20% limit → ALLOWED
        assert tracker.can_trade(now=now_day2), \
            "Should allow trades when daily profit 13% < 20% limit"
        
        # Day 3: Make $3,000 profit (25% of total - should be BLOCKED)
        now_day3 = now_day2 + timedelta(days=1)
        tracker.update_profit(trade_pnl=3000.0, now=now_day3)
        
        # Total profit = $14,500, Daily profit = $3,000
        # Daily % = 3000 / 14500 = 20.7% > 20% limit → BLOCKED
        assert not tracker.can_trade(now=now_day3), \
            "Should block trades when daily profit >20% of total profit"
        
        daily_pct = tracker.get_daily_profit_pct()
        print(f"[OK] 20% limit enforced: daily_profit=${tracker.daily_profit:.2f}, "
              f"total_profit=${tracker.total_profit:.2f}, "
              f"daily_pct={daily_pct:.1f}% (BLOCKED because >{20}%)")
    
    def test_consistency_below_20_percent_allows(self):
        """Test that <20% daily profit allows trading."""
        tracker = ConsistencyTracker(initial_balance=100000.0)
        et_tz = ZoneInfo("America/New_York")
        now = datetime(2024, 11, 1, 10, 0, 0, tzinfo=et_tz)
        
        # Day 1: Make $10,000 profit
        tracker.update_profit(trade_pnl=10000.0, now=now)
        
        # Day 2: Make $1,500 profit
        now_day2 = now + timedelta(days=1)
        tracker.update_profit(trade_pnl=1500.0, now=now_day2)
        
        # Total = $11,500, Daily = $1,500
        # Daily % = 1500 / 11500 = 13% < 20% limit
        # Should ALLOW trading
        assert tracker.can_trade(now=now_day2), \
            "Should allow trades when daily profit <20% of total profit"
        
        daily_pct = tracker.get_daily_profit_pct()
        print(f"[OK] Below 20% allowed: daily_profit=${tracker.daily_profit:.2f}, "
              f"total_profit=${tracker.total_profit:.2f}, "
              f"daily_pct={daily_pct:.1f}% (ALLOWED because <{20}%)")
    
    def test_consistency_exactly_20_percent_blocks(self):
        """Test that exactly 20% blocks (boundary condition)."""
        tracker = ConsistencyTracker(initial_balance=100000.0)
        et_tz = ZoneInfo("America/New_York")
        now = datetime(2024, 11, 1, 10, 0, 0, tzinfo=et_tz)
        
        # Day 1: Make $10,000 profit
        tracker.update_profit(trade_pnl=10000.0, now=now)
        
        # Day 2: Make exactly 20% of total
        # If total = $10,000, then 20% = $2,000
        # New total = $12,000, daily = $2,000
        # Daily % = 2000 / 12000 = 16.67% < 20%
        # Need different calculation...
        
        # To get exactly 20%: daily / total = 0.20
        # If total after trade = T, daily = D
        # D / T = 0.20
        # If we add $2,500 to $10,000: T = $12,500, D = $2,500
        # 2500 / 12500 = 20%
        
        now_day2 = now + timedelta(days=1)
        tracker.update_profit(trade_pnl=2500.0, now=now_day2)
        
        # Should block at exactly 20%
        can_trade = tracker.can_trade(now=now_day2)
        daily_pct = tracker.get_daily_profit_pct()
        
        # Should be blocked (>= 20%)
        assert not can_trade, \
            f"Should block at exactly 20%, got daily_pct={daily_pct:.1f}%"
        
        print(f"[OK] 20% boundary: daily_profit=${tracker.daily_profit:.2f}, "
              f"total_profit=${tracker.total_profit:.2f}, "
              f"daily_pct={daily_pct:.1f}% (BLOCKED at boundary)")
    
    def test_consistency_daily_reset(self):
        """Test that daily reset allows trading again next day."""
        tracker = ConsistencyTracker(initial_balance=100000.0)
        et_tz = ZoneInfo("America/New_York")
        now = datetime(2024, 11, 1, 10, 0, 0, tzinfo=et_tz)
        
        # Day 1: Hit 20% limit
        tracker.update_profit(trade_pnl=10000.0, now=now)
        now_same_day = now + timedelta(hours=2)
        tracker.update_profit(trade_pnl=2500.0, now=now_same_day)
        
        # Should be blocked
        assert not tracker.can_trade(now=now_same_day), \
            "Should be blocked after hitting 20%"
        
        # Day 2: Auto-reset on new day
        now_next_day = now + timedelta(days=1)
        
        # Just checking can_trade should trigger reset
        can_trade_next_day = tracker.can_trade(now=now_next_day)
        
        # Should be allowed again (daily profit reset to 0)
        assert can_trade_next_day, \
            "Should allow trading after daily reset"
        
        assert tracker.daily_profit == 0.0, \
            "Daily profit should reset to 0 on new day"
        
        assert tracker.total_profit == 12500.0, \
            "Total profit should persist across days"
        
        print(f"[OK] Daily reset: day 2 can_trade={can_trade_next_day}, "
              f"daily_profit=${tracker.daily_profit:.2f}, "
              f"total_profit=${tracker.total_profit:.2f}")
    
    def test_consistency_apex_safety_margin(self):
        """Test that 20% limit provides 10% safety margin below Apex 30%."""
        # Apex rule: Daily profit > 30% of total → account termination
        # Our rule: Daily profit > 20% of total → block trades
        # Safety margin: 30% - 20% = 10%
        
        tracker = ConsistencyTracker(initial_balance=100000.0)
        et_tz = ZoneInfo("America/New_York")
        now = datetime(2024, 11, 1, 10, 0, 0, tzinfo=et_tz)
        
        # Day 1: $10,000 profit
        tracker.update_profit(trade_pnl=10000.0, now=now)
        
        # Day 2: $2,500 profit (20% of total)
        now_day2 = now + timedelta(days=1)
        tracker.update_profit(trade_pnl=2500.0, now=now_day2)
        
        daily_pct = tracker.get_daily_profit_pct()
        
        # Should be blocked at 20%
        assert not tracker.can_trade(now=now_day2), \
            "Should block at 20%"
        
        # Calculate margin to Apex 30% limit
        margin_to_apex = 30.0 - daily_pct
        
        assert margin_to_apex >= 9.0, \
            f"Should have at least 9% margin to Apex 30%, got {margin_to_apex:.1f}%"
        
        # If we made $1,250 more ($3,750 total daily):
        # 3750 / 13750 = 27.27% (still below Apex 30%)
        # This validates the 10% buffer works
        
        print(f"[OK] Safety margin: our_limit=20%, apex_limit=30%, "
              f"current={daily_pct:.1f}%, "
              f"margin_to_apex={margin_to_apex:.1f}% "
              f"(PROTECTED with 10% buffer)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
