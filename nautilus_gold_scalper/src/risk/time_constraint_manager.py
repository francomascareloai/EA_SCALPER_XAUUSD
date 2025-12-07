"""
TimeConstraintManager
Enforces Apex daily cutoff (4:59 PM ET) with staged warnings.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Set

try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    from datetime import timezone
    ET_TZ = timezone.utc


class TimeConstraintManager:
    """Enforces Apex 4:59 PM ET cutoff with warnings and forced flatten."""

    def __init__(
        self,
        strategy,
        cutoff: time = time(16, 59),
        warning: time = time(16, 0),
        urgent: time = time(16, 30),
        emergency: time = time(16, 55),
        allow_overnight: bool = False,
    ) -> None:
        self.strategy = strategy
        self.cutoff = cutoff
        self.warnings = {
            "warning": warning,
            "urgent": urgent,
            "emergency": emergency,
        }
        self.allow_overnight = allow_overnight
        self._issued: Set[str] = set()

    def check(self, ts_ns: int) -> bool:
        """
        Check time constraints for a given timestamp in nanoseconds.
        Returns True if trading is allowed, False if trading must stop.
        """
        if self.allow_overnight:
            return True

        dt_et = datetime.fromtimestamp(ts_ns / 1e9, tz=ET_TZ)
        now_time = dt_et.time()

        # Warnings
        for level, when in self.warnings.items():
            if now_time >= when and level not in self._issued:
                self._log_warning(level, dt_et)
                self._issued.add(level)

        # Hard cutoff: force close + block
        if now_time >= self.cutoff:
            self._force_close_all(dt_et)
            return False

        return True

    def reset_daily(self) -> None:
        """Reset warning flags for a new trading day."""
        self._issued.clear()

    # -------- internal helpers --------
    def _force_close_all(self, dt_et: datetime) -> None:
        """Flatten all positions and stop further trading."""
        try:
            self.strategy.close_all_positions(self.strategy.config.instrument_id)
        except Exception:
            # Fail-safe: try generic cache walk
            for pos in self.strategy.cache.positions_open():
                try:
                    self.strategy.close_position(pos)
                except Exception:
                    pass

        self.strategy._is_trading_allowed = False  # internal guard
        setattr(self.strategy, "_trading_blocked_today", True)
        self.strategy.log.critical(
            f"APEX TIME CUTOFF {dt_et.strftime('%Y-%m-%d %H:%M:%S %Z')} - all positions closed"
        )

    def _log_warning(self, level: str, dt_et: datetime) -> None:
        msg = f"APEX cutoff warning ({level}) at {dt_et.strftime('%H:%M:%S %Z')} -> 16:59 cutoff"
        if level == "emergency":
            self.strategy.log.critical(msg)
        elif level == "urgent":
            self.strategy.log.error(msg)
        else:
            self.strategy.log.warning(msg)
