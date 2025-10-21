import re
from pathlib import Path
from typing import Dict, Optional


def _find_number(text: str, patterns) -> Optional[float]:
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(',', '').replace('%', ''))
            except Exception:
                pass
    return None


def parse_mt5_html_report(path: Path) -> Dict[str, float]:
    """
    Parse a standard MT5 Strategy Tester HTML report (best-effort; build differences handled heuristically).
    Returns dict with keys: net_profit, max_drawdown_pct, profit_factor, total_trades, win_rate_pct.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")

    # Net profit
    net_profit = _find_number(text, [r"Total net profit\s*</td>\s*<td[^>]*>\s*([-0-9.,]+)",
                                   r"Net profit\s*</td>\s*<td[^>]*>\s*([-0-9.,]+)"]) or 0.0

    # Max drawdown percent
    max_dd_pct = _find_number(text, [r"Maximal drawdown\s*</td>\s*<td[^>]*>\s*[-0-9.,]+\s*\(([-0-9.,]+)%\)",
                                   r"Maximum drawdown\s*</td>\s*<td[^>]*>\s*[-0-9.,]+\s*\(([-0-9.,]+)%\)"])
    if max_dd_pct is None:
        # fallback: sometimes only percentage appears on a separate column
        max_dd_pct = _find_number(text, r"Drawdown[^%]*\(([-0-9.,]+)%\)") or 0.0

    # Profit factor
    pf = _find_number(text, [r"Profit factor\s*</td>\s*<td[^>]*>\s*([-0-9.,]+)"]) or 0.0

    # Total trades
    total_trades = _find_number(text, [r"Total trades\s*</td>\s*<td[^>]*>\s*([-0-9.,]+)"]) or 0.0

    # Win rate
    win_rate = _find_number(text, [r"Profit trades[^%]*\(([-0-9.,]+)%\)",
                                   r"Win[^%]*\(([-0-9.,]+)%\)"]) or 0.0

    return {
        "net_profit": float(net_profit),
        "max_drawdown_pct": float(max_dd_pct),
        "profit_factor": float(pf),
        "total_trades": float(total_trades),
        "win_rate_pct": float(win_rate),
    }

