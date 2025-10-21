import os
from pathlib import Path


def write_tester_ini(path: Path, *, expert_ex5: str, expert_set: str, symbol: str, period: str,
                     model: int, deposit: int, leverage: int, from_date: str, to_date: str,
                     report_path: str, use_local: int = 1, visual: int = 0) -> None:
    """
    Create a tester.ini file for MT5 terminal CLI.
    - expert_ex5: path (or MQL5\Experts path mapping) to EA .ex5
    - expert_set: .set file with inputs
    - report_path: absolute path for HTML report (Terminal will add .html)
    """
    ini = [
        "[Common]",
        "Symbols=",
        "Login=",
        "Password=",
        "Server=",
        "",
        "[Tester]",
        f"Expert={expert_ex5}",
        f"ExpertParameters={expert_set}",
        f"Symbol={symbol}",
        f"Period={period}",
        f"Model={model}",
        f"FromDate={from_date}",
        f"ToDate={to_date}",
        f"Deposit={deposit}",
        f"Leverage={leverage}",
        f"UseLocal={use_local}",
        f"ForwardMode=0",
        f"Optimization=0",
        f"Report={report_path}",
        f"ReplaceReport=1",
        f"ShutdownTerminal=0",
        f"Visual={visual}",
        "",
    ]
    path.write_text("\n".join(ini), encoding="utf-8")

