import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from mt5_tester_config import write_tester_ini
from mt5_report_parser import parse_mt5_html_report


def compile_ea(metaeditor: Path, ea_mq5: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "compile.log"
    cmd = [
        str(metaeditor),
        f"/compile:{ea_mq5}",
        f"/log:{log_path}",
        
    ]
    print(f"[compile] {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"MetaEditor compile failed: code {res.returncode}")

    # Heuristic: .ex5 next to source (MetaEditor puts in MQL5/Experts by default if mapped)
    ex5_guess = ea_mq5.with_suffix('.ex5')
    if ex5_guess.exists():
        return ex5_guess
    # Fallback: search under terminal data folder (not available from here). Require user to set EA path mapped under MQL5/Experts.
    return ex5_guess


def run_dataset(terminal: Path, tester_dir: Path, expert_ex5: str, expert_set: Path,
                symbol: str, period: str, model: int, deposit: int, leverage: int,
                from_date: str, to_date: str, out_report: Path) -> Path:
    tester_dir.mkdir(parents=True, exist_ok=True)
    ini_path = tester_dir / "tester.ini"
    write_tester_ini(
        ini_path,
        expert_ex5=expert_ex5,
        expert_set=str(expert_set.resolve()),
        symbol=symbol,
        period=period,
        model=model,
        deposit=deposit,
        leverage=leverage,
        from_date=from_date,
        to_date=to_date,
        report_path=str(out_report.with_suffix('')),
        use_local=1,
        visual=0,
    )

    cmd = [
        str(terminal),
        f"/portable",
        f"/config:{ini_path}",
    ]
    print(f"[tester] {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"Terminal run failed: code {res.returncode}")
    # MT5 writes report automatically; allow a brief delay for IO
    time.sleep(1.5)
    return out_report


def evaluate_thresholds(metrics: dict, thresholds: dict, deposit: float, days: int) -> dict:
    # Approximate monthly return from net profit for the tested period. If the period is ~30 days use as-is; else scale.
    net_profit = metrics.get('net_profit', 0.0)
    monthly_return_pct = (net_profit / deposit) * 100.0
    # Drawdown is already percent in metrics
    max_dd = metrics.get('max_drawdown_pct', 0.0)
    pf = metrics.get('profit_factor', 0.0)
    wr = metrics.get('win_rate_pct', 0.0)

    ok = True
    reasons = []
    if monthly_return_pct < thresholds['monthly_return_min_pct']:
        ok = False
        reasons.append(f"monthly_return {monthly_return_pct:.2f}% < {thresholds['monthly_return_min_pct']:.2f}%")
    if max_dd > thresholds['max_drawdown_max_pct']:
        ok = False
        reasons.append(f"max_drawdown {max_dd:.2f}% > {thresholds['max_drawdown_max_pct']:.2f}%")
    if pf < thresholds['profit_factor_min']:
        ok = False
        reasons.append(f"profit_factor {pf:.2f} < {thresholds['profit_factor_min']:.2f}")
    if wr < thresholds['win_rate_min_pct']:
        ok = False
        reasons.append(f"win_rate {wr:.2f}% < {thresholds['win_rate_min_pct']:.2f}%")

    return {"pass": ok, "reasons": reasons, "monthly_return_pct": monthly_return_pct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metaeditor', required=True, help='Path to metaeditor64.exe')
    ap.add_argument('--terminal', required=True, help='Path to terminal64.exe')
    ap.add_argument('--spec', required=True, help='Spec JSON path')
    ap.add_argument('--out', default='MCP/out', help='Output directory')
    args = ap.parse_args()

    metaeditor = Path(args.metaeditor)
    terminal = Path(args.terminal)
    spec_path = Path(args.spec)
    out_root = Path(args.out)

    spec = json.loads(spec_path.read_text(encoding='utf-8'))

    run_id = f"{spec.get('name','run')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ea_mq5 = Path(spec['ea_path'])
    # Compile EA
    try:
        ex5_path = compile_ea(metaeditor, ea_mq5, run_dir)
    except Exception as e:
        print(f"[error] compile failed: {e}")
        sys.exit(2)

    # Determine expert path mapping for terminal (recommend placing ex5 under MQL5\\Experts)
    expert_ex5 = str(ex5_path)

    results = []
    thresholds = spec['thresholds']

    for idx, ds in enumerate(spec['datasets']):
        ds_dir = run_dir / f"ds_{idx+1}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        report = ds_dir / f"report_{idx+1}.html"

        try:
            run_dataset(
                terminal=terminal,
                tester_dir=ds_dir,
                expert_ex5=expert_ex5,
                expert_set=Path(spec['ea_set']),
                symbol=spec['symbol'],
                period=spec['period'],
                model=spec.get('model', 0),
                deposit=spec['deposit'],
                leverage=spec.get('leverage', 100),
                from_date=ds['from'],
                to_date=ds['to'],
                out_report=report,
            )
        except Exception as e:
            print(f"[error] tester failed on dataset {idx+1}: {e}")
            continue

        metrics = parse_mt5_html_report(report)
        # Rough day count for the period (approx). More exact calc is possible but not needed here.
        eval_res = evaluate_thresholds(metrics, thresholds, spec['deposit'], 30)
        results.append({
            "dataset": ds,
            "report": str(report),
            "metrics": metrics,
            "evaluation": eval_res
        })

    # Consolidate pass if all datasets pass
    overall_pass = all(r['evaluation']['pass'] for r in results) if results else False
    summary = {
        "spec": spec,
        "results": results,
        "overall_pass": overall_pass
    }
    (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"[done] summary: {run_dir / 'summary.json'} | PASS={overall_pass}")


if __name__ == '__main__':
    main()

