from pathlib import Path

import pandas as pd

from optimizer.data_loader import generate_synthetic
from optimizer.schema import ParamSpace


def main():
    out_dir = Path(__file__).parent
    df = generate_synthetic(ParamSpace(), n=500, seed=42)
    out_path = out_dir / "backtest_results_sample.csv"
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved to: {out_path}")


if __name__ == "__main__":
    main()

