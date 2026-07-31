from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bist100_pipeline import (
    PROCESSED_DIR,
    SCENARIOS,
    TABLE_DIR,
    naive_baseline_results,
    plot_performance,
    train_ml_models,
)


def load_scenarios() -> dict[str, pd.DataFrame]:
    scenarios = {}
    for name in SCENARIOS:
        scenarios[name] = pd.read_csv(PROCESSED_DIR / f"{name}.csv", parse_dates=["Tarih", "Target_Tarih"])
    return scenarios


def main() -> None:
    scenarios = load_scenarios()
    ml_result, _, _ = train_ml_models(scenarios, quick=False)

    dl_result_path = TABLE_DIR / "table_5_2_dl_scenarios.csv"
    hist_path = TABLE_DIR / "table_5_2_dl_learning_curves.csv"
    dl_result = pd.read_csv(dl_result_path) if dl_result_path.exists() else pd.DataFrame()
    hist_df = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()

    baseline = naive_baseline_results(scenarios)
    plot_performance(ml_result, dl_result, hist_df, baseline)

    print("Refreshed ML results and 5.2 performance figures.")
    print((TABLE_DIR / "table_5_2_ml_scenarios.csv").resolve())
    print((TABLE_DIR / "table_5_2_all_model_scenarios.csv").resolve())


if __name__ == "__main__":
    main()
