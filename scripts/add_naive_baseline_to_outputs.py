from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_np = actual.to_numpy(dtype=float)
    predicted_np = predicted.to_numpy(dtype=float)
    return {
        "RMSE": float(mean_squared_error(actual_np, predicted_np, squared=False)),
        "MAE": float(mean_absolute_error(actual_np, predicted_np)),
        "MAPE": float(np.mean(np.abs((actual_np - predicted_np) / actual_np)) * 100),
        "R2": float(r2_score(actual_np, predicted_np)),
    }


def make_baseline_table() -> pd.DataFrame:
    rows = []
    for path in sorted(PROCESSED_DIR.glob("scenario_*.csv")):
        scenario = path.stem
        df = pd.read_csv(path)
        test = df[df["Set"].eq("Test")].dropna(subset=["Close_Next"]).copy()
        row = {
            "Senaryo": scenario,
            "Model": "Naive Baseline",
            "Model_Tipi": "Baseline",
            "Egitim_Suresi_Sn": 0.0,
            "Ozellik_Sayisi": 1,
            "Test_Gozlem": len(test),
        }
        row.update(metrics(test["Close_Next"], test["Close"]))
        rows.append(row)
    baseline = pd.DataFrame(rows)
    baseline.to_csv(TABLE_DIR / "table_5_2_naive_baseline.csv", index=False, encoding="utf-8-sig")
    return baseline


def update_combined_table(baseline: pd.DataFrame) -> pd.DataFrame:
    combined_path = TABLE_DIR / "table_5_2_all_model_scenarios.csv"
    combined = pd.read_csv(combined_path)
    combined = combined[combined["Model"].ne("Naive Baseline")].copy()
    combined = pd.concat([baseline, combined], ignore_index=True, sort=False)
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    return combined


def plot_r2(combined: pd.DataFrame) -> Path:
    df = combined[combined["Senaryo"].eq("scenario_3_all")].copy()
    df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    df = df.dropna(subset=["R2"]).sort_values("R2", ascending=True)

    palette = {"Baseline": "#7F8C8D", "ML": "#2E86C1", "DL": "#27AE60"}
    colors = df["Model_Tipi"].map(palette).fillna("#7F8C8D")
    output = FIG_DIR / "fig_5_12_r2_horizontal_scenario_3_all_with_baseline.png"

    _, ax = plt.subplots(figsize=(11, max(6, 0.45 * len(df) + 1.5)))
    bars = ax.barh(df["Model"], df["R2"], color=colors, edgecolor="white", linewidth=0.8)
    xmin = max(0, min(df["R2"].min() - 0.02, 0.74))
    xmax = min(1.01, max(df["R2"].max() + 0.01, 1.0))
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("R² (Belirlilik Katsayısı)", fontweight="bold")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, df["R2"]):
        ax.text(min(value + 0.0015, xmax - 0.001), bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=8, fontweight="bold")
    ax.legend(
        handles=[
            Patch(facecolor=palette["Baseline"], label="Naive Baseline"),
            Patch(facecolor=palette["ML"], label="Makine Öğrenmesi"),
            Patch(facecolor=palette["DL"], label="Derin Öğrenme"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=8,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    return output


def main() -> None:
    baseline = make_baseline_table()
    combined = update_combined_table(baseline)
    output = plot_r2(combined)
    print(output)
    print(combined[combined["Senaryo"].eq("scenario_3_all")].sort_values("RMSE")[["Model", "Model_Tipi", "RMSE", "MAE", "MAPE", "R2"]].to_string(index=False))


if __name__ == "__main__":
    main()
