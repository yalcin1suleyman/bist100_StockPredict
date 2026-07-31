from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "tables" / "table_5_2_all_model_scenarios.csv"
OUTPUT = ROOT / "outputs" / "figures" / "fig_5_12_r2_horizontal_scenario_3_all.png"


def main() -> None:
    df = pd.read_csv(INPUT)
    df = df[df["Senaryo"].eq("scenario_3_all")].copy()
    df["R2"] = pd.to_numeric(df["R2"], errors="coerce")
    df = df.dropna(subset=["R2"]).sort_values("R2", ascending=True)

    colors = df["Model_Tipi"].map({"ML": "#2E86C1", "DL": "#27AE60"}).fillna("#7F8C8D")
    fig_height = max(6, 0.45 * len(df) + 1.5)
    _, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(df["Model"], df["R2"], color=colors, edgecolor="white", linewidth=0.8)

    xmin = max(0, min(df["R2"].min() - 0.02, 0.94))
    xmax = min(1.01, max(df["R2"].max() + 0.01, 1.0))
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("R² (Belirlilik Katsayısı)", fontweight="bold")
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="-", alpha=0.22)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, df["R2"]):
        ax.text(
            min(value + 0.0015, xmax - 0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
        )

    ax.legend(
        handles=[
            Patch(facecolor="#2E86C1", label="Makine Öğrenmesi"),
            Patch(facecolor="#27AE60", label="Derin Öğrenme"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=8,
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close()

    print(OUTPUT)
    print(df[["Model", "Model_Tipi", "R2"]].sort_values("R2", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
