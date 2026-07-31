from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "bist100_features_28.csv"
TABLE_DIR = ROOT / "outputs" / "tables"
OUT_DIR = ROOT / "outputs" / "article_figures"

RAW_FEATURES = ["Open", "High", "Low", "Close", "Volume", "USD_TRY", "VIX"]
DERIVED_FEATURES = [
    "Gunluk_Getiri",
    "Volatilite_10",
    "MA_10",
    "MA_50",
    "Momentum_10",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "EMA_20",
    "ATR_14",
    "BB_Upper",
    "BB_Lower",
    "BB_Width",
    "ADX_14",
    "CCI_20",
    "Williams_R",
    "OBV",
    "Stoch_K",
    "Stoch_D",
    "VWAP",
]
ALL_FEATURES = RAW_FEATURES + DERIVED_FEATURES


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def scaled_long(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    clean = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    scaled = pd.DataFrame(MinMaxScaler().fit_transform(clean), columns=columns)
    return scaled.melt(var_name="Ozellik", value_name="Olceklenmis_Deger")


def save_quality_table(df: pd.DataFrame) -> None:
    rows = []
    raw = pd.read_csv(ROOT / "data" / "bist100_ham_veri.csv")
    for column in RAW_FEATURES:
        series = pd.to_numeric(raw[column], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        rows.append(
            {
                "Ozellik": column,
                "Min": series.min(),
                "Max": series.max(),
                "Eksik_Veri_Sayisi": int(series.isna().sum()),
                "Aykiri_Deger_IQR": int(((series < lower) | (series > upper)).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "table_3_1_data_quality_full.csv", index=False, encoding="utf-8-sig")


def make_boxplots(df: pd.DataFrame) -> None:
    groups = {
        "sekil_3_1_box_temel_makro.png": RAW_FEATURES,
        "sekil_3_2_box_trend_hacim_volatilite.png": [
            "Volume",
            "Volatilite_10",
            "MA_10",
            "MA_50",
            "EMA_20",
            "ATR_14",
            "BB_Width",
            "OBV",
            "VWAP",
        ],
        "sekil_3_3_box_momentum_osilator.png": [
            "Gunluk_Getiri",
            "Momentum_10",
            "RSI_14",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "ADX_14",
            "CCI_20",
            "Williams_R",
            "Stoch_K",
            "Stoch_D",
        ],
        "sekil_5_4_tum_ozellikler_boxplot.png": ALL_FEATURES,
    }
    for filename, columns in groups.items():
        data = scaled_long(df, columns)
        width = 14 if len(columns) > 15 else 11
        plt.figure(figsize=(width, 6))
        sns.boxplot(data=data, x="Ozellik", y="Olceklenmis_Deger", color="#8fb9dd", fliersize=1.8)
        plt.xlabel("")
        plt.ylabel("Min-Max Olceklenmis Deger")
        plt.xticks(rotation=45, ha="right")
        save(OUT_DIR / filename)


def make_time_series(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    for code, group in df.groupby("Hisse_Kodu"):
        plt.plot(group["Tarih"], group["Close"], label=code, linewidth=1.0)
    plt.xlabel("Tarih")
    plt.ylabel("Kapanis Fiyati")
    plt.legend(ncol=3, fontsize=8)
    save(OUT_DIR / "sekil_5_1_kapanis_fiyatlari.png")

    plt.figure(figsize=(12, 6))
    for code, group in df.groupby("Hisse_Kodu"):
        plt.plot(group["Tarih"], group["Volume"], label=code, linewidth=0.85, alpha=0.85)
    plt.xlabel("Tarih")
    plt.ylabel("Islem Hacmi")
    plt.legend(ncol=3, fontsize=8)
    save(OUT_DIR / "sekil_5_2_islem_hacimleri.png")

    code = "THYAO.IS" if "THYAO.IS" in set(df["Hisse_Kodu"]) else sorted(df["Hisse_Kodu"].unique())[0]
    one = df[df["Hisse_Kodu"].eq(code)].copy()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(one["Tarih"], one["Close"], label="Close", linewidth=1.1, color="#2E86C1")
    ax1.plot(one["Tarih"], one["MA_10"], label="MA_10", linewidth=1.0, color="#27AE60")
    ax1.plot(one["Tarih"], one["MA_50"], label="MA_50", linewidth=1.0, color="#8E44AD")
    ax1.set_xlabel("Tarih")
    ax1.set_ylabel("Fiyat")
    ax2 = ax1.twinx()
    ax2.plot(one["Tarih"], one["Volatilite_10"], label="Volatilite_10", linewidth=0.9, color="#C0392B", alpha=0.75)
    ax2.set_ylabel("Volatilite")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper left")
    save(OUT_DIR / "sekil_5_7_hareketli_ortalama_volatilite.png")


def make_imputation_plot() -> None:
    path = TABLE_DIR / "table_5_0_imputation_comparison.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["RMSE"] = pd.to_numeric(df["RMSE"], errors="coerce")
    df = df.sort_values("RMSE", ascending=True)
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x="Yontem", y="RMSE", color="#3f7fbf")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8)
    plt.xlabel("Doldurma Yontemi")
    plt.ylabel("RMSE")
    save(OUT_DIR / "sekil_5_3_imputasyon_karsilastirma.png")


def make_correlation_and_distributions(df: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 10))
    corr = df[ALL_FEATURES].corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, cbar_kws={"shrink": 0.75})
    plt.xlabel("")
    plt.ylabel("")
    save(OUT_DIR / "sekil_5_5_korelasyon_matrisi.png")

    fig, axes = plt.subplots(7, 4, figsize=(14, 16))
    axes = axes.ravel()
    for ax, column in zip(axes, ALL_FEATURES):
        values = df[column].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(values, bins=35, color="#4f8fc9", alpha=0.8)
        ax.set_xlabel(column, fontsize=8)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=7)
    for ax in axes[len(ALL_FEATURES) :]:
        ax.axis("off")
    save(OUT_DIR / "sekil_5_6_ozellik_dagilimlari.png")


def make_feature_importance() -> None:
    path = TABLE_DIR / "table_5_1_preliminary_feature_importance.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).head(15)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=df, y="Degisken", x="Onem", color="#3f7fbf")
    plt.xlabel("Onem")
    plt.ylabel("")
    save(OUT_DIR / "sekil_5_8_on_ozellik_onemi.png")


def make_performance_plots() -> None:
    path = TABLE_DIR / "table_5_2_all_model_scenarios.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["Senaryo"].eq("scenario_3_all")].copy()
    for col in ["MAE", "RMSE", "MAPE", "R2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    palette = {"Baseline": "#7F8C8D", "ML": "#2E86C1", "DL": "#27AE60"}

    for metric, ylabel, filename, ascending in [
        ("MAE", "MAE", "sekil_5_9_mae_karsilastirma.png", True),
        ("RMSE", "RMSE", "sekil_5_10_rmse_karsilastirma.png", True),
        ("MAPE", "MAPE (%)", "sekil_5_11_mape_karsilastirma.png", True),
        ("R2", "R² (Belirlilik Katsayisi)", "sekil_5_12_r2_karsilastirma_baseline.png", True),
    ]:
        data = df.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
        colors = data["Model_Tipi"].map(palette).fillna("#7F8C8D")
        plt.figure(figsize=(11, max(6, 0.45 * len(data) + 1.5)))
        bars = plt.barh(data["Model"], data[metric], color=colors, edgecolor="white", linewidth=0.8)
        plt.xlabel(ylabel)
        plt.ylabel("")
        plt.grid(axis="x", alpha=0.22)
        ax = plt.gca()
        ax.set_axisbelow(True)
        if metric == "R2":
            ax.set_xlim(max(0, data[metric].min() - 0.02), min(1.01, data[metric].max() + 0.01))
        for bar, value in zip(bars, data[metric]):
            offset = 0.0015 if metric == "R2" else max(data[metric].max() * 0.006, 0.001)
            ax.text(value + offset, bar.get_y() + bar.get_height() / 2, f"{value:.4f}", va="center", fontsize=8)
        ax.legend(
            handles=[
                Patch(facecolor=palette["Baseline"], label="Naive Baseline"),
                Patch(facecolor=palette["ML"], label="Makine Ogrenmesi"),
                Patch(facecolor=palette["DL"], label="Derin Ogrenme"),
            ],
            loc="lower right",
            fontsize=8,
            frameon=True,
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        save(OUT_DIR / filename)


def main() -> None:
    sns.set_theme(style="whitegrid")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH, parse_dates=["Tarih", "Target_Tarih"])
    save_quality_table(df)
    make_boxplots(df)
    make_time_series(df)
    make_imputation_plot()
    make_correlation_and_distributions(df)
    make_feature_importance()
    make_performance_plots()

    manifest = pd.DataFrame(
        {
            "Dosya": sorted(p.name for p in OUT_DIR.glob("*.png")),
            "Basliksiz": True,
            "Veri_Tarih_Baslangic": str(df["Tarih"].min().date()),
            "Veri_Tarih_Bitis": str(df["Tarih"].max().date()),
        }
    )
    manifest.to_csv(OUT_DIR / "article_figures_manifest.csv", index=False, encoding="utf-8-sig")
    print(OUT_DIR.resolve())
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
