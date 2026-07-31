from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs" / "predictions"
OUT_DIR = ROOT / "outputs" / "article_figures"

SCENARIO = "scenario_3_all"
STOCK_CODE = "TUPRS.IS"
VOLATILITY_START = pd.Timestamp("2023-09-01")
VOLATILITY_END = pd.Timestamp("2023-10-31")

ML_MODELS = ["Linear Regression", "SVR", "Random Forest", "XGBoost", "LightGBM", "CatBoost"]
DL_MODELS = ["MLP", "CNN", "LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Transformer"]
HIGH_VOL_MODELS = [
    "Linear Regression",
    "SVR",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "LSTM",
    "CNN-LSTM",
    "Transformer",
]

MODEL_COLORS = {
    "Linear Regression": "#2E86C1",
    "SVR": "#E67E22",
    "Random Forest": "#27AE60",
    "XGBoost": "#C0392B",
    "LightGBM": "#8E44AD",
    "CatBoost": "#16A085",
    "MLP": "#E74C3C",
    "CNN": "#3498DB",
    "LSTM": "#27AE60",
    "GRU": "#9B59B6",
    "BiLSTM": "#F39C12",
    "CNN-LSTM": "#1ABC9C",
    "Transformer": "#D35400",
}


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Tarih", "Target_Tarih"])
    df = df[
        df["Senaryo"].eq(SCENARIO)
        & df["Hisse_Kodu"].eq(STOCK_CODE)
        & df["Set"].eq("Test")
    ].copy()
    numeric_columns = ["Close_Next", "Predicted_Close"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["Target_Tarih", "Close_Next", "Predicted_Close"])
    return df.sort_values(["Target_Tarih", "Model"])


def series_for_models(
    df: pd.DataFrame,
    models: list[str],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    data = df[df["Model"].isin(models)].copy()
    if start is not None:
        data = data[data["Target_Tarih"].ge(start)]
    if end is not None:
        data = data[data["Target_Tarih"].le(end)]

    missing = sorted(set(models) - set(data["Model"].unique()))
    if missing:
        raise ValueError(f"Eksik model tahmini: {missing}")

    actual = (
        data.drop_duplicates("Target_Tarih")
        .set_index("Target_Tarih")["Close_Next"]
        .sort_index()
    )
    predicted = data.pivot_table(
        index="Target_Tarih",
        columns="Model",
        values="Predicted_Close",
        aggfunc="mean",
    ).sort_index()
    return actual, predicted.reindex(actual.index)


def format_axis(ax: plt.Axes, daily_ticks: bool = False) -> None:
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Kapanis Fiyati (TL)")
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    locator = mdates.DayLocator(interval=7) if daily_ticks else mdates.MonthLocator(interval=2)
    formatter = mdates.DateFormatter("%d.%m.%Y" if daily_ticks else "%m.%Y")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_model_group(ax: plt.Axes, actual: pd.Series, predicted: pd.DataFrame, models: list[str]) -> None:
    ax.plot(actual.index, actual.values, label="Gercek Kapanis Fiyati", color="#1F2937", linewidth=1.35)
    for model in models:
        if model not in predicted:
            continue
        ax.plot(
            predicted.index,
            predicted[model],
            label=f"{model} Tahmin",
            color=MODEL_COLORS.get(model, "#4F8FC9"),
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
        )
    ax.set_xlim(actual.index.min(), actual.index.max())
    ax.margins(x=0)
    ax.legend(fontsize=7.5, ncol=2, frameon=True, loc="upper left")


def make_combined_chart(ml_df: pd.DataFrame, dl_df: pd.DataFrame) -> None:
    ml_actual, ml_predicted = series_for_models(ml_df, ML_MODELS)
    dl_actual, dl_predicted = series_for_models(dl_df, DL_MODELS)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=False)
    plot_model_group(axes[0], ml_actual, ml_predicted, ML_MODELS)
    plot_model_group(axes[1], dl_actual, dl_predicted, DL_MODELS)
    format_axis(axes[0])
    format_axis(axes[1])
    fig.autofmt_xdate(rotation=0)
    save(OUT_DIR / "sekil_5_13_tum_modeller_gercek_tahmin.png")


def make_single_model_chart(df: pd.DataFrame, model: str, filename: str) -> None:
    actual, predicted = series_for_models(df, [model])
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(actual.index, actual.values, label="Gercek Kapanis Fiyati", color="#1F2937", linewidth=1.35)
    ax.plot(
        predicted.index,
        predicted[model],
        label=f"{model} Tahmin Fiyati",
        color="#C0392B",
        linewidth=1.15,
        linestyle="--",
    )
    ax.set_xlim(actual.index.min(), actual.index.max())
    ax.margins(x=0)
    format_axis(ax)
    ax.legend(fontsize=8, frameon=True, loc="upper left")
    fig.autofmt_xdate(rotation=0)
    save(OUT_DIR / filename)


def make_high_volatility_chart(ml_df: pd.DataFrame, dl_df: pd.DataFrame) -> None:
    combined = pd.concat([ml_df, dl_df], ignore_index=True)
    actual, predicted = series_for_models(combined, HIGH_VOL_MODELS, VOLATILITY_START, VOLATILITY_END)

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(actual.index, actual.values, label="Gercek Kapanis Fiyati", color="#1F2937", linewidth=1.35)
    for model in HIGH_VOL_MODELS:
        ax.plot(
            predicted.index,
            predicted[model],
            label=f"{model} Tahmin",
            color=MODEL_COLORS.get(model, "#4F8FC9"),
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
        )
    ax.set_xlim(actual.index.min(), actual.index.max())
    ax.margins(x=0)
    format_axis(ax, daily_ticks=True)
    ax.legend(fontsize=7.2, ncol=2, frameon=True, loc="best")
    fig.autofmt_xdate(rotation=0)
    save(OUT_DIR / "sekil_5_17_yuksek_volatilite_gercek_tahmin.png")


def write_manifest() -> None:
    rows = [
        {
            "Dosya": "sekil_5_13_tum_modeller_gercek_tahmin.png",
            "Bolum": "5.3",
            "Hisse_Kodu": STOCK_CODE,
            "Senaryo": SCENARIO,
            "Tarih_Baslangic": "2023-06-22",
            "Tarih_Bitis": "2024-12-31",
            "Basliksiz": True,
        },
        {
            "Dosya": "sekil_5_14_xgboost_gercek_tahmin.png",
            "Bolum": "5.3",
            "Hisse_Kodu": STOCK_CODE,
            "Senaryo": SCENARIO,
            "Tarih_Baslangic": "2023-06-22",
            "Tarih_Bitis": "2024-12-31",
            "Basliksiz": True,
        },
        {
            "Dosya": "sekil_5_15_lstm_gercek_tahmin.png",
            "Bolum": "5.3",
            "Hisse_Kodu": STOCK_CODE,
            "Senaryo": SCENARIO,
            "Tarih_Baslangic": "2023-07-10",
            "Tarih_Bitis": "2024-12-31",
            "Basliksiz": True,
        },
        {
            "Dosya": "sekil_5_16_transformer_gercek_tahmin.png",
            "Bolum": "5.3",
            "Hisse_Kodu": STOCK_CODE,
            "Senaryo": SCENARIO,
            "Tarih_Baslangic": "2023-07-10",
            "Tarih_Bitis": "2024-12-31",
            "Basliksiz": True,
        },
        {
            "Dosya": "sekil_5_17_yuksek_volatilite_gercek_tahmin.png",
            "Bolum": "5.3",
            "Hisse_Kodu": STOCK_CODE,
            "Senaryo": SCENARIO,
            "Tarih_Baslangic": str(VOLATILITY_START.date()),
            "Tarih_Bitis": str(VOLATILITY_END.date()),
            "Basliksiz": True,
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "section_5_3_figures_manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    sns.set_theme(style="whitegrid")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ml_df = load_predictions(PRED_DIR / "ml_predictions.csv")
    dl_df = load_predictions(PRED_DIR / "dl_predictions.csv")

    make_combined_chart(ml_df, dl_df)
    make_single_model_chart(ml_df, "XGBoost", "sekil_5_14_xgboost_gercek_tahmin.png")
    make_single_model_chart(dl_df, "LSTM", "sekil_5_15_lstm_gercek_tahmin.png")
    make_single_model_chart(dl_df, "Transformer", "sekil_5_16_transformer_gercek_tahmin.png")
    make_high_volatility_chart(ml_df, dl_df)
    write_manifest()

    print(OUT_DIR.resolve())
    print("5.3 basliksiz grafikler guncel test tahminlerinden uretildi.")


if __name__ == "__main__":
    main()
