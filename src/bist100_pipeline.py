from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONHASHSEED", "42")


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
PRED_DIR = OUTPUT_DIR / "predictions"
XAI_DIR = OUTPUT_DIR / "xai"
REPORT_DIR = OUTPUT_DIR / "report"

RAW_FILE = DATA_DIR / "bist100_ham_veri.csv"
STUDY_START = pd.Timestamp("2015-01-01")
STUDY_END_EXCLUSIVE = pd.Timestamp("2025-01-01")
EXPECTED_LAST_DATE = pd.Timestamp("2024-12-31")
EXPECTED_STOCKS = ["AKBNK.IS", "BIMAS.IS", "EREGL.IS", "GARAN.IS", "THYAO.IS", "TUPRS.IS"]

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

SCENARIOS = {
    "scenario_1_basic": RAW_FEATURES,
    "scenario_2_derived": DERIVED_FEATURES,
    "scenario_3_all": ALL_FEATURES,
}

META_COLUMNS = ["Tarih", "Target_Tarih", "Hisse_Kodu", "Set", "Close", "Close_Next", "Target_Return"]


@dataclass
class ModelRun:
    scenario: str
    model: str
    rmse: float
    mae: float
    mape: float
    r2: float
    train_seconds: float
    n_features: int
    n_test: int
    available: bool = True
    note: str = ""


def ensure_dirs() -> None:
    for directory in [PROCESSED_DIR, FIG_DIR, TABLE_DIR, PRED_DIR, XAI_DIR, REPORT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            ordered.append(column)
    return ordered


def read_raw_data() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw data not found: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE)
    required = ["Tarih", "Hisse_Kodu"] + RAW_FEATURES
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw file: {missing}")
    df = df[required].copy()
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    for column in RAW_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values(["Hisse_Kodu", "Tarih"]).reset_index(drop=True)
    df = df[(df["Tarih"] >= STUDY_START) & (df["Tarih"] < STUDY_END_EXCLUSIVE)].copy()
    validate_expected_stocks(df)
    validate_study_period(df)
    df = df[df["Hisse_Kodu"].isin(EXPECTED_STOCKS)].copy()
    return df


def validate_expected_stocks(df: pd.DataFrame) -> None:
    observed = set(df["Hisse_Kodu"].dropna().unique())
    missing = [stock for stock in EXPECTED_STOCKS if stock not in observed]
    unexpected = sorted(observed.difference(EXPECTED_STOCKS))
    if missing:
        raise ValueError(f"Ham veride beklenen hisseler eksik: {missing}")
    if unexpected:
        warnings.warn(f"Ham veride kapsam disi hisseler var ve filtrelenecek: {unexpected}", RuntimeWarning)


def validate_study_period(df: pd.DataFrame) -> None:
    observed_start = df["Tarih"].min()
    observed_end = df["Tarih"].max()
    if pd.isna(observed_start) or pd.isna(observed_end):
        raise ValueError("Ham veri tarih alani bos; 2015-01-01 / 2025-01-01 araligi dogrulanamadi.")
    if observed_start > STUDY_START:
        raise ValueError(
            f"Ham veri gec basliyor: {observed_start.date()}. Beklenen baslangic: {STUDY_START.date()}."
        )
    if observed_end < EXPECTED_LAST_DATE:
        raise ValueError(
            f"Ham veri erken bitiyor: {observed_end.date()}. "
            f"Makaledeki 2015-2025 donemi icin beklenen son gozlem: {EXPECTED_LAST_DATE.date()}."
        )


def missing_table(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in RAW_FEATURES:
        rows.append(
            {
                "Degisken": column,
                "Eksik_Gozlem": int(raw[column].isna().sum()),
                "Eksik_Oran": float(raw[column].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def impute_raw(df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
    out = df.copy()
    if method == "knn":
        imputer = KNNImputer(n_neighbors=5)
        out[RAW_FEATURES] = imputer.fit_transform(out[RAW_FEATURES])
        return out

    def _fill(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        if method == "ffill":
            group[RAW_FEATURES] = group[RAW_FEATURES].ffill().bfill()
        elif method == "interpolate":
            group[RAW_FEATURES] = group[RAW_FEATURES].interpolate(method="linear", limit_direction="both")
            group[RAW_FEATURES] = group[RAW_FEATURES].ffill().bfill()
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        return group

    return out.groupby("Hisse_Kodu", group_keys=False).apply(_fill).reset_index(drop=True)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    plus_di = 100 * plus_dm.rolling(window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window).mean() / atr.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(window).mean()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    def _one(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values("Tarih").copy()
        close = g["Close"]
        high = g["High"]
        low = g["Low"]
        volume = g["Volume"]
        typical = (high + low + close) / 3

        g["Gunluk_Getiri"] = np.log(close / close.shift(1))
        g["Volatilite_10"] = g["Gunluk_Getiri"].rolling(10).std()
        g["MA_10"] = close.rolling(10).mean()
        g["MA_50"] = close.rolling(50).mean()
        g["Momentum_10"] = close - close.shift(10)
        g["RSI_14"] = _rsi(close)

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        g["MACD"] = ema_12 - ema_26
        g["MACD_Signal"] = g["MACD"].ewm(span=9, adjust=False).mean()
        g["MACD_Histogram"] = g["MACD"] - g["MACD_Signal"]
        g["EMA_20"] = close.ewm(span=20, adjust=False).mean()

        tr = pd.concat(
            [
                high - low,
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        g["ATR_14"] = tr.rolling(14).mean()

        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        g["BB_Upper"] = ma_20 + (2 * std_20)
        g["BB_Lower"] = ma_20 - (2 * std_20)
        g["BB_Width"] = (g["BB_Upper"] - g["BB_Lower"]) / ma_20.replace(0, np.nan)
        g["ADX_14"] = _adx(high, low, close)
        g["CCI_20"] = (typical - typical.rolling(20).mean()) / (0.015 * typical.rolling(20).std()).replace(0, np.nan)

        high_14 = high.rolling(14).max()
        low_14 = low.rolling(14).min()
        g["Williams_R"] = -100 * (high_14 - close) / (high_14 - low_14).replace(0, np.nan)
        direction = np.sign(close.diff()).fillna(0)
        g["OBV"] = (direction * volume).cumsum()
        g["Stoch_K"] = 100 * (close - low_14) / (high_14 - low_14).replace(0, np.nan)
        g["Stoch_D"] = g["Stoch_K"].rolling(3).mean()
        g["VWAP"] = (typical * volume).cumsum() / volume.replace(0, np.nan).cumsum()
        return g

    features = df.groupby("Hisse_Kodu", group_keys=False).apply(_one)
    features = features.sort_values(["Hisse_Kodu", "Tarih"]).reset_index(drop=True)
    features[ALL_FEATURES] = features[ALL_FEATURES].replace([np.inf, -np.inf], np.nan)

    def _fill_indicators(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group[ALL_FEATURES] = group[ALL_FEATURES].interpolate(method="linear", limit_direction="both")
        group[ALL_FEATURES] = group[ALL_FEATURES].ffill().bfill()
        return group

    features = features.groupby("Hisse_Kodu", group_keys=False).apply(_fill_indicators)
    features[ALL_FEATURES] = features[ALL_FEATURES].fillna(0)

    features["Close_Next"] = features.groupby("Hisse_Kodu")["Close"].shift(-1)
    features["Target_Tarih"] = features.groupby("Hisse_Kodu")["Tarih"].shift(-1)
    features["Target_Return"] = (features["Close_Next"] - features["Close"]) / features["Close"].replace(0, np.nan)
    return assign_splits(features.reset_index(drop=True))


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Set"] = ""
    for _, idx in out.groupby("Hisse_Kodu").groups.items():
        indices = list(idx)
        n = len(indices)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        out.loc[indices[:train_end], "Set"] = "Train"
        out.loc[indices[train_end:val_end], "Set"] = "Validation"
        out.loc[indices[val_end:], "Set"] = "Test"
    return out


def create_scenario_files(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    features[unique_columns(META_COLUMNS + ALL_FEATURES)].to_csv(
        PROCESSED_DIR / "bist100_features_28.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scenarios = {}
    for name, columns in SCENARIOS.items():
        scenario_df = features[unique_columns(META_COLUMNS + columns)].copy()
        scenario_df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
        scenarios[name] = scenario_df
    return scenarios


def descriptive_statistics(features: pd.DataFrame) -> pd.DataFrame:
    desc = features[ALL_FEATURES].describe().T.reset_index().rename(columns={"index": "Degisken"})
    return desc


def scenario_matrix(scenario_df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    x = scenario_df[feature_names].copy()
    y = scenario_df["Target_Return"].copy()
    return x, y


def price_metrics(actual_price: np.ndarray, predicted_price: np.ndarray) -> dict[str, float]:
    actual_price = np.asarray(actual_price, dtype=float)
    predicted_price = np.asarray(predicted_price, dtype=float)
    rmse = math.sqrt(mean_squared_error(actual_price, predicted_price))
    mae = mean_absolute_error(actual_price, predicted_price)
    mape = np.mean(np.abs((actual_price - predicted_price) / np.where(actual_price == 0, np.nan, actual_price))) * 100
    r2 = r2_score(actual_price, predicted_price)
    return {"rmse": rmse, "mae": mae, "mape": float(mape), "r2": r2}


def optional_ml_models(quick: bool = False) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(C=10.0, epsilon=0.001, gamma="scale"),
        "Random Forest": RandomForestRegressor(
            n_estimators=80 if quick else 220,
            max_depth=10 if quick else 16,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=80 if quick else 220,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        models["LightGBM"] = LGBMRegressor(
            n_estimators=100 if quick else 260,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=90 if quick else 230,
            depth=6,
            learning_rate=0.04,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
    except Exception:
        pass
    return models


def train_ml_models(scenarios: dict[str, pd.DataFrame], quick: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    fitted: dict[str, Any] = {}
    models = optional_ml_models(quick=quick)

    for scenario_name, scenario_df in scenarios.items():
        print(f"ML training: {scenario_name}", flush=True)
        scenario_df = scenario_df.loc[:, ~scenario_df.columns.duplicated()].copy()
        features = SCENARIOS[scenario_name]
        train_val = scenario_df[scenario_df["Set"].eq("Train")].dropna(
            subset=["Close_Next", "Target_Return"]
        ).copy()
        test = scenario_df[scenario_df["Set"].eq("Test")].dropna(subset=["Close_Next", "Target_Return"]).copy()
        x_train, y_train = scenario_matrix(train_val, features)
        x_test, _ = scenario_matrix(test, features)

        for model_name, model in models.items():
            print(f"  - {model_name}", flush=True)
            start = time.perf_counter()
            pipe = Pipeline([("scaler", MinMaxScaler()), ("model", clone(model))])
            pipe.fit(x_train, y_train)
            elapsed = time.perf_counter() - start
            predicted_return = pipe.predict(x_test)
            predicted_price = test["Close"].to_numpy() * (1 + predicted_return)
            actual_price = test["Close_Next"].to_numpy()
            metrics = price_metrics(actual_price, predicted_price)
            rows.append(
                {
                    "Senaryo": scenario_name,
                    "Model": model_name,
                    "RMSE": metrics["rmse"],
                    "MAE": metrics["mae"],
                    "MAPE": metrics["mape"],
                    "R2": metrics["r2"],
                    "Egitim_Suresi_Sn": elapsed,
                    "Ozellik_Sayisi": len(features),
                    "Test_Gozlem": len(test),
                }
            )
            pred = test[["Tarih", "Target_Tarih", "Hisse_Kodu", "Set", "Close", "Close_Next", "Target_Return"]].copy()
            pred["Senaryo"] = scenario_name
            pred["Model"] = model_name
            pred["Predicted_Return"] = predicted_return
            pred["Predicted_Close"] = predicted_price
            predictions.append(pred)
            fitted[f"ML::{scenario_name}::{model_name}"] = pipe

    result = pd.DataFrame(rows).sort_values(["Senaryo", "RMSE"]).reset_index(drop=True)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    save_table(result, TABLE_DIR / "table_5_2_ml_scenarios.csv")
    save_table(pred_df, PRED_DIR / "ml_predictions.csv")
    return result, pred_df, fitted


def import_torch() -> tuple[Any, Any, Any]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset


def make_sequences(
    df: pd.DataFrame,
    feature_names: list[str],
    scaler: MinMaxScaler,
    window: int = 10,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    meta: list[pd.Series] = []
    for _, group in df.sort_values(["Hisse_Kodu", "Tarih"]).groupby("Hisse_Kodu"):
        values = scaler.transform(group[feature_names])
        targets = group["Target_Return"].to_numpy()
        for i in range(window - 1, len(group)):
            xs.append(values[i - window + 1 : i + 1])
            ys.append(float(targets[i]))
            meta.append(group.iloc[i])
    if not xs:
        return np.empty((0, window, len(feature_names))), np.empty((0,)), pd.DataFrame()
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(meta).reset_index(drop=True)


def build_torch_model(model_name: str, n_features: int, window: int, nn: Any) -> Any:
    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(window * n_features, 96),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(96, 48),
                nn.ReLU(),
                nn.Linear(48, 1),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x).squeeze(-1)

    class CNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_features, 48, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(48, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.out = nn.Linear(32, 1)

        def forward(self, x: Any) -> Any:
            z = self.conv(x.transpose(1, 2)).squeeze(-1)
            return self.out(z).squeeze(-1)

    class RNNModel(nn.Module):
        def __init__(self, kind: str = "lstm", bidirectional: bool = False) -> None:
            super().__init__()
            rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = rnn_cls(n_features, 48, batch_first=True, bidirectional=bidirectional)
            factor = 2 if bidirectional else 1
            self.out = nn.Sequential(nn.Dropout(0.15), nn.Linear(48 * factor, 1))

        def forward(self, x: Any) -> Any:
            z, _ = self.rnn(x)
            return self.out(z[:, -1, :]).squeeze(-1)

    class CNNLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(n_features, 32, kernel_size=3, padding=1), nn.ReLU())
            self.rnn = nn.LSTM(32, 48, batch_first=True)
            self.out = nn.Linear(48, 1)

        def forward(self, x: Any) -> Any:
            z = self.conv(x.transpose(1, 2)).transpose(1, 2)
            z, _ = self.rnn(z)
            return self.out(z[:, -1, :]).squeeze(-1)

    class TransformerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(n_features, 48)
            layer = nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=96, batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.out = nn.Linear(48, 1)

        def forward(self, x: Any) -> Any:
            z = self.proj(x)
            z = self.encoder(z)
            return self.out(z[:, -1, :]).squeeze(-1)

    builders: dict[str, Callable[[], Any]] = {
        "MLP": MLP,
        "CNN": CNN,
        "LSTM": lambda: RNNModel("lstm", False),
        "GRU": lambda: RNNModel("gru", False),
        "BiLSTM": lambda: RNNModel("lstm", True),
        "CNN-LSTM": CNNLSTM,
        "Transformer": TransformerModel,
    }
    return builders[model_name]()


def train_one_torch(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    window: int,
    epochs: int,
    quick: bool,
) -> tuple[Any, list[dict[str, float]], float, int]:
    torch, nn, DataLoader, TensorDataset = import_torch()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torch_model(model_name, n_features, window, nn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    batch_size = 128 if quick else 96
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = torch.tensor(x_val).to(device)
    val_y = torch.tensor(y_val).to(device)
    best_state = None
    best_val = float("inf")
    patience = 4 if quick else 7
    wait = 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
        train_loss = float(np.mean(train_losses))
        history.append({"Epoch": epoch, "Train_Loss": train_loss, "Validation_Loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - start
    param_count = sum(p.numel() for p in model.parameters())
    return model, history, elapsed, int(param_count)


def predict_torch(model: Any, x_test: np.ndarray) -> np.ndarray:
    torch, _, _, _ = import_torch()
    device = next(model.parameters()).device
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_test), 512):
            batch = torch.tensor(x_test[start : start + 512]).to(device)
            outputs.append(model(batch).detach().cpu().numpy())
    return np.concatenate(outputs)


def train_dl_models(
    scenarios: dict[str, pd.DataFrame],
    epochs: int = 25,
    quick: bool = False,
    window: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import_torch()
    except Exception as exc:
        note = pd.DataFrame([{"Not": f"PyTorch unavailable, DL models skipped: {exc}"}])
        save_table(note, TABLE_DIR / "table_5_3_dl_scenarios.csv")
        return note, pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    complexity: list[dict[str, Any]] = []
    model_names = ["MLP", "CNN", "LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Transformer"]
    if quick:
        model_names = ["MLP", "LSTM", "GRU", "Transformer"]

    for scenario_name, scenario_df in scenarios.items():
        print(f"DL training: {scenario_name}", flush=True)
        feature_names = SCENARIOS[scenario_name]
        train_df = scenario_df[scenario_df["Set"].eq("Train")].dropna(subset=["Close_Next", "Target_Return"]).copy()
        val_df = scenario_df[scenario_df["Set"].eq("Validation")].dropna(subset=["Close_Next", "Target_Return"]).copy()
        test_df = scenario_df[scenario_df["Set"].eq("Test")].dropna(subset=["Close_Next", "Target_Return"]).copy()
        scaler = MinMaxScaler().fit(train_df[feature_names])
        x_train, y_train, _ = make_sequences(train_df, feature_names, scaler, window)
        x_val, y_val, _ = make_sequences(val_df, feature_names, scaler, window)
        x_test, _, test_meta = make_sequences(test_df, feature_names, scaler, window)
        if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
            continue

        for model_name in model_names:
            print(f"  - {model_name}", flush=True)
            model, history, elapsed, param_count = train_one_torch(
                model_name,
                x_train,
                y_train,
                x_val,
                y_val,
                len(feature_names),
                window,
                epochs,
                quick,
            )
            predicted_return = predict_torch(model, x_test)
            predicted_price = test_meta["Close"].to_numpy() * (1 + predicted_return)
            actual_price = test_meta["Close_Next"].to_numpy()
            metrics = price_metrics(actual_price, predicted_price)
            rows.append(
                {
                    "Senaryo": scenario_name,
                    "Model": model_name,
                    "RMSE": metrics["rmse"],
                    "MAE": metrics["mae"],
                    "MAPE": metrics["mape"],
                    "R2": metrics["r2"],
                    "Egitim_Suresi_Sn": elapsed,
                    "Ozellik_Sayisi": len(feature_names),
                    "Pencere": window,
                    "Parametre_Sayisi": param_count,
                    "Test_Gozlem": len(test_meta),
                }
            )
            pred = test_meta[["Tarih", "Target_Tarih", "Hisse_Kodu", "Set", "Close", "Close_Next", "Target_Return"]].copy()
            pred["Senaryo"] = scenario_name
            pred["Model"] = model_name
            pred["Predicted_Return"] = predicted_return
            pred["Predicted_Close"] = predicted_price
            predictions.append(pred)
            hist = pd.DataFrame(history)
            hist["Senaryo"] = scenario_name
            hist["Model"] = model_name
            histories.append(hist)
            complexity.append(
                {
                    "Model_Tipi": "DL",
                    "Senaryo": scenario_name,
                    "Model": model_name,
                    "Parametre_Sayisi": param_count,
                    "Egitim_Suresi_Sn": elapsed,
                }
            )

    result = pd.DataFrame(rows).sort_values(["Senaryo", "RMSE"]).reset_index(drop=True)
    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    hist_df = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
    complexity_df = pd.DataFrame(complexity)
    save_table(result, TABLE_DIR / "table_5_2_dl_scenarios.csv")
    save_table(pred_df, PRED_DIR / "dl_predictions.csv")
    save_table(hist_df, TABLE_DIR / "table_5_2_dl_learning_curves.csv")
    save_table(complexity_df, TABLE_DIR / "table_5_2_dl_training_summary.csv")
    return result, pred_df, hist_df


def imputation_benchmark_models() -> dict[str, Any]:
    try:
        from lightgbm import LGBMRegressor

        return {
            "LightGBM": LGBMRegressor(
                n_estimators=140,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        }
    except Exception:
        return {
            "Random Forest": RandomForestRegressor(
                n_estimators=90,
                max_depth=12,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        }


def imputation_comparison(raw: pd.DataFrame, quick: bool = False) -> pd.DataFrame:
    rows = []
    models = imputation_benchmark_models()
    for method in ["ffill", "interpolate", "knn"]:
        print(f"Imputation benchmark: {method}", flush=True)
        start = time.perf_counter()
        feature_df = engineer_features(impute_raw(raw, method=method))
        scenario_df = feature_df[unique_columns(META_COLUMNS + ALL_FEATURES)].copy()
        train_val = scenario_df[scenario_df["Set"].eq("Train")].dropna(
            subset=["Close_Next", "Target_Return"]
        )
        test = scenario_df[scenario_df["Set"].eq("Test")].dropna(subset=["Close_Next", "Target_Return"])
        best_row: dict[str, Any] | None = None
        for model_name, model in models.items():
            pipe = Pipeline([("scaler", MinMaxScaler()), ("model", clone(model))])
            pipe.fit(train_val[ALL_FEATURES], train_val["Target_Return"])
            predicted_return = pipe.predict(test[ALL_FEATURES])
            predicted_price = test["Close"].to_numpy() * (1 + predicted_return)
            metrics = price_metrics(test["Close_Next"].to_numpy(), predicted_price)
            candidate = {
                "Yontem": method,
                "Model": model_name,
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "MAPE": metrics["mape"],
                "R2": metrics["r2"],
                "Sure_Sn": time.perf_counter() - start,
            }
            if best_row is None or candidate["RMSE"] < best_row["RMSE"]:
                best_row = candidate
        if best_row is not None:
            rows.append(best_row)
    result = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    save_table(result, TABLE_DIR / "table_5_0_imputation_comparison.csv")
    return result


def plot_eda(features: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    sample_codes = sorted(features["Hisse_Kodu"].unique())

    plt.figure(figsize=(11, 6))
    for code in sample_codes:
        group = features[features["Hisse_Kodu"].eq(code)]
        plt.plot(group["Tarih"], group["Close"], linewidth=1.1, label=code)
    plt.xlabel("Tarih")
    plt.ylabel("Kapanis Fiyati")
    plt.legend(ncol=3, fontsize=8)
    save_fig(FIG_DIR / "fig_5_1_closing_prices.png")

    plt.figure(figsize=(11, 6))
    sns.boxplot(data=features, x="Hisse_Kodu", y="Gunluk_Getiri")
    plt.xlabel("Hisse")
    plt.ylabel("Gunluk Getiri")
    save_fig(FIG_DIR / "fig_5_2_daily_return_boxplot.png")

    plt.figure(figsize=(12, 10))
    corr = features[ALL_FEATURES].corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, square=False, cbar_kws={"shrink": 0.75})
    plt.xlabel("")
    plt.ylabel("")
    save_fig(FIG_DIR / "fig_5_3_correlation_matrix.png")

    plt.figure(figsize=(11, 6))
    for code in sample_codes[:3]:
        group = features[features["Hisse_Kodu"].eq(code)]
        plt.plot(group["Tarih"], group["Volatilite_10"], linewidth=1.0, label=code)
    plt.xlabel("Tarih")
    plt.ylabel("10 Gunluk Volatilite")
    plt.legend(fontsize=8)
    save_fig(FIG_DIR / "fig_5_4_volatility_examples.png")

    plt.figure(figsize=(11, 6))
    one = features[features["Hisse_Kodu"].eq(sample_codes[0])]
    plt.plot(one["Tarih"], one["Close"], label="Close", linewidth=1.2)
    plt.plot(one["Tarih"], one["MA_10"], label="MA_10", linewidth=1.0)
    plt.plot(one["Tarih"], one["MA_50"], label="MA_50", linewidth=1.0)
    plt.xlabel("Tarih")
    plt.ylabel("Fiyat")
    plt.legend(fontsize=8)
    save_fig(FIG_DIR / "fig_5_5_moving_average_example.png")


def naive_baseline_results(scenarios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, scenario_df in scenarios.items():
        test = scenario_df.loc[:, ~scenario_df.columns.duplicated()]
        test = test[test["Set"].eq("Test")].dropna(subset=["Close_Next", "Target_Return"]).copy()
        if test.empty:
            continue
        metrics = price_metrics(test["Close_Next"].to_numpy(), test["Close"].to_numpy())
        rows.append(
            {
                "Senaryo": scenario_name,
                "Model": "Naive Baseline",
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "MAPE": metrics["mape"],
                "R2": metrics["r2"],
                "Egitim_Suresi_Sn": 0.0,
                "Ozellik_Sayisi": 1,
                "Test_Gozlem": len(test),
                "Model_Tipi": "Baseline",
            }
        )
    result = pd.DataFrame(rows)
    save_table(result, TABLE_DIR / "table_5_2_naive_baseline.csv")
    return result


def plot_performance(
    ml_result: pd.DataFrame,
    dl_result: pd.DataFrame,
    hist_df: pd.DataFrame,
    baseline_result: pd.DataFrame | None = None,
) -> None:
    combined = []
    if baseline_result is not None and not baseline_result.empty:
        combined.append(baseline_result.copy())
    if not ml_result.empty and "RMSE" in ml_result.columns:
        a = ml_result.copy()
        a["Model_Tipi"] = "ML"
        combined.append(a)
    if not dl_result.empty and "RMSE" in dl_result.columns:
        b = dl_result.copy()
        b["Model_Tipi"] = "DL"
        combined.append(b)
    combined_df = pd.concat(combined, ignore_index=True) if combined else pd.DataFrame()
    if not combined_df.empty:
        save_table(combined_df, TABLE_DIR / "table_5_2_all_model_scenarios.csv")
        best_all = combined_df[combined_df["Senaryo"].eq("scenario_3_all")].copy()
        for metric, ylabel, filename in [
            ("MAE", "MAE", "fig_5_9_model_mae_comparison.png"),
            ("RMSE", "RMSE", "fig_5_10_model_rmse_comparison.png"),
            ("MAPE", "MAPE (%)", "fig_5_11_model_mape_comparison.png"),
            ("R2", "R2", "fig_5_12_model_r2_comparison.png"),
        ]:
            if not best_all.empty and metric in best_all.columns:
                plt.figure(figsize=(12, 6))
                sns.barplot(data=best_all.sort_values(metric), x="Model", y=metric, hue="Model_Tipi")
                plt.xlabel("Model")
                plt.ylabel(ylabel)
                plt.xticks(rotation=35, ha="right")
                save_fig(FIG_DIR / filename)

    if not ml_result.empty and "RMSE" in ml_result.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=ml_result, x="Model", y="RMSE", hue="Senaryo")
        plt.xlabel("Model")
        plt.ylabel("RMSE")
        plt.xticks(rotation=35, ha="right")
        save_fig(FIG_DIR / "fig_5_6_ml_rmse_by_scenario.png")

        plt.figure(figsize=(12, 6))
        sns.barplot(data=ml_result, x="Model", y="MAPE", hue="Senaryo")
        plt.xlabel("Model")
        plt.ylabel("MAPE (%)")
        plt.xticks(rotation=35, ha="right")
        save_fig(FIG_DIR / "fig_5_7_ml_mape_by_scenario.png")

    if not dl_result.empty and "RMSE" in dl_result.columns:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=dl_result, x="Model", y="RMSE", hue="Senaryo")
        plt.xlabel("Model")
        plt.ylabel("RMSE")
        plt.xticks(rotation=35, ha="right")
        save_fig(FIG_DIR / "fig_5_8_dl_rmse_by_scenario.png")

    if not hist_df.empty:
        best_hist = hist_df[hist_df["Senaryo"].eq("scenario_3_all")].copy()
        if not best_hist.empty:
            plt.figure(figsize=(11, 6))
            for model_name, group in best_hist.groupby("Model"):
                plt.plot(group["Epoch"], group["Validation_Loss"], label=model_name, linewidth=1.2)
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.legend(fontsize=8)
            save_fig(FIG_DIR / "fig_5_9_dl_validation_loss.png")


def preliminary_feature_importance(features: pd.DataFrame) -> None:
    model_df = features.dropna(subset=["Target_Return"]).copy()
    train_val = model_df[model_df["Set"].eq("Train")]
    if train_val.empty:
        return
    pipe = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=160,
                    max_depth=12,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(train_val[ALL_FEATURES], train_val["Target_Return"])
    importance = pd.DataFrame(
        {
            "Degisken": ALL_FEATURES,
            "Onem": pipe.named_steps["model"].feature_importances_,
        }
    ).sort_values("Onem", ascending=False)
    save_table(importance, TABLE_DIR / "table_5_1_preliminary_feature_importance.csv")
    plt.figure(figsize=(9, 7))
    sns.barplot(data=importance.head(15), y="Degisken", x="Onem", color="#3f7fbf")
    plt.xlabel("Onem")
    plt.ylabel("")
    save_fig(FIG_DIR / "fig_5_8_preliminary_feature_importance.png")


def plot_actual_vs_pred(pred_df: pd.DataFrame, result_df: pd.DataFrame, prefix: str) -> None:
    if pred_df.empty or result_df.empty or "RMSE" not in result_df.columns:
        return
    best = result_df.sort_values("RMSE").iloc[0]
    subset = pred_df[
        pred_df["Senaryo"].eq(best["Senaryo"]) & pred_df["Model"].eq(best["Model"])
    ].sort_values(["Hisse_Kodu", "Tarih"])
    if subset.empty:
        return
    code = subset["Hisse_Kodu"].value_counts().index[0]
    one = subset[subset["Hisse_Kodu"].eq(code)].copy()
    plt.figure(figsize=(11, 6))
    plt.plot(one["Tarih"], one["Close_Next"], label="Gercek", linewidth=1.3)
    plt.plot(one["Tarih"], one["Predicted_Close"], label="Tahmin", linewidth=1.2)
    plt.xlabel("Tarih")
    plt.ylabel("Kapanis Fiyati")
    plt.legend(fontsize=8)
    save_fig(FIG_DIR / f"{prefix}_actual_vs_predicted.png")


def run_xai(features: pd.DataFrame, fitted_models: dict[str, Any], ml_result: pd.DataFrame) -> None:
    all_key_rows = ml_result[ml_result["Senaryo"].eq("scenario_3_all")].sort_values("RMSE")
    if all_key_rows.empty:
        return
    model_name = str(all_key_rows.iloc[0]["Model"])
    key = f"ML::scenario_3_all::{model_name}"
    if key not in fitted_models:
        return
    pipe = fitted_models[key]
    model = pipe.named_steps["model"]
    train_test = features[features["Set"].isin(["Train", "Validation", "Test"])].copy()
    x_scaled = pipe.named_steps["scaler"].transform(train_test[ALL_FEATURES])

    importance = None
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({"Degisken": ALL_FEATURES, "Onem": model.feature_importances_})
    elif hasattr(model, "coef_"):
        importance = pd.DataFrame({"Degisken": ALL_FEATURES, "Onem": np.abs(np.ravel(model.coef_))})
    if importance is not None:
        importance = importance.sort_values("Onem", ascending=False).reset_index(drop=True)
        save_table(importance, XAI_DIR / "feature_importance_scenario_3_all.csv")
        plt.figure(figsize=(9, 7))
        sns.barplot(data=importance.head(15), y="Degisken", x="Onem", color="#3f7fbf")
        plt.xlabel("Onem")
        plt.ylabel("")
        save_fig(FIG_DIR / "fig_5_10_xai_feature_importance.png")

    try:
        import shap

        if model_name in {"Random Forest", "XGBoost", "LightGBM", "CatBoost"}:
            sample_size = min(600, len(x_scaled))
            sample_idx = np.linspace(0, len(x_scaled) - 1, sample_size).astype(int)
            explainer = shap.Explainer(model)
            shap_values = explainer(x_scaled[sample_idx])
            shap.summary_plot(shap_values, features=train_test.iloc[sample_idx][ALL_FEATURES], show=False, plot_type="bar")
            plt.xlabel("mean(|SHAP value|)")
            save_fig(FIG_DIR / "fig_5_11_shap_summary_bar.png")
    except Exception as exc:
        (XAI_DIR / "shap_note.txt").write_text(f"SHAP skipped: {exc}", encoding="utf-8")

    try:
        if importance is not None:
            top_features = importance.head(2)["Degisken"].tolist()
        else:
            top_features = ALL_FEATURES[:2]
        display = PartialDependenceDisplay.from_estimator(pipe, train_test[ALL_FEATURES], top_features, kind="average")
        for ax in display.axes_.ravel():
            ax.set_title("")
        save_fig(FIG_DIR / "fig_5_12_pdp_top_features.png")
    except Exception as exc:
        (XAI_DIR / "pdp_note.txt").write_text(f"PDP skipped: {exc}", encoding="utf-8")

    try:
        from lime.lime_tabular import LimeTabularExplainer

        train = features[features["Set"].isin(["Train", "Validation"])].copy()
        scaler = pipe.named_steps["scaler"]
        explainer = LimeTabularExplainer(
            scaler.transform(train[ALL_FEATURES]),
            feature_names=ALL_FEATURES,
            mode="regression",
            discretize_continuous=True,
            random_state=42,
        )
        test = features[features["Set"].eq("Test")].copy()
        instance = scaler.transform(test[ALL_FEATURES].iloc[[0]])[0]
        explanation = explainer.explain_instance(instance, pipe.named_steps["model"].predict, num_features=10)
        lime_rows = pd.DataFrame(explanation.as_list(), columns=["Degisken_Aralik", "Katki"])
        save_table(lime_rows, XAI_DIR / "lime_single_instance.csv")
    except Exception as exc:
        (XAI_DIR / "lime_note.txt").write_text(f"LIME skipped: {exc}", encoding="utf-8")


def diebold_mariano(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    d = d[np.isfinite(d)]
    if len(d) < 3 or np.std(d) == 0:
        return np.nan, np.nan
    stat = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    p_value = 2 * (1 - stats.t.cdf(abs(stat), df=len(d) - 1))
    return float(stat), float(p_value)


def statistical_tests(ml_pred: pd.DataFrame, dl_pred: pd.DataFrame, ml_result: pd.DataFrame, dl_result: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if ml_pred.empty or dl_pred.empty or dl_result.empty:
        out = pd.DataFrame(rows)
        save_table(out, TABLE_DIR / "table_5_4_statistical_tests.csv")
        return out

    best_ml = ml_result.sort_values("RMSE").iloc[0]
    best_dl = dl_result.sort_values("RMSE").iloc[0]
    a = ml_pred[(ml_pred["Senaryo"].eq(best_ml["Senaryo"])) & (ml_pred["Model"].eq(best_ml["Model"]))].copy()
    b = dl_pred[(dl_pred["Senaryo"].eq(best_dl["Senaryo"])) & (dl_pred["Model"].eq(best_dl["Model"]))].copy()
    merged = a.merge(
        b,
        on=["Tarih", "Hisse_Kodu"],
        suffixes=("_ML", "_DL"),
    )
    if not merged.empty:
        err_ml = merged["Close_Next_ML"] - merged["Predicted_Close_ML"]
        err_dl = merged["Close_Next_DL"] - merged["Predicted_Close_DL"]
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(np.abs(err_ml), np.abs(err_dl))
        except Exception:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan
        dm_stat, dm_p = diebold_mariano(err_ml, err_dl)
        rows.extend(
            [
                {
                    "Test": "Wilcoxon",
                    "Model_A": f"{best_ml['Model']} ({best_ml['Senaryo']})",
                    "Model_B": f"{best_dl['Model']} ({best_dl['Senaryo']})",
                    "Istatistik": wilcoxon_stat,
                    "p_degeri": wilcoxon_p,
                    "Gozlem": len(merged),
                },
                {
                    "Test": "Diebold-Mariano",
                    "Model_A": f"{best_ml['Model']} ({best_ml['Senaryo']})",
                    "Model_B": f"{best_dl['Model']} ({best_dl['Senaryo']})",
                    "Istatistik": dm_stat,
                    "p_degeri": dm_p,
                    "Gozlem": len(merged),
                },
            ]
        )
    out = pd.DataFrame(rows)
    save_table(out, TABLE_DIR / "table_5_4_statistical_tests.csv")
    return out


def model_complexity(ml_result: pd.DataFrame, dl_result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not ml_result.empty:
        for _, row in ml_result.iterrows():
            rows.append(
                {
                    "Model_Tipi": "ML",
                    "Senaryo": row["Senaryo"],
                    "Model": row["Model"],
                    "Ozellik_Sayisi": row["Ozellik_Sayisi"],
                    "Parametre_Sayisi": np.nan,
                    "Egitim_Suresi_Sn": row["Egitim_Suresi_Sn"],
                }
            )
    if not dl_result.empty and "Parametre_Sayisi" in dl_result.columns:
        for _, row in dl_result.iterrows():
            rows.append(
                {
                    "Model_Tipi": "DL",
                    "Senaryo": row["Senaryo"],
                    "Model": row["Model"],
                    "Ozellik_Sayisi": row["Ozellik_Sayisi"],
                    "Parametre_Sayisi": row["Parametre_Sayisi"],
                    "Egitim_Suresi_Sn": row["Egitim_Suresi_Sn"],
                }
            )
    out = pd.DataFrame(rows)
    save_table(out, TABLE_DIR / "table_5_5_model_complexity.csv")
    if not out.empty:
        plt.figure(figsize=(11, 6))
        sns.barplot(data=out, x="Model", y="Egitim_Suresi_Sn", hue="Model_Tipi")
        plt.xlabel("Model")
        plt.ylabel("Egitim Suresi (sn)")
        plt.xticks(rotation=35, ha="right")
        save_fig(FIG_DIR / "fig_5_13_model_training_time.png")
    return out


def write_manifest(raw: pd.DataFrame, features: pd.DataFrame, ml_result: pd.DataFrame, dl_result: pd.DataFrame) -> None:
    manifest = {
        "raw_file": str(RAW_FILE),
        "raw_shape": list(raw.shape),
        "feature_shape": list(features.shape),
        "feature_date_min": str(features["Tarih"].min().date()),
        "feature_date_max": str(features["Tarih"].max().date()),
        "model_input_date_max": str(features.dropna(subset=["Target_Return"])["Tarih"].max().date()),
        "prediction_target_date_max": str(features.dropna(subset=["Target_Tarih"])["Target_Tarih"].max().date()),
        "study_start_inclusive": str(STUDY_START.date()),
        "study_end_exclusive": str(STUDY_END_EXCLUSIVE.date()),
        "expected_last_observation": str(EXPECTED_LAST_DATE.date()),
        "stocks": sorted(features["Hisse_Kodu"].unique().tolist()),
        "expected_stocks": EXPECTED_STOCKS,
        "feature_count": len(ALL_FEATURES),
        "basic_feature_count": len(RAW_FEATURES),
        "derived_feature_count": len(DERIVED_FEATURES),
        "scenarios": {name: columns for name, columns in SCENARIOS.items()},
        "split_counts": features.groupby(["Hisse_Kodu", "Set"]).size().reset_index(name="n").to_dict(orient="records"),
        "ml_models": sorted(ml_result["Model"].unique().tolist()) if not ml_result.empty and "Model" in ml_result else [],
        "dl_models": sorted(dl_result["Model"].unique().tolist()) if not dl_result.empty and "Model" in dl_result else [],
        "figures_have_titles": False,
    }
    (REPORT_DIR / "pipeline_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(args: argparse.Namespace) -> None:
    set_seed()
    ensure_dirs()
    print("Step 1/7: reading raw data", flush=True)
    raw = read_raw_data()
    save_table(missing_table(raw), TABLE_DIR / "table_3_1_missing_data.csv")

    if not args.skip_imputation_comparison:
        print("Step 2/7: imputation comparison", flush=True)
        imputation_comparison(raw, quick=True)

    print("Step 3/7: feature engineering and scenarios", flush=True)
    imputed = impute_raw(raw, method="interpolate")
    features = engineer_features(imputed)
    scenarios = create_scenario_files(features)
    save_table(descriptive_statistics(features), TABLE_DIR / "table_5_1_descriptive_statistics.csv")

    print("Step 4/7: 5.1 figures", flush=True)
    plot_eda(features)
    preliminary_feature_importance(features)
    print("Step 5/7: ML models", flush=True)
    ml_result, ml_pred, fitted = train_ml_models(scenarios, quick=args.quick)
    print("Step 6/7: DL models", flush=True)
    dl_result, dl_pred, hist_df = train_dl_models(scenarios, epochs=args.epochs, quick=args.quick, window=args.window)
    baseline_result = naive_baseline_results(scenarios)
    print("Step 7/7: 5.2 performance figures", flush=True)
    plot_performance(ml_result, dl_result, hist_df, baseline_result)
    if args.include_post_5_2:
        print("Optional: post-5.2 outputs", flush=True)
        plot_actual_vs_pred(ml_pred, ml_result, "fig_5_14_ml")
        plot_actual_vs_pred(dl_pred, dl_result, "fig_5_15_dl")
        run_xai(features, fitted, ml_result)
        statistical_tests(ml_pred, dl_pred, ml_result, dl_result)
        model_complexity(ml_result, dl_result)
    write_manifest(raw, features, ml_result, dl_result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean BIST 100 research pipeline")
    parser.add_argument("--quick", action="store_true", help="Run fewer tree/DL models for a fast smoke test.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum epoch count for deep learning models.")
    parser.add_argument("--window", type=int, default=10, help="Sequence window length for deep learning models.")
    parser.add_argument(
        "--skip-imputation-comparison",
        action="store_true",
        help="Skip the ffill/interpolate/knn comparison step.",
    )
    parser.add_argument(
        "--include-post-5-2",
        action="store_true",
        help="Also generate draft outputs for sections after 5.2.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)
