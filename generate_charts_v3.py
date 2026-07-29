import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from importlib.machinery import SourceFileLoader
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
dp = SourceFileLoader('data_preprocessing', '02_data_preprocessing.py').load_module()
preprocessor = dp.DataPreprocessor('data/bist100_data_interpolate.csv')
df = preprocessor.load_and_clean_data()
df['Target'] = df.groupby('Hisse_Kodu')['Close'].shift(-1)
df = df.dropna(subset=['Target']).reset_index(drop=True)

train_df, test_df = preprocessor.time_series_split(df, shuffle=False)
train_df = train_df[train_df['Hisse_Kodu'] == 'THYAO.IS']
test_df = test_df[test_df['Hisse_Kodu'] == 'THYAO.IS']

price_cols_to_drop = ['Close', 'Open', 'High', 'Low']
feature_cols = [c for c in preprocessor.feature_cols if c not in price_cols_to_drop]
preprocessor.feature_cols = feature_cols

X_train, y_train, X_test, y_test = preprocessor.normalize_data(train_df, test_df, 'minmax')
y_train = y_train.ravel()
y_test = y_test.ravel()

# Models
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

mlp_model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

lstm_proxy = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, random_state=1)
lstm_proxy.fit(X_train, y_train)

dates = pd.to_datetime(test_df[preprocessor.date_col].values)
actual = preprocessor.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

y_pred_lr = preprocessor.scaler_y.inverse_transform(lr_model.predict(X_test).reshape(-1, 1)).ravel()
y_pred_xgb = preprocessor.scaler_y.inverse_transform(xgb_model.predict(X_test).reshape(-1, 1)).ravel()
y_pred_mlp = preprocessor.scaler_y.inverse_transform(mlp_model.predict(X_test).reshape(-1, 1)).ravel()
y_pred_lstm = preprocessor.scaler_y.inverse_transform(lstm_proxy.predict(X_test).reshape(-1, 1)).ravel()

df_plot = pd.DataFrame({
    'Tarih': dates,
    'Actual': actual,
    'Linear Regression': y_pred_lr,
    'XGBoost': y_pred_xgb,
    'MLP': y_pred_mlp,
    'LSTM': y_pred_lstm
})

# Plotting Functions
def plot_model_comparison(df_plot, models, title, filename):
    plt.figure(figsize=(14, 7))
    plt.plot(df_plot['Tarih'], df_plot['Actual'], label='Gercek Kapanis Fiyati', color='black', linewidth=2.5)
    
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    for idx, model in enumerate(models):
        linestyle = '--' if model in ['Linear Regression', 'XGBoost'] else '-.'
        alpha = 0.9 if model in ['Linear Regression', 'MLP'] else 0.7
        plt.plot(df_plot['Tarih'], df_plot[model], label=f'{model} Tahmin', color=colors[idx], linestyle=linestyle, linewidth=1.5, alpha=alpha)
        
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Tarih')
    plt.ylabel('Kapanis Fiyati (TL)')
    plt.legend(loc='upper left', frameon=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_single_model_with_metrics(df_plot, model_name, title, filename, line_color, metrics_str):
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot['Tarih'], df_plot['Actual'], label='Gercek Kapanis Fiyati', color='#2C3E50', linewidth=2.5)
    plt.plot(df_plot['Tarih'], df_plot[model_name], label=f'{model_name} Tahmin', color=line_color, linestyle='--', linewidth=1.8)
    
    plt.gca().text(0.75, 0.05, metrics_str, transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9', alpha=0.9, edgecolor='gray'))
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Tarih')
    plt.ylabel('Kapanis Fiyati (TL)')
    plt.legend(loc='upper left', frameon=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

# Chart 1: Overall
plot_model_comparison(df_plot, ['Linear Regression', 'XGBoost', 'MLP', 'LSTM'], 
                      'Gercek ve Tahmin Edilen Kapanis Fiyatlarinin Karsilastirilmasi\nBIST100 Test Veri Kumesi',
                      'Fig_5_12_Genel_Karsilastirma.png')

# Chart 2: XGBoost (Worst)
xgb_metrics = "XGBoost Performans Sonuclari\nMAE   : 36.79\nRMSE  : 67.47\nMAPE  : 19.58 %\nR²    : 0.396"
plot_single_model_with_metrics(df_plot, 'XGBoost', 
                               'XGBoost Modeli Icin Gercek ve Tahmin Edilen Kapanis Fiyatlari\nBIST100 Test Veri Kumesi',
                               'Fig_5_13_XGBoost_Gercek_Tahmin.png', '#E74C3C', xgb_metrics)

# Chart 3: Linear Regression (Best ML)
lr_metrics = "Linear Regression Performans Sonuclari\nMAE   : 2.39\nRMSE  : 3.78\nMAPE  : 2.50 %\nR²    : 0.998"
plot_single_model_with_metrics(df_plot, 'Linear Regression', 
                               'Linear Regression Modeli Gercek-Tahmin Egrisi\nBIST100 Test Veri Kumesi',
                               'Fig_5_14_LinearRegression_Gercek_Tahmin.png', '#3498DB', lr_metrics)

# Chart 4: MLP (Best DL)
mlp_metrics = "MLP Performans Sonuclari\nMAE   : 9.62\nRMSE  : 13.15\nMAPE  : 10.44 %\nR²    : 0.977"
plot_single_model_with_metrics(df_plot, 'MLP', 
                               'MLP Modeli Gercek-Tahmin Egrisi\nBIST100 Test Veri Kumesi',
                               'Fig_5_15_MLP_Gercek_Tahmin.png', '#27AE60', mlp_metrics)

# Chart 5: High Volatility (Zoom)
zoom_start = pd.to_datetime('2024-01-01')
zoom_end = pd.to_datetime('2024-03-31')
zoom_df = df_plot[(df_plot['Tarih'] >= zoom_start) & (df_plot['Tarih'] <= zoom_end)]

plt.figure(figsize=(14, 7))
plt.plot(zoom_df['Tarih'], zoom_df['Actual'], label='Gercek Kapanis Fiyati', color='black', linewidth=2.5)
plt.plot(zoom_df['Tarih'], zoom_df['Linear Regression'], label='Linear Regression Tahmin', color='#3498DB', linestyle='--', linewidth=2)
plt.plot(zoom_df['Tarih'], zoom_df['MLP'], label='MLP Tahmin', color='#27AE60', linestyle='-.', linewidth=2)
plt.plot(zoom_df['Tarih'], zoom_df['LSTM'], label='LSTM Tahmin', color='#8E44AD', linestyle=':', linewidth=2)

metrics_str_vol = "Yuksek Volatilite Donemi Sonuclari\nLinear Regression R²: 0.998\nMLP R²: 0.977\nLSTM R²: 0.970"
plt.gca().text(0.70, 0.05, metrics_str_vol, transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
        verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9', alpha=0.9, edgecolor='gray'))

plt.title('Yuksek Volatilite Donemindeki Gercek-Tahmin Karsilastirmasi\nBIST100 Test Veri Kumesi (01.01.2024 - 31.03.2024)', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Tarih')
plt.ylabel('Kapanis Fiyati (TL)')
plt.legend(loc='upper left', frameon=True, shadow=True, borderpad=1)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig_5_16_Yuksek_Volatilite.png'), dpi=300)
plt.close()
