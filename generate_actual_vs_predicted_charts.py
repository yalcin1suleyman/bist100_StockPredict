import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from importlib.machinery import SourceFileLoader
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

# 1. Load Data
dp = SourceFileLoader('data_preprocessing', '02_data_preprocessing.py').load_module()
preprocessor = dp.DataPreprocessor('data/bist100_data_interpolate.csv')
df = preprocessor.load_and_clean_data()
df['Target'] = df.groupby('Hisse_Kodu')['Close'].shift(-1)
df = df.dropna(subset=['Target']).reset_index(drop=True)
preprocessor.target_col = 'Target'

train_df, test_df = preprocessor.time_series_split(df, shuffle=False)

price_cols_to_drop = ['Close', 'Open', 'High', 'Low']
feature_cols = [c for c in preprocessor.feature_cols if c not in price_cols_to_drop]
preprocessor.feature_cols = feature_cols

X_train, y_train, X_test, y_test = preprocessor.normalize_data(train_df, test_df, 'minmax')
y_train = y_train.ravel()
y_test = y_test.ravel()

print("ML Modelleri Egitiliyor...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

ml_dates = test_df[preprocessor.date_col].values
ml_hisse = test_df['Hisse_Kodu'].values

y_pred_lr = preprocessor.scaler_y.inverse_transform(lr_model.predict(X_test).reshape(-1, 1)).ravel()
y_pred_xgb = preprocessor.scaler_y.inverse_transform(xgb_model.predict(X_test).reshape(-1, 1)).ravel()

ml_preds = pd.DataFrame({
    'Tarih': ml_dates,
    'Hisse_Kodu': ml_hisse,
    'Linear Regression': y_pred_lr,
    'XGBoost': y_pred_xgb
})

# 2. Get DL Predictions and align with dates
dl_preds = pd.read_csv('outputs/dl_predictions.csv')
dates_list = []
hisse_list = []
window_size = 10
for hisse in df['Hisse_Kodu'].unique():
    test_mask = (test_df['Hisse_Kodu'] == hisse).values
    dates = test_df[test_mask][preprocessor.date_col].values[window_size:]
    dates_list.extend(dates)
    hisse_list.extend([hisse] * len(dates))

dl_preds['Tarih'] = dates_list
dl_preds['Hisse_Kodu'] = hisse_list

# Merge for THYAO
thyao_dl = dl_preds[dl_preds['Hisse_Kodu'] == 'THYAO.IS'].copy()
thyao_dl['Tarih'] = pd.to_datetime(thyao_dl['Tarih'])
thyao_ml = ml_preds[ml_preds['Hisse_Kodu'] == 'THYAO.IS'].copy()
thyao_ml['Tarih'] = pd.to_datetime(thyao_ml['Tarih'])

merged = pd.merge(thyao_dl, thyao_ml, on=['Tarih', 'Hisse_Kodu'], how='inner')
merged = merged.sort_values('Tarih')

# 3. Plotting Functions
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

def plot_model_comparison(df_plot, models, title, filename):
    plt.figure(figsize=(14, 7))
    plt.plot(df_plot['Tarih'], df_plot['Actual'], label='Gercek Kapanis Fiyati', color='black', linewidth=2)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, model in enumerate(models):
        linestyle = '--' if model in ['Linear Regression', 'XGBoost'] else '-.'
        plt.plot(df_plot['Tarih'], df_plot[model], label=f'{model} Tahmin', color=colors[idx], linestyle=linestyle, linewidth=1.5)
        
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Tarih')
    plt.ylabel('Kapanis Fiyati (TL)')
    plt.legend(loc='upper left', frameon=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def plot_single_model_with_metrics(df_plot, model_name, title, filename, line_color='red'):
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot['Tarih'], df_plot['Actual'], label='Gercek Kapanis Fiyati', color='#2C3E50', linewidth=2)
    plt.plot(df_plot['Tarih'], df_plot[model_name], label=f'{model_name} Tahmin', color=line_color, linestyle='--', linewidth=1.5)
    
    mae = mean_absolute_error(df_plot['Actual'], df_plot[model_name])
    rmse = np.sqrt(mean_squared_error(df_plot['Actual'], df_plot[model_name]))
    mape = calculate_mape(df_plot['Actual'], df_plot[model_name])
    r2 = r2_score(df_plot['Actual'], df_plot[model_name])
    
    metrics_text = f"{model_name} Performans Sonuclari\n"
    metrics_text += f"MAE   : {mae:.2f}\n"
    metrics_text += f"RMSE  : {rmse:.2f}\n"
    metrics_text += f"MAPE  : {mape:.2f} %\n"
    metrics_text += f"R²    : {r2:.3f}"
    
    plt.gca().text(0.75, 0.05, metrics_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Tarih')
    plt.ylabel('Kapanis Fiyati (TL)')
    plt.legend(loc='upper left', frameon=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

print("Grafikler Olusturuluyor...")
# Chart 1: Overall
plot_model_comparison(merged, ['Linear Regression', 'XGBoost', 'MLP', 'LSTM'], 
                      'Gercek ve Tahmin Edilen Kapanis Fiyatlarinin Karsilastirilmasi',
                      'Fig_5_12_Genel_Karsilastirma.png')

# Chart 2: XGBoost (Worst)
plot_single_model_with_metrics(merged, 'XGBoost', 
                               'XGBoost Modeli Icin Gercek ve Tahmin Edilen Kapanis Fiyatlari',
                               'Fig_5_13_XGBoost_Gercek_Tahmin.png', line_color='#E74C3C')

# Chart 3: Linear Regression (Best ML)
plot_single_model_with_metrics(merged, 'Linear Regression', 
                               'Linear Regression Modeli Gercek-Tahmin Egrisi',
                               'Fig_5_14_LinearRegression_Gercek_Tahmin.png', line_color='#3498DB')

# Chart 4: MLP (Best DL)
plot_single_model_with_metrics(merged, 'MLP', 
                               'MLP Modeli Gercek-Tahmin Egrisi',
                               'Fig_5_15_MLP_Gercek_Tahmin.png', line_color='#27AE60')

# Chart 5: High Volatility (Zoom)
zoom_start = pd.to_datetime('2023-10-01')
zoom_end = pd.to_datetime('2023-12-31')
zoom_df = merged[(merged['Tarih'] >= zoom_start) & (merged['Tarih'] <= zoom_end)]

plot_model_comparison(zoom_df, ['Linear Regression', 'XGBoost', 'MLP'], 
                      'Yuksek Volatilite Donemindeki Gercek-Tahmin Karsilastirmasi (Ekim-Aralik 2023)',
                      'Fig_5_16_Yuksek_Volatilite.png')

print("Tum grafikler basariyla kaydedildi.")
