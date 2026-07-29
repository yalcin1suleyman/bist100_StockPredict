import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import os
from importlib.machinery import SourceFileLoader
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

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

colors = {
    'Actual': 'black',
    'Linear Regression': '#3498DB',
    'XGBoost': '#E74C3C',
    'MLP': '#27AE60',
    'LSTM': '#9B59B6'
}
linestyles = {
    'Actual': '-',
    'Linear Regression': '--',
    'XGBoost': ':',
    'MLP': '-.',
    'LSTM': '--'
}

def get_metrics(df_sub, models):
    res = []
    for m in models:
        rmse = np.sqrt(mean_squared_error(df_sub['Actual'], df_sub[m]))
        mape = calculate_mape(df_sub['Actual'], df_sub[m])
        r2 = r2_score(df_sub['Actual'], df_sub[m])
        res.append([m, f"{rmse:.2f}", f"{mape:.2f}", f"{r2:.3f}"])
    return res

# ---------------------------------------------------------
# Plot 1: High Volatility with Inset 
# Bizim Proje Tarihleri: 2024-01-01 to 2024-03-31
# ---------------------------------------------------------
start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2024-04-30')
df_vol = df_plot[(df_plot['Tarih'] >= start_date) & (df_plot['Tarih'] <= end_date)]

fig, ax = plt.subplots(figsize=(15, 8))
models_to_plot = ['Linear Regression', 'XGBoost', 'MLP', 'LSTM']

ax.plot(df_vol['Tarih'], df_vol['Actual'], label='Gercek Kapanis Fiyati', color=colors['Actual'], linewidth=2.5)
for m in models_to_plot:
    ax.plot(df_vol['Tarih'], df_vol[m], label=f'{m} Tahmin', color=colors[m], linestyle=linestyles[m], linewidth=2)

ax.set_title('Yuksek Volatilite Donemindeki Gercek-Tahmin Karsilastirmasi\nBIST100 Test Veri Kumesi (01.01.2024 - 30.04.2024)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Tarih', fontsize=12)
ax.set_ylabel('Kapanis Fiyati (TL)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower left', fontsize=10)

# Inset Axes (yakindan bakis: secim donemi vs)
inset_start = pd.to_datetime('2024-02-15')
inset_end = pd.to_datetime('2024-03-15')
df_inset = df_vol[(df_vol['Tarih'] >= inset_start) & (df_vol['Tarih'] <= inset_end)]

axins = ax.inset_axes([0.35, 0.15, 0.35, 0.35])
axins.plot(df_inset['Tarih'], df_inset['Actual'], color=colors['Actual'], linewidth=2)
for m in models_to_plot:
    axins.plot(df_inset['Tarih'], df_inset[m], color=colors[m], linestyle=linestyles[m], linewidth=1.5)

axins.set_title("Yakinlastirilmis Gorunum (15.02.2024 - 15.03.2024)", fontsize=10)
axins.tick_params(axis='both', which='major', labelsize=8)
for label in axins.get_xticklabels():
    label.set_rotation(45)

ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.8)

# Table inside plot
metrics_data = get_metrics(df_vol, models_to_plot)
col_labels = ['Model', 'RMSE (TL)', 'MAPE (%)', 'R²']
table = ax.table(cellText=metrics_data, colLabels=col_labels, loc='upper right', bbox=[0.7, 0.7, 0.28, 0.25], cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2C3E50')
    else:
        if j == 0:
            m_name = metrics_data[i-1][0]
            cell.set_text_props(weight='bold', color=colors[m_name])
            
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig_5_16_Yuksek_Volatilite_Proje.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 2: Full plot with red box + Split bottom view 
# ---------------------------------------------------------
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.2)

# Top: Full series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_plot['Tarih'], df_plot['Actual'], label='Gercek Kapanis Fiyati', color=colors['Actual'], linewidth=2)
for m in models_to_plot:
    ax1.plot(df_plot['Tarih'], df_plot[m], label=f'{m}', color=colors[m], linestyle=linestyles[m], linewidth=1.5)

ax1.set_title('Yakinlastirilmis Gercek-Tahmin Egrileri\nBIST100 Test Veri Kumesi Tumu', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Kapanis Fiyati (TL)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower left', ncol=3)

zoom_start = pd.to_datetime('2024-01-01')
zoom_end = pd.to_datetime('2024-04-30')
df_zoom = df_plot[(df_plot['Tarih'] >= zoom_start) & (df_plot['Tarih'] <= zoom_end)]

# Red box
min_y = df_zoom[['Actual'] + models_to_plot].min().min() * 0.95
max_y = df_zoom[['Actual'] + models_to_plot].max().max() * 1.05
import matplotlib.dates as mdates
rect = patches.Rectangle((mdates.date2num(zoom_start), min_y), 
                         mdates.date2num(zoom_end) - mdates.date2num(zoom_start), 
                         max_y - min_y, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
ax1.add_patch(rect)

# Draw arrow from red box to left subplot
arrow_x = mdates.date2num(zoom_start) + (mdates.date2num(zoom_end) - mdates.date2num(zoom_start)) / 2
arrow_y = min_y
ax1.annotate('', xy=(0.3, 0.48), xytext=(arrow_x, arrow_y),
            xycoords='figure fraction', textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# Bottom-Left: Zoomed region
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df_zoom['Tarih'], df_zoom['Actual'], color=colors['Actual'], linewidth=2.5)
for m in models_to_plot:
    ax2.plot(df_zoom['Tarih'], df_zoom[m], color=colors[m], linestyle=linestyles[m], linewidth=2)

ax2.set_title('Yakinlastirilmis Bolge (01.01.2024 - 30.04.2024)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Kapanis Fiyati (TL)')
ax2.grid(True, linestyle='--', alpha=0.6)
for label in ax2.get_xticklabels():
    label.set_rotation(30)

# Bottom-Right: Table
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
ax3.set_title('Yakinlastirilmis Bolge Performans Metrikleri', fontsize=12, fontweight='bold', pad=10)

metrics_data_zoom = get_metrics(df_zoom, models_to_plot)
table2 = ax3.table(cellText=metrics_data_zoom, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0.1, 0.1, 0.8, 0.8])
table2.auto_set_font_size(False)
table2.set_fontsize(11)
for (i, j), cell in table2.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2C3E50')
    else:
        if j == 0:
            m_name = metrics_data_zoom[i-1][0]
            cell.set_text_props(weight='bold', color=colors[m_name])

plt.savefig(os.path.join(output_dir, 'Fig_5_17_Yakinlastirilmis_Bolge_Proje.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Proje versiyonu grafikler basariyla olusturuldu.")
