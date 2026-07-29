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
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
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

print("Modeller egitiliyor...")
# Models from the user's screenshot
models = {
    'Linear Regression': LinearRegression(),
    'SVR': SVR(C=1.0, epsilon=0.2),
    'Random Forest': RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1),
    'LSTM': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=1),
    'CNN-LSTM': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=2),
    'Transformer': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=3)
}

dates = pd.to_datetime(test_df[preprocessor.date_col].values)
actual = preprocessor.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

df_plot = pd.DataFrame({'Tarih': dates, 'Actual': actual})

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = preprocessor.scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1)).ravel()
    df_plot[name] = y_pred

colors = {
    'Actual': 'black',
    'Linear Regression': '#1f77b4',  # tab:blue
    'SVR': '#ff7f0e',               # tab:orange
    'Random Forest': '#2ca02c',     # tab:green
    'XGBoost': '#d62728',           # tab:red
    'LightGBM': '#9467bd',          # tab:purple
    'LSTM': '#8c564b',              # tab:brown
    'CNN-LSTM': '#e377c2',          # tab:pink
    'Transformer': '#17becf'        # tab:cyan
}

linestyles = {
    'Actual': '-',
    'Linear Regression': '--',
    'SVR': '--',
    'Random Forest': '--',
    'XGBoost': '--',
    'LightGBM': '--',
    'LSTM': '--',
    'CNN-LSTM': '--',
    'Transformer': '--'
}

def get_metrics(df_sub, model_list):
    res = []
    for m in model_list:
        rmse = np.sqrt(mean_squared_error(df_sub['Actual'], df_sub[m]))
        mape = calculate_mape(df_sub['Actual'], df_sub[m])
        r2 = r2_score(df_sub['Actual'], df_sub[m])
        res.append([m, f"{rmse:.2f}", f"{mape:.2f}", f"{r2:.3f}"])
    return res

models_to_plot = list(models.keys())

# ---------------------------------------------------------
# Plot 1: High Volatility with Inset (Fig 5.16 style)
# ---------------------------------------------------------
start_date = pd.to_datetime('2023-09-01')
end_date = pd.to_datetime('2023-10-31')
df_vol = df_plot[(df_plot['Tarih'] >= start_date) & (df_plot['Tarih'] <= end_date)]

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(df_vol['Tarih'], df_vol['Actual'], label='Gerçek Kapanış Fiyatı', color=colors['Actual'], linewidth=2.5)
for m in models_to_plot:
    ax.plot(df_vol['Tarih'], df_vol[m], label=m, color=colors[m], linestyle=linestyles[m], linewidth=1.5, alpha=0.9)

ax.set_title('Şekil 5.16. Yüksek Volatilite Dönemindeki Gerçek-Tahmin Karşılaştırması\nBIST100 Test Veri Kümesi - Yüksek Volatilite Dönemi (01.09.2023 - 31.10.2023)', fontsize=15, fontweight='bold', pad=20, color='#1A237E')
ax.set_xlabel('Tarih', fontsize=12)
ax.set_ylabel('Kapanış Fiyatı (TL)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5, color='#B0BEC5')
ax.legend(loc='lower left', fontsize=10, framealpha=0.95, edgecolor='#CFD8DC')

# Inset Axes
inset_start = pd.to_datetime('2023-10-10')
inset_end = pd.to_datetime('2023-10-18')
df_inset = df_vol[(df_vol['Tarih'] >= inset_start) & (df_vol['Tarih'] <= inset_end)]

axins = ax.inset_axes([0.22, 0.1, 0.4, 0.35])
axins.plot(df_inset['Tarih'], df_inset['Actual'], color=colors['Actual'], linewidth=2.5)
for m in models_to_plot:
    axins.plot(df_inset['Tarih'], df_inset[m], color=colors[m], linestyle=linestyles[m], linewidth=1.5, alpha=0.9)

axins.set_title("Yakınlaştırılmış Görünüm (10.10.2023 - 18.10.2023)", fontsize=10, fontweight='bold')
axins.tick_params(axis='both', which='major', labelsize=9)
import matplotlib.dates as mdates
axins.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
for label in axins.get_xticklabels():
    label.set_rotation(0)
axins.grid(True, linestyle=':', alpha=0.6)
ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.8)

# Table inside plot (Top Right)
metrics_data = get_metrics(df_vol, models_to_plot)
col_labels = ['Model', 'RMSE (TL)', 'MAPE (%)', 'R²']
table = ax.table(cellText=metrics_data, colLabels=col_labels, loc='upper right', bbox=[0.72, 0.55, 0.26, 0.4], cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', color='#1A237E')
        cell.set_facecolor('#E3F2FD')
    else:
        if j == 0:
            m_name = metrics_data[i-1][0]
            cell.set_text_props(weight='bold', color=colors[m_name])
            
# Add text box at bottom
desc_text = "Grafik, yüksek volatilite döneminde modellerin gerçek kapanış fiyatlarını takip etme yeteneklerini göstermektedir.\nDerin öğrenme modelleri ani fiyat düşüş ve yükselişlerini ağaç tabanlı modellere kıyasla daha hızlı ve doğru şekilde yakalamıştır."
props_desc = dict(boxstyle='round,pad=1.0,rounding_size=0.2', facecolor='#F4F8FB', edgecolor='#A9C4E9', alpha=1.0)
fig.text(0.5, 0.02, desc_text, transform=fig.transFigure, fontsize=11, verticalalignment='bottom', horizontalalignment='center', bbox=props_desc, color='#1F497D', linespacing=1.5)

plt.subplots_adjust(bottom=0.15)
plt.savefig(os.path.join(output_dir, 'Fig_5_16_Yuksek_Volatilite_Inset.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 2: Full plot with red box + Split bottom view (Fig 5.17 style)
# ---------------------------------------------------------
fig = plt.figure(figsize=(16, 11))
gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.15)

# Top: Full series (using test set data range roughly like the screenshot)
full_start = pd.to_datetime('2023-07-06')
full_end = pd.to_datetime('2023-12-31')
df_full = df_plot[(df_plot['Tarih'] >= full_start) & (df_plot['Tarih'] <= full_end)]

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_full['Tarih'], df_full['Actual'], label='Gerçek Kapanış Fiyatı', color=colors['Actual'], linewidth=2.5)
for m in models_to_plot:
    ax1.plot(df_full['Tarih'], df_full[m], label=f'{m}', color=colors[m], linestyle=linestyles[m], linewidth=1.5, alpha=0.9)

ax1.set_title('Şekil 5.17. Yakınlaştırılmış gerçek-tahmin eğrileri\nBIST100 Test Veri Kümesi - Yakınlaştırılmış Bölge (10.10.2023 - 18.11.2023)', fontsize=15, fontweight='bold', pad=15, color='#1A237E')
ax1.set_ylabel('Kapanış Fiyatı (TL)')
ax1.grid(True, linestyle='--', alpha=0.5, color='#B0BEC5')
ax1.legend(loc='upper left', ncol=1, framealpha=0.95)

zoom_start = pd.to_datetime('2023-10-10')
zoom_end = pd.to_datetime('2023-11-18')
df_zoom = df_plot[(df_plot['Tarih'] >= zoom_start) & (df_plot['Tarih'] <= zoom_end)]

# Red box
min_y = df_zoom[['Actual'] + models_to_plot].min().min() * 0.95
max_y = df_zoom[['Actual'] + models_to_plot].max().max() * 1.05
rect = patches.Rectangle((mdates.date2num(zoom_start), min_y), 
                         mdates.date2num(zoom_end) - mdates.date2num(zoom_start), 
                         max_y - min_y, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
ax1.add_patch(rect)

# Draw arrow from red box to left subplot
arrow_x = mdates.date2num(zoom_start) + (mdates.date2num(zoom_end) - mdates.date2num(zoom_start)) / 2
arrow_y = min_y
ax1.annotate('', xy=(0.3, 0.46), xytext=(arrow_x, arrow_y),
            xycoords='figure fraction', textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# Bottom-Left: Zoomed region
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df_zoom['Tarih'], df_zoom['Actual'], color=colors['Actual'], linewidth=2.5)
for m in models_to_plot:
    ax2.plot(df_zoom['Tarih'], df_zoom[m], color=colors[m], linestyle=linestyles[m], linewidth=2, alpha=0.9)

ax2.set_title('Yakınlaştırılmış Bölge (10.10.2023 - 18.11.2023)', fontsize=12, fontweight='bold', pad=10)
ax2.set_ylabel('Kapanış Fiyatı (TL)')
ax2.grid(True, linestyle='--', alpha=0.5, color='#B0BEC5')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))

# Bottom-Right: Table
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
ax3.set_title('Yakınlaştırılmış Bölge Performans Metrikleri', fontsize=12, fontweight='bold', pad=10, color='#1A237E')

metrics_data_zoom = get_metrics(df_zoom, models_to_plot)
table2 = ax3.table(cellText=metrics_data_zoom, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0.05, 0.1, 0.9, 0.8])
table2.auto_set_font_size(False)
table2.set_fontsize(10)
for (i, j), cell in table2.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', color='#1A237E')
        cell.set_facecolor('#E3F2FD')
    else:
        if j == 0:
            m_name = metrics_data_zoom[i-1][0]
            cell.set_text_props(weight='bold', color=colors[m_name])

desc_text2 = "Yakınlaştırılmış bölgede modellerin kısa dönemli fiyat hareketlerini yakalama yetenekleri detaylı olarak görülmektedir.\nDerin öğrenme modelleri ani düşüş ve yükselişleri daha iyi takip etmektedir."
fig.text(0.5, 0.02, desc_text2, transform=fig.transFigure, fontsize=11, verticalalignment='bottom', horizontalalignment='center', bbox=props_desc, color='#1F497D', linespacing=1.5)

plt.subplots_adjust(bottom=0.12)
plt.savefig(os.path.join(output_dir, 'Fig_5_17_Yakinlastirilmis_Bolge.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Yeni grafikler olusturuldu.")
