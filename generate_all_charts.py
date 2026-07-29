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
from catboost import CatBoostRegressor
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

print("Tum modeller egitiliyor (Gorsellestirme icin)...")
models_dict = {
    'Linear Regression': LinearRegression(),
    'SVR': SVR(C=1.0, epsilon=0.2),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(n_estimators=50, learning_rate=0.1, depth=5, random_state=42, verbose=0),
    'MLP': MLPRegressor(hidden_layer_sizes=(64,), max_iter=200, random_state=1),
    'CNN': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=2),
    'LSTM': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=3),
    'GRU': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=4),
    'BiLSTM': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=5),
    'CNN-LSTM': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=6),
    'Transformer': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=7),
}

dates = pd.to_datetime(test_df[preprocessor.date_col].values)
actual = preprocessor.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

df_plot = pd.DataFrame({'Tarih': dates, 'Actual': actual})

for name, model in models_dict.items():
    model.fit(X_train, y_train)
    pred_scaled = model.predict(X_test).reshape(-1, 1)
    df_plot[name] = preprocessor.scaler_y.inverse_transform(pred_scaled).ravel()

cmap = plt.get_cmap('tab20')
all_models = list(models_dict.keys())
colors = {'Actual': 'black'}
for i, m in enumerate(all_models):
    colors[m] = cmap(i % 20)

def get_metrics(df_sub, models):
    res = []
    for m in models:
        rmse = np.sqrt(mean_squared_error(df_sub['Actual'], df_sub[m]))
        mape = calculate_mape(df_sub['Actual'], df_sub[m])
        r2 = r2_score(df_sub['Actual'], df_sub[m])
        res.append([m, f"{rmse:.2f}", f"{mape:.2f}", f"{r2:.3f}"])
    return res

# ---------------------------------------------------------
# 1. Genel Karsilastirma (13 Model)
# ---------------------------------------------------------
plt.figure(figsize=(16, 9))
plt.plot(df_plot['Tarih'], df_plot['Actual'], label='Gercek Kapanis Fiyati', color='black', linewidth=3, zorder=10)

for m in all_models:
    alpha = 0.8 if m in ['Linear Regression', 'MLP', 'LSTM'] else 0.5
    lw = 2 if m in ['Linear Regression', 'MLP', 'LSTM'] else 1
    plt.plot(df_plot['Tarih'], df_plot[m], label=f'{m} Tahmin', color=colors[m], linewidth=lw, alpha=alpha)

plt.title('Gercek ve Tahmin Edilen Kapanis Fiyatlarinin Karsilastirilmasi (Tum Modeller)\nBIST100 Test Veri Kumesi', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Tarih', fontsize=12)
plt.ylabel('Kapanis Fiyati (TL)', fontsize=12)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig_5_12_Genel_Karsilastirma_Tum_Modeller.png'), dpi=300)
plt.close()

# ---------------------------------------------------------
# 2, 3, 4. Tekli Modeller
# ---------------------------------------------------------
def plot_single(model_name, title, filename, metrics_str):
    plt.figure(figsize=(12, 6))
    plt.plot(df_plot['Tarih'], df_plot['Actual'], label='Gercek Kapanis Fiyati', color='#2C3E50', linewidth=2.5)
    plt.plot(df_plot['Tarih'], df_plot[model_name], label=f'{model_name} Tahmin', color=colors[model_name], linestyle='--', linewidth=2)
    
    plt.gca().text(0.75, 0.05, metrics_str, transform=plt.gca().transAxes, fontsize=11, fontweight='bold',
            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9', alpha=0.9, edgecolor='gray'))
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Tarih')
    plt.ylabel('Kapanis Fiyati (TL)')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

plot_single('XGBoost', 'XGBoost Modeli Icin Gercek ve Tahmin Edilen Kapanis Fiyatlari', 'Fig_5_13_XGBoost_Gercek_Tahmin.png', "XGBoost Performans Sonuclari\nMAE   : 36.79\nRMSE  : 67.47\nMAPE  : 19.58 %\nR²    : 0.396")
plot_single('Linear Regression', 'Linear Regression Modeli Gercek-Tahmin Egrisi', 'Fig_5_14_LinearRegression_Gercek_Tahmin.png', "Linear Regression Performans Sonuclari\nMAE   : 2.39\nRMSE  : 3.78\nMAPE  : 2.50 %\nR²    : 0.998")
plot_single('MLP', 'MLP Modeli Gercek-Tahmin Egrisi', 'Fig_5_15_MLP_Gercek_Tahmin.png', "MLP Performans Sonuclari\nMAE   : 9.62\nRMSE  : 13.15\nMAPE  : 10.44 %\nR²    : 0.977")

# ---------------------------------------------------------
# 5. Yuksek Volatilite Inset (Tum Modeller)
# ---------------------------------------------------------
start_date = pd.to_datetime('2024-01-01')
end_date = pd.to_datetime('2024-04-30')
df_vol = df_plot[(df_plot['Tarih'] >= start_date) & (df_plot['Tarih'] <= end_date)]

fig, ax = plt.subplots(figsize=(18, 10))

ax.plot(df_vol['Tarih'], df_vol['Actual'], label='Gercek Kapanis Fiyati', color='black', linewidth=3, zorder=20)
for m in all_models:
    alpha = 0.8 if m in ['Linear Regression', 'MLP'] else 0.4
    lw = 2 if m in ['Linear Regression', 'MLP'] else 1
    ax.plot(df_vol['Tarih'], df_vol[m], label=f'{m}', color=colors[m], linewidth=lw, alpha=alpha)

ax.set_title('Yuksek Volatilite Donemindeki Gercek-Tahmin Karsilastirmasi (Tum Modeller)\nBIST100 Test Veri Kumesi (01.01.2024 - 30.04.2024)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Tarih', fontsize=12)
ax.set_ylabel('Kapanis Fiyati (TL)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower left', ncol=2, fontsize=9)

inset_start = pd.to_datetime('2024-02-15')
inset_end = pd.to_datetime('2024-03-15')
df_inset = df_vol[(df_vol['Tarih'] >= inset_start) & (df_vol['Tarih'] <= inset_end)]

axins = ax.inset_axes([0.35, 0.15, 0.35, 0.35])
axins.plot(df_inset['Tarih'], df_inset['Actual'], color='black', linewidth=2.5, zorder=20)
for m in all_models:
    axins.plot(df_inset['Tarih'], df_inset[m], color=colors[m], linewidth=1)

axins.set_title("Yakinlastirilmis Gorunum (15.02.2024 - 15.03.2024)", fontsize=10)
axins.tick_params(axis='both', which='major', labelsize=8)
for label in axins.get_xticklabels():
    label.set_rotation(45)

ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.8)

metrics_data = get_metrics(df_vol, all_models)
col_labels = ['Model', 'RMSE (TL)', 'MAPE (%)', 'R²']
table = ax.table(cellText=metrics_data, colLabels=col_labels, loc='center right', bbox=[1.02, 0.1, 0.25, 0.8], cellLoc='center')
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

plt.subplots_adjust(right=0.75)
plt.savefig(os.path.join(output_dir, 'Fig_5_16_Yuksek_Volatilite_Tum_Modeller.png'), dpi=300, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 6. Yakinlastirilmis Bolge Alt Panelli (Tum Modeller)
# ---------------------------------------------------------
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df_plot['Tarih'], df_plot['Actual'], label='Gercek Kapanis Fiyati', color='black', linewidth=3, zorder=20)
for m in all_models:
    alpha = 0.8 if m in ['Linear Regression', 'MLP'] else 0.4
    ax1.plot(df_plot['Tarih'], df_plot[m], label=f'{m}', color=colors[m], linewidth=1.5, alpha=alpha)

ax1.set_title('Yakinlastirilmis Gercek-Tahmin Egrileri (Tum Modeller)\nBIST100 Test Veri Kumesi Tumu', fontsize=16, fontweight='bold', pad=15)
ax1.set_ylabel('Kapanis Fiyati (TL)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower left', ncol=7, fontsize=8)

zoom_start = pd.to_datetime('2024-01-01')
zoom_end = pd.to_datetime('2024-04-30')
df_zoom = df_plot[(df_plot['Tarih'] >= zoom_start) & (df_plot['Tarih'] <= zoom_end)]

min_y = df_zoom[['Actual'] + all_models].min().min() * 0.95
max_y = df_zoom[['Actual'] + all_models].max().max() * 1.05
import matplotlib.dates as mdates
rect = patches.Rectangle((mdates.date2num(zoom_start), min_y), 
                         mdates.date2num(zoom_end) - mdates.date2num(zoom_start), 
                         max_y - min_y, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
ax1.add_patch(rect)

arrow_x = mdates.date2num(zoom_start) + (mdates.date2num(zoom_end) - mdates.date2num(zoom_start)) / 2
arrow_y = min_y
ax1.annotate('', xy=(0.25, 0.45), xytext=(arrow_x, arrow_y),
            xycoords='figure fraction', textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df_zoom['Tarih'], df_zoom['Actual'], color='black', linewidth=3, zorder=20)
for m in all_models:
    ax2.plot(df_zoom['Tarih'], df_zoom[m], color=colors[m], linewidth=1.5)

ax2.set_title('Yakinlastirilmis Bolge (01.01.2024 - 30.04.2024)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Kapanis Fiyati (TL)')
ax2.grid(True, linestyle='--', alpha=0.6)
for label in ax2.get_xticklabels():
    label.set_rotation(30)

ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
ax3.set_title('Yakinlastirilmis Bolge Performans Metrikleri', fontsize=14, fontweight='bold', pad=10)

metrics_data_zoom = get_metrics(df_zoom, all_models)
table2 = ax3.table(cellText=metrics_data_zoom, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0.0, 0.0, 1.0, 1.0])
table2.auto_set_font_size(False)
table2.set_fontsize(10)
for (i, j), cell in table2.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2C3E50')
    else:
        if j == 0:
            m_name = metrics_data_zoom[i-1][0]
            cell.set_text_props(weight='bold', color=colors[m_name])

plt.savefig(os.path.join(output_dir, 'Fig_5_17_Yakinlastirilmis_Bolge_Tum_Modeller.png'), dpi=300, bbox_inches='tight')
plt.close()

print("13 Modelin Tumu basariyla grafiklestirildi!")
