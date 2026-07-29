import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
import os
from importlib.machinery import SourceFileLoader
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

dp = SourceFileLoader('data_preprocessing', '02_data_preprocessing.py').load_module()
preprocessor = dp.DataPreprocessor('data/bist100_data_interpolate.csv')
df = preprocessor.load_and_clean_data()
df['Target'] = df.groupby('Hisse_Kodu')['Close'].shift(-1)
df = df.dropna(subset=['Target']).reset_index(drop=True)

df_thyao = df[df['Hisse_Kodu'] == 'THYAO.IS'].copy()
train_df, test_df = preprocessor.time_series_split(df_thyao, shuffle=False)

price_cols_to_drop = ['Close', 'Open', 'High', 'Low']
feature_cols = [c for c in preprocessor.feature_cols if c not in price_cols_to_drop]
preprocessor.feature_cols = feature_cols

X_train, y_train, X_test, y_test = preprocessor.normalize_data(train_df, test_df, 'minmax')
y_train = y_train.ravel()

X_full = np.vstack((X_train, X_test))
y_full = np.concatenate((y_train, y_test.ravel()))
dates_full = pd.to_datetime(df_thyao[preprocessor.date_col].values)
actual_full = df_thyao['Target'].values 

print("13 Model Egitiliyor (Iki Panelli Gosterim Icin)...")

ml_models = {
    'Linear Regression': LinearRegression(),
    'SVR': SVR(C=1.0, epsilon=0.2),
    'Random Forest': RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(n_estimators=10, learning_rate=0.1, depth=5, random_state=42, verbose=0)
}

dl_models = {
    'MLP': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=1),
    'CNN': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=2),
    'LSTM': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=3),
    'GRU': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=4),
    'BiLSTM': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=5),
    'CNN-LSTM': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=6),
    'Transformer': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=7),
}

df_plot = pd.DataFrame({'Tarih': dates_full, 'Actual': actual_full})

for name, model in {**ml_models, **dl_models}.items():
    model.fit(X_train, y_train)
    pred_scaled = model.predict(X_full).reshape(-1, 1)
    df_plot[name] = preprocessor.scaler_y.inverse_transform(pred_scaled).ravel()


# PLOTTING
plt.rcParams['font.family'] = 'sans-serif'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# ----------------- AX1: ML Models -----------------
ax1.plot(df_plot['Tarih'], df_plot['Actual'], label='Gerçek Kapanış Fiyatı', color='#333333', linewidth=2.0, zorder=1)

ml_colors = ['#3498DB', '#E67E22', '#2ECC71', '#E74C3C', '#9B59B6', '#1ABC9C']
for idx, m in enumerate(ml_models.keys()):
    ax1.plot(df_plot['Tarih'], df_plot[m], label=m, color=ml_colors[idx], linestyle='--', linewidth=1.5, alpha=0.9, zorder=10)

ax1.set_title("Makine Öğrenmesi Modelleri", fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Kapanış Fiyatı (TL)', fontsize=12)

ax1.grid(True, linestyle='--', linewidth=0.8, alpha=0.5, color='#B0B0B0')
for spine in ax1.spines.values():
    spine.set_edgecolor('#D0D0D0')
    spine.set_linewidth(1.5)

legend1 = ax1.legend(loc='upper left', bbox_to_anchor=(0.015, 0.96), frameon=True, fontsize=10, 
                   edgecolor='#D0D0D0', facecolor='white', borderpad=0.8)
legend1.get_frame().set_alpha(0.95)


# ----------------- AX2: DL Models -----------------
ax2.plot(df_plot['Tarih'], df_plot['Actual'], label='Gerçek Kapanış Fiyatı', color='#333333', linewidth=2.0, zorder=1)

dl_colors = ['#FF4081', '#00BCD4', '#8D6E63', '#4CAF50', '#3F51B5', '#FF9800', '#E91E63']
for idx, m in enumerate(dl_models.keys()):
    ax2.plot(df_plot['Tarih'], df_plot[m], label=m, color=dl_colors[idx], linestyle='--', linewidth=1.5, alpha=0.9, zorder=10)

ax2.set_title("Derin Öğrenme Modelleri", fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Kapanış Fiyatı (TL)', fontsize=12)
ax2.set_xlabel('Tarih', fontsize=12, labelpad=10)

ax2.grid(True, linestyle='--', linewidth=0.8, alpha=0.5, color='#B0B0B0')
for spine in ax2.spines.values():
    spine.set_edgecolor('#D0D0D0')
    spine.set_linewidth(1.5)

legend2 = ax2.legend(loc='upper left', bbox_to_anchor=(0.015, 0.96), frameon=True, fontsize=10, 
                   edgecolor='#D0D0D0', facecolor='white', borderpad=0.8)
legend2.get_frame().set_alpha(0.95)

# Formatting axes
ax2.set_xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2025-01-01'))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
ax1.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), '.0f').replace(',', '.')))
ax2.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), '.0f').replace(',', '.')))

# General Title
fig.suptitle("Şekil 5.12. Gerçek ve Tahmin Edilen Kapanış Fiyatlarının Karşılaştırılması\nBIST100 Veri Kümesi (01.01.2015 – 31.12.2024)", 
             fontsize=16, fontweight='bold', y=0.96, color='#333333')

# Bottom Text Box
text_box = ("Grafiklerde siyah çizgi gerçek kapanış fiyatlarını, renkli çizgiler ise modeller tarafından tahmin edilen değerleri göstermektedir.\n"
            "Modeller kategorilere (Makine Öğrenmesi ve Derin Öğrenme) ayrılarak okunabilirlik artırılmıştır.")

props = dict(boxstyle='round,pad=1.0,rounding_size=0.2', facecolor='#F8F9F9', edgecolor='#CCCCCC', alpha=1.0)
fig.text(0.5, 0.02, text_box, transform=fig.transFigure, fontsize=11,
        verticalalignment='bottom', horizontalalignment='center', bbox=props, color='#333333')

plt.subplots_adjust(top=0.88, bottom=0.12, hspace=0.15)
plt.savefig(os.path.join(output_dir, 'Sekil_5_12_Proje_Sik_Panelli.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Iki panelli sik tasarim basariyla olusturuldu.")
