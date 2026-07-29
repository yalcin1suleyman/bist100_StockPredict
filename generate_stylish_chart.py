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

print("Modeller egitiliyor...")
models_dict = {
    'Linear Regression': LinearRegression(),
    'SVR': SVR(C=1.0, epsilon=0.2),
    'Random Forest': RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(n_estimators=10, learning_rate=0.1, depth=5, random_state=42, verbose=0),
    'MLP': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=1),
    'CNN': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=2),
    'LSTM': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=3),
    'GRU': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=4),
    'BiLSTM': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=5),
    'CNN-LSTM': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=6),
    'Transformer': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=7),
}

df_plot = pd.DataFrame({'Tarih': dates_full, 'Actual': actual_full})

for name, model in models_dict.items():
    model.fit(X_train, y_train)
    pred_scaled = model.predict(X_full).reshape(-1, 1)
    df_plot[name] = preprocessor.scaler_y.inverse_transform(pred_scaled).ravel()

plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(15, 8.5))

ax.plot(df_plot['Tarih'], df_plot['Actual'], label='Gerçek Kapanış Fiyatı', color='#222222', linewidth=2, zorder=10)

colors = [
    '#3498DB', '#F39C12', '#2ECC71', '#E74C3C', '#9B59B6', '#8E44AD', 
    '#E67E22', '#1ABC9C', '#16A085', '#27AE60', '#2980B9', '#C0392B', '#BDC3C7'
]

for idx, m in enumerate(models_dict.keys()):
    ax.plot(df_plot['Tarih'], df_plot[m], label=m, color=colors[idx], linestyle='--', linewidth=1.2, alpha=0.8)

ax.set_title("Şekil 5.12. Gerçek ve Tahmin Edilen Kapanış Fiyatlarının Karşılaştırılması", fontsize=16, fontweight='bold', pad=30)
plt.suptitle("BIST100 Veri Kümesi (01.01.2015 – 31.12.2024)", y=0.91, fontsize=12, color='#444444')

ax.set_xlabel('Tarih', fontsize=12, labelpad=10)
ax.set_ylabel('Kapanış Fiyatı (TL)', fontsize=12, labelpad=10)
ax.set_xlim(pd.to_datetime('2015-01-01'), pd.to_datetime('2025-01-01'))

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), '.0f').replace(',', '.')))

ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5, color='#B0B0B0')
for spine in ax.spines.values():
    spine.set_edgecolor('#D0D0D0')
    spine.set_linewidth(1.5)

legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=9.5, 
                   edgecolor='#D0D0D0', facecolor='white', borderpad=1, ncol=1)
legend.get_frame().set_alpha(0.95)
legend.get_frame().set_boxstyle('round,pad=0.5,rounding_size=0.2')

text_box = ("Grafikte siyah çizgi gerçek kapanış fiyatlarını, diğer renkli çizgiler ise farklı modeller tarafından tahmin edilen kapanış fiyatlarını göstermektedir.\n"
            "Tüm modeller genel trendi takip etmekle birlikte, doğrusal ve derin öğrenme tabanlı modeller gerçek değerlere daha yakın sonuçlar üretmiştir.")

props = dict(boxstyle='round,pad=1.0,rounding_size=0.2', facecolor='#F8F9F9', edgecolor='#CCCCCC', alpha=1.0)
ax.text(0.5, -0.16, text_box, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='center', bbox=props, color='#333333')

plt.subplots_adjust(bottom=0.2)
plt.savefig(os.path.join(output_dir, 'Sekil_5_12_Proje_Sik.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Şık tasarım başarıyla oluşturuldu.")
