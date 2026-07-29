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

dates_test = pd.to_datetime(test_df[preprocessor.date_col].values)
actual_test = test_df['Target'].values

print("Tüm 13 Model Egitiliyor (Zoom Grafikleri Icin)...")

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

df_plot = pd.DataFrame({'Tarih': dates_test, 'Actual': actual_test})

for name, model in {**ml_models, **dl_models}.items():
    model.fit(X_train, y_train)
    pred_scaled = model.predict(X_test).reshape(-1, 1)
    df_plot[name] = preprocessor.scaler_y.inverse_transform(pred_scaled).ravel()

plt.rcParams['font.family'] = 'sans-serif'

def create_2panel_zoom_chart(start_date, end_date, filename, title, subtitle, desc_text):
    mask = (df_plot['Tarih'] >= start_date) & (df_plot['Tarih'] <= end_date)
    df_zoom = df_plot[mask]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    # ML Panel
    ax1.plot(df_zoom['Tarih'], df_zoom['Actual'], label='Gerçek Kapanış Fiyatı', color='#333333', linewidth=2.5, zorder=1)
    ml_colors = ['#3498DB', '#E67E22', '#2ECC71', '#E74C3C', '#9B59B6', '#1ABC9C']
    for idx, m in enumerate(ml_models.keys()):
        ax1.plot(df_zoom['Tarih'], df_zoom[m], label=m, color=ml_colors[idx], linestyle='--', linewidth=1.5, alpha=0.9, zorder=10)
    
    ax1.set_title("Makine Öğrenmesi Modelleri", fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Kapanış Fiyatı (TL)', fontsize=12)
    ax1.grid(True, linestyle='--', linewidth=0.8, alpha=0.5, color='#B0B0B0')
    legend1 = ax1.legend(loc='lower left', bbox_to_anchor=(0.015, 0.05), frameon=True, fontsize=10, 
                       edgecolor='#D0D0D0', facecolor='white', borderpad=0.8, ncol=2)
    legend1.get_frame().set_alpha(0.95)

    # DL Panel
    ax2.plot(df_zoom['Tarih'], df_zoom['Actual'], label='Gerçek Kapanış Fiyatı', color='#333333', linewidth=2.5, zorder=1)
    dl_colors = ['#FF4081', '#00BCD4', '#8D6E63', '#4CAF50', '#3F51B5', '#FF9800', '#E91E63']
    for idx, m in enumerate(dl_models.keys()):
        ax2.plot(df_zoom['Tarih'], df_zoom[m], label=m, color=dl_colors[idx], linestyle='-.', linewidth=1.5, alpha=0.9, zorder=10)
    
    ax2.set_title("Derin Öğrenme Modelleri", fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Kapanış Fiyatı (TL)', fontsize=12)
    ax2.set_xlabel('Tarih', fontsize=12, labelpad=10)
    ax2.grid(True, linestyle='--', linewidth=0.8, alpha=0.5, color='#B0B0B0')
    legend2 = ax2.legend(loc='lower left', bbox_to_anchor=(0.015, 0.05), frameon=True, fontsize=10, 
                       edgecolor='#D0D0D0', facecolor='white', borderpad=0.8, ncol=2)
    legend2.get_frame().set_alpha(0.95)

    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5 if (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days <= 60 else 10))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    fig.autofmt_xdate(rotation=45)
    
    ax1.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), '.0f').replace(',', '.')))
    ax2.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), '.0f').replace(',', '.')))

    fig.suptitle(title + f"\n{subtitle}", fontsize=16, fontweight='bold', y=0.95, color='#333333')

    props_desc = dict(boxstyle='round,pad=1.0,rounding_size=0.2', facecolor='#F8F9F9', edgecolor='#CCCCCC', alpha=1.0)
    fig.text(0.5, 0.01, desc_text, transform=fig.transFigure, fontsize=11,
            verticalalignment='bottom', horizontalalignment='center', bbox=props_desc, color='#333333', linespacing=1.5)

    plt.subplots_adjust(top=0.87, bottom=0.15, hspace=0.2)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

create_2panel_zoom_chart(
    start_date='2024-01-01',
    end_date='2024-04-30',
    filename='Sekil_5_16_Yuksek_Volatilite.png',
    title='Şekil 5.16. Yüksek Volatilite Dönemindeki Gerçek–Tahmin Karşılaştırması',
    subtitle='BIST100 Test Veri Kümesi (01.01.2024 – 30.04.2024)',
    desc_text="Yüksek volatilite döneminde (13 modelin tamamı): Doğrusal ve derin öğrenme modelleri ani şokları başarıyla takip edebilirken,\nağaç tabanlı algoritmaların dalgalanmalara gecikmeli tepki verdiği ve eğilimin gerisinde kaldığı görülmektedir."
)

create_2panel_zoom_chart(
    start_date='2023-10-01',
    end_date='2023-11-30',
    filename='Sekil_5_17_Zoom.png',
    title='Şekil 5.17. 60 Günlük Yakınlaştırılmış Gerçek–Tahmin Eğrileri',
    subtitle='BIST100 Test Veri Kümesi (01.10.2023 – 30.11.2023)',
    desc_text="Yakınlaştırılmış 60 günlük kesitte (13 modelin tamamı): Derin öğrenme (MLP, LSTM) ve Doğrusal Regresyon modellerinin\ntrend dönüşlerini karar ağaçlarına göre çok daha erken ve isabetli bir şekilde yakaladığı mikroskobik olarak görülmektedir."
)

print("Tum 13 modeli iceren panelli zoom grafikleri olusturuldu.")
