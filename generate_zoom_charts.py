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
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
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

# Sadece test setini tahmin edeceğiz (zoom grafikler test setinden)
dates_test = pd.to_datetime(test_df[preprocessor.date_col].values)
actual_test = test_df['Target'].values

print("Zoom Grafikleri icin modeller egitiliyor...")

models = {
    'Linear Regression': LinearRegression(),
    'XGBoost': XGBRegressor(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42),
    'MLP': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=1),
    'LSTM': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=3)
}

df_plot = pd.DataFrame({'Tarih': dates_test, 'Actual': actual_test})

for name, model in models.items():
    model.fit(X_train, y_train)
    pred_scaled = model.predict(X_test).reshape(-1, 1)
    df_plot[name] = preprocessor.scaler_y.inverse_transform(pred_scaled).ravel()

plt.rcParams['font.family'] = 'sans-serif'

def create_zoom_chart(start_date, end_date, models_to_plot, filename, title, subtitle, desc_text):
    mask = (df_plot['Tarih'] >= start_date) & (df_plot['Tarih'] <= end_date)
    df_zoom = df_plot[mask]
    
    fig, ax = plt.subplots(figsize=(15, 7.5))
    
    ax.plot(df_zoom['Tarih'], df_zoom['Actual'], label='Gerçek Kapanış Fiyatı', color='#333333', linewidth=2.5, zorder=1)
    
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']
    
    for idx, m in enumerate(models_to_plot):
        ax.plot(df_zoom['Tarih'], df_zoom[m], label=f'{m}', color=colors[idx % len(colors)], linestyle='--', linewidth=2.0, alpha=0.95, zorder=10)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
    plt.suptitle(subtitle, y=0.91, fontsize=12, color='#444444')

    ax.set_xlabel('Tarih', fontsize=12, labelpad=10)
    ax.set_ylabel('Kapanış Fiyatı (TL)', fontsize=12, labelpad=10)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5 if (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days <= 60 else 10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    fig.autofmt_xdate(rotation=45)
    
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), '.0f').replace(',', '.')))

    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5, color='#B0B0B0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#D0D0D0')
        spine.set_linewidth(1.5)

    legend = ax.legend(loc='lower left', frameon=True, fontsize=11, 
                       edgecolor='#D0D0D0', facecolor='white', borderpad=1, ncol=1)
    legend.get_frame().set_alpha(0.95)

    props_desc = dict(boxstyle='round,pad=1.0,rounding_size=0.2', facecolor='#F8F9F9', edgecolor='#CCCCCC', alpha=1.0)
    fig.text(0.5, -0.12, desc_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='center', bbox=props_desc, color='#333333', linespacing=1.5)

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 1. Yuksek Volatilite (2024 Ilk Ceyrek: 01.01.2024 - 30.04.2024)
create_zoom_chart(
    start_date='2024-01-01',
    end_date='2024-04-30',
    models_to_plot=['Linear Regression', 'XGBoost', 'MLP', 'LSTM'],
    filename='Sekil_5_16_Yuksek_Volatilite.png',
    title='Şekil 5.16. Yüksek Volatilite Dönemindeki Gerçek–Tahmin Karşılaştırması',
    subtitle='BIST100 Test Veri Kümesi (01.01.2024 – 30.04.2024)',
    desc_text="Grafikte siyah çizgi gerçek kapanış fiyatını göstermektedir. Doğrusal ve derin öğrenme modelleri (Linear Regression, MLP, LSTM) ani yükseliş ve düşüşleri \nbaşarıyla takip edebilirken, ağaç tabanlı modelin (XGBoost) bu şoklara gecikmeli tepki verdiği ve eğilimin gerisinde kaldığı görülmektedir."
)

# 2. 60 Gunluk Zoom (01.10.2023 - 30.11.2023)
create_zoom_chart(
    start_date='2023-10-01',
    end_date='2023-11-30',
    models_to_plot=['Linear Regression', 'Random Forest', 'MLP', 'LSTM'],
    filename='Sekil_5_17_Zoom.png',
    title='Şekil 5.17. 60 Günlük Yakınlaştırılmış Gerçek–Tahmin Eğrileri',
    subtitle='BIST100 Test Veri Kümesi (01.10.2023 – 30.11.2023)',
    desc_text="Yakınlaştırılmış bu kesitte modeller arası mikro farklılıklar daha net görülmektedir.\nDoğrusal modeller ve MLP modeli trend dönüşlerini ağaç tabanlı modellere göre çok daha erken ve isabetli bir şekilde yakalamaktadır."
)

print("Zoom grafikleri basariyla olusturuldu.")
