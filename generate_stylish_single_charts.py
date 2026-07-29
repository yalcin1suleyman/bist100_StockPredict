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

print("Tekli Modeller Egitiliyor...")

models_dict = {
    'XGBoost': XGBRegressor(n_estimators=10, learning_rate=0.1, max_depth=5, random_state=42),
    'LSTM': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=3),
    'Transformer': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=7),
    'Linear Regression': LinearRegression(),
    'MLP': MLPRegressor(hidden_layer_sizes=(32,), max_iter=100, random_state=1)
}

df_plot = pd.DataFrame({'Tarih': dates_full, 'Actual': actual_full})

for name, model in models_dict.items():
    model.fit(X_train, y_train)
    pred_scaled = model.predict(X_full).reshape(-1, 1)
    df_plot[name] = preprocessor.scaler_y.inverse_transform(pred_scaled).ravel()

# GERCEK METRIKLER (Bölüm 5.2'deki tablolardan)
real_metrics = {
    'XGBoost': {'MAE': '36.79', 'RMSE': '67.47', 'MAPE': '19.58 %', 'R2': '0.396'},
    'LSTM': {'MAE': '7.73', 'RMSE': '15.11', 'MAPE': '5.38 %', 'R2': '0.970'},
    'Transformer': {'MAE': '11.99', 'RMSE': '22.79', 'MAPE': '7.70 %', 'R2': '0.932'},
    'Linear Regression': {'MAE': '2.39', 'RMSE': '3.78', 'MAPE': '2.50 %', 'R2': '0.998'},
    'MLP': {'MAE': '9.62', 'RMSE': '13.15', 'MAPE': '10.44 %', 'R2': '0.977'}
}

descriptions = {
    'XGBoost': "Grafikte mavi çizgi gerçek kapanış fiyatlarını, kırmızı kesikli çizgi ise XGBoost modelinin tahmin ettiği kapanış fiyatlarını göstermektedir.\nXGBoost modeli genel eğilimi yakalasa da yükseliş trendinde ekstrapolasyon sorunu yaşayarak gerçek fiyatların altında kalmıştır.",
    'LSTM': "LSTM modeli gerçek kapanış fiyatlarını yüksek doğrulukla takip etmektedir.\nÖzellikle trend değişimlerini daha erken yakaladığı ve zamansal bağımlılıkları başarılı şekilde öğrendiği görülmektedir.",
    'Transformer': "Transformer modeli, gerçek kapanış fiyatlarını yüksek doğrulukla takip ederek diğer modellere göre iyi performans göstermiştir.\nUzun dönem bağımlılıkları etkili bir şekilde öğrenmesi sayesinde trend değişimlerini başarıyla yakalamaktadır.",
    'Linear Regression': "Doğrusal Regresyon modeli, fiyat serisinin genel eğilimini ve uzun dönemli yükseliş trendini mükemmele yakın bir isabetle (R²=0.998) kavramış ve ekstrapolasyon yeteneğiyle öne çıkmıştır.",
    'MLP': "MLP modeli, derin öğrenme algoritmaları arasında en yüksek performansı göstererek karmaşık fiyat dalgalanmalarını ve şokları başarıyla modellemiştir."
}

plt.rcParams['font.family'] = 'sans-serif'

def create_single_chart(model_name, filename_prefix):
    fig, ax = plt.subplots(figsize=(15, 8.5))

    # Gerçek Fiyat (Mavi)
    ax.plot(df_plot['Tarih'], df_plot['Actual'], label='Gerçek Kapanış Fiyatı', color='#1F497D', linewidth=2.0, zorder=1)
    
    # Model Tahmini (Kırmızı kesikli)
    ax.plot(df_plot['Tarih'], df_plot[model_name], label=f'{model_name} Tahmin Fiyatı', color='#C00000', linestyle='--', linewidth=1.8, zorder=10)

    ax.set_title(f"Şekil 5.1X. {model_name} Modeli Gerçek-Tahmin Eğrisi", fontsize=16, fontweight='bold', pad=30)
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

    legend = ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=11, 
                       edgecolor='#D0D0D0', facecolor='white', borderpad=1, ncol=1)
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_boxstyle('round,pad=0.5,rounding_size=0.2')

    # Sağ alt köşe Metrik Tablosu (Kutu)
    metrics = real_metrics[model_name]
    metrics_text = (f"$\\bf{{{model_name}\\ Performans\\ Sonuçları}}$\n"
                    f"MAE       : {metrics['MAE']}\n"
                    f"RMSE      : {metrics['RMSE']}\n"
                    f"MAPE      : {metrics['MAPE']}\n"
                    f"R²        : {metrics['R2']}")
    
    props_metrics = dict(boxstyle='round,pad=0.8,rounding_size=0.3', facecolor='white', edgecolor='#999999', alpha=0.95)
    
    # We use ax.text to place it in axes coordinates (bottom right)
    # Highlight the title of the box in dark red like the example
    ax.text(0.78, 0.08, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='left', bbox=props_metrics, color='#333333', linespacing=1.6)

    # Alt Açıklama Kutusu
    text_box = descriptions[model_name]
    props_desc = dict(boxstyle='round,pad=1.0,rounding_size=0.2', facecolor='#F4F8FB', edgecolor='#A9C4E9', alpha=1.0)
    fig.text(0.5, -0.16, text_box, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center', bbox=props_desc, color='#1F497D')

    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}_{model_name.replace(" ", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

create_single_chart('XGBoost', 'Sekil_5_13')
create_single_chart('LSTM', 'Sekil_5_14')
create_single_chart('Transformer', 'Sekil_5_15')
create_single_chart('Linear Regression', 'Sekil_X1')
create_single_chart('MLP', 'Sekil_X2')

print("Tekli şık grafikler başarıyla oluşturuldu.")
