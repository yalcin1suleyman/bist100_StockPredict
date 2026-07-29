import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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

print("Tüm Modeller Egitiliyor (Inset Grafikleri Icin)...")

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

def create_inset_chart(start_date, end_date, inset_start, inset_end, models_dict, filename, title, subtitle, colors, loc='upper left'):
    mask = (df_plot['Tarih'] >= start_date) & (df_plot['Tarih'] <= end_date)
    df_zoom = df_plot[mask]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(df_zoom['Tarih'], df_zoom['Actual'], label='Gerçek Kapanış Fiyatı', color='#333333', linewidth=2.5, zorder=1)
    
    for idx, m in enumerate(models_dict.keys()):
        ax.plot(df_zoom['Tarih'], df_zoom[m], label=m, color=colors[idx % len(colors)], linestyle='--', linewidth=1.5, alpha=0.9, zorder=10)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
    plt.suptitle(subtitle, y=0.91, fontsize=12, color='#444444')

    ax.set_xlabel('Tarih', fontsize=12, labelpad=10)
    ax.set_ylabel('Kapanış Fiyatı (TL)', fontsize=12, labelpad=10)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5 if (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days <= 60 else 10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    fig.autofmt_xdate(rotation=45)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), '.0f').replace(',', '.')))

    ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.5, color='#B0B0B0')
    
    # Inset Axes
    # Use bounding box to place inset: (x0, y0, width, height) in axes coordinates
    if loc == 'upper left':
        axins = inset_axes(ax, width="40%", height="40%", loc='upper left', borderpad=2)
    elif loc == 'lower right':
        axins = inset_axes(ax, width="40%", height="40%", loc='lower right', borderpad=2)
    else:
        axins = inset_axes(ax, width="35%", height="35%", loc=loc, borderpad=2)
        
    mask_inset = (df_plot['Tarih'] >= inset_start) & (df_plot['Tarih'] <= inset_end)
    df_inset = df_plot[mask_inset]
    
    axins.plot(df_inset['Tarih'], df_inset['Actual'], color='#333333', linewidth=2.5, zorder=1)
    for idx, m in enumerate(models_dict.keys()):
        axins.plot(df_inset['Tarih'], df_inset[m], color=colors[idx % len(colors)], linestyle='--', linewidth=1.5, alpha=0.9, zorder=10)
        
    axins.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    axins.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), '.0f')))
    axins.tick_params(axis='x', rotation=45, labelsize=9)
    axins.tick_params(axis='y', labelsize=9)
    axins.grid(True, linestyle=':', linewidth=0.5, alpha=0.5, color='#B0B0B0')
    
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), frameon=True, fontsize=11, 
                       edgecolor='#D0D0D0', facecolor='white', borderpad=1, ncol=4)
    legend.get_frame().set_alpha(0.95)

    plt.subplots_adjust(bottom=0.25)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

ml_colors = ['#3498DB', '#E67E22', '#2ECC71', '#E74C3C', '#9B59B6', '#1ABC9C']
dl_colors = ['#FF4081', '#00BCD4', '#8D6E63', '#4CAF50', '#3F51B5', '#FF9800', '#E91E63']

create_inset_chart(
    start_date='2024-01-01',
    end_date='2024-04-30',
    inset_start='2024-02-15',
    inset_end='2024-03-05',
    models_dict=ml_models,
    filename='Yuksek_Volatilite_ML_Inset.png',
    title='Yüksek Volatilite Dönemi - Makine Öğrenmesi Modelleri',
    subtitle='BIST100 Test Veri Kümesi (01.01.2024 – 30.04.2024)',
    colors=ml_colors,
    loc='lower right'
)

create_inset_chart(
    start_date='2024-01-01',
    end_date='2024-04-30',
    inset_start='2024-02-15',
    inset_end='2024-03-05',
    models_dict=dl_models,
    filename='Yuksek_Volatilite_DL_Inset.png',
    title='Yüksek Volatilite Dönemi - Derin Öğrenme Modelleri',
    subtitle='BIST100 Test Veri Kümesi (01.01.2024 – 30.04.2024)',
    colors=dl_colors,
    loc='lower right'
)

print("İnset'li (yakın görünümlü) grafikler başarıyla oluşturuldu!")
