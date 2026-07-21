import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import os

# Uyarıları gizle
import warnings
warnings.filterwarnings("ignore")

# Kendi veri ön işleme sınıfımızı içe aktarıyoruz
from importlib.machinery import SourceFileLoader
data_module = SourceFileLoader("data_preprocessing", "02_data_preprocessing.py").load_module()
DataPreprocessor = data_module.DataPreprocessor

# Grafik ayarları
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def plot_horizontal_bar(data, metric_name, file_name, title):
    # R2 için sıralama büyükten küçüğe, diğerleri için küçükten büyüğe
    ascending = False if metric_name == 'R²' else True
    data_sorted = data.sort_values(by=metric_name, ascending=ascending)
    
    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("Set2", len(data_sorted))
    
    bars = plt.barh(data_sorted['Model'], data_sorted[metric_name], color=colors)
    plt.xlabel(metric_name)
    plt.ylabel('Modeller')
    plt.title(title)
    
    # Barların yanına değerleri yazdır
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                 va='center', ha='left', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()

def train_and_evaluate_ml_models(file_path):
    print("--- 03: HIZLI MAKİNE ÖĞRENMESİ MODELLERİ (TASLAK UYUMLU) ---")
    
    preprocessor = DataPreprocessor(file_path=file_path)
    X_train, y_train, X_test, y_test, y_test_unscaled, scaler_y, feature_cols = preprocessor.get_ml_data(scaler_type='minmax')
    
    # Zaman serisi çizimi için tarihleri alalım
    df = preprocessor.load_and_clean_data()
    # Hedef değişkende shift işlemi olduğu için indeksleri yeniden ayarlayarak split edeceğiz
    df['Target'] = df.groupby('Hisse_Kodu')['Close'].shift(-1)
    df = df.dropna(subset=['Target']).reset_index(drop=True)
    _, test_df = preprocessor.time_series_split(df, shuffle=False)
    test_dates = pd.to_datetime(test_df[preprocessor.date_col]).values
    
    # 2. Modellerin ve Hiperparametre Izgaralarının Tanımlanması (Grid)
    models_and_params = {
        'Linear Regression': {
            'model': LinearRegression(),
            'params': {}
        },
        'SVR': {
            'model': SVR(),
            'params': {'kernel': ['rbf', 'linear'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        },
        'Random Forest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        },
        'XGBoost': {
            'model': XGBRegressor(random_state=42, n_jobs=-1),
            'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
        },
        'LightGBM': {
            'model': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'num_leaves': [31, 50, 100]}
        },
        'CatBoost': {
            'model': CatBoostRegressor(random_state=42, verbose=0),
            'params': {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'depth': [4, 6, 8]}
        }
    }
    
    results = []
    predictions_dict = {}
    
    # 3. Model Eğitimi ve Optimizasyonu
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    
    for name, mp in models_and_params.items():
        print(f"\n[{name}] modeli optimize ediliyor ve eğitiliyor...")
        start_time = time.time()
        
        if not mp['params']:
            best_model = mp['model']
            best_model.fit(X_train, y_train)
            print(f"[{name}] Parametre ızgarası boş, varsayılan parametrelerle eğitildi.")
        else:
            search = RandomizedSearchCV(mp['model'], mp['params'], n_iter=5, cv=tscv, 
                                        scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            print(f"[{name}] En iyi parametreler: {search.best_params_}")
            
        y_pred_scaled = best_model.predict(X_test)
        
        if len(y_pred_scaled.shape) == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled).ravel()
        
        end_time = time.time()
        
        mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
        mape = calculate_mape(y_test_unscaled, y_pred_unscaled)
        r2 = r2_score(y_test_unscaled, y_pred_unscaled)
        
        results.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'R²': r2,
            'Time (s)': end_time - start_time
        })
        
        predictions_dict[name] = y_pred_unscaled
        print(f"[{name}] tamamlandı. R²: {r2:.4f}, RMSE: {rmse:.4f}")

    # 4. Tablo 5.2
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("Tablo 5.2. Makine öğrenmesi performans karşılaştırması")
    print("="*60)
    print(results_df.set_index('Model').round(4))
    print("="*60)
    results_df.to_csv("Table_5_2_ML_Performance.csv", index=False)
    
    # 5. Şekil 5.8 - 5.11 (Yatay Barlar)
    print("Metrik grafikleri (Yatay Bar) çizdiriliyor...")
    plot_horizontal_bar(results_df, 'MAE', 'Fig_5_8_MAE_Comparison.png', 'Şekil 5.8. Modellerin MAE Karşılaştırması')
    plot_horizontal_bar(results_df, 'RMSE', 'Fig_5_9_RMSE_Comparison.png', 'Şekil 5.9. Modellerin RMSE Karşılaştırması')
    plot_horizontal_bar(results_df, 'MAPE (%)', 'Fig_5_10_MAPE_Comparison.png', 'Şekil 5.10. Modellerin MAPE Karşılaştırması')
    plot_horizontal_bar(results_df, 'R²', 'Fig_5_11_R2_Comparison.png', 'Şekil 5.11. Modellerin R² Karşılaştırması')
    
    # 6. Şekil 5.12: Tüm Modeller Gerçek vs Tahmin (Aynı Grafik)
    print("Şekil 5.12 (Tüm modeller gerçek vs tahmin) çizdiriliyor...")
    plt.figure(figsize=(16, 8))
    plt.plot(test_dates, y_test_unscaled, label='Gerçek Kapanış Fiyatı', color='black', linewidth=2)
    
    for name, y_pred in predictions_dict.items():
        plt.plot(test_dates, y_pred, label=f'{name}', linestyle='--', alpha=0.7)
        
    plt.title('Şekil 5.12. Gerçek ve Tahmin Edilen Kapanış Fiyatlarının Karşılaştırılması')
    plt.xlabel('Tarih')
    plt.ylabel('Kapanış Fiyatı (TL)')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Fig_5_12_All_Models_vs_Actual.png', dpi=300)
    plt.close()
    
    # 7. Şekil 5.13: Tekil Modeller (XGBoost örneği)
    print("Şekil 5.13 (Tekil modeller gerçek vs tahmin) çizdiriliyor...")
    for name, y_pred in predictions_dict.items():
        plt.figure(figsize=(14, 6))
        plt.plot(test_dates, y_test_unscaled, label='Gerçek Kapanış Fiyatı', color='black', alpha=0.8)
        plt.plot(test_dates, y_pred, label=f'{name} Tahmin Fiyatı', color='red', linestyle='--', alpha=0.8)
        
        plt.title(f'Şekil 5.13. {name} Modeli İçin Gerçek ve Tahmin Edilen Kapanış Fiyatları')
        plt.xlabel('Tarih')
        plt.ylabel('Kapanış Fiyatı (TL)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrik tablosunu grafiğin bir köşesine iliştirme
        metrics_text = f"MAE: {results_df[results_df['Model']==name]['MAE'].values[0]:.2f}\n" \
                       f"RMSE: {results_df[results_df['Model']==name]['RMSE'].values[0]:.2f}\n" \
                       f"MAPE: {results_df[results_df['Model']==name]['MAPE (%)'].values[0]:.2f}%\n" \
                       f"R²: {results_df[results_df['Model']==name]['R²'].values[0]:.3f}"
        plt.text(0.95, 0.05, metrics_text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                 
        plt.tight_layout()
        # İsimlendirme örneğin: Fig_5_13_XGBoost_vs_Actual.png
        safe_name = name.replace(" ", "_")
        plt.savefig(f'Fig_5_13_{safe_name}_vs_Actual.png', dpi=300)
        plt.close()

if __name__ == "__main__":
    file_path = "bist100_data_interpolate.csv"
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} dosyası bulunamadı.")
    else:
        train_and_evaluate_ml_models(file_path)
