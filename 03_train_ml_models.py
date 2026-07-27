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
import warnings

warnings.filterwarnings("ignore")

from importlib.machinery import SourceFileLoader
data_module = SourceFileLoader("data_preprocessing", "02_data_preprocessing.py").load_module()
DataPreprocessor = data_module.DataPreprocessor

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    epsilon = np.finfo(np.float64).eps
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def plot_horizontal_bar(data, metric_name, file_name, title):
    ascending = False if metric_name == 'R²' else True
    # Get top 15 to avoid clutter
    data_sorted = data.sort_values(by=metric_name, ascending=ascending).head(15)
    
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("Set2", len(data_sorted))
    bars = plt.barh(data_sorted['Model_Scenario'], data_sorted[metric_name], color=colors)
    plt.xlabel(metric_name)
    plt.ylabel('Modeller ve Senaryolar')
    plt.title(title)
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', ha='left', fontsize=10)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)
    plt.close()

def train_and_evaluate_ml_models(file_path):
    print("--- 03: HIZLI MAKİNE ÖĞRENMESİ MODELLERİ (SENARYO BAZLI) ---")
    
    scenarios = ['raw', 'technical', 'all']
    all_results = []
    
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    
    for scenario in scenarios:
        print(f"\n{'='*50}\nSENARYO: {scenario.upper()}\n{'='*50}")
        preprocessor = DataPreprocessor(file_path=file_path)
        X_train, y_train, X_test, y_test, y_test_unscaled, scaler_y, feature_cols = preprocessor.get_ml_data(scaler_type='minmax', feature_set=scenario)
        
        # Test dates
        df = preprocessor.load_and_clean_data()
        df['Target'] = df.groupby('Hisse_Kodu')['Close'].shift(-1)
        df = df.dropna(subset=['Target']).reset_index(drop=True)
        _, test_df = preprocessor.time_series_split(df, shuffle=False)
        test_dates = pd.to_datetime(test_df[preprocessor.date_col]).values
        
        models_and_params = {
            'Linear Regression': {'model': LinearRegression(), 'params': {}},
            'SVR': {'model': SVR(), 'params': {'kernel': ['rbf', 'linear'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}},
            'Random Forest': {'model': RandomForestRegressor(random_state=42, n_jobs=-1), 'params': {'n_estimators': [50, 100], 'max_depth': [None, 10]}},
            'XGBoost': {'model': XGBRegressor(random_state=42, n_jobs=-1), 'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}},
            'LightGBM': {'model': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1), 'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 50]}},
            'CatBoost': {'model': CatBoostRegressor(random_state=42, verbose=0), 'params': {'iterations': [50, 100], 'learning_rate': [0.01, 0.1], 'depth': [4, 6]}}
        }
        
        predictions_dict = {}
        
        for name, mp in models_and_params.items():
            print(f"[{name}] eğitiliyor ({scenario})...")
            start_time = time.time()
            if not mp['params']:
                best_model = mp['model']
                best_model.fit(X_train, y_train)
            else:
                search = RandomizedSearchCV(mp['model'], mp['params'], n_iter=3, cv=tscv, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                
            y_pred_scaled = best_model.predict(X_test)
            if len(y_pred_scaled.shape) == 1:
                y_pred_scaled = y_pred_scaled.reshape(-1, 1)
            y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled).ravel()
            
            end_time = time.time()
            mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
            rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
            mape = calculate_mape(y_test_unscaled, y_pred_unscaled)
            r2 = r2_score(y_test_unscaled, y_pred_unscaled)
            
            all_results.append({
                'Scenario': scenario,
                'Model': name,
                'Model_Scenario': f"{name} ({scenario})",
                'MAE': mae,
                'RMSE': rmse,
                'MAPE (%)': mape,
                'R²': r2,
                'Time (s)': end_time - start_time
            })
            if scenario == 'all':
                predictions_dict[name] = y_pred_unscaled
            
        if scenario == 'all':
            plt.figure(figsize=(16, 8))
            plt.plot(test_dates, y_test_unscaled, label='Gerçek Kapanış Fiyatı', color='black', linewidth=2)
            for name, y_pred in predictions_dict.items():
                plt.plot(test_dates, y_pred, label=f'{name}', linestyle='--', alpha=0.7)
            plt.title('Şekil 5.12. Gerçek ve Tahmin Edilen Kapanış Fiyatlarının Karşılaştırılması (Tüm Özellikler)')
            plt.xlabel('Tarih')
            plt.ylabel('Kapanış Fiyatı (TL)')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'outputs/Fig_5_12_All_Models_vs_Actual_{scenario}.png', dpi=300)
            plt.close()
            
            # Save predictions for statistical tests
            pred_df = pd.DataFrame(predictions_dict)
            pred_df['Actual'] = y_test_unscaled
            pred_df.to_csv("outputs/ml_predictions.csv", index=False)

    results_df = pd.DataFrame(all_results)
    print("\n" + "="*60)
    print("Tablo 5.2. Senaryo Bazlı Makine Öğrenmesi Performansı")
    print("="*60)
    print(results_df[['Model_Scenario', 'R²', 'RMSE', 'MAE']].set_index('Model_Scenario').round(4))
    print("="*60)
    
    results_df.to_csv("outputs/Table_5_2_ML_Performance_Scenarios.csv", index=False)
    
    # En iyi modelleri çizdir
    plot_horizontal_bar(results_df, 'R²', 'outputs/Fig_5_11_R2_Comparison_Scenarios.png', 'En İyi 15 Model/Senaryo R² Karşılaştırması')
    plot_horizontal_bar(results_df, 'RMSE', 'outputs/Fig_5_9_RMSE_Comparison_Scenarios.png', 'En İyi 15 Model/Senaryo RMSE Karşılaştırması')
    plot_horizontal_bar(results_df, 'MAPE (%)', 'outputs/Fig_5_10_MAPE_Comparison_Scenarios.png', 'En İyi 15 Model/Senaryo MAPE Karşılaştırması')
    plot_horizontal_bar(results_df, 'MAE', 'outputs/Fig_5_8_MAE_Comparison_Scenarios.png', 'En İyi 15 Model/Senaryo MAE Karşılaştırması')

if __name__ == "__main__":
    file_path = "data/bist100_data_interpolate.csv"
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} dosyası bulunamadı.")
    else:
        train_and_evaluate_ml_models(file_path)
