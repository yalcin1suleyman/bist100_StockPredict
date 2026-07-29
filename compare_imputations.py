import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from importlib.machinery import SourceFileLoader
import warnings

warnings.filterwarnings("ignore")

# Load DataPreprocessor
data_module = SourceFileLoader("data_preprocessing", "02_data_preprocessing.py").load_module()
DataPreprocessor = data_module.DataPreprocessor

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def compare_imputations():
    print("--- EKSİK VERİ DOLDURMA YÖNTEMLERİNİN KARŞILAŞTIRILMASI ---")
    datasets = {
        'Forward Fill': 'data/bist100_data_ffill.csv',
        'KNN Imputer': 'data/bist100_data_knn.csv',
        'Doğrusal Enterpolasyon': 'data/bist100_data_interpolate.csv'
    }
    
    results = []
    
    for name, file_path in datasets.items():
        print(f"\n[{name}] veri seti test ediliyor...")
        
        # Load data using preprocessor logic (all features)
        preprocessor = DataPreprocessor(file_path=file_path)
        X_train, y_train, X_test, y_test, y_test_unscaled, scaler_y, feature_cols = preprocessor.get_ml_data(scaler_type='minmax', feature_set='all')
        
        # Model: LightGBM (Literature suggests it's robust and fast)
        model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1, n_estimators=100)
        model.fit(X_train, y_train)
        
        y_pred_scaled = model.predict(X_test)
        if len(y_pred_scaled.shape) == 1:
            y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled).ravel()
        
        rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
        mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
        r2 = r2_score(y_test_unscaled, y_pred_unscaled)
        
        print(f"R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        
        results.append({
            'Yöntem': name,
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae
        })
        
    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/Table_5_0_Imputation_Comparison.csv', index=False)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results_df['Yöntem']))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, results_df['R²'], width, label='R² (Yüksek Daha İyi)', color='steelblue')
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, results_df['RMSE'], width, label='RMSE (Düşük Daha İyi)', color='salmon')
    
    ax1.set_xlabel('Doldurma (Imputation) Yöntemi', fontweight='bold', fontsize=12)
    ax1.set_ylabel('R² Skoru', fontweight='bold', fontsize=12)
    ax2.set_ylabel('RMSE Hata', fontweight='bold', fontsize=12)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Yöntem'], fontweight='bold')
    
    # Add values on top of bars
    ax1.bar_label(rects1, fmt='%.4f', padding=3)
    ax2.bar_label(rects2, fmt='%.2f', padding=3)
    
    plt.title('Şekil 5.0. Farklı Doldurma Yöntemlerinin LightGBM ile Performans Karşılaştırması', fontweight='bold', pad=15)
    
    # Custom legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('outputs/Fig_5_0_Imputation_Comparison.png', dpi=300)
    print("\n--- İŞLEM TAMAMLANDI! Çıktılar 'outputs' klasörüne kaydedildi. ---")

if __name__ == "__main__":
    compare_imputations()
