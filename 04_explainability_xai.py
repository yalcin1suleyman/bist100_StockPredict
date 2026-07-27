import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import os
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

from importlib.machinery import SourceFileLoader
data_module = SourceFileLoader("data_preprocessing", "02_data_preprocessing.py").load_module()
DataPreprocessor = data_module.DataPreprocessor
from lightgbm import LGBMRegressor
from sklearn.inspection import PartialDependenceDisplay

def run_shap_analysis(file_path):
    print("--- 04: AÇIKLANABİLİR YAPAY ZEKA (XAI) VE GRUPLANDIRILMIŞ ANALİZ ---")
    
    # Tüm özelliklerle (all) modeli eğitelim
    preprocessor = DataPreprocessor(file_path=file_path)
    X_train, y_train, X_test, y_test, _, _, feature_names = preprocessor.get_ml_data(scaler_type='minmax', feature_set='all')
    
    print("\nSHAP Analizi için LightGBM eğitiliyor (Tüm özellikler ile)...")
    lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    lgbm_model.fit(X_train, y_train)
    
    print("\nSHAP Grafikleri oluşturuluyor...")
    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_test)
    
    # 1. Klasik Grafikler
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Şekil 5.24. SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/Fig_5_24_SHAP_Global_Feature_Importance.png", dpi=300)
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("Şekil 5.25. SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig("outputs/Fig_5_25_SHAP_Summary_Plot.png", dpi=300)
    plt.close()
    
    # 2. HOCANIN İSTERİ: Gruplandırılmış SHAP Analizi
    print("\nFinansal Özelliklere Göre Gruplandırılmış SHAP Analizi Yapılıyor...")
    
    feature_groups = {
        'Ham Özellikler': ['Volume', 'Gunluk_Getiri'],
        'Makroekonomik': ['USD_TRY', 'VIX'],
        'Volatilite': ['Volatilite_10', 'ATR_14', 'BB_Upper', 'BB_Lower', 'BB_Width'],
        'Momentum': ['Momentum_10', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'CCI_20', 'Williams_R', 'Stoch_K', 'Stoch_D'],
        'Trend ve Hacim': ['MA_10', 'MA_50', 'EMA_20', 'ADX_14', 'OBV', 'VWAP']
    }
    
    # Calculate mean absolute SHAP value for each feature
    global_shap_vals = np.abs(shap_values).mean(0)
    feature_importance = dict(zip(feature_names, global_shap_vals))
    
    group_importances = {}
    for group_name, features in feature_groups.items():
        group_sum = 0
        for f in features:
            if f in feature_importance:
                group_sum += feature_importance[f]
        group_importances[group_name] = group_sum
        
    # Plot grouped importance
    plt.figure(figsize=(10, 6))
    groups = list(group_importances.keys())
    vals = list(group_importances.values())
    
    # Sort descending
    sorted_indices = np.argsort(vals)[::-1]
    sorted_groups = [groups[i] for i in sorted_indices]
    sorted_vals = [vals[i] for i in sorted_indices]
    
    colors = sns.color_palette("viridis", len(sorted_groups))
    bars = plt.bar(sorted_groups, sorted_vals, color=colors)
    plt.title('Şekil 5.28. Finansal Özellik Gruplarına Göre SHAP Önem Düzeyi')
    plt.ylabel('Ortalama Mutlak SHAP Değeri')
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig('outputs/Fig_5_28_SHAP_Grouped_Importance.png', dpi=300)
    plt.close()
    
    # 3. Yerel (Local) LIME Analizi
    print("\nYerel (Local) LIME analizi oluşturuluyor...")
    instance_idx = 0 
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['Kapanis_Fiyati'],
        mode='regression',
        random_state=42
    )
    
    exp = lime_explainer.explain_instance(data_row=X_test[instance_idx], predict_fn=lgbm_model.predict)
    exp.save_to_file('outputs/Fig_5_27_LIME_Local_Explanation.html')
    
    fig = exp.as_pyplot_figure()
    plt.title(f"Şekil 5.27. LIME Yerel Açıklaması")
    plt.tight_layout()
    fig.savefig("outputs/Fig_5_27_LIME_Local_Explanation.png", dpi=300)
    plt.close(fig)
    # 4. PDP (Partial Dependence Plot) ve ICE (Individual Conditional Expectation)
    print("\nPDP ve ICE grafikleri oluşturuluyor...")
    # Find top 2 features from SHAP
    top_indices = np.argsort(global_shap_vals)[-2:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    # We need a dataframe for PartialDependenceDisplay
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    PartialDependenceDisplay.from_estimator(
        lgbm_model, X_train_df, top_features,
        kind='both', # 'both' means PDP + ICE
        ax=ax,
        grid_resolution=50
    )
    plt.suptitle("Şekil 5.29. PDP ve ICE Grafikleri (En Önemli 2 Değişken)", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/Fig_5_29_PDP_ICE.png", dpi=300)
    plt.close(fig)
    
    # 5. SHAP Force Plot
    print("\nSHAP Force Plot oluşturuluyor...")
    # Force plot for the same instance as LIME
    force_plot = shap.force_plot(explainer.expected_value, shap_values[instance_idx,:], X_test[instance_idx,:], feature_names=feature_names)
    shap.save_html("outputs/Fig_5_30_SHAP_Force_Plot.html", force_plot)

    print("Tüm XAI grafikleri başarıyla kaydedildi.")

if __name__ == "__main__":
    file_path = "data/bist100_data_interpolate.csv"
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} bulunamadı.")
    else:
        run_shap_analysis(file_path)
