import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import os

# Uyarıları gizle
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Modülleri içe aktar
from importlib.machinery import SourceFileLoader
data_module = SourceFileLoader("data_preprocessing", "02_data_preprocessing.py").load_module()
DataPreprocessor = data_module.DataPreprocessor
from lightgbm import LGBMRegressor

def run_shap_analysis(file_path):
    print("--- 04: AÇIKLANABİLİR YAPAY ZEKA (XAI) - SADECE HIZLI MODELLER İÇİN ---")
    
    preprocessor = DataPreprocessor(file_path=file_path)
    X_train, y_train, X_test, y_test, _, _, feature_names = preprocessor.get_ml_data(scaler_type='minmax')
    
    print("\nSHAP Analizi için en iyi modellerden LightGBM eğitiliyor...")
    lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    lgbm_model.fit(X_train, y_train)
    
    print("\nŞekil 5.24 ve 5.25 (SHAP Grafikleri) oluşturuluyor...")
    explainer = shap.TreeExplainer(lgbm_model)
    
    # Tüm test seti üzerinde SHAP değerlerini hesapla (çok uzun sürerse sample alınabilir)
    # Taslakta "tüm test kümesi üzerinde ortalama mutlak SHAP değerleri" denmiş.
    shap_values = explainer.shap_values(X_test)
    
    # 1. Şekil 5.24: SHAP Global Feature Importance
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Şekil 5.24. SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig("Fig_5_24_SHAP_Global_Feature_Importance.png", dpi=300)
    plt.close()
    
    # 2. Şekil 5.25: SHAP Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title("Şekil 5.25. SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig("Fig_5_25_SHAP_Summary_Plot.png", dpi=300)
    plt.close()
    
    # 3. Şekil 5.26: Yerel (Local) SHAP Analizi (Waterfall Plot) - Tek bir gün (Örn: İlk test günü)
    print("\nYerel (Local) SHAP ve LIME analizleri oluşturuluyor...")
    instance_idx = 0  # Test setindeki ilk günü inceleyelim
    
    plt.figure()
    # TreeExplainer for LightGBM returns raw values. We need an Explanation object for waterfall
    # Expected value is a scalar or an array depending on the model, we ensure it's scalar here
    expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
    shap_exp = shap.Explanation(values=shap_values[instance_idx], 
                                base_values=expected_val, 
                                data=X_test[instance_idx], 
                                feature_names=feature_names)
    shap.waterfall_plot(shap_exp, show=False)
    plt.title(f"Şekil 5.26. SHAP Waterfall Plot (Yerel Açıklama)")
    plt.tight_layout()
    plt.savefig("Fig_5_26_SHAP_Waterfall_Plot.png", dpi=300)
    plt.close()

    # 4. Şekil 5.27: Yerel (Local) LIME Analizi
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['Kapanis_Fiyati'],
        mode='regression',
        random_state=42
    )
    
    exp = lime_explainer.explain_instance(
        data_row=X_test[instance_idx], 
        predict_fn=lgbm_model.predict
    )
    
    # LIME sonucunu HTML olarak kaydet
    exp.save_to_file('Fig_5_27_LIME_Local_Explanation.html')
    
    # Ayrıca LIME grafiğini PNG olarak da kaydedelim (pyplot figürü olarak döner)
    fig = exp.as_pyplot_figure()
    plt.title(f"Şekil 5.27. LIME Yerel Açıklaması")
    plt.tight_layout()
    fig.savefig("Fig_5_27_LIME_Local_Explanation.png", dpi=300)
    plt.close(fig)

    print("XAI grafikleri başarıyla kaydedildi.")

if __name__ == "__main__":
    file_path = "bist100_data_interpolate.csv"
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} bulunamadı.")
    else:
        run_shap_analysis(file_path)
