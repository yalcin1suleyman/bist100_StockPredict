import numpy as np
import pandas as pd
import os
from scipy.stats import wilcoxon

def dm_test(actual, pred1, pred2, h=1, power=1):
    """
    Diebold-Mariano test for predictive accuracy.
    """
    actual = np.array(actual)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    
    e1 = actual - pred1
    e2 = actual - pred2
    
    if power == 1:
        d = np.abs(e1) - np.abs(e2)
    else:
        d = (e1 ** 2) - (e2 ** 2)
        
    d_mean = np.mean(d)
    
    # Autocovariance of d
    gamma = []
    for lag in range(0, h):
        if lag == 0:
            gamma.append(np.var(d, ddof=1))
        else:
            gamma.append(np.cov(d[:-lag], d[lag:])[0,1])
            
    v_d = gamma[0] + 2 * sum(gamma[1:])
    
    if v_d <= 0:
        return 0, 1.0 # Cannot reject null
        
    DM_stat = d_mean / np.sqrt(v_d / len(d))
    
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(DM_stat)))
    
    return DM_stat, p_value

def run_statistical_tests():
    print("--- 06: İSTATİSTİKSEL TESTLER ---")
    
    ml_path = "outputs/ml_predictions.csv"
    dl_path = "outputs/dl_predictions.csv"
    
    if not os.path.exists(ml_path) or not os.path.exists(dl_path):
        print("HATA: ml_predictions.csv veya dl_predictions.csv bulunamadı.")
        print("Lütfen önce 03_train_ml_models.py ve 05_train_dl_models.py dosyalarını çalıştırın.")
        return
        
    ml_df = pd.read_csv(ml_path)
    dl_df = pd.read_csv(dl_path)
    
    # Use LightGBM from ML and CNN-LSTM from DL as examples of 'best' models
    best_ml_model = 'LightGBM'
    best_dl_model = 'CNN-LSTM'
    
    if best_ml_model not in ml_df.columns:
        best_ml_model = [col for col in ml_df.columns if col != 'Actual'][0]
    if best_dl_model not in dl_df.columns:
        best_dl_model = [col for col in dl_df.columns if col != 'Actual'][0]
        
    actual = ml_df['Actual'].values
    pred_ml = ml_df[best_ml_model].values
    
    # For DL, because of sliding window, the actuals might be shifted or shorter.
    # We should align them from the end.
    min_len = min(len(actual), len(dl_df))
    actual_aligned = actual[-min_len:]
    pred_ml_aligned = pred_ml[-min_len:]
    pred_dl_aligned = dl_df[best_dl_model].values[-min_len:]
    
    results = []
    
    # Wilcoxon Signed-Rank Test
    # Compare absolute errors
    err_ml = np.abs(actual_aligned - pred_ml_aligned)
    err_dl = np.abs(actual_aligned - pred_dl_aligned)
    
    stat, p_wilcoxon = wilcoxon(err_ml, err_dl)
    results.append({'Test': 'Wilcoxon', 'Statistic': stat, 'p-value': p_wilcoxon})
    
    # Diebold-Mariano Test (RMSE based -> power=2)
    dm_stat, p_dm = dm_test(actual_aligned, pred_ml_aligned, pred_dl_aligned, h=1, power=2)
    results.append({'Test': 'Diebold-Mariano', 'Statistic': dm_stat, 'p-value': p_dm})
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print(f"Tablo 5.4. İstatistiksel Test Sonuçları ({best_ml_model} vs {best_dl_model})")
    print("="*60)
    print(results_df.round(4))
    print("="*60)
    
    results_df.to_csv("outputs/Table_5_4_Statistical_Tests.csv", index=False)
    
    with open("outputs/statistical_tests_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Karşılaştırılan Modeller: {best_ml_model} (Makine Öğrenmesi) ve {best_dl_model} (Derin Öğrenme)\n\n")
        f.write(f"1. Wilcoxon İşaretli Sıralar Testi p-değeri: {p_wilcoxon:.4f}\n")
        f.write(f"2. Diebold-Mariano Testi p-değeri: {p_dm:.4f}\n\n")
        if p_dm < 0.05:
            f.write("Sonuç: İki modelin tahmin performansları arasında istatistiksel olarak anlamlı bir fark vardır (p < 0.05).")
        else:
            f.write("Sonuç: İki modelin tahmin performansları arasında istatistiksel olarak anlamlı bir fark YOKTUR (p >= 0.05).")
            
    print("İstatistiksel test sonuçları kaydedildi.")

if __name__ == "__main__":
    run_statistical_tests()
