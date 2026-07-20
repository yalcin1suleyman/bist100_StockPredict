import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor

# Uyarıları gizle
import warnings
warnings.filterwarnings("ignore")

# Grafiklerin akademik görünümü için seaborn teması
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def perform_eda(file_path):
    print("--- 01: KEŞİFSEL VERİ ANALİZİ (EDA) BAŞLIYOR ---")
    
    if not os.path.exists(file_path):
        print(f"HATA: {file_path} bulunamadı.")
        return
        
    df = pd.read_csv(file_path)
    
    # Tarihi datetime yapalım
    if 'Tarih' in df.columns:
        df['Tarih'] = pd.to_datetime(df['Tarih'])
        df = df.sort_values(by='Tarih').reset_index(drop=True)
        
    # Sayısal özellikleri belirle
    drop_cols = ['Tarih', 'Set', 'Hisse_Kodu']
    feature_cols = [col for col in df.columns if col not in drop_cols]
    
    # ---------------------------------------------------------
    # 1. Tablo 5.1: Tanımlayıcı İstatistikler
    # ---------------------------------------------------------
    print("Tablo 5.1 oluşturuluyor (Descriptive Statistics)...")
    stats_list = []
    for col in feature_cols:
        col_data = df[col].dropna()
        stats_list.append({
            'Özellik': col,
            'Ortalama': col_data.mean(),
            'Medyan': col_data.median(),
            'Std. Sapma': col_data.std(),
            'Min.': col_data.min(),
            'Maks.': col_data.max(),
            'Çarpıklık (Skewness)': skew(col_data),
            'Basıklık (Kurtosis)': kurtosis(col_data)
        })
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv("Table_5_1_Descriptive_Statistics.csv", index=False)
    
    # ---------------------------------------------------------
    # 2. Şekil 5.1 & 5.2: Fiyat ve Hacim Çizimleri
    # ---------------------------------------------------------
    print("Şekil 5.1 ve 5.2 oluşturuluyor (Fiyat ve Hacim Zaman Serisi)...")
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Tarih', y='Close', hue='Hisse_Kodu', palette='tab10', legend='brief')
    plt.title("Şekil 5.1. Kapanış fiyatlarının zaman içerisindeki değişimi")
    plt.xlabel("Tarih")
    plt.ylabel("Kapanış Fiyatı (TL)")
    plt.tight_layout()
    plt.savefig("Fig_5_1_Closing_Prices.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Tarih', y='Volume', hue='Hisse_Kodu', palette='tab10', legend='brief')
    plt.title("Şekil 5.2. İşlem hacimlerinin zamana göre dağılımı")
    plt.xlabel("Tarih")
    plt.ylabel("İşlem Hacmi")
    plt.tight_layout()
    plt.savefig("Fig_5_2_Volume.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 3. Şekil 5.3: Box Plots (Tüm Değişkenler - Şablon Stili)
    # ---------------------------------------------------------
    print("Şekil 5.3 oluşturuluyor (Bütünleşik şablon Box Plot grafiği tüm özellikler için)...")
    
    # Tüm özellikleri (feature_cols) alıp hocanın estetik tarzıyla çizdirelim
    # Renk paleti oluşturalım
    import math
    import itertools
    
    colors = ['#b3d1ff', '#b3ffcc', '#ffffb3', '#ffb3b3', '#d9b3ff', '#66c2cd', '#ffcc99', '#d1e0e0', '#ffcce6', '#b3ffb3', '#ffff66']
    color_cycle = itertools.cycle(colors)
    
    # Izgara (Grid) boyutunu ayarlayalım
    n_features = len(feature_cols)
    n_cols = 7
    n_rows = math.ceil(n_features / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4 * n_rows))
    axes = axes.flatten()
    
    for i in range(n_rows * n_cols):
        ax = axes[i]
        if i < n_features:
            col = feature_cols[i]
            data_to_plot = df[col].dropna()
            
            # Hacim veya OBV gibi çok büyük değerli olanları milyon'a bölelim ki grafik düzgün görünsün
            ylabel = "Değer"
            if 'Volume' in col or 'OBV' in col:
                data_to_plot = data_to_plot / 1e6
                ylabel = "milyon"
            elif col in ['Open', 'High', 'Low', 'Close', 'MA_10', 'MA_50', 'EMA_20', 'BB_Upper', 'BB_Lower', 'ATR_14', 'USD_TRY']:
                ylabel = "Fiyat (TL)"
                
            box_color = next(color_cycle)
                
            sns.boxplot(y=data_to_plot, ax=ax, color=box_color, width=0.5,
                        medianprops=dict(color="red", linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=4, alpha=0.5))
            
            # Başlık (Mavi arka planlı kutu)
            ax.set_title(col, fontsize=10, pad=10, fontweight='bold',
                         bbox=dict(facecolor='#e6f2ff', edgecolor='none', boxstyle='round,pad=0.4'))
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xticks([])
        else:
            ax.axis('off') # Boş kalan kısımları gizle
            
    # Ana başlıklar
    plt.suptitle(f"Özelliklere Ait Box Plot Grafikleri ({n_features} Değişken)\nBIST100 Veri Kümesi", fontsize=18, fontweight='bold', y=0.98)
    
    # Alt not
    fig.text(0.5, 0.01, "Not: Kutular %25-%75 çeyrek aralığını, kırmızı çizgi medyanı, bıyıklar ise 1.5*IQR aralığını göstermektedir.\nDaireler aykırı değerleri (outliers) temsil etmektedir.", 
             ha='center', fontsize=11, style='italic')
             
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Fig_5_3_Box_Plots_Template.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 4. Şekil 5.4: Pearson Korelasyon Matrisi (Tüm Değişkenler)
    # ---------------------------------------------------------
    print("Şekil 5.4 oluşturuluyor (Pearson Korelasyon Matrisi - Tüm Özellikler)...")
    
    # Veri setimizdeki TÜM özellikleri (feature_cols) dahil ediyoruz
    corr_matrix = df[feature_cols].corr(method='pearson')
    
    # Tüm değişkenleri sığdırmak için devasa bir çözünürlük ve boyut ayarlıyoruz
    plt.figure(figsize=(24, 20))
    
    # annot_kws ile yazı boyutunu küçültüyoruz ki sayılar birbirine girmesin
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, 
                square=True, annot_kws={"size": 7}, linewidths=.5)
                
    plt.title("Pearson Korelasyon Matrisi (Tüm Özellikler)", fontsize=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("Fig_5_4_Pearson_Correlation.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 5. Şekil 5.5: Özellik Dağılımları (Histogram ve KDE)
    # ---------------------------------------------------------
    print("Şekil 5.5 oluşturuluyor (Özellik Dağılımları)...")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    dist_cols = ['Close', 'Volume', 'Gunluk_Getiri', 'RSI_14', 'MACD', 'Momentum_10', 'ATR_14', 'Williams_R', 'CCI_20']
    dist_cols = [c for c in dist_cols if c in feature_cols]
    
    for i, col in enumerate(dist_cols[:9]):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], bins=30, color='steelblue')
        axes[i].set_title(col)
        
    plt.suptitle("Şekil 5.5. Özellik Dağılımları")
    plt.tight_layout()
    plt.savefig("Fig_5_5_Feature_Distributions.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 6. Şekil 5.6: Hareketli Ortalama ve Volatilite Grafiği
    # ---------------------------------------------------------
    print("Şekil 5.6 oluşturuluyor (Hareketli Ortalama ve Volatilite)...")
    
    # Tek bir hisse (veya endeks) üzerinden göstermek daha mantıklı, örneğin verideki ilk hisse
    hisse = df['Hisse_Kodu'].iloc[0]
    df_hisse = df[df['Hisse_Kodu'] == hisse].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Fiyat ve Ortalamalar
    ax1.plot(df_hisse['Tarih'], df_hisse['Close'], label='Kapanış Fiyatı', color='black')
    if 'MA_10' in df_hisse.columns:
        ax1.plot(df_hisse['Tarih'], df_hisse['MA_10'], label='10 Günlük MA', color='orange')
    if 'MA_50' in df_hisse.columns:
        ax1.plot(df_hisse['Tarih'], df_hisse['MA_50'], label='50 Günlük MA', color='blue')
    ax1.set_title(f"Hareketli Ortalama Grafiği ({hisse})")
    ax1.set_ylabel("Fiyat (TL)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volatilite
    if 'Volatilite_10' in df_hisse.columns:
        ax2.plot(df_hisse['Tarih'], df_hisse['Volatilite_10'], label='Volatilite (10 Günlük)', color='red')
    ax2.set_title("Volatilite Grafiği")
    ax2.set_xlabel("Tarih")
    ax2.set_ylabel("Volatilite")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Şekil 5.6. Hareketli ortalama ve volatilite grafiği")
    plt.tight_layout()
    plt.savefig("Fig_5_6_Moving_Average_Volatility.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 7. Şekil 5.7: Özellik Öneminin Ön Analizi (Random Forest)
    # ---------------------------------------------------------
    print("Şekil 5.7 oluşturuluyor (RF Özellik Önemi)...")
    
    # Hedef olarak 'Close' alalım (veri sızıntısı olmaması için Close'u t+1 yapacağız veya diğerlerini kullanacağız)
    # Taslaktaki amaç sadece "hangi değişkenler kapanışı belirliyor" sorusudur.
    X = df[feature_cols].copy()
    # Eğer X içinde Close varsa çıkaralım
    features_for_rf = [c for c in feature_cols if c != 'Close']
    X = df[features_for_rf]
    y = df['Close']
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X.fillna(0), y) # Basitçe fillna(0) dedik çünkü bu sadece keşifsel bir adım
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    
    # Sadece en önemli 15 özelliği çizdirelim
    top_n = 15
    top_indices = indices[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[top_indices], color='mediumseagreen', align='center')
    plt.yticks(range(top_n), [features_for_rf[i] for i in top_indices])
    plt.xlabel('Özellik Önemi (Mean Decrease in Impurity)')
    plt.title('Şekil 5.7. Ön özellik önem grafiği (Random Forest)')
    plt.tight_layout()
    plt.savefig("Fig_5_7_RF_Feature_Importance.png", dpi=300)
    plt.close()
    
    print("--- EDA TAMAMLANDI! TÜM GRAFİKLER VE TABLOLAR OLUŞTURULDU ---")

if __name__ == "__main__":
    perform_eda("bist100_data_interpolate.csv")
