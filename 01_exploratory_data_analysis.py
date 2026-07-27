import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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
    stats_df.to_csv("outputs/Table_5_1_Descriptive_Statistics.csv", index=False)
    
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
    plt.savefig("outputs/Fig_5_1_Closing_Prices.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Tarih', y='Volume', hue='Hisse_Kodu', palette='tab10', legend='brief')
    plt.title("Şekil 5.2. İşlem hacimlerinin zamana göre dağılımı")
    plt.xlabel("Tarih")
    plt.ylabel("İşlem Hacmi")
    plt.tight_layout()
    plt.savefig("outputs/Fig_5_2_Volume.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 3. Şekil 5.3: Box Plots (3 Mantıksal Grup)
    # ---------------------------------------------------------
    print("Şekil 5.3 oluşturuluyor (3 ayrı mantıksal gruba bölünmüş Box Plot grafikleri)...")
    
    import math
    import itertools
    
    colors = ['#b3d1ff', '#b3ffcc', '#ffffb3', '#ffb3b3', '#d9b3ff', '#66c2cd', '#ffcc99', '#d1e0e0', '#ffcce6', '#b3ffb3', '#ffff66']
    
    # 1. GRUP: Temel Veriler ve Makroekonomi (8 Özellik - 2x4 Grid)
    grid1_features = {
        'Open': 'Açılış (Open)', 'High': 'En Yüksek (High)', 'Low': 'En Düşük (Low)', 'Close': 'Kapanış (Close)', 
        'Volume': 'İşlem Hacmi\n(Volume, milyon lot)', 'Gunluk_Getiri': 'Günlük Getiri', 'USD_TRY': 'USD/TRY', 'VIX': 'VIX'
    }
    
    # 2. GRUP: Trend, Hacim ve Volatilite Göstergeleri (10 Özellik - 2x5 Grid)
    grid2_features = {
        'MA_10': 'SMA (10)', 'MA_50': 'SMA (50)', 'EMA_20': 'EMA (20)', 'VWAP': 'VWAP', 'OBV': 'OBV',
        'Volatilite_10': 'Volatilite (10)', 'ATR_14': 'ATR (14)', 'BB_Upper': 'BB Üst', 'BB_Lower': 'BB Alt', 'BB_Width': 'BB Genişlik'
    }
    
    # 3. GRUP: Momentum ve Osilatörler (10 Özellik - 2x5 Grid)
    grid3_features = {
        'RSI_14': 'RSI (14)', 'MACD': 'MACD', 'MACD_Signal': 'MACD Sinyal', 'MACD_Histogram': 'MACD Histogram', 
        'Momentum_10': 'Momentum (10)', 'Williams_R': 'Williams %R', 'Stoch_K': 'Stoch K', 'Stoch_D': 'Stoch D', 
        'CCI_20': 'CCI (20)', 'ADX_14': 'ADX (14)'
    }
    
    grids = [
        ("outputs/Fig_5_3_Box_Plots_1_Temel_Veriler.png", grid1_features, "Temel Finansal ve Makroekonomik Veriler", 2, 4, 12),
        ("outputs/Fig_5_3_Box_Plots_2_Trend_Volatilite.png", grid2_features, "Trend, Hacim ve Volatilite Göstergeleri", 2, 5, 15),
        ("outputs/Fig_5_3_Box_Plots_3_Momentum.png", grid3_features, "Momentum ve Osilatör Göstergeleri", 2, 5, 15)
    ]
             
    for file_name, features_dict, sub_title, n_rows, n_cols, width in grids:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, 8))
        axes = axes.flatten()
        color_cycle = itertools.cycle(colors)
        
        for i, (col, title) in enumerate(features_dict.items()):
            ax = axes[i]
            if col in df.columns:
                data_to_plot = df[col].dropna()
                
                ylabel = "Değer"
                if 'Volume' in col or 'OBV' in col:
                    data_to_plot = data_to_plot / 1e6
                    ylabel = "milyon"
                    if 'Volume' in col: ylabel = "milyon lot"
                elif col in ['Open', 'High', 'Low', 'Close', 'MA_10', 'MA_50', 'EMA_20', 'BB_Upper', 'BB_Lower', 'ATR_14', 'USD_TRY', 'VWAP']:
                    ylabel = "Fiyat (TL)"
                    
                box_color = next(color_cycle)
                
                sns.boxplot(y=data_to_plot, ax=ax, color=box_color, width=0.5,
                            medianprops=dict(color="red", linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor='none', markeredgecolor='black', markersize=4, alpha=0.5))
                
                ax.set_title(title, fontsize=13, pad=10, fontweight='bold',
                             bbox=dict(facecolor='#e6f2ff', edgecolor='#d9d9d9', boxstyle='round,pad=0.4'))
                ax.set_ylabel(ylabel, fontsize=12)
                ax.tick_params(axis='y', labelsize=11)
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                ax.set_xticks([])
            else:
                ax.axis('off')
                
        # Fill empty subplots if any (e.g. if we had 9 features in a 2x5 grid)
        for j in range(i + 1, n_rows * n_cols):
            axes[j].axis('off')
                
        plt.suptitle(f"{sub_title}\nBIST100 Veri Kümesi (01.01.2015 – 31.12.2024)", fontsize=18, fontweight='bold', y=0.98)
        fig.text(0.5, 0.02, "Not: Kutular %25-%75 çeyrek aralığını, kırmızı çizgi medyanı, bıyıklar ise 1.5*IQR aralığını göstermektedir.", 
                 ha='center', fontsize=14, style='italic', fontweight='bold')
                 
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        plt.savefig(file_name, dpi=300)
        plt.close()
    
    # ---------------------------------------------------------
    # 3.1 Şekil 5.3 (Ekstra): Tüm Özellikler İçin Tek Bir Box Plot (Z-Score)
    # ---------------------------------------------------------
    print("Tüm özellikler için tek bir standartlaştırılmış Box Plot oluşturuluyor (Z-Score)...")
    
    # Sadece sayısal özellikleri al
    X_num = df[feature_cols].dropna()
    
    # Standartlaştırma uygula
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns)
    
    plt.figure(figsize=(22, 10))
    sns.boxplot(data=X_scaled, palette='Set3', linewidth=1.2, fliersize=3)
    
    plt.title("Tüm Özelliklerin Karşılaştırmalı Kutu Grafiği (Standartlaştırılmış - Z Score)", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("BIST 100 Özellikleri (Makroekonomik, Teknik ve Ham Fiyat)", fontsize=14, labelpad=10)
    plt.ylabel("Standart Sapma Birimi (Z-Score)", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("outputs/Fig_5_3_Box_Plots_Tum_Ozellikler_ZScore.png", dpi=300, bbox_inches='tight')
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
    plt.savefig("outputs/Fig_5_4_Pearson_Correlation.png", dpi=300)
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
    plt.savefig("outputs/Fig_5_5_Feature_Distributions.png", dpi=300)
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
    plt.savefig("outputs/Fig_5_6_Moving_Average_Volatility.png", dpi=300)
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
    plt.savefig("outputs/Fig_5_7_RF_Feature_Importance.png", dpi=300)
    plt.close()
    
    print("--- EDA TAMAMLANDI! TÜM GRAFİKLER VE TABLOLAR OLUŞTURULDU ---")

if __name__ == "__main__":
    perform_eda("data/bist100_data_interpolate.csv")
