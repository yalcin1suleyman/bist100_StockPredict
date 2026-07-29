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
    # 5. Şekil 5.6: Özellik Dağılımları (Histogram ve KDE) - 3 Grup
    # ---------------------------------------------------------
    print("Şekil 5.6 oluşturuluyor (Özellik Dağılımları - Tüm Özellikler 3 Grup Halinde)...")
    
    chunk_size = int(np.ceil(len(feature_cols) / 3))
    
    for chunk_idx in range(3):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(feature_cols))
        cols_to_plot = feature_cols[start_idx:end_idx]
        
        if not cols_to_plot:
            continue
            
        n_cols_plot = len(cols_to_plot)
        n_rows_fig = int(np.ceil(n_cols_plot / 3))
        fig, axes = plt.subplots(n_rows_fig, 3, figsize=(15, 4 * n_rows_fig))
        axes = axes.flatten()
        
        for i, col in enumerate(cols_to_plot):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], bins=30, color='steelblue')
            axes[i].set_title(col)
            
        # Hide any unused subplots
        for i in range(n_cols_plot, len(axes)):
            axes[i].axis('off')
            
        plt.suptitle(f"Şekil 5.6.{chunk_idx+1}. Özellik Dağılımları (Bölüm {chunk_idx+1})", y=1.02)
        plt.tight_layout()
        plt.savefig(f"outputs/Fig_5_6_Ozellik_Dagilimlari_{chunk_idx+1}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # ---------------------------------------------------------
    # 6. Şekil 5.7: Hareketli Ortalama ve Volatilite Grafiği
    # ---------------------------------------------------------
    print("Şekil 5.7 oluşturuluyor (Hareketli Ortalama ve Volatilite)...")
    
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
    
    plt.suptitle("Şekil 5.7. Hareketli ortalama ve volatilite grafiği")
    plt.tight_layout()
    plt.savefig("outputs/Fig_5_7_Moving_Average_Volatility.png", dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 7. Şekil 5.8: Özellik Öneminin Ön Analizi (Random Forest)
    # ---------------------------------------------------------
    print("Şekil 5.8 oluşturuluyor (RF Özellik Önemi)...")
    
    # Hedef: Yarının yüzdesel getirisi (fiyat değişim oranı).
    # Ham fiyatı tahmin etmek yerine getiriyi tahmin edersek, fiyat seviyesini taşıyan
    # değişkenler (Close, High, Low) dominant olmaz ve teknik göstergelerin
    # gerçek katkısı ortaya çıkar (örnek grafikteki gibi dengeli dağılım).
    df_rf = df.copy()
    df_rf['Next_Return'] = df_rf.groupby('Hisse_Kodu')['Close'].pct_change().shift(-1)
    df_rf = df_rf.dropna(subset=['Next_Return'])
    
    # Tüm özellikleri kullan (Close, Open, High, Low dahil)
    features_for_rf = [c for c in feature_cols]
    
    X = df_rf[features_for_rf].fillna(0)
    y = df_rf['Next_Return']
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    
    # Türkçe etiketler
    turkish_labels = {
        'Close': 'Kapanış Fiyatı (Close)', 'Open': 'Açılış Fiyatı (Open)',
        'High': 'En Yüksek Fiyat (High)', 'Low': 'En Düşük Fiyat (Low)',
        'Volume': 'İşlem Hacmi (Volume)', 'Gunluk_Getiri': 'Günlük Getiri (%)',
        'MA_10': 'SMA (10)', 'MA_50': 'SMA (50)', 'EMA_20': 'EMA (20)',
        'RSI_14': 'RSI (14)', 'MACD': 'MACD', 'MACD_Signal': 'MACD Sinyal',
        'MACD_Histogram': 'MACD Histogram', 'BB_Upper': 'Bollinger Üst Bandı',
        'BB_Lower': 'Bollinger Alt Bandı', 'BB_Width': 'Bollinger Bant Genişliği',
        'ATR_14': 'ATR (14)', 'OBV': 'OBV', 'VWAP': 'VWAP',
        'Momentum_10': 'Momentum (10)', 'ROC_10': 'ROC (%)',
        'Stoch_K': 'Stochastic %K', 'Stoch_D': 'Stochastic %D',
        'Williams_R': 'Williams %R', 'CCI_20': 'CCI (20)',
        'ADX_14': 'ADX (14)', 'Volatilite_10': 'Volatilite (10)',
        'USD_TRY': 'USD/TRY', 'VIX': 'VIX'
    }
    
    n_features = len(features_for_rf)
    
    # Renkli barlar (örnek grafikteki gibi her bar farklı renk)
    colors = plt.cm.tab20(np.linspace(0, 1, n_features))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for rank, idx in enumerate(indices):
        label = turkish_labels.get(features_for_rf[idx], features_for_rf[idx])
        bar = ax.barh(rank, importances[idx], color=colors[rank % len(colors)], align='center', height=0.7)
        ax.text(importances[idx] + 0.002, rank, f'{importances[idx]:.3f}', va='center', fontsize=9)
    
    labels = [turkish_labels.get(features_for_rf[i], features_for_rf[i]) for i in indices]
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Özellik Önemi (Mean Decrease in Impurity)')
    ax.set_title('Ön Özellik Önem Grafiği\nRandom Forest Modeli ile Tahmin Performansı Ön Değerlendirmesi\nBIST100 Veri Kümesi (01.01.2021 – 31.12.2023)', fontsize=11, fontweight='bold')
    
    fig.text(0.5, 0.01, 
             'Not: Değerler Random Forest modelinin Mean Decrease in Impurity (MDI) yöntemine göre hesaplanmıştır.\n'
             'Değerlerin büyüklüğü, ilgili özelliğin tahmin performansına katkısının göreli ölçüsünü gösterir.',
             ha='center', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("outputs/Fig_5_8_RF_Feature_Importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("--- EDA TAMAMLANDI! TÜM GRAFİKLER VE TABLOLAR OLUŞTURULDU ---")

if __name__ == "__main__":
    perform_eda("data/bist100_data_interpolate.csv")
