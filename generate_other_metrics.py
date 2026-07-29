import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# ── Verileri oku ──
ml_df = pd.read_csv('outputs/Table_5_2_ML_Performance_Scenarios.csv')
dl_df = pd.read_csv('outputs/Table_5_3_DL_Performance.csv')

def plot_metric(metric_col, title, filename, ascending=True, x_label=''):
    # ML: sadece "all" senaryosu
    ml_all = ml_df[ml_df['Scenario'] == 'all'][['Model', metric_col]].copy()
    ml_all['Kategori'] = 'Makine Ogrenmesi'

    # DL
    dl = dl_df[['Model', metric_col]].copy()
    dl['Kategori'] = 'Derin Ogrenme'

    # Birlestir ve sirala
    combined = pd.concat([ml_all, dl], ignore_index=True)
    combined = combined.sort_values(by=metric_col, ascending=ascending).reset_index(drop=True)

    # ── Grafik ──
    fig, ax = plt.subplots(figsize=(12, 7))

    # Renkleri kategoriye gore ayarla
    colors = []
    for _, row in combined.iterrows():
        if row['Kategori'] == 'Makine Ogrenmesi':
            colors.append('#2980B9')
        else:
            colors.append('#27AE60')

    bars = ax.barh(
        combined['Model'],
        combined[metric_col],
        color=colors,
        edgecolor='white',
        linewidth=0.8,
        height=0.65
    )

    # Bar ucuna deger yaz
    for bar in bars:
        width = bar.get_width()
        
        # Eğer değer negatifse (R2 de çok düşük değerler olabilir) 0'a yakınsa sağa yaz, yoksa uygun hizala
        align_x = width + (combined[metric_col].max() * 0.01) if width >= 0 else width - (combined[metric_col].min() * 0.01)
        ha_val = 'left' if width >= 0 else 'right'
        
        # Yüzdelik değerse ya da küçük virgüllü değerse 4 hane, genelse 2 hane
        if metric_col == 'R²':
            val_text = f'{width:.4f}'
        else:
            val_text = f'{width:.2f}'

        ax.text(
            align_x,
            bar.get_y() + bar.get_height() / 2.,
            val_text,
            ha=ha_val, va='center', fontsize=10, fontweight='bold', color='#333333'
        )

    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=10)

    # X siniri
    x_max = combined[metric_col].max()
    x_min = combined[metric_col].min() if combined[metric_col].min() < 0 else 0
    
    if x_min < 0:
        ax.set_xlim(x_min * 1.20, x_max * 1.20)
    else:
        ax.set_xlim(0, x_max * 1.15)

    # Grid
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2980B9', label='Makine Ogrenmesi'),
        Patch(facecolor='#27AE60', label='Derin Ogrenme')
    ]
    
    loc_val = 'lower right' if ascending else 'lower left'
    if metric_col == 'R²':
        loc_val = 'center right'
        
    ax.legend(handles=legend_elements, loc=loc_val, fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # ── Kaydet ──
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cıktılar")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[OK] Grafik kaydedildi: {output_path}")

# Hata metrikleri kucukten buyuge siralanmali (ascending=True)
plot_metric('RMSE', 'Sekil 5.10. Modellerin RMSE Karsilastirmasi', 'Sekil_5_10_RMSE_Karsilastirmasi.png', ascending=True, x_label='RMSE (Hata Kareler Ortalamasi Karekoku)')
plot_metric('MAPE (%)', 'Sekil 5.11. Modellerin MAPE Karsilastirmasi', 'Sekil_5_11_MAPE_Karsilastirmasi.png', ascending=True, x_label='MAPE (%) (Ortalama Mutlak Yuzde Hata)')

# Basari metrigi R2 buyukten kucuge siralanmali, bar siralamasi icin ascending=True (en iyi ustte olsun diye sort ters olacak)
# yatay barda (barh) y ekseni alttan uste cizilir. 
# En iyinin en ustte olmasi icin kucukten buyuge sort edilmeli ki en buyuk deger arrayin sonunda (y ekseninin en ustunde) olsun.
plot_metric('R²', 'Sekil 5.12. Modellerin R² Karsilastirmasi', 'Sekil_5_12_R2_Karsilastirmasi.png', ascending=True, x_label='R² (Belirlilik Katsayisi)')
