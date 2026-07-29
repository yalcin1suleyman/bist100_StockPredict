"""
Sekil 5.9 - Modellerin MAE Karsilastirmasi
Tum ML ve DL modellerinin MAE degerlerini karsilastiran yatay bar grafigi.
"""

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

# ML: sadece "all" senaryosu
ml_all = ml_df[ml_df['Scenario'] == 'all'][['Model', 'MAE']].copy()
ml_all['Kategori'] = 'Makine Ogrenmesi'

# DL
dl = dl_df[['Model', 'MAE']].copy()
dl['Kategori'] = 'Derin Ogrenme'

# Birlestir ve MAE'ye gore sirala (kucukten buyuge)
combined = pd.concat([ml_all, dl], ignore_index=True)
combined = combined.sort_values(by='MAE', ascending=True).reset_index(drop=True)

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
    combined['MAE'],
    color=colors,
    edgecolor='white',
    linewidth=0.8,
    height=0.65
)

# Bar ucuna deger yaz
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.3,
        bar.get_y() + bar.get_height() / 2.,
        f'{width:.2f}',
        ha='left', va='center', fontsize=10, fontweight='bold', color='#333333'
    )

ax.set_xlabel('MAE (Ortalama Mutlak Hata)', fontsize=12, fontweight='bold')
ax.set_title('Sekil 5.9. Modellerin MAE Karsilastirmasi', fontsize=14, fontweight='bold', pad=15)
ax.tick_params(axis='y', labelsize=11)
ax.tick_params(axis='x', labelsize=10)

# X siniri
x_max = combined['MAE'].max()
ax.set_xlim(0, x_max * 1.20)

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
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)

plt.tight_layout()

# ── Kaydet ──
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cıktılar")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Sekil_5_9_MAE_Karsilastirmasi.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"[OK] Grafik kaydedildi: {output_path}")
