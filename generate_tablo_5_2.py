"""
Tablo 5.2 - Tum Makine Ogrenmesi ve Derin Ogrenme Modellerinin
Test Veri Kumesi Uzerindeki Performans Sonuclari
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Verileri oku ──
ml_df = pd.read_csv('outputs/Table_5_2_ML_Performance_Scenarios.csv')
dl_df = pd.read_csv('outputs/Table_5_3_DL_Performance.csv')

# ML: sadece "all" senaryosu
ml_all = ml_df[ml_df['Scenario'] == 'all'][['Model', 'MAE', 'RMSE', 'MAPE (%)', 'R²']].copy()
ml_all['Kategori'] = 'Makine Ogrenmesi'

# DL
dl = dl_df[['Model', 'MAE', 'RMSE', 'MAPE (%)', 'R²']].copy()
dl['Kategori'] = 'Derin Ogrenme'

# Birlestir
combined = pd.concat([ml_all, dl], ignore_index=True)
combined = combined[['Kategori', 'Model', 'MAE', 'RMSE', 'MAPE (%)', 'R²']]

# Siralama: Kategori icinde R2 azalan
combined = combined.sort_values(by=['Kategori', 'R²'], ascending=[True, False]).reset_index(drop=True)

print("Tablo 5.2 Verileri:")
print(combined.to_string(index=False))

# ── Tablo Gorseli Olustur ──
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Tablo baslik
fig.suptitle('Tablo 5.2. Makine Ogrenmesi ve Derin Ogrenme Modellerinin\nTest Veri Kumesi Uzerindeki Performans Sonuclari',
             fontsize=14, fontweight='bold', y=0.97)

# Tablo verileri
col_labels = ['Kategori', 'Model', 'MAE', 'RMSE', 'MAPE (%)', 'R\u00B2']

cell_data = []
for _, row in combined.iterrows():
    cell_data.append([
        row['Kategori'],
        row['Model'],
        f"{row['MAE']:.4f}",
        f"{row['RMSE']:.4f}",
        f"{row['MAPE (%)']:.4f}",
        f"{row['R²']:.4f}"
    ])

table = ax.table(
    cellText=cell_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    colColours=['#2C3E50'] * len(col_labels)
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.8)

# Stil: Header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_text_props(color='white', fontweight='bold', fontsize=11)
    cell.set_edgecolor('#2C3E50')

# En iyi R2 degerini bul
best_r2_idx = combined['R²'].idxmax()

# Stil: Satir satirlar
ml_color_1 = '#EBF5FB'
ml_color_2 = '#D6EAF8'
dl_color_1 = '#EAFAF1'
dl_color_2 = '#D5F5E3'
best_color = '#F9E79F'

ml_count = len(ml_all)

for i in range(len(cell_data)):
    if i < ml_count:
        bg = ml_color_1 if i % 2 == 0 else ml_color_2
    else:
        bg = dl_color_1 if i % 2 == 0 else dl_color_2

    # En iyi R2 satirini vurgula
    if i == best_r2_idx:
        bg = best_color

    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor('#BDC3C7')

        # Kategori sutunu kalin
        if j == 0 and cell_data[i][0] != '':
            cell.set_text_props(fontweight='bold', fontsize=10)

        # Model ismi sola yasla
        if j == 1:
            cell.set_text_props(ha='left')
            cell._loc = 'left'

# Sutun genislikleri
col_widths = [0.20, 0.20, 0.12, 0.12, 0.12, 0.12]
for j, w in enumerate(col_widths):
    for i in range(len(cell_data) + 1):
        table[i, j].set_width(w)

# Not ekle
fig.text(0.5, 0.03,
         'Not: Sari ile vurgulanan satir en yuksek R\u00B2 degerine sahip modeldir.\n'
         'Tum modeller ayni veri on isleme adimlari ve kronolojik bolunme (%85-%15) ile degerlendirilmistir.',
         ha='center', fontsize=9, style='italic', color='#555555')

plt.tight_layout(rect=[0, 0.07, 1, 0.93])

# ── Kaydet ──
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cıktılar")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Tablo_5_2_Tum_Model_Performanslari.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Excel olarak da kaydet
excel_path = os.path.join(output_dir, "Tablo_5_2_Tum_Model_Performanslari.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    combined.to_excel(writer, index=False, sheet_name='Performans')
    
print(f"\n[OK] Tablo gorseli kaydedildi: {output_path}")
print(f"[OK] Excel dosyasi kaydedildi: {excel_path}")
