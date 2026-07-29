import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 1. MAKİNE ÖĞRENMESİ TABLOSU
df_ml = pd.read_csv('outputs/Table_5_2_ML_Performance.csv')
df_ml = df_ml[['Model', 'MAE', 'RMSE', 'MAPE (%)', 'R²']].copy()
df_ml = df_ml.sort_values(by='R²', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')
fig.suptitle('Tablo 5.2. Makine Ogrenmesi Modellerinin Performans Sonuclari', fontsize=14, fontweight='bold', y=0.9)

col_labels = ['Model', 'MAE', 'RMSE', 'MAPE (%)', 'R\u00B2']
cell_data = []
for _, row in df_ml.iterrows():
    cell_data.append([
        row['Model'],
        f"{row['MAE']:.4f}",
        f"{row['RMSE']:.4f}",
        f"{row['MAPE (%)']:.4f}",
        f"{row['R²']:.4f}"
    ])

table = ax.table(cellText=cell_data, colLabels=col_labels, cellLoc='center', loc='center', colColours=['#2C3E50']*len(col_labels))
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0)

for j in range(len(col_labels)):
    table[0, j].set_text_props(color='white', fontweight='bold', fontsize=12)
    table[0, j].set_edgecolor('#2C3E50')

for i in range(len(cell_data)):
    bg = '#EBF5FB' if i % 2 == 0 else '#D6EAF8'
    if cell_data[i][0] == 'Linear Regression':
        bg = '#F9E79F' # sari vurgu
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor('#BDC3C7')
        if j == 0:
            cell.set_text_props(fontweight='bold', ha='left')
            cell._loc = 'left'

for j, w in enumerate([0.3, 0.15, 0.15, 0.15, 0.15]):
    for i in range(len(cell_data) + 1):
        table[i, j].set_width(w)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Tablo_5_2_Makine_Ogrenmesi.png"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 2. DERİN ÖĞRENME TABLOSU
df_dl = pd.read_csv('outputs/Table_5_3_DL_Performance.csv')
df_dl = df_dl[['Model', 'MAE', 'RMSE', 'MAPE (%)', 'R²']].copy()
df_dl = df_dl.sort_values(by='R²', ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 7))
ax.axis('off')
fig.suptitle('Tablo 5.3. Derin Ogrenme Modellerinin Performans Sonuclari', fontsize=14, fontweight='bold', y=0.9)

cell_data_dl = []
for _, row in df_dl.iterrows():
    cell_data_dl.append([
        row['Model'],
        f"{row['MAE']:.4f}",
        f"{row['RMSE']:.4f}",
        f"{row['MAPE (%)']:.4f}",
        f"{row['R²']:.4f}"
    ])

table_dl = ax.table(cellText=cell_data_dl, colLabels=col_labels, cellLoc='center', loc='center', colColours=['#2C3E50']*len(col_labels))
table_dl.auto_set_font_size(False)
table_dl.set_fontsize(11)
table_dl.scale(1.0, 2.0)

for j in range(len(col_labels)):
    table_dl[0, j].set_text_props(color='white', fontweight='bold', fontsize=12)
    table_dl[0, j].set_edgecolor('#2C3E50')

for i in range(len(cell_data_dl)):
    bg = '#EAFAF1' if i % 2 == 0 else '#D5F5E3'
    if cell_data_dl[i][0] == 'MLP': # DL in en iyisi
        bg = '#F9E79F'
    for j in range(len(col_labels)):
        cell = table_dl[i + 1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor('#BDC3C7')
        if j == 0:
            cell.set_text_props(fontweight='bold', ha='left')
            cell._loc = 'left'

for j, w in enumerate([0.3, 0.15, 0.15, 0.15, 0.15]):
    for i in range(len(cell_data_dl) + 1):
        table_dl[i, j].set_width(w)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Tablo_5_3_Derin_Ogrenme.png"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
