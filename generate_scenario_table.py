import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Veriyi oku
df = pd.read_csv('outputs/Table_5_2_ML_Performance_Scenarios.csv')

# Senaryo isimlerini Turkcelestir
scenario_map = {
    'raw': 'Senaryo 1 (Ham Veri)',
    'technical': 'Senaryo 2 (Teknik Gostergeler)',
    'all': 'Senaryo 3 (Tum Ozellikler)'
}
df['Senaryo'] = df['Scenario'].map(scenario_map)

# Tablo icin sutunlari sec
table_df = df[['Senaryo', 'Model', 'MAE', 'RMSE', 'MAPE (%)', 'R²']].copy()

# R2 ye gore siralayalim ama once Senaryo
table_df['Senaryo_Cat'] = pd.Categorical(table_df['Senaryo'], categories=[
    'Senaryo 1 (Ham Veri)', 'Senaryo 2 (Teknik Gostergeler)', 'Senaryo 3 (Tum Ozellikler)'
], ordered=True)
table_df = table_df.sort_values(by=['Senaryo_Cat', 'R²'], ascending=[True, False]).drop(columns=['Senaryo_Cat']).reset_index(drop=True)

# Gorsel olusturma
fig, ax = plt.subplots(figsize=(14, 12)) 
ax.axis('off')

# Baslik
fig.suptitle('Tablo 5.2.1. Makine Ogrenmesi Modellerinin Farkli Ozellik Senaryolarindaki Performanslari',
             fontsize=14, fontweight='bold', y=0.96)

col_labels = ['Senaryo', 'Model', 'MAE', 'RMSE', 'MAPE (%)', 'R\u00B2']

cell_data = []
for _, row in table_df.iterrows():
    cell_data.append([
        row['Senaryo'],
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

# Header stili
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_text_props(color='white', fontweight='bold', fontsize=11)
    cell.set_edgecolor('#2C3E50')

# Renklendirme
color_map = {
    'Senaryo 1 (Ham Veri)': ('#FADBD8', '#F5B7B1'), # Kirmizi tonlari (Basarisiz)
    'Senaryo 2 (Teknik Gostergeler)': ('#EBF5FB', '#D6EAF8'), # Mavi tonlari (Basarili)
    'Senaryo 3 (Tum Ozellikler)': ('#EAFAF1', '#D5F5E3') # Yesil tonlari (En iyi)
}

for i in range(len(cell_data)):
    senaryo = cell_data[i][0]
    c1, c2 = color_map[senaryo]
    bg = c1 if i % 2 == 0 else c2
    
    for j in range(len(col_labels)):
        cell = table[i + 1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor('#BDC3C7')
        
        if j == 0:
            cell.set_text_props(fontweight='bold', fontsize=9)
        if j == 1:
            cell.set_text_props(ha='left')
            cell._loc = 'left'

# Sutun genislikleri
col_widths = [0.25, 0.20, 0.12, 0.12, 0.12, 0.12]
for j, w in enumerate(col_widths):
    for i in range(len(cell_data) + 1):
        table[i, j].set_width(w)

fig.text(0.5, 0.05,
         'Not: Senaryo 1 sadece ham fiyat verilerini, Senaryo 2 fiyat ve teknik gostergeleri, Senaryo 3 ise tum ozellikleri icermektedir.\n'
         'Ham veri senaryosunda modellerin R\u00B2 degerlerinin negatif cikmasi, egitimin basarisiz oldugunu gostermektedir.',
         ha='center', fontsize=10, style='italic', color='#555555')

plt.tight_layout(rect=[0, 0.08, 1, 0.94])

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Tablo_5_2_1_Senaryo_Karsilastirmasi.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Excel kaydet
excel_path = os.path.join(output_dir, "Tablo_5_2_1_Senaryo_Karsilastirmasi.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    table_df.to_excel(writer, index=False, sheet_name='Senaryolar')
    
print(f"[OK] Senaryo tablosu kaydedildi: {output_path}")
