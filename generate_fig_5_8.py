"""
Şekil 5.8 - Random Forest Tabanlı Ön Özellik Önem Analizi
===========================================================
Açıklanabilir yapay zekâ (XAI) analizlerinden önce, değişkenlerin modelin
tahmin performansına olası katkılarını ön değerlendirme amacıyla inceler.
Tüm özellikler (ham fiyat + teknik göstergeler) dahil edilmiştir.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

warnings.filterwarnings("ignore")

# ── Stil Ayarları ──
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# ── Veri Yükleme ──
file_path = "data/bist100_data_interpolate.csv"
print(f"Veri yükleniyor: {file_path}")
df = pd.read_csv(file_path)
df['Tarih'] = pd.to_datetime(df['Tarih'])
df = df.sort_values(by='Tarih').reset_index(drop=True)

# ── Hedef Değişken: Bir sonraki günün kapanışı ──
df['Target'] = df.groupby('Hisse_Kodu')['Close'].shift(-1)
df = df.dropna(subset=['Target']).reset_index(drop=True)

# ── Özellik Sütunları (Tüm ham fiyatlar DAHİL) ──
drop_cols = ['Tarih', 'Set', 'Hisse_Kodu', 'Target']
feature_cols = [col for col in df.columns if col not in drop_cols]
print(f"Kullanılan özellikler ({len(feature_cols)}): {feature_cols}")

# ── Train/Test Bölme (Kronolojik %85-%15) ──
split_index = int(len(df) * 0.85)
train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

# ── Normalizasyon ──
scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(train_df[feature_cols].values)
y_train = train_df['Target'].values

# ── Random Forest Eğitimi ──
print("Random Forest modeli eğitiliyor (ön özellik önem analizi)...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# ── Özellik Önemleri ──
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Özellik': feature_cols,
    'Önem': importances
}).sort_values(by='Önem', ascending=True)

print("\n--- Özellik Önem Sıralaması ---")
for _, row in importance_df.iterrows():
    print(f"  {row['Özellik']:25s} : {row['Önem']:.6f}")

# ── Grafik Oluşturma ──
fig, ax = plt.subplots(figsize=(10, 9))

# Renk paleti: en önemliden en az önemliye gradient
colors = sns.color_palette("viridis", n_colors=len(importance_df))

bars = ax.barh(
    importance_df['Özellik'],
    importance_df['Önem'],
    color=colors,
    edgecolor='white',
    linewidth=0.5,
    height=0.7
)

# Bar uçlarına değer yaz
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.002,
        bar.get_y() + bar.get_height() / 2.,
        f'{width:.4f}',
        ha='left', va='center', fontsize=8, color='#333333'
    )

ax.set_xlabel('Özellik Önemi (Gini Importance)', fontsize=12, fontweight='bold')
ax.set_title('Şekil 5.8. Random Forest Tabanlı Ön Özellik Önem Analizi', fontsize=13, fontweight='bold', pad=15)
ax.tick_params(axis='y', labelsize=9)
ax.tick_params(axis='x', labelsize=9)

# X ekseni sınırlarını biraz genişlet (bar uçlarındaki yazılar için)
x_max = importance_df['Önem'].max()
ax.set_xlim(0, x_max * 1.18)

# Grid
ax.xaxis.grid(True, alpha=0.3)
ax.yaxis.grid(False)
ax.set_axisbelow(True)

plt.tight_layout()

# ── Kaydet ──
output_dir = "cıktılar"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Sekil_5_8_RF_On_Ozellik_Onem_Analizi.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n[OK] Grafik basariyla kaydedildi: {output_path}")
