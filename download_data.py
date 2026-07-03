import yfinance as yf
import pandas as pd
import numpy as np

# 1. Aşama: Parametrelerin Tanımlanması
# Veri çekilecek hedef hisse senetleri (BIST100)
hisseler = ['THYAO.IS', 'EREGL.IS', 'AKBNK.IS', 'GARAN.IS', 'BIMAS.IS', 'TUPRS.IS']
# Makroekonomik dışsal değişkenler
dis_degiskenler = ['TRY=X', '^VIX']

# 10 yıllık tarih aralığı (yfinance bitiş tarihini hariç tuttuğu için 2025-01-01 yazılmıştır)
start_date = '2015-01-01'
end_date = '2025-01-01'

# Tüm ticker'ları birleştirerek yfinance üzerinden tek seferde çekiyoruz
tum_tickerlar = hisseler + dis_degiskenler
print("Veriler Yahoo Finance üzerinden indiriliyor...")
df_raw = yf.download(tum_tickerlar, start=start_date, end=end_date)

# 2. Aşama: Eksik Verilerin Doldurulması
# Farklı takvimler ve resmi tatiller sebebiyle oluşabilecek NaN değerleri dolduruyoruz.
# Önce önceki günün verisiyle (forward fill), eğer başta hala boşluk varsa sonraki günün verisiyle (backward fill) doldurulur.
df_filled = df_raw.ffill().bfill()

# 3. Aşama: Verilerin Uzun Formata (Long Format) Dönüştürülmesi
# Her hisse senedi için ayrı bir DataFrame oluşturup, bunları dikeyde birleştireceğiz.
uzun_format_listesi = []

for hisse in hisseler:
    # Hisse senedine ait fiyat ve hacim verilerini alıyoruz
    hisse_df = pd.DataFrame({
        'Open': df_filled[('Open', hisse)],
        'High': df_filled[('High', hisse)],
        'Low': df_filled[('Low', hisse)],
        'Close': df_filled[('Close', hisse)],
        'Volume': df_filled[('Volume', hisse)]
    })
    
    # Tarih indeksini sütun haline getiriyoruz
    hisse_df = hisse_df.reset_index()
    hisse_df.rename(columns={'Date': 'Tarih'}, inplace=True)
    
    # Tarih formatını sadeleştiriyoruz (YYYY-MM-DD)
    hisse_df['Tarih'] = pd.to_datetime(hisse_df['Tarih']).dt.date
    
    # Hisse kodunu ekliyoruz
    hisse_df['Hisse_Kodu'] = hisse
    
    # Dışsal makro değişkenlerin (USD_TRY ve VIX) kapanış değerlerini ekliyoruz
    hisse_df['USD_TRY'] = df_filled[('Close', 'TRY=X')].values
    hisse_df['VIX'] = df_filled[('Close', '^VIX')].values
    
    # Listeye ekliyoruz
    uzun_format_listesi.append(hisse_df)

# Tüm hisse DataFrame'lerini tek bir veri setinde birleştiriyoruz
bist100_df = pd.concat(uzun_format_listesi, ignore_index=True)

# Sütunları istenen sırada düzenliyoruz
kolon_sirasi = ['Tarih', 'Hisse_Kodu', 'Open', 'High', 'Low', 'Close', 'Volume', 'USD_TRY', 'VIX']
bist100_df = bist100_df[kolon_sirasi]

# 4. Aşama: Verilerin Doğrulanması ve Kaydedilmesi
# Veri setinin düzgün oluştuğunu doğrulamak için ilk ve son satırları konsola yazdırıyoruz.
print("\n--- Veri Seti İlk 5 Satır (df.head()) ---")
print(bist100_df.head())

print("\n--- Veri Seti Son 5 Satır (df.tail()) ---")
print(bist100_df.tail())

# Sonucu CSV dosyası olarak kaydediyoruz
csv_dosya_adi = "bist100_sektorel_veri.csv"
bist100_df.to_csv(csv_dosya_adi, index=False, encoding='utf-8')
print(f"\nVeriler başarıyla çekildi ve '{csv_dosya_adi}' dosyasına kaydedildi.")
