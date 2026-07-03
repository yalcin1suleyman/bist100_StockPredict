import yfinance as yf
import pandas as pd
import numpy as np

# Test parametreleri
hisseler = ['THYAO.IS', 'EREGL.IS', 'AKBNK.IS', 'GARAN.IS', 'BIMAS.IS', 'TUPRS.IS']
dis_degiskenler = ['TRY=X', '^VIX']
start_date = '2023-01-01'
end_date = '2023-01-15'

tickers = hisseler + dis_degiskenler
print("Veriler çekiliyor...")
df_raw = yf.download(tickers, start=start_date, end=end_date)
print("Çekilen verilerin kolonları:")
print(df_raw.columns)

# Eksik verileri doldurma
df_filled = df_raw.ffill().bfill()

# Uzun formata dönüştürme
long_dfs = []
for stock in hisseler:
    stock_df = pd.DataFrame({
        'Open': df_filled[('Open', stock)],
        'High': df_filled[('High', stock)],
        'Low': df_filled[('Low', stock)],
        'Close': df_filled[('Close', stock)],
        'Volume': df_filled[('Volume', stock)]
    })
    stock_df = stock_df.reset_index()
    stock_df.rename(columns={'Date': 'Tarih'}, inplace=True)
    stock_df['Hisse_Kodu'] = stock
    stock_df['USD_TRY'] = df_filled[('Close', 'TRY=X')].values
    stock_df['VIX'] = df_filled[('Close', '^VIX')].values
    long_dfs.append(stock_df)

final_df = pd.concat(long_dfs, ignore_index=True)
column_order = ['Tarih', 'Hisse_Kodu', 'Open', 'High', 'Low', 'Close', 'Volume', 'USD_TRY', 'VIX']
final_df = final_df[column_order]

print("\nBirleştirilmiş DataFrame (İlk 5 Satır):")
print(final_df.head())
print("\nBirleştirilmiş DataFrame (Son 5 Satır):")
print(final_df.tail())
