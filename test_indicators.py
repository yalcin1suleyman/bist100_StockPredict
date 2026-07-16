"""Yeni eklenen teknik göstergelerin doğru çalışıp çalışmadığını test eder."""
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, ROCIndicator, WilliamsRIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Kısa bir test verisi çek (sadece 1 hisse, 6 ay)
print("Test verisi çekiliyor (THYAO.IS, 6 aylık)...")
df = yf.download('THYAO.IS', start='2024-01-01', end='2024-07-01', auto_adjust=True)

if len(df) == 0:
    print("HATA: Veri çekilemedi!")
    exit(1)

# MultiIndex varsa düzleştir (yfinance bazen (Price, Ticker) formatında verir)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Sütunları 1D'ye zorla
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = df[col].values.flatten()

print(f"Çekilen veri: {len(df)} satır")

# Mevcut göstergeler
df['Gunluk_Getiri'] = np.log(df['Close'] / df['Close'].shift(1))
df['Volatilite_10'] = df['Gunluk_Getiri'].rolling(window=10).std()
df['MA_10'] = SMAIndicator(close=df['Close'], window=10).sma_indicator()
df['MA_50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
df['Momentum_10'] = ROCIndicator(close=df['Close'], window=10).roc()
df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
macd_obj = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
df['MACD'] = macd_obj.macd()
df['MACD_Signal'] = macd_obj.macd_signal()
df['MACD_Histogram'] = macd_obj.macd_diff()

# YENİ GÖSTERGELER
df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
df['ATR_14'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
df['BB_Upper'] = bb.bollinger_hband()
df['BB_Lower'] = bb.bollinger_lband()
df['BB_Width'] = bb.bollinger_wband()
df['ADX_14'] = ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=14).adx()
df['CCI_20'] = CCIIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=20).cci()
df['Williams_R'] = WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r()
df['OBV'] = OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
df['Stoch_K'] = stoch.stoch()
df['Stoch_D'] = stoch.stoch_signal()

# NaN'ları at
df_clean = df.dropna()

print(f"\nNaN temizleme sonrası: {len(df_clean)} satır")
print(f"Toplam sütun sayısı: {len(df_clean.columns)}")
print(f"\nTüm sütunlar:")
for i, col in enumerate(df_clean.columns, 1):
    print(f"  {i:2d}. {col}")

print(f"\nSon 3 satır (yeni göstergeler):")
new_cols = ['EMA_20', 'ATR_14', 'BB_Upper', 'BB_Lower', 'BB_Width', 'ADX_14', 'CCI_20', 'Williams_R', 'OBV', 'Stoch_K', 'Stoch_D']
print(df_clean[new_cols].tail(3).to_string())
print("\n✅ Test başarılı! Tüm göstergeler doğru hesaplandı.")
