import yfinance as yf
import pandas as pd
import numpy as np
import sys
from sklearn.impute import KNNImputer

# ta kütüphanesi modülleri
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, ROCIndicator, WilliamsRIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# --- 1. Parametrelerin Tanımlanması ---
HISSELER = ['THYAO.IS', 'EREGL.IS', 'AKBNK.IS', 'GARAN.IS', 'BIMAS.IS', 'TUPRS.IS']
DIS_DEGISKENLER = ['TRY=X', '^VIX']
START_DATE = '2015-01-01'
END_DATE = '2025-01-01'

def fetch_raw_data():
    """Yahoo Finance'ten ham fiyat ve makro verileri çeker."""
    print("Veriler Yahoo Finance üzerinden indiriliyor...")
    df_stocks = yf.download(HISSELER, start=START_DATE, end=END_DATE, auto_adjust=True)
    
    if len(df_stocks) == 0:
        print("HATA: Yahoo Finance'ten hisse verisi çekilemedi. Bağlantınızı veya ticker'ları kontrol edin.")
        sys.exit(1)
        
    assert isinstance(df_stocks.columns, pd.MultiIndex), "Beklenmeyen veri formatı: Sütunlar MultiIndex değil!"
    
    df_macro = yf.download(DIS_DEGISKENLER, start=START_DATE, end=END_DATE, auto_adjust=True)
    
    if len(df_macro) == 0:
        print("UYARI: Makro değişkenler çekilemedi. Sütunlar boş kalacak.")
        
    return df_stocks, df_macro

def align_and_prepare_raw_df(df_stocks, df_macro):
    """Verileri uzun formata getirir ve master takvime oturtur (NaN doldurma henüz yapılmaz)."""
    master_calendar = df_stocks.index
    df_macro_aligned = df_macro.reindex(master_calendar)
    
    uzun_format_listesi = []
    
    for hisse in HISSELER:
        hisse_df = pd.DataFrame({
            'Tarih': master_calendar,
            'Open': df_stocks[('Open', hisse)].values,
            'High': df_stocks[('High', hisse)].values,
            'Low': df_stocks[('Low', hisse)].values,
            'Close': df_stocks[('Close', hisse)].values,
            'Volume': df_stocks[('Volume', hisse)].values
        })
        
        # Makro değişkenleri NaN ile eklenecek. Doldurma fonksiyonları buraları da dolduracak.
        if len(df_macro_aligned) > 0 and ('Close', 'TRY=X') in df_macro_aligned.columns:
            hisse_df['USD_TRY'] = df_macro_aligned[('Close', 'TRY=X')].values
        else:
            hisse_df['USD_TRY'] = np.nan
            
        if len(df_macro_aligned) > 0 and ('Close', '^VIX') in df_macro_aligned.columns:
            hisse_df['VIX'] = df_macro_aligned[('Close', '^VIX')].values
        else:
            hisse_df['VIX'] = np.nan
            
        hisse_df['Hisse_Kodu'] = hisse
        uzun_format_listesi.append(hisse_df)
        
    final_df = pd.concat(uzun_format_listesi, ignore_index=True)
    final_df['Tarih'] = pd.to_datetime(final_df['Tarih']) # interpolate için datetime şart
    return final_df

# --- 2. Eksik Veri Doldurma (Imputation) Fonksiyonları ---

def apply_ffill(df):
    """Versiyon A: Yalnızca ffill (baseline) ile eksik verileri doldurur."""
    print("Versiyon A (Forward Fill) hesaplanıyor...")
    filled_df = df.copy()
    columns_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume', 'USD_TRY', 'VIX']
    
    # Sadece ilgili kolonları ffill ile güncelliyoruz, diğer sütunlara (Hisse_Kodu, Tarih) dokunmuyoruz.
    filled_df[columns_to_fill] = filled_df.groupby('Hisse_Kodu')[columns_to_fill].ffill()
    
    filled_df = filled_df.dropna(subset=['Close']) # Başta ffill alamayıp boş kalanları at
    return filled_df.reset_index(drop=True)

def apply_interpolation(df):
    """Versiyon B: Zaman endeksine duyarlı doğrusal (linear) interpolasyon."""
    print("Versiyon B (Linear Interpolation) hesaplanıyor...")
    filled_df = df.copy()
    columns_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume', 'USD_TRY', 'VIX']
    
    dfs_to_concat = []
    
    for hisse, group in filled_df.groupby('Hisse_Kodu'):
        group = group.set_index('Tarih')
        group[columns_to_fill] = group[columns_to_fill].interpolate(method='time')
        group = group.reset_index()
        dfs_to_concat.append(group)
        
    filled_df = pd.concat(dfs_to_concat, ignore_index=True)
    filled_df = filled_df.dropna(subset=['Close'])
    return filled_df.reset_index(drop=True)

def apply_knn_imputation(df):
    """Versiyon C: Makine öğrenmesi tabanlı KNN Imputation (k=5)."""
    print("Versiyon C (KNN Imputer) hesaplanıyor...")
    filled_df = df.copy()
    columns_to_fill = ['Open', 'High', 'Low', 'Close', 'Volume', 'USD_TRY', 'VIX']
    imputer = KNNImputer(n_neighbors=5)
    
    dfs_to_concat = []
    
    for hisse, group in filled_df.groupby('Hisse_Kodu'):
        if group[columns_to_fill].isna().all().all():
            dfs_to_concat.append(group)
            continue
            
        imputed_values = imputer.fit_transform(group[columns_to_fill])
        group_copy = group.copy()
        for i, col in enumerate(columns_to_fill):
            group_copy[col] = imputed_values[:, i]
        dfs_to_concat.append(group_copy)
        
    filled_df = pd.concat(dfs_to_concat, ignore_index=True)
    filled_df = filled_df.dropna(subset=['Close'])
    return filled_df.reset_index(drop=True)

# --- 3. Teknik Gösterge Fonksiyonu ---

def calculate_technical_indicators(df):
    """Eksik verileri doldurulmuş dataframe'e teknik göstergeleri hesaplar."""
    dfs_to_concat = []
    
    for hisse, group in df.groupby('Hisse_Kodu'):
        hisse_df = group.copy()
        
        # Logaritmik Getiri
        hisse_df['Gunluk_Getiri'] = np.log(hisse_df['Close'] / hisse_df['Close'].shift(1))
        
        # Volatilite (Ham fiyat değil, getiri üzerinden standart sapma)
        hisse_df['Volatilite_10'] = hisse_df['Gunluk_Getiri'].rolling(window=10).std()
        
        # Hareketli Ortalamalar (MA)
        hisse_df['MA_10'] = SMAIndicator(close=hisse_df['Close'], window=10).sma_indicator()
        hisse_df['MA_50'] = SMAIndicator(close=hisse_df['Close'], window=50).sma_indicator()
        
        # Momentum (Rate of Change)
        hisse_df['Momentum_10'] = ROCIndicator(close=hisse_df['Close'], window=10).roc()
        
        # RSI (14)
        hisse_df['RSI_14'] = RSIIndicator(close=hisse_df['Close'], window=14).rsi()
        
        # MACD (12, 26, 9)
        macd_obj = MACD(close=hisse_df['Close'], window_slow=26, window_fast=12, window_sign=9)
        hisse_df['MACD'] = macd_obj.macd()
        hisse_df['MACD_Signal'] = macd_obj.macd_signal()
        hisse_df['MACD_Histogram'] = macd_obj.macd_diff()
        
        # --- Yeni Eklenen Teknik Göstergeler ---
        
        # EMA (Üstel Hareketli Ortalama - 20 günlük)
        hisse_df['EMA_20'] = EMAIndicator(close=hisse_df['Close'], window=20).ema_indicator()
        
        # ATR (Ortalama Gerçek Aralık - 14 günlük volatilite ölçüsü)
        hisse_df['ATR_14'] = AverageTrueRange(high=hisse_df['High'], low=hisse_df['Low'], close=hisse_df['Close'], window=14).average_true_range()
        
        # Bollinger Bantları (20 günlük, 2 standart sapma)
        bb = BollingerBands(close=hisse_df['Close'], window=20, window_dev=2)
        hisse_df['BB_Upper'] = bb.bollinger_hband()
        hisse_df['BB_Lower'] = bb.bollinger_lband()
        hisse_df['BB_Width'] = bb.bollinger_wband()
        
        # ADX (Ortalama Yönsel Hareket Endeksi - trend gücü)
        hisse_df['ADX_14'] = ADXIndicator(high=hisse_df['High'], low=hisse_df['Low'], close=hisse_df['Close'], window=14).adx()
        
        # CCI (Emtia Kanal Endeksi - fiyatın ortalamadan sapması)
        hisse_df['CCI_20'] = CCIIndicator(high=hisse_df['High'], low=hisse_df['Low'], close=hisse_df['Close'], window=20).cci()
        
        # Williams %R (aşırı alım/satım göstergesi)
        hisse_df['Williams_R'] = WilliamsRIndicator(high=hisse_df['High'], low=hisse_df['Low'], close=hisse_df['Close'], lbp=14).williams_r()
        
        # OBV (Denge Hacmi - hacim bazlı trend teyidi)
        hisse_df['OBV'] = OnBalanceVolumeIndicator(close=hisse_df['Close'], volume=hisse_df['Volume']).on_balance_volume()
        
        # Stochastic Oscillator (momentum göstergesi)
        # Stochastic Oscillator (momentum göstergesi)
        stoch = StochasticOscillator(high=hisse_df['High'], low=hisse_df['Low'], close=hisse_df['Close'], window=14, smooth_window=3)
        hisse_df['Stoch_K'] = stoch.stoch()
        hisse_df['Stoch_D'] = stoch.stoch_signal()

        # VWAP (Hacim Ağırlıklı Ortalama Fiyat)
        from ta.volume import VolumeWeightedAveragePrice
        hisse_df['VWAP'] = VolumeWeightedAveragePrice(high=hisse_df['High'], low=hisse_df['Low'], close=hisse_df['Close'], volume=hisse_df['Volume'], window=14).volume_weighted_average_price()
        
        dfs_to_concat.append(hisse_df)
        
    df_out = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Göstergelerin geçmiş periyotlara ihtiyaç duymasından oluşan başlangıçtaki NaN'ları sil
    df_out = df_out.dropna()
    return df_out.reset_index(drop=True)

# --- 4. Veri Bölme ve Dosya Kaydetme Fonksiyonu ---

def generate_model_specific_files(df, suffix):
    """Ana makine öğrenmesi ve derin öğrenme modelleri için tek bir temiz dosya kaydeder."""
    dfs_to_concat = []
    
    for hisse, group in df.groupby('Hisse_Kodu'):
        group_copy = group.copy()
        split_idx = int(len(group_copy) * 0.85)  # %85 (Train+Val), %15 Test
        group_copy['Set'] = 'Train'
        group_copy.iloc[split_idx:, group_copy.columns.get_loc('Set')] = 'Test'
        dfs_to_concat.append(group_copy)
        
    df_out = pd.concat(dfs_to_concat, ignore_index=True)
    
    # Sütunları düzenle, Tarihi okunabilir formata çevir
    df_out['Tarih'] = pd.to_datetime(df_out['Tarih']).dt.date
    temel_kolonlar = ['Tarih', 'Hisse_Kodu', 'Set', 'Open', 'High', 'Low', 'Close', 'Volume', 'USD_TRY', 'VIX']
    diger_kolonlar = [col for col in df_out.columns if col not in temel_kolonlar]
    df_out = df_out[temel_kolonlar + diger_kolonlar]

    # Tek ve temiz veri seti formatı (ARIMA/Prophet kalıntıları temizlendi)
    dosya_adi = f"data/bist100_data_{suffix}.csv"
    df_out.to_csv(dosya_adi, index=False, encoding='utf-8')
    print(f" -> {dosya_adi} kaydedildi.")

# --- 5. Ana Yürütme Bloğu ---

if __name__ == "__main__":
    df_stocks, df_macro = fetch_raw_data()
    raw_df = align_and_prepare_raw_df(df_stocks, df_macro)
    
    # Ham veriyi (imputation ve indikatör öncesi saf hali) diske kaydet
    raw_df.to_csv("data/bist100_ham_veri.csv", index=False, encoding='utf-8')
    print(" -> bist100_ham_veri.csv (Ham Veri) kaydedildi.")
    
    # Versiyon A: FFill
    df_ffill = apply_ffill(raw_df)
    df_ffill_ind = calculate_technical_indicators(df_ffill)
    generate_model_specific_files(df_ffill_ind.copy(), "ffill")
    
    # Versiyon B: Interpolate
    df_interp = apply_interpolation(raw_df)
    df_interp_ind = calculate_technical_indicators(df_interp)
    generate_model_specific_files(df_interp_ind.copy(), "interpolate")
    
    # Versiyon C: KNN
    df_knn = apply_knn_imputation(raw_df)
    df_knn_ind = calculate_technical_indicators(df_knn)
    generate_model_specific_files(df_knn_ind.copy(), "knn")
    
    print("\n[TAMAM] Toplam 9 adet CSV dosyası başarıyla üretildi.")
    
    # --- Karşılaştırma Bloğu ---
    print("\n--- İMPUTASYON YÖNTEMLERİ KARŞILAŞTIRMA ÖZETİ ---")
    
    # Satırların tarih olarak mükemmel hizalanması için merge işlemi uyguluyoruz.
    compare_df = pd.merge(df_ffill_ind[['Tarih', 'Hisse_Kodu', 'Close']], 
                          df_interp_ind[['Tarih', 'Hisse_Kodu', 'Close']], 
                          on=['Tarih', 'Hisse_Kodu'], 
                          suffixes=('_ffill', '_interp'))
                          
    compare_df = pd.merge(compare_df, 
                          df_knn_ind[['Tarih', 'Hisse_Kodu', 'Close']], 
                          on=['Tarih', 'Hisse_Kodu'])
    compare_df.rename(columns={'Close': 'Close_knn'}, inplace=True)
    
    diff_ffill_interp = (compare_df['Close_ffill'] != compare_df['Close_interp']).sum()
    diff_ffill_knn = (compare_df['Close_ffill'] != compare_df['Close_knn']).sum()
    
    # Sadece fark olan satırların mutlak fark ortalaması (Yoksa sıfırlar ortalamayı çok düşürür)
    mask_interp = compare_df['Close_ffill'] != compare_df['Close_interp']
    mad_ffill_interp = np.abs(compare_df.loc[mask_interp, 'Close_ffill'] - compare_df.loc[mask_interp, 'Close_interp']).mean()
    if pd.isna(mad_ffill_interp): mad_ffill_interp = 0.0

    mask_knn = compare_df['Close_ffill'] != compare_df['Close_knn']
    mad_ffill_knn = np.abs(compare_df.loc[mask_knn, 'Close_ffill'] - compare_df.loc[mask_knn, 'Close_knn']).mean()
    if pd.isna(mad_ffill_knn): mad_ffill_knn = 0.0
    
    print(f"1. FFill vs. Linear Interpolation : {diff_ffill_interp} farklı hücre. (Farklı olanların Ortalama Mutlak Farkı: {mad_ffill_interp:.4f})")
    print(f"2. FFill vs. KNN Imputer          : {diff_ffill_knn} farklı hücre. (Farklı olanların Ortalama Mutlak Farkı: {mad_ffill_knn:.4f})")
    print("\nBu farklılık metrikleri 'eksik veri stratejisinin modele etkisi' başlığı altında ampirik veri olarak kullanılabilir.")
