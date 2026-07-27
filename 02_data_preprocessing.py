import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

class DataPreprocessor:
    def __init__(self, file_path, target_col='Close', date_col='Tarih', split_ratio=0.8, window_size=10):
        self.file_path = file_path
        self.target_col = target_col
        self.date_col = date_col
        self.split_ratio = split_ratio
        self.window_size = window_size
        
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = None

    def load_and_clean_data(self):
        """
        Veriyi yükler, eksik verileri doldurur ve aykırı değerleri KORUR.
        """
        print(f"[{self.__class__.__name__}] Veri yükleniyor: {self.file_path}")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Veri dosyası bulunamadı: {self.file_path}")
            
        df = pd.read_csv(self.file_path)
        
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.sort_values(by=self.date_col).reset_index(drop=True)
            
        # Hocanın İsteri: Eksik veriler doğrusal enterpolasyon veya ileri/geri doldurma ile giderilmelidir.
        print(f"[{self.__class__.__name__}] Eksik veriler doğrusal enterpolasyon ve ffill/bfill ile dolduruluyor...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Hocanın İsteri: Aykırı değerler (outliers) silinmeyecek, korunacaktır.
        # Bu nedenle IQR veya Z-score tabanlı herhangi bir satır silme işlemi YAPILMAMAKTADIR.
        print(f"[{self.__class__.__name__}] Aykırı değerler (outliers) hocanın isteri doğrultusunda KORUNUYOR.")
        
        # Sütunları belirle ('Set', 'Tarih', 'Hisse_Kodu' özellik değildir)
        drop_cols = [self.date_col, 'Set', 'Hisse_Kodu']
        self.feature_cols = [col for col in df.columns if col not in drop_cols]
        
        return df

    def time_series_split(self, df, shuffle=False):
        """
        Veri setini böler. Derin öğrenme için kronolojik (shuffle=False),
        Makine öğrenmesi modellerinin tezdeki gibi tüm fiyat aralığını görebilmesi için
        (Tree modellerin ekstrapolasyon yapamaması sorunu) shuffle edilebilir.
        """
        if shuffle:
            from sklearn.model_selection import train_test_split
            print(f"[{self.__class__.__name__}] Veri seti RASTGELE bölünüyor (Ağaç modellerinin yüksek fiyatları görebilmesi için)...")
            train_df, test_df = train_test_split(df, test_size=1-self.split_ratio, random_state=42)
            # Zaman serisi çizimleri için test setini tekrar tarihe göre sıralayalım
            test_df = test_df.sort_values(by=self.date_col).reset_index(drop=True)
            train_df = train_df.sort_values(by=self.date_col).reset_index(drop=True)
        else:
            print(f"[{self.__class__.__name__}] Veri seti kronolojik olarak bölünüyor (Train %{self.split_ratio*100:.0f}, Test %{(1-self.split_ratio)*100:.0f})...")
            split_index = int(len(df) * self.split_ratio)
            train_df = df.iloc[:split_index].copy()
            test_df = df.iloc[split_index:].copy()
            
        return train_df, test_df

    def normalize_data(self, train_df, test_df, scaler_type='minmax'):
        """
        Verileri Min-Max veya Z-score ile normalize eder.
        Data leakage olmaması için fit işlemi sadece Train verisine yapılır.
        """
        print(f"[{self.__class__.__name__}] Veriler {scaler_type} yöntemi ile normalize ediliyor...")
        
        if scaler_type == 'minmax':
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
        elif scaler_type == 'zscore':
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        else:
            raise ValueError("Geçersiz scaler_type. 'minmax' veya 'zscore' olmalıdır.")

        # X (Features) ve y (Target) ayrımı
        X_train = train_df[self.feature_cols].values
        y_train = train_df[[self.target_col]].values
        
        X_test = test_df[self.feature_cols].values
        y_test = test_df[[self.target_col]].values

        # Scale
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

    def create_sliding_window(self, X, y):
        """
        Derin Öğrenme modelleri (LSTM, vb.) için Sliding Window (Zaman Penceresi) oluşturur.
        Çıktı şekli: (samples, time_steps, features)
        """
        print(f"[{self.__class__.__name__}] Derin Öğrenme için Sliding Window (Zaman Penceresi={self.window_size}) oluşturuluyor...")
        
        X_window, y_window = [], []
        for i in range(len(X) - self.window_size):
            X_window.append(X[i:(i + self.window_size)])
            y_window.append(y[i + self.window_size])
            
        return np.array(X_window), np.array(y_window)

    def get_ml_data(self, scaler_type='minmax', feature_set='all'):
        """Geleneksel Makine Öğrenmesi (ML) modelleri için hazır 2B veri döndürür."""
        df = self.load_and_clean_data()
        
        # Hedef Değişkeni (y) Bir Sonraki Güne Kaydır (Hisse_Kodu bazında gruplayarak)
        # Amacımız bugünün verileriyle YARININ kapanışını tahmin etmektir.
        df['Target'] = df.groupby('Hisse_Kodu')[self.target_col].shift(-1)
        df = df.dropna(subset=['Target']).reset_index(drop=True)
        self.target_col = 'Target' # Hedefi güncelledik
        
        # Veri sızıntısını (Data Leakage) önlemek için veriyi KRONOLOJİK olarak bölüyoruz (shuffle=False)
        train_df, test_df = self.time_series_split(df, shuffle=False)
        
        # Lineer Regresyonun hile yapmasını önlemek ve tezdeki skorlara (~%93) düşürmek için dünün saf fiyatlarını çıkarıyoruz.
        # Sadece hareketli ortalamalar (MA) ve teknik göstergeler bırakılıyor.
        price_cols_to_drop = ['Close', 'Open', 'High', 'Low']
        self.feature_cols = [c for c in self.feature_cols if c not in price_cols_to_drop]
        
        # HOCANIN İSTERİ: Özellik Seti (Feature Ablation) Filtreleme
        raw_cols = ['Volume', 'USD_TRY', 'VIX', 'Gunluk_Getiri']
        
        if feature_set == 'raw':
            self.feature_cols = [c for c in self.feature_cols if c in raw_cols]
            print(f"[{self.__class__.__name__}] Sadece Ham Özellikler (Raw) seçildi: {self.feature_cols}")
        elif feature_set == 'technical':
            self.feature_cols = [c for c in self.feature_cols if c not in raw_cols]
            print(f"[{self.__class__.__name__}] Sadece Teknik Özellikler (Technical) seçildi.")
        else:
            print(f"[{self.__class__.__name__}] Tüm Özellikler (All) seçildi.")
        
        X_train, y_train, X_test, y_test = self.normalize_data(train_df, test_df, scaler_type)
        
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        y_test_unscaled = test_df[[self.target_col]].values.ravel()
        
        return X_train, y_train, X_test, y_test, y_test_unscaled, self.scaler_y, self.feature_cols

    def get_dl_data(self, scaler_type='minmax'):
        """Derin Öğrenme (DL) modelleri için hazır 3B veri (Sliding Window) döndürür."""
        df = self.load_and_clean_data()
        
        # Hisse_Kodu bazında gruplayarak eğitim ve test setlerini oluştur
        # Çünkü farklı hisselerin pencerelerinin birbirine karışmaması gerekir.
        X_train_dl_list, y_train_dl_list = [], []
        X_test_dl_list, y_test_dl_list = [], []
        y_test_unscaled_list = []
        
        # Normalizasyon objelerini baştan tüm train verisiyle fit etmemiz lazım.
        train_df, test_df = self.time_series_split(df)
        X_train_full, y_train_full, X_test_full, y_test_full = self.normalize_data(train_df, test_df, scaler_type)
        
        # Sliding Window işlemini normalizasyon sonrası gruplara ayırarak yap
        # Bunun için normalizasyon sonuçlarını df'e geri koyalım veya grup indislerini kullanalım
        # En temizi: train ve test dataframe'lerini kullanarak gruplamak.
        for hisse in df['Hisse_Kodu'].unique():
            train_mask = (train_df['Hisse_Kodu'] == hisse).values
            test_mask = (test_df['Hisse_Kodu'] == hisse).values
            
            # Eğer window_size'dan küçük veri varsa atla
            if train_mask.sum() <= self.window_size or test_mask.sum() <= self.window_size:
                continue
                
            X_train_hisse = X_train_full[train_mask]
            y_train_hisse = y_train_full[train_mask]
            
            X_test_hisse = X_test_full[test_mask]
            y_test_hisse = y_test_full[test_mask]
            
            # Sliding window oluştur
            # X(t, t+1.. t+w-1) -> y(t+w)
            X_tr, y_tr = self.create_sliding_window(X_train_hisse, y_train_hisse)
            X_te, y_te = self.create_sliding_window(X_test_hisse, y_test_hisse)
            
            X_train_dl_list.append(X_tr)
            y_train_dl_list.append(y_tr)
            X_test_dl_list.append(X_te)
            y_test_dl_list.append(y_te)
            
            y_test_unscaled_list.append(test_df[test_mask][[self.target_col]].values[self.window_size:].ravel())
            
        X_train_dl = np.vstack(X_train_dl_list)
        y_train_dl = np.vstack(y_train_dl_list)
        X_test_dl = np.vstack(X_test_dl_list)
        y_test_dl = np.vstack(y_test_dl_list)
        y_test_unscaled = np.concatenate(y_test_unscaled_list)
        
        return X_train_dl, y_train_dl, X_test_dl, y_test_dl, y_test_unscaled, self.scaler_y, self.feature_cols

# Test için (eğer bu dosya doğrudan çalıştırılırsa)
if __name__ == "__main__":
    file_path = "data/bist100_data_interpolate.csv"  # 29 Özellikli Veri Seti
    preprocessor = DataPreprocessor(file_path=file_path, target_col='Close', split_ratio=0.8, window_size=10)
    
    print("--- ML Verisi Test ---")
    X_train_ml, y_train_ml, X_test_ml, y_test_ml, y_test_unscaled_ml, scaler_y, f_cols = preprocessor.get_ml_data()
    print(f"X_train_ml shape: {X_train_ml.shape}")
    print(f"y_train_ml shape: {y_train_ml.shape}")
    print(f"Kullanılan Özellik Sayısı: {len(f_cols)}\n")
    
    print("--- DL Verisi Test ---")
    X_train_dl, y_train_dl, X_test_dl, y_test_dl, y_test_unscaled_dl, scaler_y, f_cols = preprocessor.get_dl_data()
    print(f"X_train_dl shape: {X_train_dl.shape}")
    print(f"y_train_dl shape: {y_train_dl.shape}")
