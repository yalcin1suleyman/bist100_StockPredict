# BIST 100 Hisse Senedi Tahmini Projesi

Bu proje, BIST 100 hisse senetlerinin makine öğrenmesi algoritmalarıyla tahmin edilmesi ve bu tahminlerin Açıklanabilir Yapay Zeka (XAI) teknikleriyle yorumlanmasını amaçlar.

## 📁 Proje Dosya Yapısı

Proje karmaşıklığı önlemek adına 3 temel klasöre ve numaralandırılmış Python kodlarına bölünmüştür:

### 1. Ana Dizin (Çalıştırılabilir Kodlar)
Modelleri ve işlemleri çalıştırmak için bu numaralandırılmış dosyaları sırasıyla kullanabilirsiniz:
- `download_data.py`: Yahoo Finance gibi kaynaklardan ham borsa ve sektör verilerini indirip `data/` klasörüne kaydeder.
- `01_exploratory_data_analysis.py`: Keşifçi veri analizi (EDA) yapar, veriyi tanımak için grafikler çıkarır.
- `02_data_preprocessing.py`: Verideki eksiklikleri giderir (ffill, bfill, interpolate) ve makine öğrenmesi için temiz bir alt yapı sunar.
- `03_train_ml_models.py`: Belirlenen özellikleri (Ham, Teknik ve Tümü) senaryolar halinde kullanarak LightGBM, XGBoost, Random Forest gibi modelleri eğitir ve başarı oranlarını test eder.
- `04_explainability_xai.py`: Eğitilen en iyi modellerin aldığı kararları SHAP ve LIME algoritmaları ile açıklayarak finansal gruplara göre (Makroekonomik, Volatilite vb.) analiz eder.
- `05_train_dl_models.py`: Makine öğrenmesi modellerine ek olarak Derin Öğrenme (Deep Learning) modellerini eğitir ve değerlendirir.
- `06_statistical_tests.py`: Modellerin tahmin başarıları arasındaki farkların istatistiksel olarak anlamlı olup olmadığını test eder.

### 2. `data/` (Veri Setleri)
Projenin tüm girdileri ve ön işlemden geçmiş veri setleri burada saklanır:
- `bist100_ham_veri.csv`
- `bist100_data_interpolate.csv` (Modellerin kullandığı ana işlenmiş veri) vb.

### 3. `outputs/` (Grafikler ve Sonuçlar)
Kodları çalıştırdığınızda üretilen **tüm grafikler (Fig_...)** ve **başarı tabloları (Table_...)** otomatik olarak bu klasöre kaydedilir. Ana dizin kalabalıklaşmaz.
- Örnek: `Table_5_2_ML_Performance_Scenarios.csv` (Senaryo analiz sonuçları)
- Örnek: `Fig_5_28_SHAP_Grouped_Importance.png` (Gruplandırılmış SHAP sonuçları)
