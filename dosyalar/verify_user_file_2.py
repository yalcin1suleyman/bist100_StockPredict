"""
Doğrulama Scripti: Kullanıcının güncellediği "bist_100 (1).docx" dosyasını kontrol eder.
"""
import docx

FILE = r"c:\stajProje\dosyalar\bist_100 (1).docx"
doc = docx.Document(FILE)
full_text = "\n".join([p.text for p in doc.paragraphs])

print("=" * 60)
print("USER FILE KONTROL RAPORU (YENI)")
print("=" * 60)

checks = [
    ("4: Sekil 5.13'te",
     "Şekil 5.13'te XGBoost",
     True),
    
    ("6: SHAP yazim hatasi duzeltildi",
     "SHAP analizleri yalnızca matematiksel",
     True),
    
    ("7: 5.10 Model Karmasikligi",
     "5.10 Model Karmaşıklığı",
     True),
]

# Olmamasi gereken seyler
negative_checks = [
    ("Kesilmis CNN-LST kalmasin (em-dash veya normal)",
     "CNN\u2013LST",
     False),
    
    ("5.12 numaralandirma kalmasin",
     "5.12 Model Karmaşıklığı",
     False),
     
    ("HAP analizleri kalmasin",
     "HAP analizleri",
     False),
]

all_pass = True

for desc, text, should_exist in checks:
    found = text in full_text
    if found == should_exist:
        print(f"  [OK] {desc}")
    else:
        all_pass = False
        print(f"  [FAIL] {desc} - BEKLENEN DURUM SAGLANAMADI")

print()
for desc, text, should_exist in negative_checks:
    found = text in full_text
    if found == should_exist:
        print(f"  [OK] {desc}")
    else:
        all_pass = False
        print(f"  [FAIL] {desc} - Eski metin hala belgede!")

# Özel CNN-LST kontrolü (tamamen silinip CNN-LSTM oldu mu)
cnn_lstm_found = "CNN-LSTM ve Transformer modellerinin öğrenme eğrileri incelenmiştir" in full_text or \
                 "CNN–LSTM ve Transformer modellerinin öğrenme eğrileri incelenmiştir" in full_text

if cnn_lstm_found:
    print("  [OK] CNN-LSTM cumlesi basariyla duzeltilmis.")
else:
    print("  [FAIL] CNN-LSTM cumlesi hatali veya bulunamadi.")
    all_pass = False

print("\n" + "=" * 60)
if all_pass:
    print("TUM KONTROLLER BASARILI! Kullanicinin dosyasi tamamen dogru.")
else:
    print("BAZI KONTROLLER BASARISIZ! Lutfen FAIL satirlarini inceleyin.")
print("=" * 60)
