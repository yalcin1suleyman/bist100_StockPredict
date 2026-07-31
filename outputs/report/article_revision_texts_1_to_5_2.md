# Makale Duzeltme Metinleri ve Proje Denetim Notu

Bu dosya, proje ciktisi ile makalenin 1-5.2 araliginin uyumlu hale
getirilmesi icin hazirlanmistir.

## Proje Denetim Sonucu

Veri ve bolme yapisi kontrol edildi:

- Hisse kapsami: AKBNK.IS, BIMAS.IS, EREGL.IS, GARAN.IS, THYAO.IS, TUPRS.IS
- Ozellik tarihi: 2015-01-01 ile 2024-12-31
- Model girdi son tarihi: 2024-12-30
- Son hedef tarih: 2024-12-31
- Ozellik sayisi: 28
- Temel ozellik: 7
- Turetilmis/teknik ozellik: 21
- Split: her hisse icin 1788 train, 383 validation, 384 test
- Split kronolojiktir; test araligi 2023-06-21 ile 2024-12-31 arasindadir.
- Close_Next, Target_Tarih ve Target_Return model girdilerine dahil edilmemistir.

Makine ogrenmesi modelleri guncellendi:

- Model ve Min-Max scaler yalnizca Train seti uzerinde fit edilmektedir.
- Validation seti egitimden ayri tutulmustur.
- Test seti nihai performans olcumu icin kullanilmaktadir.

Derin ogrenme modelleri:

- Train seti ile egitilmekte, validation seti erken durdurma icin kullanilmakta,
  test seti nihai performans olcumu icin ayrilmaktadir.

## Makalede Duzeltilmesi Gereken Ana Noktalar

1. Giris bolumundeki `!!!` taslak notlari akademik metne donusturulmelidir.

2. `yfinance ile otomatik veri toplandi` ifadesi dikkatli yazilmalidir. Projede
   su anda ham veri dosyasi uzerinden isleme yapilmaktadir. Veri indirme scripti
   eklenmezse metinde "Yahoo Finance kaynakli ham veri kullanilmistir" denmesi
   daha dogrudur.

3. Hedef degisken netlestirilmelidir. Projede model T+1 getirisini tahmin eder;
   tahmin edilen getiri bugunku kapanis fiyatina uygulanarak T+1 kapanis fiyatina
   donusturulur.

4. Makalede "RandomizedSearchCV" veya "TimeSeriesCV ile hiperparametre
   optimizasyonu" denecekse kodda bu adim gercekten eklenmelidir. Mevcut proje
   sabit ve makul hiperparametrelerle calismaktadir.

5. "Yuksek dogruluk" ifadesi temkinli kullanilmalidir. Fiyat seviyesi R2 degerleri
   cok yuksek gorunse de naive baseline, modellerden daha iyi sonuc vermektedir.

6. XAI, Wilcoxon ve Diebold-Mariano gibi 5.2 sonrasi analizler su an varsayilan
   pipeline kapsaminda degildir. 1-4 bolumlerinde "gerceklestirilmistir" yerine
   "calisma cercevesine dahil edilmistir" veya "ilerleyen bolumlerde
   degerlendirilecektir" gibi ifadeler tercih edilmelidir.

## 3. Bolum Icin Onerilen Metin

Bu calismada kullanilan veri seti, Yahoo Finance kaynakli gunluk finansal
verilerden olusmaktadir. Veri kapsami 2015-01-01 tarihinde baslamakta ve
2024-12-31 tarihli son islem gunune kadar uzanmaktadir. Makalede belirtilen
2015-2025 donemi, 2015-01-01 baslangic tarihini ve 2025-01-01 haric bitis
sinirini ifade etmektedir. Bu nedenle modelleme acisindan son hedef tarih
2024-12-31'dir.

Calisma kapsaminda BIST100 endeksinde farkli sektorleri temsil eden alti hisse
senedi kullanilmistir: THYAO, EREGL, AKBNK, GARAN, BIMAS ve TUPRS. Bu hisseler
ulastirma, sanayi, bankacilik, perakende ve enerji sektorlerini temsil edecek
sekilde secilmistir. Bu tercih, modellerin yalnizca tek bir endeks seviyesi
uzerinde degil, farkli sektor dinamiklerine sahip pay senetleri uzerinde de
sinanabilmesine olanak saglamaktadir.

Veri setinde yedi temel degisken bulunmaktadir: Open, High, Low, Close, Volume,
USD_TRY ve VIX. Bu degiskenlere ek olarak fiyat serilerinin trend, momentum,
volatilite ve hacim yapisini temsil eden 21 teknik/turetilmis gosterge
hesaplanmistir. Boylece her bir islem gunu icin toplam 28 ozellikten olusan
butunlesik bir veri yapisi olusturulmustur.

Eksik gozlemler incelendiginde en yuksek eksik deger sayisinin VIX degiskeninde
oldugu gorulmustur. Eksik veri islemede ileri doldurma, KNN Imputer ve dogrusal
enterpolasyon yontemleri karsilastirilmis; LightGBM tabanli kisa benchmark
sonucunda en dusuk RMSE degerini dogrusal enterpolasyon verdigi icin ana veri
setinde bu yontem kullanilmistir. Serinin uc noktalarinda kalan eksikler ileri ve
geri doldurma ile tamamlanmistir.

Veri sizintisini onlemek amaciyla veri seti rastgele ayrilmamis, her hisse icin
kronolojik sirasi korunarak bolunmustur. Her hisse icin 1788 gozlem egitim,
383 gozlem dogrulama ve 384 gozlem test setine ayrilmistir. Test donemi
2023-06-21 ile 2024-12-31 tarihleri arasini kapsamaktadir.

## 4. Bolum Icin Onerilen Metin

Bu calismada onerilen tahmin cercevesi, ham finansal verinin islenmesinden
model performansinin yorumlanmasina kadar birbirini izleyen moduler adimlardan
olusmaktadir. Ilk asamada secilen alti hisse senedine ait fiyat ve hacim
verileri, USD/TRY kuru ve VIX endeksi ile birlestirilmis; ardindan eksik
gozlemler dogrusal enterpolasyon temelli yaklasimla tamamlanmistir.

Ozellik muhendisligi asamasinda yedi temel degisken korunmus ve bunlara
21 teknik/turetilmis gosterge eklenmistir. Boyut indirgeme uygulanmamis,
boylece ham fiyat bilgisi ile trend, momentum, volatilite ve hacim tabanli
teknik bilgi ayni veri yapisi icinde korunmustur.

Modelleme asamasinda uc ayri senaryo olusturulmustur. Birinci senaryo yalnizca
temel/makro degiskenleri, ikinci senaryo yalnizca turetilmis teknik gostergeleri,
ucuncu senaryo ise 28 ozelligin tamamini icermektedir. Bu senaryo tasarimi,
teknik gostergelerin modele gercekten ek bilgi saglayip saglamadigini ampirik
olarak incelemek amaciyla kullanilmistir.

Makine ogrenmesi tarafinda Dogrusal Regresyon, SVR, Random Forest, XGBoost,
LightGBM ve CatBoost modelleri; derin ogrenme tarafinda ise MLP, CNN, LSTM,
GRU, BiLSTM, CNN-LSTM ve Transformer mimarileri degerlendirmeye alinmistir.
Makine ogrenmesi modellerinde olceklendirici ve model yalnizca egitim setinde
fit edilmis; derin ogrenme modellerinde dogrulama seti erken durdurma icin
kullanilmistir. Test seti ise tum modeller icin yalnizca nihai performans
olcumu amaciyla kullanilmistir.

## 5.2 Icin Onerilen Yorum Metni

Kapanis fiyati seviyesi uzerinden hesaplanan R2 degerleri tum modeller icin
oldukca yuksek gorunmektedir. Ancak bu durum, finansal fiyat serilerinin guclu
otokorelasyon yapisi nedeniyle dikkatli yorumlanmalidir. Nitekim "yarinin
kapanisi bugunun kapanisina esittir" varsayimina dayanan naive baseline modeli,
scenario_3_all uzerinde RMSE=3.3517, MAE=2.1089, MAPE=%1.8802 ve R2=0.9986
degerleriyle tum makine ogrenmesi ve derin ogrenme modellerinden daha dusuk hata
uretmis durumdadir.

Bu bulgu, fiyat seviyesi tahmininde yuksek R2 degerinin tek basina modelin piyasa
hareketlerini basariyla ogrendigini gostermedigini ortaya koymaktadir. Bu nedenle
model performansi yalnizca fiyat seviyesi metrikleriyle degil, getiri tahmini ve
yon dogrulugu gibi ek olcutlerle birlikte degerlendirilmelidir.

Getiri bazli diagnostik sonuclar, modellerin fiyat degisimlerini aciklama
gucunun sinirli oldugunu gostermektedir. En iyi modellerde dahi getiri R2
degerleri negatif bolgede kalmakta, yon dogrulugu ise yaklasik %50-52 araliginda
seyretmektedir. Bu nedenle calismanin bulgulari, modellerin fiyat seviyesini
yakindan takip edebildigini; ancak kisa vadeli fiyat hareketinin yonunu guvenilir
sekilde tahmin etme konusunda sinirli basari gosterdigini ortaya koymaktadir.

## Uretilen Makale Grafik Klasoru

Basliksiz ve dogru tarih araligina sahip grafikler:

`C:/Users/asusl/Desktop/bist100/outputs/article_figures`

Bu klasordeki `article_figures_manifest.csv` dosyasi her grafigin tarih araligini
ve basliksiz uretildigini gosterir.
