# BIST 100 Prediction Project

Bu proje, makaledeki literatur taramasi ve yontem kurgusuna uygun olarak
BIST 100 kapsamindan secilen 6 hisse icin uc ayri veri senaryosu uretir,
modelleri egitir ve makaleye eklenebilir basliksiz grafikler ile tablo
ciktilari olusturur.

## Temiz Proje Yapisi

```text
bist100/
  data/
    bist100_ham_veri.csv
    processed/
  outputs/
    figures/
    predictions/
    report/
    tables/
    xai/
  src/
    __init__.py
    bist100_pipeline.py
  run_pipeline.py
  requirements.txt
  README.md
```

## Calisma Donemi

Makaledeki `01.01.2015 ve 01.01.2025 tarihleri arasi` ifadesi kodda
acik bicimde su sekilde uygulanir:

- Baslangic dahil: `2015-01-01`
- Bitis haric: `2025-01-01`
- Beklenen son gozlem: `2024-12-31`

Pipeline bu tarih araligini manifest dosyasina yazar ve ham veri bu kapsami
saglamazsa hata verir.

## Hisse Kapsami

Calisma yalnizca makalede belirtilen 6 hisseyi kullanir:

`AKBNK.IS`, `BIMAS.IS`, `EREGL.IS`, `GARAN.IS`, `THYAO.IS`, `TUPRS.IS`

Ham veride bu hisselerden biri eksikse pipeline hata verir. Kapsam disi hisse
varsa filtrelenir.

## Veri Senaryolari

Pipeline ham veriden baslar ve 28 ozellik uretir.

- Senaryo 1: 7 temel/makro ozellik
  `Open, High, Low, Close, Volume, USD_TRY, VIX`
- Senaryo 2: 21 turetilmis/teknik ozellik
  `Gunluk_Getiri` ve teknik gostergeler
- Senaryo 3: Tum 28 ozellik

Bolme islemi her hisse icinde kronolojik olarak yapilir:

- Egitim: %70
- Dogrulama: %15
- Test: %15

## Calistirma

```powershell
python run_pipeline.py
```

Daha hizli kontrol icin:

```powershell
python run_pipeline.py --epochs 8 --quick
```

## Uretilen Ciktilar

- `data/processed/`: temiz 28 ozellikli veri ve 3 senaryo CSV dosyasi
- `outputs/tables/`: eksik veri, tanimlayici istatistik, model sonuclari,
  istatistiksel test ve karmasiklik tablolari
- `outputs/figures/`: makaleye basliksiz eklenebilir grafikler
- `outputs/predictions/`: test tahminleri
- `outputs/xai/`: ozellik onemi, SHAP/LIME/PDP destek ciktisi
- `outputs/report/pipeline_manifest.json`: calisma ozeti

Grafiklerin uzerinde baslik yoktur; makalede altlarina Sekil vb. aciklama
eklenmesi icin hazirlanmistir.
