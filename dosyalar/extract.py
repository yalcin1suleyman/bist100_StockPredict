import docx

doc = docx.Document(r"c:\stajProje\dosyalar\bist_100_guncellenmis.docx")

with open(r"c:\stajProje\dosyalar\guncellenmis_icerik.txt", "w", encoding="utf-8") as f:
    for i, p in enumerate(doc.paragraphs):
        f.write(f"[{i}] {p.text}\n")

print("Icerik yazildi.")
