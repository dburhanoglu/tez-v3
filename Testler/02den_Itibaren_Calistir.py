"""Mevcut 02, 03, 04, 05 ve 06 kodlarini sirayla calistirir.
Test yapmaz, hesaplama yontemini degistirmez, mevcut ciktilarin uzerine yazmaz.
"""
import subprocess
import sys
from pathlib import Path

PROJE = Path(__file__).resolve().parent
VERI = PROJE / "01_VeriHazirlama" / "2_veri"
KOD = PROJE / "01_VeriHazirlama" / "1kodlar"
PMI_KOD = PROJE / "02_PMI_CinsiyetAnalizi" / "1kodlar"
SONUCLAR = PROJE / "02_PMI_CinsiyetAnalizi" / "2_sonuclar"
SEED = "SEED_1"

HAM = VERI / "01_Ham_Veri" / "turkudostlari_arsiv.xlsx"
REFERANS = PROJE / "01_VeriHazirlama" / "3_Kaynaklar" / "eski_kayit_sirasi.csv"
ORTAK = VERI / "02_Islenmis_Veri" / "turkuler_ortak_veri.xlsx"
STANZA = VERI / "02_Islenmis_Veri" / "turkuler_stanza_islenmis.xlsx"
LEMMA = VERI / "03_Nihai_Veri" / "turkuler_lemma_duzeltilmis.xlsx"

ADIMLAR = [
    (KOD / "02_OrtakVeriyiHazirla.py", "--girdi", HAM, "--referans", REFERANS, "--cikti", ORTAK),
    (KOD / "03_MetniTemizle_Stanza.py", "--girdi", ORTAK, "--cikti", STANZA, "--limit", "0"),
    (KOD / "04_LemmalariDuzelt.py", "--girdi", STANZA, "--cikti", LEMMA),
    (PMI_KOD / "05_TurkuBazliPMI_Hesapla.py", "--girdi", LEMMA, "--sonuclar", SONUCLAR, "--seed", SEED),
    (PMI_KOD / "06_PMI_BubbleGrafikOlustur.py", "--sonuclar", SONUCLAR, "--seed", SEED),
]

# Eski analiz sonucunu yanlislikla yeniden olusturmayalim.
modlar = ("SEED_POS_ACIK_BAGLAM_POS_AYRI", "SEED_POS_ACIK_BAGLAM_POS_BIRLESIK",
          "SEED_POS_KAPALI_BAGLAM_POS_AYRI", "SEED_POS_KAPALI_BAGLAM_POS_BIRLESIK")
mevcut = [p for p in (ORTAK, STANZA, LEMMA) if p.exists()]
for mod in modlar:
    for onek, uzanti in (("PMI", "xlsx"), ("Bubble", "png")):
        dosya = SONUCLAR / SEED / f"{onek}_{mod}.{uzanti}"
        if dosya.exists():
            mevcut.append(dosya)

if mevcut:
    print("DURDU: Daha once uretilmis ciktilar var; uzerlerine yazilmadi.", flush=True)
    for dosya in mevcut:
        print(" -", dosya.relative_to(PROJE))
    print("Yeni bir tam analiz icin once mevcut sonuclari guvenle arsivleyin.")
    sys.exit(1)

for sira, adim in enumerate(ADIMLAR, start=2):
    print(f"\n{sira:02d} calisiyor: {adim[0].name}", flush=True)
    subprocess.run([sys.executable, *map(str, adim)], check=True)

print("\n02-06 tam analiz tamamlandi. Sonuclar:", SONUCLAR / SEED)
