"""05: Türkü bazında tekilleştirilmiş kadın/erkek ilişkilerinden NLTK PMI.

Bir türkünün aynı (kadın/erkek grubu, bağlam kelimesi) ilişkisine
katkısı en fazla 1'dir. İki POS anahtarı birbirinden bağımsızdır.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import unicodedata

import pandas as pd
from nltk.metrics import BigramAssocMeasures

from seed_listeleri import seedleri_al


# ============================================================
# 1. DENEY AYARLARI
# ============================================================
SEED_DENEYI = "SEED_1"

# Hedef (seed) kelime NOUN (isim) veya ADJ (sıfat) ise kabul edilsin mi?
# True: gelin/NOUN kabul, gelin/VERB ret. Bağlamı filtrelemez.
SEED_POS_KONTROLU = True

# Seed çevresindeki bağlam kelimelerinin POS kullanımları ayrı sayılsın mı?
# True: gül_NOUN ve gül_VERB ayrı satır/balon; False: tek "gül".
BAGLAM_POS_AYRIMI = True

PENCERE = 5                       # Hedefin sağında ve solunda 5'er kelime
MIN_GENEL_TURKU = 15              # Kelime genel olarak en az 15 türküde
MIN_BIRLIKTE_TURKU = 15           # En az bir seed grubuyla 15 FARKLI türküde
YUMUSATMA = 0.5                  # Sıfır birliktelikleri de hesaplayabilmek için

KOLONLAR = [
    "Kelime", "POS", "Genel_Turku_Frekansi",
    "Kadin_Birlikte_Turku_Sayisi", "Erkek_Birlikte_Turku_Sayisi",
    "Toplam_Grup_Kelime_Eslesmesi", "PMI_Kadin", "PMI_Erkek", "PMI_Farki",
]

PROJE = Path(__file__).resolve().parents[2]
VARSAYILAN_GIRDI = (PROJE / "01_VeriHazirlama" / "2_veri"
                   / "03_Nihai_Veri" / "turkuler_lemma_duzeltilmis.xlsx")
VARSAYILAN_SONUCLAR = PROJE / "02_PMI_CinsiyetAnalizi" / "2_sonuclar"


def liste_oku(hucre, ad, sira):
    """Excel'de JSON olarak saklanan token/lemma/POS listesini okur."""
    if pd.isna(hucre):
        raise ValueError(f"{sira}. türküde {ad} boş")
    try:
        liste = json.loads(str(hucre))
    except (ValueError, TypeError) as hata:
        raise ValueError(f"{sira}. türküde {ad} geçerli JSON değil") from hata
    if not isinstance(liste, list) or not all(isinstance(x, str) for x in liste):
        raise ValueError(f"{sira}. türküde {ad} metin listesi değil")
    return liste


def noktalama_mi(token, pos):
    """Noktalamayı pencere hesabından ÖNCE çıkarır; sayı silmez."""
    return pos == "PUNCT" or (bool(token) and all(
        unicodedata.category(harf).startswith("P") for harf in token))


def veriyi_oku(dosya):
    """8 sütunlu Excel'den hizalı lemma ve POS dizilerini okur."""
    tablo = pd.read_excel(dosya, engine="openpyxl")
    gerekli = {"Link", "Lemmalar", "Tokenler", "POS_Etiketleri"}
    eksik = gerekli - set(tablo.columns)
    if eksik:
        raise ValueError(f"PMI Excel'inde eksik sütunlar: {sorted(eksik)}")
    if tablo["Link"].isna().any() or tablo["Link"].duplicated().any():
        raise ValueError("Türkü Link sütunu boş/tekrarlı olamaz")

    korpus = []
    for sira, (_, satir) in enumerate(tablo.iterrows(), 1):
        tokenler = liste_oku(satir["Tokenler"], "Tokenler", sira)
        lemmalar = liste_oku(satir["Lemmalar"], "Lemmalar", sira)
        poslar = liste_oku(satir["POS_Etiketleri"], "POS_Etiketleri", sira)
        if not (len(tokenler) == len(lemmalar) == len(poslar)):
            raise ValueError(f"{sira}. türküde token/lemma/POS hizası bozuk")
        # Pencere şimdi noktalamasız token dizisinde kurulacak.
        kelimeler = [(lemma, pos) for token, lemma, pos in zip(tokenler, lemmalar, poslar)
                    if not noktalama_mi(token, pos)]
        korpus.append(kelimeler)
    return korpus


def sayimlari_yap(korpus, kadin_seedleri, erkek_seedleri,
                  seed_pos_kontrolu, baglam_pos_ayrimi,
                  pencere=PENCERE):
    """Her türküde aynı grup–bağlam ilişkisine en fazla 1 puan verir."""
    seedler = kadin_seedleri | erkek_seedleri

    def baglam_anahtari(lemma, pos):
        return (lemma, pos) if baglam_pos_ayrimi else (lemma, "")

    # Bir lemma+POS (veya sadece lemma) kaç FARKLI türküde görülüyor?
    genel = Counter()
    for turku in korpus:
        genel.update({baglam_anahtari(lemma, pos) for lemma, pos in turku
                      if lemma not in seedler})
    gecerli = {birim for birim, adet in genel.items() if adet >= MIN_GENEL_TURKU}

    kadin_sayisi, erkek_sayisi = Counter(), Counter()
    for turku in korpus:
        bu_turkude_kadin, bu_turkude_erkek = set(), set()
        for i, (lemma, pos) in enumerate(turku):
            if lemma not in seedler:
                continue
            # Bu filtre YALNIZCA hedef/seed tokenına uygulanır.
            if seed_pos_kontrolu and pos not in {"NOUN", "ADJ"}:
                continue
            baslangic = max(0, i - pencere)
            bitis = min(len(turku), i + pencere + 1)
            for j in range(baslangic, bitis):
                if i == j:
                    continue
                diger_lemma, diger_pos = turku[j]
                if diger_lemma in seedler:  # Seed kelimeleri bağlam olamaz.
                    continue
                birim = baglam_anahtari(diger_lemma, diger_pos)
                if birim not in gecerli:
                    continue
                if lemma in kadin_seedleri:
                    bu_turkude_kadin.add(birim)
                if lemma in erkek_seedleri:
                    bu_turkude_erkek.add(birim)
        kadin_sayisi.update(bu_turkude_kadin)
        erkek_sayisi.update(bu_turkude_erkek)

    return genel, gecerli, kadin_sayisi, erkek_sayisi


def pmi_sonuclari(genel, gecerli, kadin_sayisi, erkek_sayisi,
                  min_birlikte=MIN_BIRLIKTE_TURKU, alfa=YUMUSATMA):
    """2 x kelime matrisi ve α yumuşatmasıyla NLTK PMI hesaplar.

    V: geçerli bağlam birimi sayısı.
    Her (kadın/erkek, bağlam) hücresine alfa eklenir.
    Yumuşatma gerçek türkü sayılarını DEĞİŞTİRMEZ.
    """
    if not gecerli:
        raise ValueError("Genel türkü eşiğini geçen bağlam birimi yok")
    v = len(gecerli)
    toplam_k = sum(kadin_sayisi.values())
    toplam_e = sum(erkek_sayisi.values())
    toplam_k_duzeltilmis = toplam_k + v * alfa
    toplam_e_duzeltilmis = toplam_e + v * alfa
    n = toplam_k_duzeltilmis + toplam_e_duzeltilmis

    satirlar = []
    for kelime, pos in sorted(gecerli):
        birim = (kelime, pos)
        k = kadin_sayisi[birim]
        e = erkek_sayisi[birim]
        if max(k, e) < min_birlikte:
            continue  # Yalnız sonuç seçimi; PMI referans evrenini değiştirmez.

        # Aynı marjinaller iki kez kullanılır: kadın ve erkek PMI.
        # NLTK: pmi(ortak, (grup_toplamı, kelime_toplamı), genel_toplam)
        kelime_toplami = k + e + 2 * alfa
        pmi_k = BigramAssocMeasures.pmi(
            k + alfa, (toplam_k_duzeltilmis, kelime_toplami), n)
        pmi_e = BigramAssocMeasures.pmi(
            e + alfa, (toplam_e_duzeltilmis, kelime_toplami), n)
        satirlar.append({
            "Kelime": kelime,
            "POS": pos,
            "Genel_Turku_Frekansi": genel[birim],
            "Kadin_Birlikte_Turku_Sayisi": k,
            "Erkek_Birlikte_Turku_Sayisi": e,
            # Aynı türkü her iki gruba da katkı yapabilir.
            "Toplam_Grup_Kelime_Eslesmesi": k + e,
            "PMI_Kadin": pmi_k,
            "PMI_Erkek": pmi_e,
            "PMI_Farki": pmi_k - pmi_e,
        })

    sonuc = pd.DataFrame(satirlar, columns=KOLONLAR)
    if not sonuc.empty:
        sonuc = sonuc.sort_values(
            ["PMI_Farki", "Kelime", "POS"],
            ascending=[False, True, True]).reset_index(drop=True)
    bilgiler = {"V": v, "T_K": toplam_k, "T_E": toplam_e,
                "N_Yumusatilmis": n, "Sonuc_Kelimesi": len(sonuc)}
    return sonuc, bilgiler


def dosya_adi(seed_pos, baglam_pos):
    a = "ACIK" if seed_pos else "KAPALI"
    b = "AYRI" if baglam_pos else "BIRLESIK"
    return f"SEED_POS_{a}_BAGLAM_POS_{b}"


def hesapla_ve_kaydet(korpus, kadin, erkek, deney, sonuc_klasoru, modlar):
    """Dört modu bağımsız Excel'e yazar; mevcut dosyaları değiştirmez."""
    hedef_klasor = sonuc_klasoru / deney
    hedefler = [hedef_klasor / f"PMI_{dosya_adi(sp, bp)}.xlsx" for sp, bp in modlar]
    varolan = [str(x) for x in hedefler if x.exists()]
    if varolan:
        raise FileExistsError(f"PMI çıktıları zaten var, değiştirilmedi: {varolan}")
    hedef_klasor.mkdir(parents=True, exist_ok=True)

    for (seed_pos, baglam_pos), hedef in zip(modlar, hedefler):
        genel, gecerli, ks, es = sayimlari_yap(
            korpus, kadin, erkek, seed_pos, baglam_pos)
        sonuc, sayilar = pmi_sonuclari(genel, gecerli, ks, es)
        # Hangi seed ve ayarların kullanıldığı sonuç Excel'inde kayıtlı.
        ayarlar = {
            "Seed_Deneyi": deney,
            "Kadin_Seedleri": ", ".join(sorted(kadin)),
            "Erkek_Seedleri": ", ".join(sorted(erkek)),
            "SEED_POS_KONTROLU": seed_pos,
            "BAGLAM_POS_AYRIMI": baglam_pos,
            "Pencere": PENCERE,
            "Genel_Turku_Esigi": MIN_GENEL_TURKU,
            "Birlikte_Turku_Esigi": MIN_BIRLIKTE_TURKU,
            "Yumusatma_Alfa": YUMUSATMA,
            "Tekillestirme": "Turku + grup + baglam birimi",
            "Islenen_Turku": len(korpus),
            **sayilar,
        }
        bilgi_tablo = pd.DataFrame(list(ayarlar.items()), columns=["Parametre", "Deger"])
        with pd.ExcelWriter(hedef, engine="openpyxl") as yazici:
            sonuc.to_excel(yazici, sheet_name="PMI_Sonuclari", index=False)
            bilgi_tablo.to_excel(yazici, sheet_name="Deney_Bilgisi", index=False)
        print(f"05: {hedef.name} | {len(sonuc)} sonuç | "+
              f"kadın/erkek ilişki: {sayilar['T_K']}/{sayilar['T_E']}")


def main():
    parser = argparse.ArgumentParser(description="Türkü başına tekilleştirilmiş NLTK PMI")
    parser.add_argument("--girdi", type=Path, default=VARSAYILAN_GIRDI)
    parser.add_argument("--sonuclar", type=Path, default=VARSAYILAN_SONUCLAR)
    parser.add_argument("--seed", default=SEED_DENEYI)
    parser.add_argument("--mod", choices=["DORT", "AYARLAR", "SEED_ACIK_BAGLAM_AYRI",
                                            "SEED_ACIK_BAGLAM_BIRLESIK",
                                            "SEED_KAPALI_BAGLAM_AYRI",
                                            "SEED_KAPALI_BAGLAM_BIRLESIK"], default="DORT")
    args = parser.parse_args()
    tum_modlar = [(True, True), (True, False), (False, True), (False, False)]
    adlar = ["SEED_ACIK_BAGLAM_AYRI", "SEED_ACIK_BAGLAM_BIRLESIK",
             "SEED_KAPALI_BAGLAM_AYRI", "SEED_KAPALI_BAGLAM_BIRLESIK"]
    if args.mod == "DORT":
        modlar = tum_modlar
    elif args.mod == "AYARLAR":
        modlar = [(SEED_POS_KONTROLU, BAGLAM_POS_AYRIMI)]
    else:
        modlar = [tum_modlar[adlar.index(args.mod)]]
    kadin, erkek = seedleri_al(args.seed)
    korpus = veriyi_oku(args.girdi)
    hesapla_ve_kaydet(korpus, kadin, erkek, args.seed, args.sonuclar, modlar)


if __name__ == "__main__":
    main()
