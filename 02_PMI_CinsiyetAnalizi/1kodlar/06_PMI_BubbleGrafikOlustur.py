"""06: Dört PMI Excel'inden aynı ölçekli dört bubble grafik üretir."""
import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Windows'ta pencere açmadan PNG oluştur.
import matplotlib.colors as renkler
import matplotlib.pyplot as plt
import pandas as pd

# İki aç/kapa ayarının dosya adı. Excel ve PNG'de aynı kurala uy.
def mod_adi(seed_pos, baglam_pos):
    a = "ACIK" if seed_pos else "KAPALI"
    b = "AYRI" if baglam_pos else "BIRLESIK"
    return f"SEED_POS_{a}_BAGLAM_POS_{b}"


PROJE = Path(__file__).resolve().parents[2]
SONUCLAR = PROJE / "02_PMI_CinsiyetAnalizi" / "2_sonuclar"
MODLAR = [(True, True), (True, False), (False, True), (False, False)]


def etiket(satir):
    """POS ayrımı açıkken örn. gül_NOUN, kapalıyken gül."""
    pos = satir.get("POS", "")
    return str(satir["Kelime"]) + (f"_{pos}" if pd.notna(pos) and pos else "")


def sec(tablo):
    """En yüksek 50 pozitif ve en düşük 50 negatif; sıfırlar seçilmez."""
    veri = tablo.dropna(subset=["PMI_Farki"]).copy()
    veri["Etiket"] = veri.apply(etiket, axis=1)
    pozitif = veri[veri.PMI_Farki > 0].sort_values(
        ["PMI_Farki", "Etiket"], ascending=[False, True]).head(50)
    negatif = veri[veri.PMI_Farki < 0].sort_values(
        ["PMI_Farki", "Etiket"], ascending=[True, True]).head(50)
    return pd.concat([pozitif, negatif], ignore_index=True)


def dikey_kaydirma(yazi):
    """Etiketler her çalıştırmada aynı konumda başlasın."""
    sayi = int.from_bytes(hashlib.sha256(yazi.encode("utf-8")).digest()[:4], "big")
    return (sayi / (2**32 - 1) - .5) * .28


def ciz(tablo, baslik, hedef, x_sinir, min_boyut, max_boyut):
    fig, ax = plt.subplots(figsize=(17, 8))
    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.set_xlim(-x_sinir, x_sinir)
    ax.set_ylim(-.35, .35)
    ax.set_yticks([])
    ax.set_xlabel("← Erkek bağlamı | PMI farkı | Kadın bağlamı →")
    ax.set_title(baslik)
    ax.grid(axis="x", alpha=.2)

    if tablo.empty:
        ax.text(.5, .5, "Bu örneklemde eşikleri geçen karşılaştırılabilir kelime yok.",
                transform=ax.transAxes, ha="center", va="center")
    else:
        boyutlar = []
        for frekans in tablo["Toplam_Grup_Kelime_Eslesmesi"]:
            oran = 0.5 if max_boyut == min_boyut else (
                (frekans - min_boyut) / (max_boyut - min_boyut))
            boyutlar.append(55 + 330 * oran)
        yler = [dikey_kaydirma(s) for s in tablo["Etiket"]]
        ax.scatter(tablo["PMI_Farki"], yler, s=boyutlar,
                   c=tablo["PMI_Farki"], cmap="coolwarm",
                   norm=renkler.Normalize(vmin=-x_sinir, vmax=x_sinir),
                   alpha=.72, edgecolors="grey", linewidths=.4)

        # Her balonun kelime + POS adını yaz. Varsa adjustText ile yerleştir.
        yazilar = [ax.text(x, y, ad, fontsize=7, ha="center")
                   for x, y, ad in zip(tablo["PMI_Farki"], yler, tablo["Etiket"])]
        try:
            from adjustText import adjust_text
            adjust_text(yazilar, ax=ax, time_lim=2,
                        arrowprops={"arrowstyle": "-", "color": "gray", "lw": .35})
        except ImportError:
            pass  # adjustText olmadan da PNG üretilebilir; etiketler çakışabilir.

    fig.tight_layout()
    fig.savefig(hedef, dpi=240)
    plt.close(fig)


def grafikleri_olustur(klasor):
    """Dört grafiği aynı x/renk/bubble ölçeğinde üret."""
    girdiler = [klasor / f"PMI_{mod_adi(a, b)}.xlsx" for a, b in MODLAR]
    ciktilar = [klasor / f"Bubble_{mod_adi(a, b)}.png" for a, b in MODLAR]
    if any(not p.exists() for p in girdiler):
        raise FileNotFoundError("Dört PMI Excel'i de bulunmalı: " + str(klasor))
    if any(p.exists() for p in ciktilar):
        raise FileExistsError("Grafiklerden biri mevcut, üzerine yazılmadı")
    tablolar = [pd.read_excel(p, sheet_name="PMI_Sonuclari", engine="openpyxl")
               for p in girdiler]
    gereken = {"Kelime", "POS", "PMI_Farki", "Toplam_Grup_Kelime_Eslesmesi"}
    for tablo in tablolar:
        if not gereken.issubset(tablo.columns):
            raise ValueError(f"Eksik grafik sütunu: {gereken - set(tablo.columns)}")
    secimler = [sec(t) for t in tablolar]
    toplam = pd.concat(secimler, ignore_index=True)
    x_sinir = max(1., float(toplam["PMI_Farki"].abs().max()) * 1.10) if len(toplam) else 1.
    min_b = float(toplam["Toplam_Grup_Kelime_Eslesmesi"].min()) if len(toplam) else 0.
    max_b = float(toplam["Toplam_Grup_Kelime_Eslesmesi"].max()) if len(toplam) else 1.
    for (sp, bp), secim, hedef in zip(MODLAR, secimler, ciktilar):
        ciz(secim, mod_adi(sp, bp), hedef, x_sinir, min_b, max_b)
        print(f"06: {hedef.name} | {len(secim)} balon")


def main():
    parser = argparse.ArgumentParser(description="Dört karşılaştırılabilir bubble grafiği")
    parser.add_argument("--sonuclar", type=Path, default=SONUCLAR)
    parser.add_argument("--seed", default="SEED_1")
    args = parser.parse_args()
    grafikleri_olustur(args.sonuclar / args.seed)


if __name__ == "__main__":
    main()
