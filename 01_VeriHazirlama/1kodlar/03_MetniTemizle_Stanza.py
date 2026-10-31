"""03: Sözleri temizler, Stanza ile token/lemma/POS üretir (8 sütun)."""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


BIBLIYOGRAFIK = re.compile(
    r"\b(yayınları|neşriyat|matbaası|yayınevi|aktarılmaktadır|aktarılıyor|antolojisi)\b",
    re.IGNORECASE,
)
PARANTEZ = re.compile(r"\(([\s\S]*?)\)")


def turkce_kucult(text):
    return unicodedata.normalize("NFC", text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower())


def temiz_sozler(raw):
    """PMI metninden kaynak satirlarini ve parantezleri cikarir."""
    if pd.isna(raw):
        return ""
    satirlar = [s for s in str(raw).splitlines() if not BIBLIYOGRAFIK.search(s.strip())]
    yalin = PARANTEZ.sub("", "\n".join(satirlar))
    return "\n".join(s.strip() for s in yalin.splitlines() if s.strip())


def stanza_metni(yalin):
    # 03 noktalama ve sayilari korur; PMI penceresinden once noktalama cikarilir.
    t = re.sub(r"<[^>]+>", " ", yalin)
    t = re.sub(r"\b(serbest|bağlantı)\s*:", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    # Sayilar ve â/î/û korunur. Noktalama 05 PMI kodunda kaldirilir.
    return turkce_kucult(re.sub(r"\s+", " ", t).strip())


def hizali_birimler(kelime, lemma, upos):
    """Stanza'nin bosluklu kelimelerini, lemma/POS sirasini bozmadan duzenler."""
    kelimeler = kelime.split()
    lemmalar = lemma.split()
    if not kelimeler or not lemmalar:
        raise ValueError(f"Stanza bos token/lemma dondurdu: {kelime!r} -> {lemma!r}")
    if len(kelimeler) == len(lemmalar) == 1:
        return [(kelimeler[0], lemmalar[0], upos or "X")], None
    if len(kelimeler) == len(lemmalar):
        # Tek POS birden cok sozcugun her biri icin guvenilir degildir.
        return [(k, l, "X") for k, l in zip(kelimeler, lemmalar)], "ayrildi"
    # Parca sayilari farkli: hayali bir eslestirme yapmak yerine
    # birlesik birimi koru. Bosluk icermesin ki PMI .split() ile hizali olsun.
    return [("_".join(kelimeler), "_".join(lemmalar), "X")], "uyusmadi"


def main():
    ap = argparse.ArgumentParser(description="PMI metin hazirligi ve Stanza")
    ap.add_argument("--girdi", type=Path, required=True)
    ap.add_argument("--cikti", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="Kisa test: ilk N turku; 0=tumu")
    args = ap.parse_args()
    if args.cikti.exists():
        raise FileExistsError(f"03 cikti zaten var: {args.cikti} (uzerine yazilmadi)")

    df = pd.read_excel(args.girdi, engine="openpyxl")
    gereken = {"Link", "Başlık", "Yöre", "Sözler"}
    if not gereken.issubset(df.columns):
        raise ValueError(f"03 girdisinde eksik sutun: {sorted(gereken - set(df.columns))}")
    if args.limit > 0:
        df = df.head(args.limit).copy()

    df["Yalın Sözler (Temizlik Sonrası)"] = [temiz_sozler(t) for t in df["Sözler"]]
    # Parantez iclerini ayri sutuna yazmiyoruz: sonraki kodlar okumuyor.
    metinler = [stanza_metni(t) for t in df["Yalın Sözler (Temizlik Sonrası)"]]

    # Stanza modeli YERELDEN acilir; otomatik internetten indirme yok.
    try:
        import stanza
    except ImportError as hata:
        raise RuntimeError("Stanza yuklu degil. Bu kod Stanza modelini otomatik indirmez.") from hata
    import torch
    from stanza.models.common.doc import Document

    nlp = stanza.Pipeline(
        lang="tr", processors="tokenize,pos,lemma", download_method=None,
        use_gpu=torch.cuda.is_available(), verbose=False,
    )

    alanlar = {k: [] for k in ("Tokenler", "Lemmalar", "POS_Etiketleri")}
    batch_size = 128
    cok_kelime_sayisi = 0
    uyumsuz_sayisi = 0
    for start in range(0, len(metinler), batch_size):
        parca = metinler[start:start + batch_size]
        docs = [Document([], text=t) for t in parca if t]
        cozumler = iter(nlp(docs) if docs else [])
        for text in parca:
            kelimeler, lemmalar, poslar = [], [], []
            if text:
                doc = next(cozumler)
                for cumle in doc.sentences:
                    for word in cumle.words:
                        kelime = turkce_kucult(word.text)
                        if word.upos == "PROPN" or not word.lemma:
                            lemma = kelime
                        else:
                            lemma = turkce_kucult(word.lemma)
                        # Karar: isalpha(), 3 harf ve PUNCT/SYM/NUM filtresi YOK.
                        birimler, durum = hizali_birimler(
                            kelime, lemma, word.upos
                        )
                        if durum == "ayrildi":
                            cok_kelime_sayisi += 1
                        elif durum == "uyusmadi":
                            uyumsuz_sayisi += 1
                        for yuzey, kok, pos in birimler:
                            kelimeler.append(yuzey)
                            lemmalar.append(kok)
                            poslar.append(pos)
            for ad, deger in zip(alanlar, (kelimeler, lemmalar, poslar)):
                yazi = json.dumps(deger, ensure_ascii=False, separators=(",", ":"))
                if len(yazi) > 32767:
                    raise ValueError(f"Excel hucre siniri asildi: {ad}, turku {len(alanlar[ad]) + 1}")
                alanlar[ad].append(yazi)
        print(f"03 Stanza: {min(start + batch_size, len(metinler))}/{len(metinler)} turku")

    # Excel'de tek birlesik tuple listesi degil, 3 AYRI liste vardir.
    # Her listede ayni indeks ayni Stanza tokenini temsil eder.
    # FEATS ve tekrar eden temiz_metin_lemma saklanmaz.
    df = df[["Link", "Başlık", "Yöre", "Sözler", "Yalın Sözler (Temizlik Sonrası)"]].copy()
    for ad, degerler in alanlar.items():
        df[ad] = degerler

    args.cikti.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.cikti, index=False, engine="openpyxl")
    print(f"03 TAMAM | {len(df)} turku | {args.cikti}")
    print(f"Stanza bosluklu kelime (ayrilan): {cok_kelime_sayisi}")
    print(f"Stanza bosluklu kelime (birim korundu, POS=X): {uyumsuz_sayisi}")


if __name__ == "__main__":
    main()
