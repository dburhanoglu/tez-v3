"""02: Ortak korpus. Orijinal sözler korunur; PMI temizliği burada yapılmaz."""
import argparse
import hashlib
from pathlib import Path

import pandas as pd


def metin(deger):
    return "" if pd.isna(deger) else str(deger)


def ozet_hash(satir):
    alanlar = [metin(satir[k]) for k in ("Başlık", "Yöre", "Sözler")]
    return hashlib.sha256("\x1f".join(alanlar).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Ham turku arsivini ortak Excel'e donusturur")
    ap.add_argument("--girdi", type=Path, required=True)
    ap.add_argument("--referans", type=Path, required=True,
                    help="Eski nihai veri sirasi ve metin ozeti: TEST guvenlik referansi")
    ap.add_argument("--cikti", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_excel(args.girdi, engine="openpyxl")
    gerekli = {"Link", "Başlık", "Yöre", "Sözler"}
    if not gerekli.issubset(df.columns):
        raise ValueError(f"Eksik sutunlar: {sorted(gerekli - set(df.columns))}")
    if df["Link"].isna().any() or df["Link"].duplicated().any():
        raise ValueError("Bos veya tekrarlanan Link var: islem durduruldu")

    ref = pd.read_csv(args.referans, dtype=str, keep_default_na=False)
    if len(ref) != len(df) or ref["Link"].duplicated().any():
        raise ValueError("Referans ile ham arsivin kayit sayisi/tekilligi uyusmuyor")

    # Eski genellestirme kimlikleri satir sirasina bagli oldugu icin,
    # tarihsel sirayi Link uzerinden KESIN olarak koruyoruz.
    df = df.set_index("Link", drop=False)
    if set(ref["Link"]) != set(df.index):
        raise ValueError("Referans ile arşivdeki Link kumeleri farkli")
    df = df.loc[ref["Link"].tolist()].reset_index(drop=True)

    # Yalnizca sira degil, orijinal metnin de ayni kaldigini test ediyoruz.
    for i, (_, row) in enumerate(df.iterrows()):
        if ozet_hash(row) != ref.iloc[i]["Icerik_SHA256"]:
            raise ValueError(f"Baslik/Yore/Sozler degismis! Satir {i + 2}, Link={row['Link']}")

    if args.cikti.exists():
        raise FileExistsError(f"02 cikti zaten var: {args.cikti} (uzerine yazilmadi)")
    args.cikti.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.cikti, index=False, engine="openpyxl")
    print(f"02 TAMAM | {len(df)} turku | Orijinal sozler ve eski sira dogrulandi")
    print(f"Cikti: {args.cikti}")


if __name__ == "__main__":
    main()
