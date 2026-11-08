
import pandas as pd
from pathlib import Path

veriler = {}
for f in Path("..").rglob("PMI_SEED_POS_*.xlsx"):
    seed = "ON" if "SEED_POS_ACIK" in f.name else "OFF"
    context = "ON" if "BAGLAM_POS_AYRI" in f.name else "OFF"
    df = pd.read_excel(f)

    veriler[seed, context] = [
        set(df.sort_values("PMI_Farki", ascending=asc)
            .drop_duplicates("Kelime")["Kelime"].head(50))
        for asc in (False, True)
    ]
if len(veriler) != 4:
    raise SystemExit(f"4 Excel gerekli, bulunan: {len(veriler)}")
for i, grup in enumerate(["KADIN (ilk 50)", "ERKEK (son 50)"]):
    print(f"\n--- {grup} ---")
    ortak = set.intersection(*(v[i] for v in veriler.values()))
    print(f"4 ayarda ortak: {len(ortak)}/50 = %{len(ortak)*2}")

    for a, b in [
        (("ON","ON"), ("OFF","ON")),
        (("ON","OFF"), ("OFF","OFF")),
        (("ON","ON"), ("ON","OFF")),
        (("OFF","ON"), ("OFF","OFF"))
    ]:
        x, y = veriler[a][i], veriler[b][i]
        print(f"\n{a} -> {b}: %{len(x & y)*2} ortak")
        print("Çıkan:", sorted(x-y))
        print("Giren:", sorted(y-x))
