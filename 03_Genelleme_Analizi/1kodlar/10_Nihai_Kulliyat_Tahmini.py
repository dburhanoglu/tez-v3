import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ayarlar import EXCEL_DIR, RANDOM_STATE, FINAL_DATA


# ============================================================
# AYARLAR
# ============================================================

CORPUS = FINAL_DATA

DOSYALAR = [
    EXCEL_DIR / "01_Baslangic_Etiketleme.xlsx",
    EXCEL_DIR / "03_Tur0_Etiketle.xlsx",
    EXCEL_DIR / "04_Tur1_Etiketle.xlsx",
    EXCEL_DIR / "06_Tur2_Etiketle.xlsx",
    EXCEL_DIR / "08_Tur3_Etiketle.xlsx"
]

OUT_TUM = EXCEL_DIR / "10_Tum_Tahminler.xlsx"
OUT_GEN = EXCEL_DIR / "10_Yuksek_Genelleme_Adaylari.xlsx"
GRAFIK = EXCEL_DIR / "10_Nihai_Kulliyat_Gorseli.png"
OUT_YORE = EXCEL_DIR / "10_Yuksek_Genelleme_Yore_Sayilari.xlsx"
GRAFIK_YORE = EXCEL_DIR / "10_Yuksek_Genelleme_Ilk15_Yore.png"

MIN_KELIME = 2
BATCH = 20000
P1_ESIK = 0.20


# ============================================================
# KÜLLİYATI DİZELERE AYIR
# ============================================================

corpus = pd.read_excel(CORPUS)

id_col = next(
    (c for c in ["Turku_ID", "Türkü_ID", "Kaynak_ID", "ID"]
     if c in corpus.columns),
    None
)

corpus["Turku_ID"] = (
    corpus[id_col]
    if id_col
    else np.arange(1, len(corpus) + 1)
)

soz_col = next(
    (c for c in ["Sözler", "Sozler", "sözler", "sozler", "Metin", "metin"]
     if c in corpus.columns),
    None
)

if soz_col is None:
    raise ValueError("Sözler sütunu bulunamadı.")

satirlar = []

for _, row in corpus.iterrows():
    if pd.isna(row[soz_col]):
        continue

    for no, dize in enumerate(str(row[soz_col]).splitlines(), 1):
        dize = dize.strip()

        if len(dize.split()) < MIN_KELIME:
            continue

        satirlar.append({
            "Turku_ID": row["Turku_ID"],
            "Başlık": row.get("Başlık", ""),
            "Yöre": row.get("Yöre", ""),
            "Dize": dize,
            "Dize_No": no
        })

kulliyat = (
    pd.DataFrame(satirlar)
    .drop_duplicates(["Turku_ID", "Dize"])
    .reset_index(drop=True)
)


# ============================================================
# EĞİTİM VERİSİ
# ============================================================

baslangic = pd.read_excel(DOSYALAR[0])

eslesme = kulliyat[
    ["Turku_ID", "Başlık", "Yöre", "Dize"]
].drop_duplicates(["Başlık", "Yöre", "Dize"])

baslangic = baslangic.merge(
    eslesme,
    on=["Başlık", "Yöre", "Dize"],
    how="left"
)

etiketliler = [baslangic] + [
    pd.read_excel(dosya)
    for dosya in DOSYALAR[1:]
]

kolonlar = ["Turku_ID", "Başlık", "Yöre", "Dize", "Etiket"]

egitim = pd.concat(
    [df[kolonlar] for df in etiketliler],
    ignore_index=True
)

egitim["Etiket"] = pd.to_numeric(
    egitim["Etiket"]
).astype(int)

egitim = (
    egitim
    .drop_duplicates(["Turku_ID", "Dize"], keep="last")
    .reset_index(drop=True)
)


# ============================================================
# MODEL
# ============================================================

model = Pipeline([
    ("features", FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=2,
            sublinear_tf=True
        ))
    ])),
    ("lr", LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE
    ))
])

model.fit(
    egitim["Dize"].astype(str),
    egitim["Etiket"]
)


# ============================================================
# KÜLLİYATI SKORLA
# ============================================================

classes = list(model.named_steps["lr"].classes_)

if set(classes) != {1, 2, 3}:
    raise ValueError(f"Beklenmeyen sınıflar: {classes}")

p1, p2, p3 = [], [], []
toplam = len(kulliyat)

for start in range(0, toplam, BATCH):
    end = min(start + BATCH, toplam)

    p = model.predict_proba(
        kulliyat["Dize"].iloc[start:end].astype(str)
    )

    p1.extend(p[:, classes.index(1)])
    p2.extend(p[:, classes.index(2)])
    p3.extend(p[:, classes.index(3)])

    print(
        f"\rKülliyat skorlanıyor: %{end / toplam * 100:5.1f}",
        end="",
        flush=True
    )

print("\nSkorlama tamamlandı.")

kulliyat["P1"] = p1
kulliyat["P2"] = p2
kulliyat["P3"] = p3

proba = kulliyat[["P1", "P2", "P3"]].to_numpy()

kulliyat["Tahmin_Etiket"] = np.array([1, 2, 3])[
    np.argmax(proba, axis=1)
]

assert np.allclose(
    kulliyat[["P1", "P2", "P3"]].sum(axis=1),
    1.0
)


# ============================================================
# 1) TÜM TAHMİNLER
# 3 -> P3 azalan
# 2 -> P2 azalan
# 1 -> P1 azalan
# ============================================================

tum = kulliyat.copy()

tum["_Skor"] = np.select(
    [
        tum["Tahmin_Etiket"] == 1,
        tum["Tahmin_Etiket"] == 2,
        tum["Tahmin_Etiket"] == 3
    ],
    [tum["P1"], tum["P2"], tum["P3"]]
)

tum["_Sira"] = tum["Tahmin_Etiket"].map({
    3: 1,
    2: 2,
    1: 3
})

tum = (
    tum
    .sort_values(
        ["_Sira", "_Skor"],
        ascending=[True, False]
    )
    [["Dize", "Tahmin_Etiket", "P1", "P2", "P3"]]
    .reset_index(drop=True)
)

tum.to_excel(
    OUT_TUM,
    index=False
)


# ============================================================
# 2) YÜKSEK GENELLEME ADAYLARI
# P3 > P1 AND P3 > P2 AND P1 < 0.20
# ============================================================

gen = kulliyat[
    (kulliyat["P3"] > kulliyat["P1"]) &
    (kulliyat["P3"] > kulliyat["P2"]) &
    (kulliyat["P1"] < P1_ESIK)
].copy()

gen = (
    gen
    .sort_values("P3", ascending=False)
    [["Yöre","Dize", "Tahmin_Etiket", "P1", "P2", "P3"]]
    .reset_index(drop=True)
)

gen.to_excel(
    OUT_GEN,
    index=False
)
# ============================================================
# YÜKSEK GENELLEME ADAYLARININ YÖRE DAĞILIMI
# ============================================================

yore_sayilari = (
    gen["Yöre"]
    .fillna("Bilinmeyen")
    .replace("", "Bilinmeyen")
    .value_counts()
    .rename_axis("Yöre")
    .reset_index(name="Genelleme_Sayisi")
)

yore_sayilari.to_excel(
    OUT_YORE,
    index=False
)


# ============================================================
# İLK 15 YÖRE - YATAY ÇUBUK GRAFİK
# ============================================================

ilk15 = yore_sayilari.head(15)

plt.figure(figsize=(9, 7))

plt.barh(
    ilk15["Yöre"],
    ilk15["Genelleme_Sayisi"]
)

# En yüksek sayı üstte görünsün
plt.gca().invert_yaxis()

plt.xlabel("Yüksek genelleme adayı sayısı")
plt.ylabel("Yöre")
plt.title("Yüksek Genelleme Adaylarının İlk 15 Yöreye Göre Dağılımı")

plt.tight_layout()

plt.savefig(
    GRAFIK_YORE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TERNARY YOĞUNLUK GÖRSELİ
# ============================================================

p1 = kulliyat["P1"].to_numpy()
p2 = kulliyat["P2"].to_numpy()
p3 = kulliyat["P3"].to_numpy()

h = np.sqrt(3) / 2
x = p2 + 0.5 * p3
y = h * p3

plt.figure(figsize=(9, 8))

hb = plt.hexbin(
    x,
    y,
    gridsize=75,
    bins="log",
    mincnt=1,
    linewidths=0
)

plt.colorbar(
    hb,
    label="Dize yoğunluğu (log ölçek)"
)

# Üçgen
plt.plot(
    [0, 1, 0.5, 0],
    [0, 0, h, 0],
    linewidth=1.5
)

# Sınıf eşitlik sınırları
t = np.linspace(0, 0.5, 200)

for a1, a2, a3 in [
    (t, t, 1 - 2*t),       # P1=P2
    (t, 1 - 2*t, t),       # P1=P3
    (1 - 2*t, t, t)        # P2=P3
]:
    plt.plot(
        a2 + 0.5*a3,
        h*a3,
        linestyle="--",
        linewidth=1
    )

plt.text(
    -0.02, -0.035,
    "P1 = 1\nGenelleme değil",
    ha="center",
    va="top"
)

plt.text(
    1.02, -0.035,
    "P2 = 1\nKararsız",
    ha="center",
    va="top"
)

plt.text(
    0.5, h + 0.035,
    "P3 = 1\nGenelleme",
    ha="center",
    va="bottom"
)

plt.title(
    "Külliyatın P1-P2-P3 Olasılık Yoğunluğu"
)

plt.axis("equal")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    GRAFIK,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SONUÇ
# ============================================================

print("\nNİHAİ TAHMİN TAMAMLANDI")
print("Eğitim :", len(egitim))
print("Taranan:", len(kulliyat))

print("\nTahmin dağılımı:")
print(
    kulliyat["Tahmin_Etiket"]
    .value_counts()
    .sort_index()
)

print(
    "\nYüksek genelleme adayı:",
    len(gen)
)

print("\nFiltre: P3 > P1 AND P3 > P2 AND P1 < 0.20")

print("\nDosyalar:")
print(OUT_TUM)
print(OUT_GEN)
print(GRAFIK)