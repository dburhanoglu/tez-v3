import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ayarlar import EXCEL_DIR, RANDOM_STATE, FINAL_DATA


# ============================================================
# DOSYALAR
# ============================================================

BASLANGIC_FILE = EXCEL_DIR / "01_Baslangic_Etiketleme.xlsx"
TUR0_FILE = EXCEL_DIR / "03_Tur0_Etiketle.xlsx"
TUR1_FILE = EXCEL_DIR / "04_Tur1_Etiketle.xlsx"
CORPUS_FILE = FINAL_DATA

POOL_FILE = EXCEL_DIR / "03_Aktif_Ogrenme_Havuzu.xlsx"

OUTPUT_FILE = EXCEL_DIR / "06_Tur2_Etiketle.xlsx"
KAYIT_FILE = EXCEL_DIR / "06_Tur2_Secim_Kaydi.xlsx"


# ============================================================
# VERİLER
# ============================================================

baslangic = pd.read_excel(BASLANGIC_FILE)
tur0 = pd.read_excel(TUR0_FILE)
tur1 = pd.read_excel(TUR1_FILE)
corpus = pd.read_excel(CORPUS_FILE)

for df in [baslangic, tur0, tur1]:
    df["Etiket"] = pd.to_numeric(df["Etiket"]).astype(int)


# ============================================================
# BAŞLANGIÇ 300 İÇİN TURKU_ID
# ============================================================

id_sutunu = next(
    (s for s in ["Turku_ID", "Türkü_ID", "Kaynak_ID", "ID"]
     if s in corpus.columns),
    None
)

corpus["Turku_ID"] = (
    corpus[id_sutunu]
    if id_sutunu
    else np.arange(1, len(corpus) + 1)
)

eslesme = corpus[
    ["Turku_ID", "Başlık", "Yöre", "Sözler"]
].dropna(subset=["Sözler"]).copy()

eslesme["Dize"] = eslesme["Sözler"].map(
    lambda x: str(x).splitlines()
)

eslesme = eslesme.explode("Dize")

eslesme["Dize"] = eslesme["Dize"].str.strip()

eslesme = (
    eslesme[["Turku_ID", "Başlık", "Yöre", "Dize"]]
    .drop_duplicates(["Başlık", "Yöre", "Dize"])
)

baslangic = baslangic.merge(
    eslesme,
    on=["Başlık", "Yöre", "Dize"],
    how="left"
)

baslangic["Turku_ID"] = baslangic["Turku_ID"].astype(int)


# ============================================================
# EĞİTİM = 01 + TUR0 + TUR1
# ============================================================

kolonlar = [
    "Turku_ID",
    "Başlık",
    "Yöre",
    "Dize",
    "Etiket"
]

egitim = pd.concat(
    [
        baslangic[kolonlar],
        tur0[kolonlar],
        tur1[kolonlar]
    ],
    ignore_index=True
)

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
# HAVUZ
# ============================================================

pool = pd.read_excel(POOL_FILE)

# Daha önce eğitimde kullanılan bütün türküleri çıkar
pool = pool[
    ~pool["Turku_ID"].isin(set(egitim["Turku_ID"]))
].copy()


# ============================================================
# OLASILIKLAR
# ============================================================

proba = model.predict_proba(
    pool["Dize"].astype(str)
)

classes = list(model.named_steps["lr"].classes_)

pool["P1"] = proba[:, classes.index(1)]
pool["P2"] = proba[:, classes.index(2)]
pool["P3"] = proba[:, classes.index(3)]

pool["Entropy"] = -np.sum(
    proba * np.log(proba + 1e-12),
    axis=1
)


# 1 ile 3'ün ikisinin de yüksek olduğu örnekler
pool["Sinir_13"] = np.minimum(
    pool["P1"],
    pool["P3"]
)

# 2 ile 3'ün ikisinin de yüksek olduğu örnekler
pool["Sinir_23"] = np.minimum(
    pool["P2"],
    pool["P3"]
)


# ============================================================
# TUR 2 SEÇİMİ
#
# 45 → 1↔3 sınırı
# 38 → 2↔3 sınırı
# 30 → P3 yüksek
# 22 → entropy
# 15 → random
#
# Aynı türküden maksimum 1 dize
# ============================================================

selected = []
neden = {}
used_ids = set()


def sec(n, sebep, skor=None, random=False):

    kalan = pool[
        ~pool.index.isin(selected)
        & ~pool["Turku_ID"].isin(used_ids)
    ]

    if random:
        aday = kalan.sample(
            frac=1,
            random_state=RANDOM_STATE
        )
    else:
        aday = kalan.sort_values(
            skor,
            ascending=False
        )

    secilen = []

    for idx, row in aday.iterrows():

        turku_id = row["Turku_ID"]

        if turku_id in used_ids:
            continue

        secilen.append(idx)
        used_ids.add(turku_id)
        neden[idx] = sebep

        if len(secilen) == n:
            break

    selected.extend(secilen)


sec(45, "1-3 Siniri", "Sinir_13")
sec(38, "2-3 Siniri", "Sinir_23")
sec(30, "P3 Yuksek", "P3")
sec(22, "Entropy", "Entropy")
sec(15, "Random", random=True)


# ============================================================
# KAYIT DOSYASI
# ============================================================

tur2 = pool.loc[selected].copy()

tur2["Secim_Sebebi"] = [
    neden[i]
    for i in tur2.index
]

tur2.to_excel(
    KAYIT_FILE,
    index=False
)


# ============================================================
# KÖR ETİKETLEME DOSYASI
# ============================================================

etiketleme = (
    tur2[
        ["Turku_ID", "Başlık", "Yöre", "Dize"]
    ]
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)

etiketleme.insert(
    0,
    "ID",
    range(1, len(etiketleme) + 1)
)

etiketleme["Etiket"] = ""
etiketleme["Not"] = ""

etiketleme.to_excel(
    OUTPUT_FILE,
    index=False
)


print("\nTUR 2 HAZIR")
print("Eğitim :", len(egitim))
print("Havuz  :", len(pool))
print("Seçilen:", len(tur2))

print("\nSeçim dağılımı:")
print(tur2["Secim_Sebebi"].value_counts())