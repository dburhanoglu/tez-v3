import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ayarlar import EXCEL_DIR, RANDOM_STATE, FINAL_DATA


# ============================================================
# DOSYALAR
# ============================================================

BASLANGIC = EXCEL_DIR / "01_Baslangic_Etiketleme.xlsx"
TUR0 = EXCEL_DIR / "03_Tur0_Etiketle.xlsx"
TUR1 = EXCEL_DIR / "04_Tur1_Etiketle.xlsx"
TUR2 = EXCEL_DIR / "06_Tur2_Etiketle.xlsx"
CORPUS = FINAL_DATA
POOL = EXCEL_DIR / "03_Aktif_Ogrenme_Havuzu.xlsx"

OUT = EXCEL_DIR / "08_Tur3_Etiketle.xlsx"
KAYIT = EXCEL_DIR / "08_Tur3_Secim_Kaydi.xlsx"


# ============================================================
# VERİ
# ============================================================

baslangic = pd.read_excel(BASLANGIC)
tur0 = pd.read_excel(TUR0)
tur1 = pd.read_excel(TUR1)
tur2 = pd.read_excel(TUR2)
corpus = pd.read_excel(CORPUS)

for d in [baslangic, tur0, tur1, tur2]:
    d["Etiket"] = pd.to_numeric(d["Etiket"]).astype(int)


# 01 için Turku_ID
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

eslesme = corpus[
    ["Turku_ID", "Başlık", "Yöre", "Sözler"]
].dropna(subset=["Sözler"]).copy()

eslesme["Dize"] = eslesme["Sözler"].map(
    lambda x: str(x).splitlines()
)

eslesme = eslesme.explode("Dize")
eslesme["Dize"] = eslesme["Dize"].str.strip()

eslesme = eslesme[
    ["Turku_ID", "Başlık", "Yöre", "Dize"]
].drop_duplicates(["Başlık", "Yöre", "Dize"])

baslangic = baslangic.merge(
    eslesme,
    on=["Başlık", "Yöre", "Dize"],
    how="left"
)


# ============================================================
# EĞİTİM = 01 + TUR0 + TUR1 + TUR2
# ============================================================

kolonlar = ["Turku_ID", "Başlık", "Yöre", "Dize", "Etiket"]

df = pd.concat([
    baslangic[kolonlar],
    tur0[kolonlar],
    tur1[kolonlar],
    tur2[kolonlar]
], ignore_index=True)

df = df.drop_duplicates(
    ["Turku_ID", "Dize"],
    keep="last"
).reset_index(drop=True)


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
    df["Dize"].astype(str),
    df["Etiket"]
)


# ============================================================
# HAVUZ
# ============================================================

pool = pd.read_excel(POOL)

pool = pool[
    ~pool["Turku_ID"].isin(set(df["Turku_ID"]))
].copy()


# ============================================================
# SKORLAR
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

pool["Sinir_13"] = np.minimum(
    pool["P1"],
    pool["P3"]
)

pool["Sinir_23"] = np.minimum(
    pool["P2"],
    pool["P3"]
)


# ============================================================
# TUR 3 SEÇİMİ
#
# 45 → 1↔3 sınırı
# 38 → 2↔3 sınırı
# 30 → P3 yüksek
# 22 → entropy
# 15 → random
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

    for idx, row in aday.iterrows():

        turku_id = row["Turku_ID"]

        if turku_id in used_ids:
            continue

        selected.append(idx)
        used_ids.add(turku_id)
        neden[idx] = sebep

        if sum(neden[i] == sebep for i in selected) == n:
            break


sec(45, "1-3 Siniri", "Sinir_13")
sec(38, "2-3 Siniri", "Sinir_23")
sec(30, "P3 Yuksek", "P3")
sec(22, "Entropy", "Entropy")
sec(15, "Random", random=True)


# ============================================================
# KAYDET
# ============================================================

tur3 = pool.loc[selected].copy()

tur3["Secim_Sebebi"] = [
    neden[i]
    for i in tur3.index
]

if len(tur3) != 150:
    raise ValueError(f"150 yerine {len(tur3)} örnek seçildi.")

tur3.to_excel(
    KAYIT,
    index=False
)

etiket = (
    tur3[
        ["Turku_ID", "Başlık", "Yöre", "Dize"]
    ]
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)

etiket.insert(
    0,
    "ID",
    range(1, 151)
)

etiket["Etiket"] = ""
etiket["Not"] = ""

etiket.to_excel(
    OUT,
    index=False
)


print("\nTUR 3 HAZIR")
print("Eğitim :", len(df))
print("Havuz  :", len(pool))
print("Seçilen:", len(tur3))

print("\nSeçim dağılımı:")
print(tur3["Secim_Sebebi"].value_counts())