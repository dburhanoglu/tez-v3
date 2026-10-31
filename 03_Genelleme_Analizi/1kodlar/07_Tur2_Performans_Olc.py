import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, matthews_corrcoef,
    confusion_matrix
)

from ayarlar import EXCEL_DIR, RANDOM_STATE, FINAL_DATA


# ============================================================
# DOSYALAR
# ============================================================

BASLANGIC = EXCEL_DIR / "01_Baslangic_Etiketleme.xlsx"
TUR0 = EXCEL_DIR / "03_Tur0_Etiketle.xlsx"
TUR1 = EXCEL_DIR / "04_Tur1_Etiketle.xlsx"
TUR2 = EXCEL_DIR / "06_Tur2_Etiketle.xlsx"
CORPUS = FINAL_DATA

PERF_FILE = EXCEL_DIR / "04_Performans_Gecmisi.xlsx"


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
# 01 + TUR0 + TUR1 + TUR2
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

X = df["Dize"].astype(str)
y = df["Etiket"].astype(int)
groups = df["Turku_ID"]


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

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

pred = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=cv,
    n_jobs=-1
)


# ============================================================
# PERFORMANS
# ============================================================

f1s = f1_score(
    y, pred,
    labels=[1, 2, 3],
    average=None,
    zero_division=0
)

precision3 = precision_score(
    y, pred,
    labels=[3],
    average="macro",
    zero_division=0
)

recall3 = recall_score(
    y, pred,
    labels=[3],
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    y, pred,
    labels=[1, 2, 3]
)

gercek3 = (y == 3)

kacis_1 = ((gercek3) & (pred == 1)).sum() / gercek3.sum()
kacis_2 = ((gercek3) & (pred == 2)).sum() / gercek3.sum()


yeni = pd.DataFrame([{
    "Tur": 2,
    "N": len(df),

    "Sinif_1": int((y == 1).sum()),
    "Sinif_2": int((y == 2).sum()),
    "Sinif_3": int((y == 3).sum()),

    "Accuracy": accuracy_score(y, pred),
    "Balanced_Accuracy": balanced_accuracy_score(y, pred),
    "Macro_F1": f1_score(y, pred, average="macro"),

    "F1_Sinif_1": f1s[0],
    "F1_Sinif_2": f1s[1],
    "F1_Sinif_3": f1s[2],

    "Precision_Sinif_3": precision3,
    "Recall_Sinif_3": recall3,

    "MCC": matthews_corrcoef(y, pred),

    "Gercek3_1e_Kacis": kacis_1,
    "Gercek3_2ye_Kacis": kacis_2
}])


# ============================================================
# PERFORMANS GEÇMİŞİ
# ============================================================

perf = pd.read_excel(PERF_FILE)
perf = perf[perf["Tur"] != 2]

perf = pd.concat(
    [perf, yeni],
    ignore_index=True
).sort_values("Tur")

perf.to_excel(PERF_FILE, index=False)


# ============================================================
# ÇIKTI
# ============================================================

print("\nEtiket dağılımı:")
print(y.value_counts().sort_index())

print("\nPERFORMANS")
print(perf.round(3).to_string(index=False))

print("\nCONFUSION MATRIX")
print(pd.DataFrame(
    cm,
    index=["Gerçek 1", "Gerçek 2", "Gerçek 3"],
    columns=["Tahmin 1", "Tahmin 2", "Tahmin 3"]
))

print("\nSINIF 3")
print(f"Precision    : {precision3:.3f}")
print(f"Recall       : {recall3:.3f}")
print(f"3 -> 1 kaçış: {kacis_1:.3f}")
print(f"3 -> 2 kaçış: {kacis_2:.3f}")