#sadecemodelin performansını ölçer
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

from ayarlar import EXCEL_DIR, RANDOM_STATE, FINAL_DATA


# ============================================================
# DOSYALAR
# ============================================================

BASLANGIC_FILE = EXCEL_DIR / "01_Baslangic_Etiketleme.xlsx"
TUR0_FILE = EXCEL_DIR / "03_Tur0_Etiketle.xlsx"
TUR1_FILE = EXCEL_DIR / "04_Tur1_Etiketle.xlsx"
CORPUS_FILE = FINAL_DATA

PERF_FILE = EXCEL_DIR / "04_Performans_Gecmisi.xlsx"


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
# BAŞLANGIÇ 300 İÇİN TURKU_ID BUL
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

eslesmeler = []

for _, row in corpus.iterrows():

    sozler = row.get("Sözler", "")

    if pd.isna(sozler):
        continue

    for dize in str(sozler).splitlines():

        dize = dize.strip()

        if dize:
            eslesmeler.append({
                "Turku_ID": row["Turku_ID"],
                "Başlık": row.get("Başlık", ""),
                "Yöre": row.get("Yöre", ""),
                "Dize": dize
            })

eslesme_df = (
    pd.DataFrame(eslesmeler)
    .drop_duplicates(["Başlık", "Yöre", "Dize"])
)

baslangic = baslangic.merge(
    eslesme_df,
    on=["Başlık", "Yöre", "Dize"],
    how="left"
)

baslangic["Turku_ID"] = baslangic["Turku_ID"].astype(int)


# ============================================================
# 01 + TUR0 + TUR1
# ============================================================

kolonlar = [
    "Turku_ID",
    "Başlık",
    "Yöre",
    "Dize",
    "Etiket"
]

df = pd.concat(
    [
        baslangic[kolonlar],
        tur0[kolonlar],
        tur1[kolonlar]
    ],
    ignore_index=True
)

df = (
    df
    .drop_duplicates(["Turku_ID", "Dize"])
    .reset_index(drop=True)
)


# ============================================================
# 5-FOLD GROUP CV
# ============================================================

X = df["Dize"].astype(str)
y = df["Etiket"].astype(int)
groups = df["Turku_ID"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

tahmin = cross_val_predict(
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

f1_sinif = f1_score(
    y,
    tahmin,
    labels=[1, 2, 3],
    average=None,
    zero_division=0
)

precision_3 = precision_score(
    y,
    tahmin,
    labels=[3],
    average="macro",
    zero_division=0
)

recall_3 = recall_score(
    y,
    tahmin,
    labels=[3],
    average="macro",
    zero_division=0
)

yeni = pd.DataFrame([{
    "Tur": 1,
    "N": len(df),

    "Sinif_1": int((y == 1).sum()),
    "Sinif_2": int((y == 2).sum()),
    "Sinif_3": int((y == 3).sum()),

    "Accuracy": accuracy_score(y, tahmin),
    "Balanced_Accuracy": balanced_accuracy_score(y, tahmin),
    "Macro_F1": f1_score(
        y,
        tahmin,
        average="macro",
        zero_division=0
    ),

    "F1_Sinif_1": f1_sinif[0],
    "F1_Sinif_2": f1_sinif[1],
    "F1_Sinif_3": f1_sinif[2],

    "Precision_Sinif_3": precision_3,
    "Recall_Sinif_3": recall_3,

    "MCC": matthews_corrcoef(y, tahmin)
}])


# ============================================================
# PERFORMANS GEÇMİŞİ
# ============================================================

perf = pd.read_excel(PERF_FILE)

perf = perf[
    perf["Tur"] != 1
]

perf = (
    pd.concat([perf, yeni], ignore_index=True)
    .sort_values("Tur")
)

perf.to_excel(
    PERF_FILE,
    index=False
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    tahmin,
    labels=[1, 2, 3]
)

cm_df = pd.DataFrame(
    cm,
    index=["Gerçek 1", "Gerçek 2", "Gerçek 3"],
    columns=["Tahmin 1", "Tahmin 2", "Tahmin 3"]
)


# ============================================================
# SONUÇ
# ============================================================

print("\nEtiket dağılımı:")
print(y.value_counts().sort_index())

print("\nPERFORMANS")
print(perf.round(3).to_string(index=False))

print("\nCONFUSION MATRIX")
print(cm_df)

print("\nSINIF 3")
print(f"Precision : {precision_3:.3f}")
print(f"Recall    : {recall_3:.3f}")