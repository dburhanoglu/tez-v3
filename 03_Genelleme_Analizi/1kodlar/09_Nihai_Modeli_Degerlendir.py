import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

BASLANGIC = EXCEL_DIR / "01_Baslangic_Etiketleme.xlsx"
TUR0 = EXCEL_DIR / "03_Tur0_Etiketle.xlsx"
TUR1 = EXCEL_DIR / "04_Tur1_Etiketle.xlsx"
TUR2 = EXCEL_DIR / "06_Tur2_Etiketle.xlsx"
TUR3 = EXCEL_DIR / "08_Tur3_Etiketle.xlsx"
CORPUS = FINAL_DATA

PERF_FILE = EXCEL_DIR / "04_Performans_Gecmisi.xlsx"
OUTPUT = EXCEL_DIR / "09_Nihai_Model_Sonuclari.xlsx"


# ============================================================
# VERİ
# ============================================================

baslangic = pd.read_excel(BASLANGIC)
tur0 = pd.read_excel(TUR0)
tur1 = pd.read_excel(TUR1)
tur2 = pd.read_excel(TUR2)
tur3 = pd.read_excel(TUR3)
corpus = pd.read_excel(CORPUS)

for d in [baslangic, tur0, tur1, tur2, tur3]:
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
# TÜM ETİKETLİ VERİ
# ============================================================

kolonlar = [
    "Turku_ID",
    "Başlık",
    "Yöre",
    "Dize",
    "Etiket"
]

df = pd.concat([
    baslangic[kolonlar],
    tur0[kolonlar],
    tur1[kolonlar],
    tur2[kolonlar],
    tur3[kolonlar]
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


# ============================================================
# 5-FOLD GROUP CV
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

proba = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)

classes = np.array([1, 2, 3])
pred = classes[np.argmax(proba, axis=1)]


# ============================================================
# PERFORMANS
# ============================================================

f1s = f1_score(
    y,
    pred,
    labels=[1, 2, 3],
    average=None,
    zero_division=0
)

precision3 = precision_score(
    y,
    pred,
    labels=[3],
    average="macro",
    zero_division=0
)

recall3 = recall_score(
    y,
    pred,
    labels=[3],
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    y,
    pred,
    labels=[1, 2, 3]
)

gercek3 = (y == 3)

kacis1 = (
    ((gercek3) & (pred == 1)).sum()
    / gercek3.sum()
)

kacis2 = (
    ((gercek3) & (pred == 2)).sum()
    / gercek3.sum()
)


# ============================================================
# PERFORMANS GEÇMİŞİ
# ============================================================

yeni = pd.DataFrame([{
    "Tur": 3,
    "N": len(df),

    "Sinif_1": int((y == 1).sum()),
    "Sinif_2": int((y == 2).sum()),
    "Sinif_3": int((y == 3).sum()),

    "Accuracy": accuracy_score(y, pred),
    "Balanced_Accuracy": balanced_accuracy_score(y, pred),
    "Macro_F1": f1_score(
        y, pred,
        average="macro",
        zero_division=0
    ),

    "F1_Sinif_1": f1s[0],
    "F1_Sinif_2": f1s[1],
    "F1_Sinif_3": f1s[2],

    "Precision_Sinif_3": precision3,
    "Recall_Sinif_3": recall3,

    "MCC": matthews_corrcoef(y, pred),

    "Gercek3_1e_Kacis": kacis1,
    "Gercek3_2ye_Kacis": kacis2
}])

perf = pd.read_excel(PERF_FILE)

perf = perf.drop(
    columns=["Gercek3_2veya3"],
    errors="ignore"
)

perf = perf[perf["Tur"] != 3]

perf = pd.concat(
    [perf, yeni],
    ignore_index=True
).sort_values("Tur")

perf.to_excel(
    PERF_FILE,
    index=False
)


# ============================================================
# CV TAHMİNLERİ
# ============================================================

tahminler = df.copy()

tahminler = tahminler.rename(
    columns={"Etiket": "Gercek_Etiket"}
)

tahminler["Tahmin_Etiket"] = pred
tahminler["P1"] = proba[:, 0]
tahminler["P2"] = proba[:, 1]
tahminler["P3"] = proba[:, 2]

tahminler["Dogru"] = (
    tahminler["Gercek_Etiket"]
    == tahminler["Tahmin_Etiket"]
).astype(int)


# ============================================================
# ÖZET
# ============================================================

ozet = pd.DataFrame([
    ["Toplam veri", len(df)],
    ["Sınıf 1", int((y == 1).sum())],
    ["Sınıf 2", int((y == 2).sum())],
    ["Sınıf 3", int((y == 3).sum())],

    ["Accuracy", accuracy_score(y, pred)],
    ["Balanced Accuracy", balanced_accuracy_score(y, pred)],
    ["Macro F1", f1_score(y, pred, average="macro")],
    ["MCC", matthews_corrcoef(y, pred)],

    ["F1 Sınıf 1", f1s[0]],
    ["F1 Sınıf 2", f1s[1]],
    ["F1 Sınıf 3", f1s[2]],

    ["Precision Sınıf 3", precision3],
    ["Recall Sınıf 3", recall3],

    ["Gerçek 3 -> 1 kaçış", kacis1],
    ["Gerçek 3 -> 2 kaçış", kacis2]
], columns=["Metrik", "Deger"])


sinif_matris = pd.DataFrame(
    cm,
    index=[
        "Gercek_1",
        "Gercek_2",
        "Gercek_3"
    ],
    columns=[
        "Tahmin_1",
        "Tahmin_2",
        "Tahmin_3"
    ]
)


# ============================================================
# EXCEL
# ============================================================

with pd.ExcelWriter(
    OUTPUT,
    engine="openpyxl"
) as writer:

    ozet.to_excel(
        writer,
        sheet_name="Genel_Ozet",
        index=False
    )

    perf.to_excel(
        writer,
        sheet_name="Tur_Gecmisi",
        index=False
    )

    tahminler.to_excel(
        writer,
        sheet_name="CV_Tahminleri",
        index=False
    )

    sinif_matris.to_excel(
        writer,
        sheet_name="3Sinif_Matris"
    )


# ============================================================
# TERMINAL
# ============================================================

print("\nNİHAİ MODEL DEĞERLENDİRMESİ")
print("Toplam veri:", len(df))

print("\nEtiket dağılımı:")
print(y.value_counts().sort_index())

print("\nPERFORMANS GEÇMİŞİ")
print(perf.round(3).to_string(index=False))

print("\nCONFUSION MATRIX")
print(sinif_matris)

print("\nSINIF 3")
print(f"Precision    : {precision3:.3f}")
print(f"Recall       : {recall3:.3f}")
print(f"F1           : {f1s[2]:.3f}")
print(f"3 -> 1 kaçış: {kacis1:.3f}")
print(f"3 -> 2 kaçış: {kacis2:.3f}")

print("\nExcel:")
print(OUTPUT)