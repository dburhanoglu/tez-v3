import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef
)

from ayarlar import EXCEL_DIR, RANDOM_STATE, AL_BATCH_SIZE, FINAL_DATA


# ============================================================
# DOSYALAR
# ============================================================

BASLANGIC_FILE = EXCEL_DIR / "01_Baslangic_Etiketleme.xlsx"
TUR0_FILE = EXCEL_DIR / "03_Tur0_Etiketle.xlsx"
POOL_FILE = EXCEL_DIR / "03_Aktif_Ogrenme_Havuzu.xlsx"
CORPUS_FILE = FINAL_DATA

PERFORMANS_FILE = EXCEL_DIR / "04_Performans_Gecmisi.xlsx"
TUR1_FILE = EXCEL_DIR / "04_Tur1_Etiketle.xlsx"
TUR1_KAYIT_FILE = EXCEL_DIR / "04_Tur1_Secim_Kaydi.xlsx"


# ============================================================
# MODEL
# Word + Char TF-IDF + Logistic Regression
# ============================================================

def model_olustur():

    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=None,
                sublinear_tf=True
            )
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=2,
                sublinear_tf=True
            )
        )
    ])

    return Pipeline([
        ("features", features),
        (
            "lr",
            LogisticRegression(
                class_weight="balanced",
                max_iter=3000,
                random_state=RANDOM_STATE
            )
        )
    ])


# ============================================================
# AYNI TURDA AYNI TÜRKÜDEN EN FAZLA 1 DİZE
# ============================================================

def cesitli_sec(
    df,
    n,
    kullanilan_turku_idleri,
    siralama=None,
    ascending=False,
    random=False
):

    if random:
        aday = df.sample(
            frac=1,
            random_state=RANDOM_STATE
        )

    elif siralama is not None:
        aday = df.sort_values(
            siralama,
            ascending=ascending
        )

    else:
        aday = df

    secilen = []

    for idx, row in aday.iterrows():

        turku_id = row["Turku_ID"]

        if turku_id in kullanilan_turku_idleri:
            continue

        secilen.append(idx)
        kullanilan_turku_idleri.add(turku_id)

        if len(secilen) >= n:
            break

    return secilen


# ============================================================
# VERİLERİ OKU
# ============================================================

baslangic = pd.read_excel(BASLANGIC_FILE)
tur0 = pd.read_excel(TUR0_FILE)
corpus = pd.read_excel(CORPUS_FILE)


# ============================================================
# ETİKET KONTROLÜ
# ============================================================

for isim, df in [
    ("Başlangıç", baslangic),
    ("Tur0", tur0)
]:

    df["Etiket"] = pd.to_numeric(
        df["Etiket"],
        errors="coerce"
    )

    if (
        df["Etiket"].isna().any()
        or not df["Etiket"].isin([1, 2, 3]).all()
    ):
        raise ValueError(
            f"{isim} dosyasında boş veya "
            "1/2/3 dışında etiket var."
        )

    df["Etiket"] = df["Etiket"].astype(int)


# ============================================================
# CORPUS TÜRKÜ ID
# 03 İLE AYNI MANTIK
# ============================================================

id_sutunu = next(
    (
        sutun
        for sutun in [
            "Turku_ID",
            "Türkü_ID",
            "Kaynak_ID",
            "ID"
        ]
        if sutun in corpus.columns
    ),
    None
)

if id_sutunu:
    corpus["Turku_ID"] = corpus[id_sutunu]
else:
    corpus["Turku_ID"] = np.arange(
        1,
        len(corpus) + 1
    )


# ============================================================
# BAŞLANGIÇ 300 ÖRNEĞİN TURKU_ID'SİNİ BUL
# ============================================================

eslesmeler = []

for _, row in corpus.iterrows():

    sozler = row.get("Sözler", "")

    if pd.isna(sozler):
        continue

    for dize in str(sozler).splitlines():

        dize = dize.strip()

        if not dize:
            continue

        eslesmeler.append({
            "Turku_ID": row["Turku_ID"],
            "Başlık": row.get("Başlık", ""),
            "Yöre": row.get("Yöre", ""),
            "Dize": dize
        })


eslesme_df = (
    pd.DataFrame(eslesmeler)
    .drop_duplicates(
        subset=["Başlık", "Yöre", "Dize"],
        keep="first"
    )
)


baslangic = baslangic.merge(
    eslesme_df,
    on=["Başlık", "Yöre", "Dize"],
    how="left"
)


if baslangic["Turku_ID"].isna().any():

    eksik = baslangic["Turku_ID"].isna().sum()

    raise ValueError(
        f"{eksik} başlangıç örneğinin "
        "Turku_ID'si bulunamadı."
    )


# ============================================================
# 01 + TUR0 = EĞİTİM
# Master Excel yok; yalnızca RAM'de birleşiyor
# ============================================================

egitim = pd.concat(
    [
        baslangic[
            [
                "Turku_ID",
                "Başlık",
                "Yöre",
                "Dize",
                "Etiket"
            ]
        ],
        tur0[
            [
                "Turku_ID",
                "Başlık",
                "Yöre",
                "Dize",
                "Etiket"
            ]
        ]
    ],
    ignore_index=True
)


# Aynı türkü + aynı dize varsa bir kez tut.
# Aynı dize farklı türkülerdeyse korunur.
egitim = (
    egitim
    .drop_duplicates(
        subset=["Turku_ID", "Dize"],
        keep="first"
    )
    .reset_index(drop=True)
)


print("=" * 70)
print("TUR 0 SONRASI EĞİTİM")
print("=" * 70)

print("Toplam eğitim:", len(egitim))
print(
    egitim["Etiket"]
    .value_counts()
    .sort_index()
)


# ============================================================
# GROUP CV
# Aynı Turku_ID farklı fold'lara gitmez
# ============================================================

X = egitim["Dize"].astype(str)
y = egitim["Etiket"].astype(int)
groups = egitim["Turku_ID"]

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

model = model_olustur()

tahmin = cross_val_predict(
    model,
    X,
    y,
    groups=groups,
    cv=cv,
    n_jobs=-1
)


# ============================================================
# TUR0 SONRASI 3 SINIF PERFORMANSI
# ============================================================

accuracy = accuracy_score(
    y,
    tahmin
)

balanced_accuracy = balanced_accuracy_score(
    y,
    tahmin
)

macro_f1 = f1_score(
    y,
    tahmin,
    average="macro",
    zero_division=0
)

f1_1 = f1_score(
    y,
    tahmin,
    labels=[1],
    average="macro",
    zero_division=0
)

f1_2 = f1_score(
    y,
    tahmin,
    labels=[2],
    average="macro",
    zero_division=0
)

f1_3 = f1_score(
    y,
    tahmin,
    labels=[3],
    average="macro",
    zero_division=0
)

mcc = matthews_corrcoef(
    y,
    tahmin
)


performans = pd.DataFrame([
    {
        "Tur": 0,
        "N": len(egitim),

        "Sinif_1": int((y == 1).sum()),
        "Sinif_2": int((y == 2).sum()),
        "Sinif_3": int((y == 3).sum()),

        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Macro_F1": macro_f1,

        "F1_Sinif_1": f1_1,
        "F1_Sinif_2": f1_2,
        "F1_Sinif_3": f1_3,

        "MCC": mcc
    }
])


performans.to_excel(
    PERFORMANS_FILE,
    index=False
)


print("\nTUR 0 SONRASI PERFORMANS")
print(
    performans
    .round(3)
    .to_string(index=False)
)


# ============================================================
# TÜM 450 VERİYLE MODELİ EĞİT
# Bundan sonrası Tur1 seçmek için
# ============================================================

model.fit(
    X,
    y
)


# ============================================================
# ACTIVE LEARNING HAVUZU
# ============================================================

pool = pd.read_excel(
    POOL_FILE
)


# Daha önce etiketlenmiş türküler artık kullanılmaz.
etiketli_turku_idleri = set(
    egitim["Turku_ID"]
)

pool = pool[
    ~pool["Turku_ID"].isin(
        etiketli_turku_idleri
    )
].copy()


print(
    "\nTur1 için kullanılabilir dize:",
    len(pool)
)


# ============================================================
# POOL'U YENİ MODELLE SKORLA
# ============================================================

proba = model.predict_proba(
    pool["Dize"].astype(str)
)

classes = model.named_steps["lr"].classes_


for i, sinif in enumerate(classes):

    pool[
        f"P_{int(sinif)}"
    ] = proba[:, i]


# Entropy yüksek → model daha kararsız
pool["Belirsizlik"] = -np.sum(
    proba * np.log(proba + 1e-12),
    axis=1
)


# Sınıf 2 olasılığı
pool["Sinirda_Skoru"] = (
    pool["P_2"]
)


# ============================================================
# TUR1 SEÇİM PLANI
# %50 belirsiz + %20 sınıf 2 + %30 rastgele
# ============================================================

N_TOTAL = AL_BATCH_SIZE

N_UNCERTAIN = round(
    N_TOTAL * 0.50
)

N_BOUNDARY = round(
    N_TOTAL * 0.20
)

N_RANDOM = (
    N_TOTAL
    - N_UNCERTAIN
    - N_BOUNDARY
)


print("\nTur1 seçim planı:")
print("Belirsiz :", N_UNCERTAIN)
print("Sınırda  :", N_BOUNDARY)
print("Rastgele :", N_RANDOM)


# ============================================================
# TUR1 ÖRNEKLERİNİ SEÇ
# Tur1 genelinde aynı türküden en fazla 1 dize
# ============================================================

kullanilan_turku_idleri = set()


# %50 belirsiz
idx_uncertain = cesitli_sec(
    pool,
    N_UNCERTAIN,
    kullanilan_turku_idleri,
    siralama="Belirsizlik",
    ascending=False
)

secilen = set(idx_uncertain)


# %20 P2 yüksek
kalan = pool[
    ~pool.index.isin(secilen)
]

idx_boundary = cesitli_sec(
    kalan,
    N_BOUNDARY,
    kullanilan_turku_idleri,
    siralama="Sinirda_Skoru",
    ascending=False
)

secilen.update(idx_boundary)


# %30 rastgele
kalan = pool[
    ~pool.index.isin(secilen)
]

idx_random = cesitli_sec(
    kalan,
    N_RANDOM,
    kullanilan_turku_idleri,
    random=True
)

secilen.update(idx_random)


if len(secilen) != N_TOTAL:
    raise ValueError(
        f"{N_TOTAL} yerine "
        f"{len(secilen)} örnek seçildi."
    )


# ============================================================
# TUR1
# ============================================================

tur1 = pool.loc[
    list(secilen)
].copy()

tur1["Secim_Turu"] = ""

tur1.loc[
    idx_uncertain,
    "Secim_Turu"
] = "Belirsiz"

tur1.loc[
    idx_boundary,
    "Secim_Turu"
] = "Sinirda"

tur1.loc[
    idx_random,
    "Secim_Turu"
] = "Rastgele"


# Model skorları + seçim sebebi
tur1.to_excel(
    TUR1_KAYIT_FILE,
    index=False
)


# ============================================================
# KÖR ETİKETLEME DOSYASI
# ============================================================

tur1 = (
    tur1
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)


etiketleme = tur1[
    [
        "Turku_ID",
        "Başlık",
        "Yöre",
        "Dize"
    ]
].copy()


etiketleme.insert(
    0,
    "ID",
    range(1, len(etiketleme) + 1)
)

etiketleme["Etiket"] = ""
etiketleme["Not"] = ""


etiketleme.to_excel(
    TUR1_FILE,
    index=False
)


# ============================================================
# SONUÇ
# ============================================================

print("\n" + "=" * 70)
print("TUR 1 HAZIR")
print("=" * 70)

print("Etiketlenecek örnek:", len(etiketleme))
print("Tur1 dosyası:", TUR1_FILE)
print("Seçim kaydı:", TUR1_KAYIT_FILE)
print("Performans:", PERFORMANS_FILE)