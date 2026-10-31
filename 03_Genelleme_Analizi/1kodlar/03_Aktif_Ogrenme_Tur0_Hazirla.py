import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from ayarlar import EXCEL_DIR, RANDOM_STATE, AL_BATCH_SIZE, FINAL_DATA


# ============================================================
# AYARLAR / DOSYALAR
# ============================================================

MIN_KELIME = 2
HOLDOUT_ORANI = 0.15

EGITIM_FILE = EXCEL_DIR / "01_Baslangic_Etiketleme.xlsx"
CORPUS_FILE = FINAL_DATA

POOL_FILE = EXCEL_DIR / "03_Aktif_Ogrenme_Havuzu.xlsx"#Sonraki turlarda modelin yeni örnek seçeceği kalan dize havuzudur.
HOLDOUT_FILE = EXCEL_DIR / "03_Final_Holdout_Havuzu.xlsx"#inal değerlendirmede kullanılmak üzere aktif öğrenmeden tamamen ayrı tutulan türkü ve dizeleri içerir.
ROUND0_FILE = EXCEL_DIR / "03_Tur0_Etiketle.xlsx"#1/2/3 etiketi vereceğin 150 Tur0 dizesini içerir.
SECIM_KAYDI_FILE = EXCEL_DIR / "03_Tur0_Secim_Kaydi.xlsx"#Tur0’a hangi dizenin neden seçildiğini ve model skorlarını metodolojik kayıt olarak saklar.


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
# SEÇİM
# Tur0 genelinde aynı türküden en fazla 1 dize
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

    elif siralama:
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

egitim = pd.read_excel(EGITIM_FILE)
corpus = pd.read_excel(CORPUS_FILE)

egitim["Etiket"] = pd.to_numeric(
    egitim["Etiket"],
    errors="coerce"
)

if (
    egitim["Etiket"].isna().any()
    or not egitim["Etiket"].isin([1, 2, 3]).all()
):
    raise ValueError(
        "Başlangıç eğitiminde boş veya 1/2/3 dışında etiket var."
    )

egitim["Etiket"] = egitim["Etiket"].astype(int)

print("Başlangıç eğitim verisi:", len(egitim))
print(egitim["Etiket"].value_counts().sort_index())


# ============================================================
# TÜRKÜ ID
# ============================================================

# Kaynak dosyada gerçek bir ID varsa onu kullan.
# Yoksa her corpus satırına sabit bir türkü ID'si ver.
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

print(
    "Türkü ID kaynağı:",
    id_sutunu if id_sutunu else "corpus satır numarası"
)


# ============================================================
# KÜLLİYATI DİZELERE AYIR
# ============================================================

kayitlar = []

for _, row in corpus.iterrows():

    sozler = row.get("Sözler", "")

    if pd.isna(sozler):
        continue

    for dize in str(sozler).splitlines():

        dize = dize.strip()

        if len(dize.split()) < MIN_KELIME:
            continue

        kayitlar.append({
            "Turku_ID": row["Turku_ID"],
            "Başlık": row.get("Başlık", ""),
            "Yöre": row.get("Yöre", ""),
            "Dize": dize
        })


tum_dizeler = pd.DataFrame(kayitlar)

print("Corpus dize sayısı:", len(tum_dizeler))


# Aynı türkü + aynı dize yalnızca 1 kez.
# Aynı dize farklı türkülerdeyse korunur.
tum_dizeler = (
    tum_dizeler
    .drop_duplicates(
        subset=["Turku_ID", "Dize"]
    )
    .reset_index(drop=True)
)

print(
    "Türkü+dize tekilleştirme sonrası:",
    len(tum_dizeler)
)


# ============================================================
# BAŞLANGIÇ ETİKETLERİNİN TÜRKÜ ID'LERİNİ BUL
# ============================================================

# 01 dosyasında Turku_ID olmadığı için,
# Başlık + Yöre + Dize üzerinden kaynak türküyü tekrar buluyoruz.
turku_eslesme = (
    tum_dizeler[
        ["Turku_ID", "Başlık", "Yöre", "Dize"]
    ]
    .drop_duplicates(
        subset=["Başlık", "Yöre", "Dize"],
        keep="first"
    )
)

egitim = egitim.merge(
    turku_eslesme,
    on=["Başlık", "Yöre", "Dize"],
    how="left"
)

if egitim["Turku_ID"].isna().any():

    eksik = egitim["Turku_ID"].isna().sum()

    raise ValueError(
        f"{eksik} başlangıç dizesinin Turku_ID'si bulunamadı."
    )


eski_turku_idleri = set(
    egitim["Turku_ID"]
)


# ============================================================
# ACTIVE LEARNING ADAY HAVUZU
# ============================================================

# Eğitimde bir dizesi görülmüş türkünün
# diğer dizeleri active learning'e alınmaz.
havuz = tum_dizeler[
    ~tum_dizeler["Turku_ID"].isin(
        eski_turku_idleri
    )
].copy()

print(
    "Eğitimdeki türküler çıkarıldıktan sonra:",
    len(havuz)
)


# ============================================================
# FINAL HOLDOUT
# ============================================================

yeni_turkuler = (
    havuz["Turku_ID"]
    .drop_duplicates()
    .to_numpy()
)

rng = np.random.default_rng(RANDOM_STATE)
rng.shuffle(yeni_turkuler)


# Veri büyük olduğu için sabit minimum kullanılmıyor.
# Yeni türkülerin %15'i final holdout.
holdout_n = int(
    round(
        len(yeni_turkuler)
        * HOLDOUT_ORANI
    )
)

holdout_idleri = set(
    yeni_turkuler[:holdout_n]
)


holdout = havuz[
    havuz["Turku_ID"].isin(
        holdout_idleri
    )
].copy()


active_pool = havuz[
    ~havuz["Turku_ID"].isin(
        holdout_idleri
    )
].copy()


print("Final holdout türkü:", holdout_n)
print("Final holdout dize:", len(holdout))
print("Active Learning havuzu:", len(active_pool))


holdout.to_excel(
    HOLDOUT_FILE,
    index=False
)


# ============================================================
# BAŞLANGIÇ MODELİNİ EĞİT
# ============================================================

# Final model değil;
# yalnızca active learning örneklerini seçiyor.
model = model_olustur()

model.fit(
    egitim["Dize"].astype(str),
    egitim["Etiket"]
)


# ============================================================
# ACTIVE POOL'U SKORLA
# ============================================================

proba = model.predict_proba(
    active_pool["Dize"].astype(str)
)

classes = model.named_steps["lr"].classes_


for i, sinif in enumerate(classes):

    active_pool[
        f"P_{int(sinif)}"
    ] = proba[:, i]


# Entropy yüksek → model daha kararsız
active_pool["Belirsizlik"] = -np.sum(
    proba * np.log(proba + 1e-12),
    axis=1
)

# Sınıf 2 olasılığı
active_pool["Sinirda_Skoru"] = (
    active_pool["P_2"]
)


# ============================================================
# TUR 0 SEÇİM PLANI
# %60 rastgele + %30 entropy + %10 P2
# ============================================================

N_TOTAL = AL_BATCH_SIZE

N_RANDOM = round(
    N_TOTAL * 0.60
)

N_UNCERTAIN = round(
    N_TOTAL * 0.30
)

N_BOUNDARY = (
    N_TOTAL
    - N_RANDOM
    - N_UNCERTAIN
)


print("\nTur 0 seçim planı:")
print("Rastgele :", N_RANDOM)
print("Belirsiz :", N_UNCERTAIN)
print("Sınırda  :", N_BOUNDARY)


# ============================================================
# TUR 0 SEÇ
# ============================================================

# Bu set üç seçim türünde ortaktır.
# Böylece aynı Turku_ID Tur0'da yalnızca 1 kez seçilir.
kullanilan_turku_idleri = set()


# %60 rastgele
idx_random = cesitli_sec(
    active_pool,
    N_RANDOM,
    kullanilan_turku_idleri,
    random=True
)

secilen = set(idx_random)


# %30 entropy yüksek
kalan = active_pool[
    ~active_pool.index.isin(secilen)
]

idx_uncertain = cesitli_sec(
    kalan,
    N_UNCERTAIN,
    kullanilan_turku_idleri,
    siralama="Belirsizlik",
    ascending=False
)

secilen.update(idx_uncertain)


# %10 P2 yüksek
kalan = active_pool[
    ~active_pool.index.isin(secilen)
]

idx_boundary = cesitli_sec(
    kalan,
    N_BOUNDARY,
    kullanilan_turku_idleri,
    siralama="Sinirda_Skoru",
    ascending=False
)

secilen.update(idx_boundary)


if len(secilen) != N_TOTAL:
    raise ValueError(
        f"{N_TOTAL} yerine {len(secilen)} örnek seçildi."
    )


# ============================================================
# TUR 0
# ============================================================

tur0 = active_pool.loc[
    list(secilen)
].copy()

tur0["Secim_Turu"] = ""

tur0.loc[
    idx_random,
    "Secim_Turu"
] = "Rastgele"

tur0.loc[
    idx_uncertain,
    "Secim_Turu"
] = "Belirsiz"

tur0.loc[
    idx_boundary,
    "Secim_Turu"
] = "Sinirda"


# Model skorları + seçim sebebi
tur0.to_excel(
    SECIM_KAYDI_FILE,
    index=False
)


# ============================================================
# KÖR ETİKETLEME DOSYASI
# ============================================================

tur0 = (
    tur0
    .sample(
        frac=1,
        random_state=RANDOM_STATE
    )
    .reset_index(drop=True)
)

etiketleme = tur0[
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
    ROUND0_FILE,
    index=False
)


# ============================================================
# ACTIVE POOL'U KAYDET
# ============================================================

active_pool.to_excel(
    POOL_FILE,
    index=False
)


print("\nTUR 0 HAZIR")
print("Etiketlenecek örnek:", len(etiketleme))
print("Etiketleme:", ROUND0_FILE)
print("Holdout:", HOLDOUT_FILE)
print("Active Pool:", POOL_FILE)