from pathlib import Path
import re
import numpy as np
import pandas as pd

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# AYARLAR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT = BASE_DIR / "2exceller" / "10_Nihai_Genelleme_Adaylari.xlsx"

OUTPUT = (
    BASE_DIR
    / "3sonucÜzerineCalismalar"
    / "01_Semantik_Benzerlik_Sonuclari.xlsx"
)

GENELLEME_ESIGI = 0.80
BENZERLIK_ESIGI = 0.85

NIDALAR = {
    "aman", "hey", "ey", "of",
    "ah", "vah", "vay", "vıy", "oy"
}


# ============================================================
# TEMİZLEME
# ============================================================

def normalize(text):
    """Benzerlik hesabı için dizeyi sadeleştirir."""
    text = str(text).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^\wçğıöşüâîû\s']", " ", text)

    return " ".join(
        k for k in text.split()
        if k not in NIDALAR
    )


def temiz_baslik(text):
    """
    Site kaynaklı başlık sonu numaralarını kaldırır.

    Çocuğa Verilen Gelin 13
    -> Çocuğa Verilen Gelin
    """
    text = str(text).strip()
    text = re.sub(r"\s+\d+\s*$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def olumsuz_mu(text):
    """Olumlu ve olumsuz ifadelerin aynı gruba girmesini engeller."""
    kelimeler = normalize(text).split()

    return (
        "değil" in kelimeler
        or "yok" in kelimeler
        or any(re.search(r"(maz|mez)$", k) for k in kelimeler)
    )


# ============================================================
# VERİ
# ============================================================

df = pd.read_excel(INPUT, sheet_name="Genelleme_Adaylari")

df = df[
    df["Genelleme_Olasiligi"] >= GENELLEME_ESIGI
].copy()

df["Norm"] = df["Dize"].fillna("").map(normalize)
df["Temiz_Baslik"] = df["Başlık"].fillna("").map(temiz_baslik)

df = df[
    (df["Norm"] != "")
    & (df["Temiz_Baslik"] != "")
].reset_index(drop=True)

print("Yüksek skorlu kayıt:", len(df))


# ============================================================
# BENZERSİZ İFADELER
# ============================================================

tekil = (
    df[["Norm", "Dize"]]
    .drop_duplicates("Norm")
    .reset_index(drop=True)
)


# ============================================================
# WORD + CHAR TF-IDF
# ============================================================

vectorizer = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )
    ),
    (
        "char",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True
        )
    )
])

X = vectorizer.fit_transform(tekil["Norm"])


# ============================================================
# COSINE >= 0.85 OLANLARI GRUPLA
# ============================================================

nn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

nn.fit(X)

distances, indices = nn.radius_neighbors(
    X,
    radius=1 - BENZERLIK_ESIGI
)


# Union-Find
parent = list(range(len(tekil)))


def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]


def union(a, b):
    a, b = find(a), find(b)

    if a != b:
        parent[b] = a


for i in range(len(tekil)):

    for distance, j in zip(distances[i], indices[i]):

        if i == j:
            continue

        similarity = 1 - distance

        if similarity < BENZERLIK_ESIGI:
            continue

        # "olur" ile "olmaz" gibi zıtları eşleştirme
        if olumsuz_mu(tekil.loc[i, "Dize"]) != olumsuz_mu(
            tekil.loc[j, "Dize"]
        ):
            continue

        union(i, j)


# ============================================================
# GRUPLARI ORİJİNAL KAYITLARA BAĞLA
# ============================================================

norm_index = {
    norm: i
    for i, norm in enumerate(tekil["Norm"])
}

df["Tekil_Index"] = df["Norm"].map(norm_index)
df["Root"] = df["Tekil_Index"].map(find)


# ============================================================
# AYNI GRUPTA AYNI TÜRKÜYÜ 1 KEZ SAY
#
# Ör:
# Çocuğa Verilen Gelin 3
# Çocuğa Verilen Gelin 8
# Çocuğa Verilen Gelin 13
#
# -> aynı eser kabul edilir.
#
# Aynı eserden en yüksek model skorlu satır bırakılır.
# ============================================================

once = len(df)

df = (
    df.sort_values(
        "Genelleme_Olasiligi",
        ascending=False
    )
    .drop_duplicates(
        subset=["Root", "Temiz_Baslik"],
        keep="first"
    )
    .copy()
)

print("Mükerrer eser olarak elenen:", once - len(df))


# ============================================================
# EN AZ 2 FARKLI TÜRKÜ İÇEREN GRUPLARI TUT
# ============================================================

grup_sayilari = df.groupby("Root").size()

gecerli = grup_sayilari[
    grup_sayilari >= 2
].index

df = df[
    df["Root"].isin(gecerli)
].copy()


# ============================================================
# GRUP ID
# En fazla farklı türkü içeren grup Grup_1
# ============================================================

sirali_rootlar = (
    df.groupby("Root")
    .size()
    .sort_values(ascending=False)
    .index
)

root_to_group = {
    root: f"Grup_{i}"
    for i, root in enumerate(sirali_rootlar, 1)
}

df["Grup_ID"] = df["Root"].map(root_to_group)


# ============================================================
# BENZERLİK ORANI
#
# Artık yalnızca KALAN farklı türküler arasında hesaplanır.
# Böylece silinen mükerrer kayıtlar oranı şişirmez.
# ============================================================

df["Benzerlik_Oranı"] = 0.0

for grup, g in df.groupby("Grup_ID"):

    satirlar = g.index.tolist()
    x_idx = g["Tekil_Index"].astype(int).tolist()

    sim = cosine_similarity(X[x_idx])

    for k, satir in enumerate(satirlar):

        # Kendisine benzerliği dikkate alma
        sim[k, k] = -1

        df.loc[satir, "Benzerlik_Oranı"] = sim[k].max()


# ============================================================
# SONUÇ
# ============================================================

sonuc = df[
    [
        "Grup_ID",
        "Dize",
        "Temiz_Baslik",
        "Yöre",
        "Genelleme_Olasiligi",
        "Benzerlik_Oranı"
    ]
].copy()

sonuc.columns = [
    "Grup_ID",
    "Cümle",
    "Başlık",
    "Yöre",
    "Genelleme_Skoru",
    "Benzerlik_Oranı"
]


# Grup_1, Grup_2... şeklinde sırala
sonuc["_No"] = (
    sonuc["Grup_ID"]
    .str.extract(r"(\d+)")[0]
    .astype(int)
)

sonuc = (
    sonuc
    .sort_values(
        ["_No", "Benzerlik_Oranı", "Genelleme_Skoru"],
        ascending=[True, False, False]
    )
    .drop(columns="_No")
    .reset_index(drop=True)
)


# ============================================================
# EXCEL
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

sonuc.to_excel(
    OUTPUT,
    index=False,
    sheet_name="Genelleme_Gruplari"
)


print("\n" + "=" * 60)
print("TAMAMLANDI")
print("=" * 60)

print("Son grup sayısı :", sonuc["Grup_ID"].nunique())
print("Farklı türkü kaydı:", len(sonuc))
print("Çıktı:", OUTPUT)