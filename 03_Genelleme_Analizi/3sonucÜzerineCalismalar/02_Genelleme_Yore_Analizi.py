from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# DOSYA YOLLARI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ANALIZ_DIR = BASE_DIR / "3sonucÜzerineCalismalar"

INPUT = (
    ANALIZ_DIR
    / "01_Semantik_Benzerlik_Sonuclari.xlsx"
)

OUTPUT_AILE = (
    ANALIZ_DIR
    / "02_Genelleme_Aileleri.xlsx"
)

OUTPUT_MATRIS = (
    ANALIZ_DIR
    / "03_Grup_Yore_Matrisi.xlsx"
)

OUTPUT_YORE = (
    ANALIZ_DIR
    / "04_Yore_Toplamlari.xlsx"
)

OUTPUT_PIE_DAHIL = (
    ANALIZ_DIR
    / "05_Yore_Dagilimi_Belirtilmemis_Dahil.png"
)

OUTPUT_PIE_BILINEN = (
    ANALIZ_DIR
    / "06_Yore_Dagilimi_Sadece_Bilinen.png"
)

TOP_YORE = 8


# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_excel(
    INPUT,
    sheet_name="Genelleme_Gruplari"
)

# Yöre temizliği
df["Yöre"] = (
    df["Yöre"]
    .fillna("Belirtilmemiş")
    .astype(str)
    .str.strip()
)

df.loc[
    df["Yöre"].isin(["", "nan", "None"]),
    "Yöre"
] = "Belirtilmemiş"


print("=" * 70)
print("VERİ OKUNDU")
print("=" * 70)
print("Toplam kayıt:", len(df))
print("Grup sayısı:", df["Grup_ID"].nunique())
print("Yöre sayısı:", df["Yöre"].nunique())


# ============================================================
# HER GRUBUN İLK CÜMLESİ
# ============================================================

ilk_cumle = (
    df.groupby(
        "Grup_ID",
        sort=False
    )["Cümle"]
    .first()
    .to_dict()
)


# ============================================================
# 1. GENELLEME AİLELERİ
# ============================================================

aile_rows = []

for grup, g in df.groupby(
    "Grup_ID",
    sort=False
):

    yore_sayilari = (
        g["Yöre"]
        .value_counts()
    )

    # Belirtilmemiş gerçek bir yöre olmadığı için
    # farklı yöre sayısına katılmıyor
    farkli_yore_sayisi = (
        g.loc[
            g["Yöre"] != "Belirtilmemiş",
            "Yöre"
        ]
        .nunique()
    )

    # BÜTÜN yöreler yazılır, ... YOK
    yore_dagilimi = ", ".join(
        f"{yore} ({sayi})"
        for yore, sayi
        in yore_sayilari.items()
    )

    aile_rows.append({
        "Grup_ID":
            grup,

        "Genelleme_Ailesi":
            ilk_cumle[grup],

        "Toplam_Kullanım":
            len(g),

        "Farklı_Yöre_Sayısı":
            farkli_yore_sayisi,

        "Yöre_Dağılımı":
            yore_dagilimi
    })


aile_ozet = pd.DataFrame(aile_rows)

aile_ozet = (
    aile_ozet
    .sort_values(
        "Toplam_Kullanım",
        ascending=False
    )
    .reset_index(drop=True)
)


aile_ozet.to_excel(
    OUTPUT_AILE,
    index=False
)


# ============================================================
# 2. GRUP × YÖRE MATRİSİ
# ============================================================

matris = pd.crosstab(
    index=df["Grup_ID"],
    columns=df["Yöre"]
)

# ------------------------------------------------------------
# Yöreleri toplam kullanım sıklığına göre sırala
# ------------------------------------------------------------

yore_sirasi = (
    matris
    .sum(axis=0)
    .sort_values(ascending=False)
    .index
)

matris = matris[
    yore_sirasi
]


# ------------------------------------------------------------
# Genelleme ailesi sütununu ekle
# ------------------------------------------------------------

matris.insert(
    0,
    "Genelleme_Ailesi",
    [
        ilk_cumle[grup]
        for grup in matris.index
    ]
)

matris = matris.reset_index()


# ------------------------------------------------------------
# Grup_1, Grup_2, Grup_3 ... şeklinde düzgün sırala
# ------------------------------------------------------------

matris["_Grup_No"] = (
    matris["Grup_ID"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .fillna(999999)
    .astype(int)
)

matris = (
    matris
    .sort_values("_Grup_No")
    .drop(columns="_Grup_No")
    .reset_index(drop=True)
)


# 0'LAR AYNEN KALIYOR
matris.to_excel(
    OUTPUT_MATRIS,
    index=False
)


# ============================================================
# 3. YÖRE TOPLAMLARI
# ============================================================

yore_ozet = (
    df["Yöre"]
    .value_counts()
    .rename_axis("Yöre")
    .reset_index(name="Toplam_Kullanım")
)


genel_toplam = (
    yore_ozet["Toplam_Kullanım"]
    .sum()
)

yore_ozet["Genel_Pay_%"] = (
    yore_ozet["Toplam_Kullanım"]
    / genel_toplam
    * 100
)


# ------------------------------------------------------------
# Sadece yöresi bilinen kayıtlar içindeki pay
# ------------------------------------------------------------

bilinen_toplam = (
    yore_ozet.loc[
        yore_ozet["Yöre"] != "Belirtilmemiş",
        "Toplam_Kullanım"
    ]
    .sum()
)


def bilinen_pay(row):

    if row["Yöre"] == "Belirtilmemiş":
        return None

    return (
        row["Toplam_Kullanım"]
        / bilinen_toplam
        * 100
    )


yore_ozet[
    "Bilinen_Yöreler_İçindeki_Pay_%"
] = yore_ozet.apply(
    bilinen_pay,
    axis=1
)


yore_ozet.to_excel(
    OUTPUT_YORE,
    index=False
)


# ============================================================
# PIE CHART İÇİN ORTAK VERİ
# ============================================================

belirtilmemis_sayi = (
    yore_ozet.loc[
        yore_ozet["Yöre"] == "Belirtilmemiş",
        "Toplam_Kullanım"
    ]
    .sum()
)

bilinen_yoreler = (
    yore_ozet[
        yore_ozet["Yöre"] != "Belirtilmemiş"
    ]
    .sort_values(
        "Toplam_Kullanım",
        ascending=False
    )
    .reset_index(drop=True)
)


# En sık 8 gerçek yöre
top8 = (
    bilinen_yoreler
    .head(TOP_YORE)
    .copy()
)

# Geri kalan gerçek yörelerin toplamı
diger_yoreler_sayi = (
    bilinen_yoreler
    .iloc[TOP_YORE:]["Toplam_Kullanım"]
    .sum()
)


# ============================================================
# 4. PIE CHART
# BELİRTİLMEMİŞ DAHİL
# ============================================================

pie_dahil_rows = []

# İlk 8 yöre
for _, row in top8.iterrows():

    pie_dahil_rows.append({
        "Kategori":
            row["Yöre"],

        "Sayı":
            row["Toplam_Kullanım"]
    })


# Diğer yöreler
if diger_yoreler_sayi > 0:

    pie_dahil_rows.append({
        "Kategori":
            "Diğer Yöreler",

        "Sayı":
            diger_yoreler_sayi
    })


# Belirtilmemiş
if belirtilmemis_sayi > 0:

    pie_dahil_rows.append({
        "Kategori":
            "Belirtilmemiş",

        "Sayı":
            belirtilmemis_sayi
    })


pie_dahil = pd.DataFrame(
    pie_dahil_rows
)


fig, ax = plt.subplots(
    figsize=(10, 9)
)

ax.pie(
    pie_dahil["Sayı"],
    labels=pie_dahil["Kategori"],
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title(
    "Tekrarlanan Genelleme Kullanımlarının "
    "Yörelere Göre Dağılımı\n"
    "Yöresi belirtilmemiş kayıtlar dahil edilmiştir.",
    pad=20
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PIE_DAHIL,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. PIE CHART
# SADECE YÖRESİ BİLİNENLER
# ============================================================

pie_bilinen_rows = []

# İlk 8 yöre
for _, row in top8.iterrows():

    pie_bilinen_rows.append({
        "Kategori":
            row["Yöre"],

        "Sayı":
            row["Toplam_Kullanım"]
    })


# Diğer gerçek yöreler
if diger_yoreler_sayi > 0:

    pie_bilinen_rows.append({
        "Kategori":
            "Diğer Yöreler",

        "Sayı":
            diger_yoreler_sayi
    })


pie_bilinen = pd.DataFrame(
    pie_bilinen_rows
)


fig, ax = plt.subplots(
    figsize=(10, 9)
)

ax.pie(
    pie_bilinen["Sayı"],
    labels=pie_bilinen["Kategori"],
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title(
    "Yöresi Belirtilmiş Tekrarlanan "
    "Genelleme Kullanımlarının Dağılımı\n"
    "Yöresi belirtilmemiş kayıtlar analiz dışında bırakılmıştır.",
    pad=20
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PIE_BILINEN,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SONUÇ
# ============================================================

print("\n" + "=" * 70)
print("ANALİZ TAMAMLANDI")
print("=" * 70)

print("\n1) Genelleme aileleri:")
print(OUTPUT_AILE)

print("\n2) Grup × Yöre matrisi:")
print(OUTPUT_MATRIS)

print("\n3) Yöre toplamları:")
print(OUTPUT_YORE)

print("\n4) Belirtilmemiş dahil pie chart:")
print(OUTPUT_PIE_DAHIL)

print("\n5) Sadece bilinen yöreler pie chart:")
print(OUTPUT_PIE_BILINEN)

print("\nGenelleme grubu sayısı:", df["Grup_ID"].nunique())
print("Toplam kullanım:", len(df))
print("Yöresi bilinen kullanım:", bilinen_toplam)
print("Belirtilmemiş kullanım:", belirtilmemis_sayi)