import pandas as pd
from ayarlar import EXCEL_DIR, normalize_text, RANDOM_STATE, FINAL_DATA
#DOSYA YOLLARI

GIRDI_DOSYASI = FINAL_DATA #bu dosyadaki  yöre başlık sözler sütununu kullıyor
CIKTI_ETIKETLEME = (
    EXCEL_DIR
    /"01_Baslangic_Etiketleme.xlsx"
)

#dosya kontrolü
if not GIRDI_DOSYASI.exists():
    raise FileNotFoundError(
        f"girdi dosyası bulunamadı:{GIRDI_DOSYASI}"
    )

print("Girdi dosyası bulundu:",GIRDI_DOSYASI)


"""print("\nSatır ve sütun sayısı:")
print(df.shape)
print("sütun adları:")
print(df.columns.tolist())
print("ilk 3 kayıt:")
print(df.head(3))
"""
#külliyatı dizelere ayır

MIN_KELIME=2 

df= pd.read_excel(GIRDI_DOSYASI)
dizeler = []

#Türküleri tek tek gez
for _, row in df.iterrows():
    baslik = row.get("Başlık","")
    yore = row.get("Yöre","")
    sozler = row.get("Sözler","")

    #Na iseyani boşsa atla
    if pd.isna(sozler):
        continue
        #satırlara böl
    for dize in str(sozler).splitlines():
        dize = dize.strip()#bosluk temizle
        #min 2 kelimelik cümle kabul
        if len(dize.split())< MIN_KELIME:
            continue
        #normalize et
        dize_normalize=normalize_text(dize)
       #normalizasyondan sonra dize boş kaldıysa atla
        if not dize_normalize:
            continue
        #bulunan dizeleri listeye ekle
        dizeler.append({
            "Başlık":baslik,
            "Yöre":yore,
            "Dize":dize,
            "Dize_Normalize":dize_normalize
        })

dize_df= pd.DataFrame(dizeler)
#aynı normalize dizeden yalnız ilkini tut
dize_df =(
    dize_df.drop_duplicates(
        subset="Dize_Normalize",
        keep="first"
    ).reset_index(drop=True)#satır numaralarını yeniden olustur
)
print("kullanılabilir benzersiz dize:", len(dize_df))

#Başlangıç örneklem ayarları

TOPLAM_ORNEK=300
RASTGELE_SAYI=150
GENELLEME_ADAY_SAYI=150
AYNI_TURKUDEN_EN_FAZLA_BIR=True

GENELLEYICI_KELIMELER = {
    "her",
    "herkes",
    "kimse",
    "hiçbir",
    "bütün",
    "tüm"
}


DESTEKLEYICI_GENELLEYICILER = {
    "hep",
    "daim",
    "asla"
}


GENEL_GRUPLAR = {
    "insan",
    "adam",
    "kadın",
    "erkek",
    "yiğit",
    "kul",
    "aşık",
    "mert",
    "namert",
    "fakir",
    "zengin"
}


NORMATIF_KELIMELER = {
    "lazım",
    "gerek",
    "gerekir",
    "şart",
    "şarttır",
    "yakışır",
    "yakışmaz"
}
GENEL_YARGI_FIILLERI = {
    "olur",
    "olmaz",

    "gelir",
    "gelmez",

    "gider",
    "gitmez",

    "eder",
    "etmez",

    "bilir",
    "bilmez",

    "kalır",
    "kalmaz",

    "çıkar",
    "çıkmaz",

    "düşer",
    "düşmez",

    "bulunur",
    "bulunmaz",

    "korkar",
    "korkmaz",

    "sever",
    "sevmez",

    "doğar",
    "doğmaz",

    "ölür",
    "ölmez",

    "yanar",
    "yanmaz",

    "söner",
    "sönmez",

    "geçer",
    "geçmez",

    "döner",
    "dönmez",

    "ağlar",
    "ağlamaz",

    "güler",
    "gülmez"
}
#GENELLEM ADAYI MI diye tek tek dizelere bak

def genelleme_adayi_mi(dize_normalize):

    kelime_listesi=dize_normalize.split()
    #kelimelerden set oluştur
    kelimeler =set(kelime_listesi)
    #normatif kelime varsa adat(& kesişim demek)
    if kelimeler & NORMATIF_KELIMELER:
        return True
    if kelimeler & GENELLEYICI_KELIMELER:
        return True
    #meli malı varsa
    if any(k.endswith(("malı", "meli")) for k in kelimeler):
        return True
    #genel grup +genel yargı fiili vatrsa
    if (
        kelimeler & GENEL_GRUPLAR
        and kelimeler & GENEL_YARGI_FIILLERI
    ):
        return True
    # hep / asla gibi ifade + genel yargı fiili
    if (
        kelimeler & DESTEKLEYICI_GENELLEYICILER
        and kelimeler & GENEL_YARGI_FIILLERI
    ):
        return True

    return False
#her dizeyi aday-değil diye işaretle
dize_df["Genelleme_Adayi"] = (
    dize_df["Dize_Normalize"]
    .apply(genelleme_adayi_mi)
)
#karşılaştırmak için başlıkları notmalize et

# Başlıkları karşılaştırmak için normalize et
dize_df["Baslik_Normalize"] = (
    dize_df["Başlık"]
    .fillna("")
    .astype(str)
    .map(normalize_text)
)
#true olanları ayrı dataframe aldık
aday_havuzu = dize_df[
    dize_df["Genelleme_Adayi"]
].copy()

genelleme_adaylari = (
    aday_havuzu
    .sample(frac=1, random_state=RANDOM_STATE)# RANDOM_STATE alıyoruz ki rastgele seçim tekrarlanabilir olsun
    .drop_duplicates("Baslik_Normalize")
    .head(GENELLEME_ADAY_SAYI)
    .copy()
)

genelleme_adaylari["Secim_Turu"] = "Genelleme_Adayi"
# Aday olmayanlardan seçiyoruz.
rastgele_havuzu = dize_df[
    ~dize_df["Genelleme_Adayi"]
].copy()

# Genelleme adaylarında kullanılan türküleri çıkar.
kullanilan_basliklar = set(
    genelleme_adaylari["Baslik_Normalize"]
)

rastgele_havuzu = rastgele_havuzu[
    ~rastgele_havuzu["Baslik_Normalize"].isin(
        kullanilan_basliklar
    )
]

rastgele_ornekler = (
    rastgele_havuzu
    .sample(frac=1, random_state=RANDOM_STATE)
    .drop_duplicates("Baslik_Normalize")
    .head(RASTGELE_SAYI) #işaretlenmiş adaylar listesinden rastgele 150 tane seç
    .copy()
)

rastgele_ornekler["Secim_Turu"] = "Rastgele"


# ============================================================
# İKİ GRUBU BİRLEŞTİR
# ============================================================

baslangic_df = pd.concat(
    [genelleme_adaylari, rastgele_ornekler],
    ignore_index=True
)

# Sıralamadan hangi grubun hangisi anlaşılmasın.
baslangic_df = (
    baslangic_df
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

baslangic_df.insert(
    0,
    "ID",
    range(1, len(baslangic_df) + 1)
)


# ============================================================
# SEÇİM KAYDI
# ============================================================

baslangic_df[
    [
        "ID",
        "Başlık",
        "Yöre",
        "Dize",
        "Secim_Turu"
    ]
].to_excel(
    CIKTI_ETIKETLEME,
    index=False
)


# ============================================================
# ETİKETLENECEK DOSYA
# ============================================================

etiketleme = baslangic_df[
    [
        "ID",
        "Başlık",
        "Yöre",
        "Dize",
        "Secim_Turu"
    ]
].copy()

etiketleme["Etiket"] = ""
etiketleme["Not"] = ""

etiketleme.to_excel(
    CIKTI_ETIKETLEME,
    index=False
)


print("Genelleme adayı:", len(genelleme_adaylari))
print("Rastgele örnek:", len(rastgele_ornekler))
print("Toplam:", len(etiketleme))
print("Tüm külliyattaki aday sayısı:", len(aday_havuzu))