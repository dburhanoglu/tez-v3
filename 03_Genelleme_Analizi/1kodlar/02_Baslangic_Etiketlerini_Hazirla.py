import pandas as pd
#başlangıç etiketli verisini hazırlama ve model karşılaştırma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion#feauture featureları birleştiren, pipline feature ile birleşmiş özellikleri modele verip tahmin alan
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from ayarlar import EXCEL_DIR,RANDOM_STATE
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.metrics import(
    accuracy_score,#tüm doğru tahminler/tüm örnek
    balanced_accuracy_score,#bu her sınıfın h
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,# modelin sınıflandırmada ne kadar iyi olduğunu ölçer
    #özellikle sınıflar dengesizse
    confusion_matrix#hangi sınıf hangisi ile karışıyor gösterir
)
GIRDI_DOSYASI = (
    EXCEL_DIR
    /"01_Baslangic_Etiketleme.xlsx"
)
RAPOR_DOSYASI =(
    EXCEL_DIR
    /"02_Baslangic_Model_Kiyaslama_Raporu.xlsx"
)

#DOSYA VAR MI KONTROL ET
if not GIRDI_DOSYASI.exists():
    raise FileNotFoundError(
        f"Girdi dosyası bulunamadı:{GIRDI_DOSYASI}"
    )
print("Girdi dosyası bulundu:",GIRDI_DOSYASI)
#veri oku ve kontrol et
df = pd.read_excel(GIRDI_DOSYASI)
df["Etiket"] = pd.to_numeric(
    df["Etiket"],
    errors="coerce"  # yanlış/boş değer varsa NaN yap
)

print("Toplam veri:", len(df))

print("\nEtiket dağılımı:")
print(
    df["Etiket"]
    .value_counts(dropna=False)#nanları da göster
    .sort_index()
)
# 1 2 3 etiketleri kullan sadece
df = df[df["Etiket"].isin([1,2,3])].copy()
df["Etiket"]=df["Etiket"].astype(int)


print("kullanılabilir veri:",len(df))
print("Tekrarlı dize:",df["Dize"].duplicated().sum())

x=df["Dize"].fillna("").astype(str)
y= df["Etiket"]

# Stratified 5-Fold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,#bölmeden önce karıştır
    random_state=RANDOM_STATE
)
#WORD VE WORD+CHAR
def word_tfidf():
    return TfidfVectorizer(
        ngram_range=(1,2),#1 ve 2 kelimelik gruplara bak 
        min_df=2,#tüm sözler külliyatında en az 2 kez görülen yapıyı al
        max_features=None,
        sublinear_tf=True#sık tekrarlı kelime etkisini azaltor
    )
def word_char_tfidf():
    return FeatureUnion([
        ("word",word_tfidf()),
        ("char",TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2,4),
            min_df=2,
            sublinear_tf =True
        ))
                ])
modeller = {
    #word özellikleri+lr
    "Word + LR": Pipeline([
        ("tfidf", word_tfidf()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ]),
#word özellikleri +linear svm
    "Word + SVM": Pipeline([
        ("tfidf", word_tfidf()),
        ("model", LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),
#word + cahr özellikleri +LR
    "Word+Char + LR": Pipeline([
        ("tfidf", word_char_tfidf()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ]),
#WORD +CHAR +LİNEAR svm
    "Word+Char + SVM": Pipeline([
        ("tfidf", word_char_tfidf()),
        ("model", LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
}

sonuclar = []
confusionlar = []
for isim, model in modeller.items():
    tahmin = cross_val_predict(
        model,
        x,
        y,
        cv=cv
    )
    sonuclar.append({
        "Model": isim,
        "Accuracy": accuracy_score(y,tahmin),
        "balanced accuracy": balanced_accuracy_score(y,tahmin),
        "Macro precision": precision_score(
            y,tahmin,average="macro",zero_division=0        
            ),
        "Macro recall":recall_score(
            y,tahmin,average="macro",zero_division=0
        ),
          "Macro F1": f1_score(
            y, tahmin, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y, tahmin)

    })
    cm = confusion_matrix(
        y,
        tahmin,
        labels=[1,2,3]
    )
    confusionlar.append(
        pd.DataFrame(
            cm,
            index=["Gerçek 1", "Gerçek 2", "Gerçek 3"],
            columns=["Tahmin 1", "Tahmin 2", "Tahmin 3"]
        ).assign(Model=isim)
    )
sonuc_df = pd.DataFrame(sonuclar)

print("\nMODEL SONUÇLARI")
print(
    sonuc_df
    .round(3)
    .to_string(index=False)
)

confusion_df = pd.concat(
    confusionlar
)

with pd.ExcelWriter(RAPOR_DOSYASI) as writer:

    sonuc_df.to_excel(
        writer,
        sheet_name="Model_Kiyaslama",
        index=False
    )

confusion_df.to_excel(
        writer,
        sheet_name="Confusion_Matrix"
    )

print("\nRapor oluşturuldu:")
print(RAPOR_DOSYASI)

