from pathlib import Path
import re


# ============================================================
# DOSYA YOLLARI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXCEL_DIR = BASE_DIR / "2exceller"

FINAL_DATA = (
    PROJECT_ROOT
    / "01_VeriHazirlama"
    / "2_veri"
    / "03_Nihai_Veri"
    / "temiz_veri_lemmatize_FINAL.xlsx"
)


# ============================================================
# GENEL AYARLAR
# ============================================================

RANDOM_STATE = 42

# Her aktif öğrenme turunda kaç yeni dize etiketleyeceğiz?
AL_BATCH_SIZE = 150


# ============================================================
# METİN NORMALİZASYONU
# ============================================================

def normalize_text(text):
    """
    Dize karşılaştırması ve duplicate kontrolü için.
    """
    if text is None:
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r"[^\wçğıöşüâîû\s']",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()