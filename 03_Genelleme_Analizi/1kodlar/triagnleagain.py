import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# DOSYA YOLLARI
# ============================================================
from pathlib import Path

EXCEL_DIR = Path(__file__).resolve().parent.parent / "2exceller"

INPUT_FILE = EXCEL_DIR / "10_Tum_Tahminler.xlsx"
OUTPUT_FIG = EXCEL_DIR / "10_Final_Corpus_Visualization_EN.png"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Excel bulunamadı: {INPUT_FILE}")
# ============================================================
# VERİYİ OKU
# ============================================================

df = pd.read_excel(INPUT_FILE)

# Gerekli sütunlar
required_cols = ["P1", "P2", "P3"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Eksik sütun: {col}")

p1 = df["P1"].to_numpy()
p2 = df["P2"].to_numpy()
p3 = df["P3"].to_numpy()

# Güvenlik kontrolü
if not np.allclose(p1 + p2 + p3, 1.0, atol=1e-6):
    print("Uyarı: Bazı satırlarda P1+P2+P3 toplamı tam 1 değil.")

# ============================================================
# TERNARY KOORDİNATLAR
# ============================================================

h = np.sqrt(3) / 2
x = p2 + 0.5 * p3
y = h * p3

# ============================================================
# GÖRSEL
# ============================================================

plt.figure(figsize=(9, 8))

hb = plt.hexbin(
    x,
    y,
    gridsize=75,
    bins="log",
    mincnt=1,
    linewidths=0
)

cbar = plt.colorbar(hb)
cbar.set_label("Line density (log scale)")

# Üçgen çerçeve
plt.plot(
    [0, 1, 0.5, 0],
    [0, 0, h, 0],
    linewidth=1.5
)

# Sınıf sınırları (eşitlik çizgileri)
t = np.linspace(0, 0.5, 200)

for a1, a2, a3 in [
    (t, t, 1 - 2*t),       # P1 = P2
    (t, 1 - 2*t, t),       # P1 = P3
    (1 - 2*t, t, t)        # P2 = P3
]:
    plt.plot(
        a2 + 0.5 * a3,
        h * a3,
        linestyle="--",
        linewidth=1
    )

# Köşe etiketleri
plt.text(
    -0.02, -0.04,
    "P1 = 1\nNon-generalization",
    ha="center",
    va="top"
)

plt.text(
    1.02, -0.04,
    "P2 = 1\nUncertain",
    ha="center",
    va="top"
)

plt.text(
    0.5, h + 0.04,
    "P3 = 1\nGeneralization",
    ha="center",
    va="bottom"
)

plt.title("Probability Density of the Full Corpus in the P1–P2–P3 Space")

plt.axis("equal")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    OUTPUT_FIG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("English figure saved to:")
print(OUTPUT_FIG)