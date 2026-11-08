from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Put the script beside these two Excel files.
folder = Path(__file__).resolve().parent.parent / "2exceller"
final_file = folder / "09_Nihai_Model_Sonuclari.xlsx"
history_file = folder / "04_Performans_Gecmisi.xlsx"
out = folder / "English_Figures"
out.mkdir(exist_ok=True)

summary_df = pd.read_excel(final_file, sheet_name="Genel_Ozet")
summary = dict(zip(summary_df.iloc[:, 0], summary_df.iloc[:, 1]))
confusion = pd.read_excel(final_file, sheet_name="3Sinif_Matris", index_col=0).to_numpy(dtype=int)
history = pd.read_excel(history_file)

def save(name):
    plt.tight_layout()
    plt.savefig(out / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved:", out / name)

# 1: Final model metrics
metric_keys = ["Accuracy", "Balanced Accuracy", "Macro F1", "MCC",
               "F1 Sınıf 1", "F1 Sınıf 2", "F1 Sınıf 3"]
labels = ["Accuracy", "Balanced\nAccuracy", "Macro F1", "MCC",
          "F1: Class 1", "F1: Class 2", "F1: Class 3"]
values = [float(summary[k]) for k in metric_keys]
colors = ["#2563eb", "#06b6d4", "#8b5cf6", "#64748b",
          "#16a34a", "#f59e0b", "#e11d48"]
plt.figure(figsize=(11, 5.5))
bars = plt.bar(labels, values, color=colors)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Final Model Performance (900 Labeled Lines)")
plt.grid(axis="y", alpha=0.2)
for bar, value in zip(bars, values):
    plt.text(bar.get_x()+bar.get_width()/2, value+0.02,
             f"{value:.3f}", ha="center")
save("01_Final_Model_Performance_EN.png")

# 2-3: Confusion matrices (counts and row percentages)
classes = ["Not a\ngeneralization", "Uncertain", "Generalization"]
for normalized, filename in [
    (False, "02_Confusion_Matrix_Counts_EN.png"),
    (True, "03_Confusion_Matrix_Percentages_EN.png"),
]:
    matrix = (100 * confusion / confusion.sum(axis=1, keepdims=True)
              if normalized else confusion)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    img = ax.imshow(matrix, cmap="YlOrRd" if normalized else "Blues")
    ax.set_xticks(range(3), classes)
    ax.set_yticks(range(3), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Normalized Confusion Matrix" if normalized
                 else "Three-Class Confusion Matrix")
    for i in range(3):
        for j in range(3):
            label = f"{matrix[i,j]:.1f}%" if normalized else str(int(matrix[i,j]))
            ax.text(j, i, label, ha="center", va="center",
                    color="white" if matrix[i,j] > matrix.max()*0.5 else "black",
                    fontweight="bold")
    fig.colorbar(img, ax=ax)
    save(filename)

# 4: Active-learning history
plt.figure(figsize=(9, 5.5))
for column, label, color in [
    ("Accuracy", "Accuracy", "#2563eb"),
    ("Balanced_Accuracy", "Balanced accuracy", "#e11d48"),
    ("Macro_F1", "Macro F1", "#059669"),
]:
    plt.plot(history["N"], history[column], marker="o", linewidth=2.5,
             label=label, color=color)
plt.xticks(history["N"])
plt.xlabel("Number of manually labeled lines")
plt.ylabel("Cross-validation score")
plt.title("Performance Across Active Learning Rounds")
plt.grid(alpha=0.2)
plt.legend()
save("04_Active_Learning_History_EN.png")
print("Done. Excel files were not changed.")
