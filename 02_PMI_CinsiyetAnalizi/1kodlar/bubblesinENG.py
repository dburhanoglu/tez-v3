from pathlib import Path
import argparse
import hashlib
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. AYARLAR VE DOSYA YOLLARI
# ============================================================

RELEASE_ID = "bubblesinENG-2026-11-05-SAME-METHOD-V2"

# AYNI: Dört POS konfigürasyonu
MODES = [
    (True, True),
    (True, False),
    (False, True),
    (False, False)
]

PROJECT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS = (
    PROJECT / "02_PMI_CinsiyetAnalizi" / "2_sonuclar"
)

# DEĞİŞEN: Görsel boyutları
FIG_SIZE = (26, 14)
WORD_FONT_SIZE = 17
TITLE_FONT_SIZE = 21
AXIS_FONT_SIZE = 18
TICK_FONT_SIZE = 16
DPI = 400

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "savefig.facecolor": "white"
})


# ============================================================
# 2. POS KONFİGÜRASYONU
# ============================================================

def mode_name(seed_pos, context_pos):

    a = "ACIK" if seed_pos else "KAPALI"
    b = "AYRI" if context_pos else "BIRLESIK"

    return f"SEED_POS_{a}_BAGLAM_POS_{b}"


# ============================================================
# 3. KELİME ETİKETLERİ
# ============================================================

def label(row):

    pos = row.get("POS", "")

    return str(row["Kelime"]) + (
        f"_{pos}" if pd.notna(pos) and pos else ""
    )


# ============================================================
# 4. KELİME SEÇİMİ - AYNI
# ============================================================

def select_words(table, top_n=50):

    data = table.dropna(subset=["PMI_Farki"]).copy()

    data["Etiket"] = data.apply(label, axis=1)

    positive = data[data.PMI_Farki > 0].sort_values(
        ["PMI_Farki", "Etiket"],
        ascending=[False, True]
    ).head(top_n)

    negative = data[data.PMI_Farki < 0].sort_values(
        ["PMI_Farki", "Etiket"],
        ascending=[True, True]
    ).head(top_n)

    return pd.concat(
        [positive, negative],
        ignore_index=True
    )


# ============================================================
# 5. DİKEY KAYDIRMA - AYNI
# ============================================================

def vertical_jitter(word):

    number = int.from_bytes(
        hashlib.sha256(
            word.encode("utf-8")
        ).digest()[:4],
        "big"
    )

    return (number / (2**32 - 1) - .5) * .28


# ============================================================
# 6. İŞLENEN TÜRKÜ SAYISINI KONTROL ET
# DEĞİŞEN: Yeni kontrol
# ============================================================

def metadata_count(excel_path):

    book = pd.ExcelFile(
        excel_path,
        engine="openpyxl"
    )

    if "Deney_Bilgisi" not in book.sheet_names:
        return None

    info = pd.read_excel(
        book,
        sheet_name="Deney_Bilgisi"
    )

    if not {"Parametre", "Deger"}.issubset(info.columns):
        return None

    row = info.loc[
        info["Parametre"].astype(str) == "Islenen_Turku",
        "Deger"
    ]

    if len(row) and pd.notna(row.iloc[0]):
        return int(row.iloc[0])

    return None


# ============================================================
# 7. GRAFİK ÇİZİMİ
# ============================================================

def draw_chart(
    table,
    figure_number,
    sp,
    cp,
    output_dir,
    xlim,
    min_frequency,
    max_frequency,
    top_n
):

    # DEĞİŞEN: Daha büyük grafik
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # AYNI: Sıfır çizgisi
    ax.axvline(
        0,
        color="black",
        lw=1.25,
        linestyle="--"
    )

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-.28, .28)   
    ax.set_yticks([])

    # DEĞİŞEN: İngilizce eksen
    ax.set_xlabel(
        "← Stronger male association (negative)"
        "   |   PMI difference (0 = neutral)   |   "
        "Stronger female association (positive) →",
        fontsize=AXIS_FONT_SIZE,
        labelpad=16
    )

    # DEĞİŞEN: İngilizce başlık
    ax.set_title(
        "Gender-Associated Words in Turkish Folk Song Lyrics\n"
        f"Seed POS: {'ON' if sp else 'OFF'}"
        "  |  "
        f"Context POS: {'ON' if cp else 'OFF'}",
        fontsize=TITLE_FONT_SIZE,
        weight="bold",
        pad=12
    )

    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)

    ax.tick_params(
        axis="x",
        labelsize=TICK_FONT_SIZE
    )

    if table.empty:

        ax.text(
            .5,
            .5,
            "No comparable words met the thresholds in this dataset.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=15
        )

    else:

        # ================================================
        # AYNI: BALON BÜYÜKLÜĞÜ HESABI
        # ================================================

        sizes = []

        for frequency in table[
            "Toplam_Grup_Kelime_Eslesmesi"
        ]:

            proportion = (
                0.5
                if max_frequency == min_frequency
                else (
                    (frequency - min_frequency)
                    / (max_frequency - min_frequency)
                )
            )

            sizes.append(110 + 520 * proportion)

        # AYNI: Dikey kaydırma
        ys = [
            vertical_jitter(w)
            for w in table["Etiket"]
        ]

        # AYNI: PMI değerleri ve balon renkleri
        ax.scatter(
            table["PMI_Farki"],
            ys,
            s=sizes,
            c=table["PMI_Farki"],
            cmap="coolwarm",
            norm=colors.Normalize(
                vmin=-xlim,
                vmax=xlim
            ),
            alpha=.72,
            edgecolors="grey",
            linewidths=.4,
            zorder=3
        )

        # ================================================
        # DEĞİŞEN: DAHA BÜYÜK KELİME YAZILARI
        # ================================================

        texts = []

        for x, y, word in zip(
            table["PMI_Farki"],
            ys,
            table["Etiket"]
        ):

            txt = ax.text(
                x,
                y,
                word,
                fontsize=WORD_FONT_SIZE,
                ha="center",
                va="center",
                color="#151515",
                zorder=4
            )

            # DEĞİŞEN: Yazının çevresine beyaz kontur
            txt.set_path_effects([
                path_effects.withStroke(
                    linewidth=2,
                    foreground="white"
                )
            ])

            texts.append(txt)

        # DEĞİŞEN: Etiket çakışmalarını azaltma
        try:
            from adjustText import adjust_text

            adjust_text(
                texts,
                ax=ax,
                time_lim=45,
                expand=(1.40, 1.60),
                arrowprops={
                    "arrowstyle": "-",
                    "color": "gray",
                    "lw": 0.45,
                    "alpha": 0.65
                }
            )

        except ImportError:

            print(
                "[WARN] Etiketler icin: pip install adjustText",
                flush=True
            )

    fig.subplots_adjust(
        left=.04,
        right=.985,
        top=.88,
        bottom=.11
    )

    suffix = f"Figure_{figure_number}_ENG_OriginalMethod"

    target = output_dir / f"{suffix}.png"

    fig.savefig(
        target,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.08
    )

    plt.close(fig)

    print(
        f"[OK] Figure {figure_number}"
        f" | Seed POS={'ON' if sp else 'OFF'}"
        f" | Context POS={'ON' if cp else 'OFF'}"
        f" | {len(table)} bubbles",
        flush=True
    )


# ============================================================
# 8. ANA FONKSİYON
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Four English PMI bubble charts"
    )

    parser.add_argument(
        "--sonuclar",
        type=Path,
        default=DEFAULT_RESULTS
    )

    parser.add_argument(
        "--seed",
        default="SEED_1"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=50
    )

    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1")

    folder = args.sonuclar / args.seed

    # DEĞİŞEN: Ayrı çıktı klasörü
    output_dir = folder / (
        "ENG_Bubble_PNG_FINAL"
        if args.top == 50
        else f"ENG_Bubble_OriginalMethod_Top{args.top}"
    )

    print(f"[START] {RELEASE_ID}", flush=True)
    print(f"[SOURCE] {folder}", flush=True)
    print(f"[DEST] {output_dir}", flush=True)

    # AYNI: Dört PMI sonuç Excel'i
    inputs = [
        folder / f"PMI_{mode_name(sp, cp)}.xlsx"
        for sp, cp in MODES
    ]

    missing = [
        str(p)
        for p in inputs
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Dort PMI Excel dosyasi gerekli:\n"
            + "\n".join(missing)
        )

    tables = [
        pd.read_excel(
            path,
            sheet_name="PMI_Sonuclari",
            engine="openpyxl"
        )
        for path in inputs
    ]

    needed = {
        "Kelime",
        "POS",
        "PMI_Farki",
        "Toplam_Grup_Kelime_Eslesmesi"
    }

    for path, table in zip(inputs, tables):

        diff = needed - set(table.columns)

        if diff:
            raise ValueError(
                f"{path.name}: {sorted(diff)}"
            )

    # DEĞİŞEN: Veri kümesi kontrolü
    song_counts = [
        metadata_count(path)
        for path in inputs
    ]

    for path, n in zip(inputs, song_counts):

        print(
            f"[INPUT] {path.name}"
            f" | Islenen_Turku = "
            f"{n if n is not None else 'unknown'}",
            flush=True
        )

    known = {
        n for n in song_counts
        if n is not None
    }

    if len(known) > 1:
        raise ValueError(
            f"Dort Excel'in turku sayisi farkli: {song_counts}"
        )

    if known and min(known) <= 1000:
        print(
            "[UYARI] Analiz 1000 veya daha az turku iceriyor.",
            flush=True
        )

    # AYNI: Kelime seçimi
    selections = [
        select_words(table, args.top)
        for table in tables
    ]

    combined = pd.concat(
        selections,
        ignore_index=True
    )

    # AYNI: Dört grafik için ortak eksen
    xlim = (
        max(
            1.,
            float(combined["PMI_Farki"].abs().max()) * 1.10
        )
        if len(combined)
        else 1.
    )

    # AYNI: Ortak balon büyüklüğü ölçeği
    min_freq = (
        float(
            combined["Toplam_Grup_Kelime_Eslesmesi"].min()
        )
        if len(combined)
        else 0.
    )

    max_freq = (
        float(
            combined["Toplam_Grup_Kelime_Eslesmesi"].max()
        )
        if len(combined)
        else 1.
    )

    print(
        f"[SCALES] X = ±{xlim:.5g}"
        f"; bubble-frequency = "
        f"{min_freq:.0f}..{max_freq:.0f}",
        flush=True
    )

    # DEĞİŞEN: Var olan grafikleri koru
    targets = [
    output_dir / f"Figure_{i}_ENG_OriginalMethod.png"
    for i in range(1, 5)
]
    existing = [
        str(p)
        for p in targets
        if p.exists()
    ]

    if existing:
        raise FileExistsError(
            "Mevcut grafiklerin uzerine yazilmadi:\n"
            + "\n".join(existing)
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # AYNI: Dört konfigürasyonun sırası
    for (sp, cp), table, figure_number in zip(
        MODES,
        selections,
        range(1, 5)
    ):

        draw_chart(
            table,
            figure_number,
            sp,
            cp,
            output_dir,
            xlim,
            min_freq,
            max_freq,
            args.top
        )

    print(
        "[DONE] Four English charts completed.",
        flush=True
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            f"[ERROR] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True
        )

        raise