"""Kadın/erkek hedef kelimeleri: Her SEED_n farklı bir deneydir.

Yeni deney eklemek için SEED_2 gibi yeni bir kayıt oluşturabilirsiniz.
Aynı sözcük iki listede bulunursa iki grupta da bağımsız sayılır.
"""

DENEYLER = {
    "SEED_1": {
        "kadin": {"kadın", "kız", "gelin", "ana", "bacı", "avrat", "hanım", "hatun"},
        "erkek": {"erkek", "oğlan", "kardeş", "gardaş", "bey", "paşa", "yiğit", "ağa"},
    },
    # "SEED_2": {"kadin": {...}, "erkek": {...}},  # Henüz belirlenmedi.
}


def seedleri_al(deney):
    if deney not in DENEYLER:
        raise ValueError(f"Seed deneyi bulunamadı: {deney}. Tanımlı: {', '.join(DENEYLER)}")
    kadin = set(DENEYLER[deney]["kadin"])
    erkek = set(DENEYLER[deney]["erkek"])
    if not kadin or not erkek:
        raise ValueError("Kadın ve erkek seed listeleri boş olamaz")
    return kadin, erkek
