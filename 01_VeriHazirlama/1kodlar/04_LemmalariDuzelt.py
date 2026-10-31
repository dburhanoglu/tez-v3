"""04: Stanza lemmalarina guvenli duzeltmeler. Seed secimi burada YOK."""
import argparse
from collections import Counter

# True: Mapping yalnızca uygun sözcük türlerinde uygulanır.
POS_LEMMA_GUVENLIK_KONTROLU = True
import json
from pathlib import Path

import pandas as pd

# Tarihsel 04 kodundaki aday esleme sozlugu korunur. Asagidaki
# guvenlik kurallari anlamca supheli donusumleri engeller.
FOLK_MAP_GROUPS = {
    'kız': ['kız', 'kızlar', 'kızım', 'kızın', 'kızları', 'kızına', 'kızlara', 'kızından', 'kızında', 'gız',
        'gızlar', 'gızım', 'gızın', 'gızları', 'gıza', 'gızına', 'gızların', 'gızını',],
    'gelin': ['gelin', 'gelini', 'gelinin', 'geline', 'gelinim', 'gelinler', 'gelinlerin', 'gelinlik',
        'geliniydi', 'gelinleriz', 'gelinleri', 'gelindi', 'gelinmez', 'gelinden',],
    'ana': ['ana', 'anam', 'anama', 'anamı', 'anamın', 'anamdan', 'anacığım', 'anamız', 'anaları', 'anası',
        'anaya', 'anasına', 'anadan', 'anasından', 'anayı', 'anasın',],
    'bacı': ['bacı', 'bacım', 'bacın', 'bacısı', 'bacılar', 'bacıma', 'bacımdan', 'bacından', 'bacında',],
    'kardeş': ['kardeş', 'kardeşim', 'kardeşin', 'kardeşler', 'kardeşi', 'kardeşlerim', 'kardeşe', 'kardaş',
        'kardaşım', 'kardaşlar', 'kardaşın', 'kardaşlarım', 'kardaşa', 'kardaşı', 'gardaş', 'gardaşım',
        'gardaşlar', 'gardaşın', 'gardaşlarım', 'gardaşa', 'gardaşı',],
    'oğul': ['oğul', 'oğlum', 'oğlun', 'oğluna', 'oğlunu', 'oğlumun', 'oğluma', 'oğluyum', 'oğlu', 'oğlunun',],
    'oğlan': ['oğlan', 'oğlana', 'oğlanı', 'oğlanın', 'oğlanlar', 'oğlancığım',],
    'canan': ['canan', 'canana', 'cananı', 'cananım', 'cananın', 'cananlar',],
    'yar': ['yar', 'yarin', 'yare', 'yardan', 'yarine', 'yarim', 'yarimi', 'yarime', 'yarimin', 'yarimden',
        'yari', 'yarisin', 'yarini', 'yarinden', 'yarem', 'yareme', 'yareler', 'yarelerim',],
    'yavru': ['yavru', 'yavrum', 'yavrusu', 'yavrular', 'yavrumu', 'yavruları', 'yavrusuna', 'yavruma',
        'yavrunun', 'yavruların',],
    'yosma': ['yosma', 'yosmam', 'yosmanın', 'yosmaya',],
    'kadın': ['kadın', 'kadınlar', 'kadını', 'kadına', 'kadınım',],
    'turna': ['turna', 'turnam', 'turnalar', 'turnanın', 'turnaya', 'turnamın', 'durna', 'durnam', 'durnalar',
        'durnamın', 'durnaya',],
    'suna': ['suna', 'sunam', 'sunalar', 'sunağım', 'sunaya', 'sunamın',],
    'bülbül': ['bülbül', 'bülbülü', 'bülbüller', 'bülbülün', 'bülbüle', 'bülbülleri', 'bülbülüm', 'bülbüldür',
        'bülbüllerin',],
    'ceylan': ['ceylan', 'ceylanı', 'ceylanın', 'ceylana', 'ceylanlar',],
    'keklik': ['keklik', 'kekliği', 'kekliğe', 'keklikler', 'kekliğin', 'kekliğim',],
    'ördek': ['ördek', 'ördeği', 'ördekler',],
    'kına': ['kına', 'kınası', 'kınalar', 'kınayı', 'kınaya', 'kınan', 'kınamayın', 'kınalı', 'gınan', 'gına',],
    'gerdan': ['gerdan', 'gerdanı', 'gerdanda', 'gerdandan', 'gerdana', 'gerdanın', 'gerdanına', 'gerdanında',],
    'güzel': ['güzel', 'gözel', 'gözeller', 'gözelin', 'gözele', 'gözelim', 'gözeldir', 'gözellerin', 'gözeldi',],
    'kurban': ['kurban', 'kurbanı', 'kurbana', 'kurbanlar', 'kurbanım', 'kurbanın', 'gurban', 'gurbanı',
        'gurbanam', 'gurbanın', 'gurbanlar', 'gurbanım',],
    'aşık': ['aşık', 'aşıklar', 'aşığın', 'aşığı', 'aşıklara', 'aşıkların', 'aşıka', 'aşıh', 'aşığ', 'aşuk',],
    'yok': ['yok', 'yoh', 'yokh', 'yohdu', 'yoğ', 'yoğumuş',],
    'öz': ['öz', 'özüm', 'özümüz', 'özün', 'özüme', 'özümden',],
    'bağır': ['bağır', 'bağrım', 'bağırım', 'bağrıma', 'bağrına', 'bağrında', 'bağrımı', 'bağrının',],
    'deli': ['deli', 'delisi', 'delidir', 'deliler', 'delisin', 'delisine',],
    'sultan': ['sultan', 'sultana', 'sultanı', 'sultanlar', 'sultanın',],
    'destan': ['destan', 'destanı', 'destanlar', 'destanın',],
    'paşa': ['paşa', 'paşalar', 'paşanın', 'paşaya', 'paşam',],
    'veli': ['veli', 'veliler', 'velinin', 'veliye',],
    'efendi': ['efendi', 'efendim', 'efendimiz',],
    'mahzuni': ['mahzuni', 'mahzun',],
    'karacaoğlan': ['karacaoğlan', 'karacoğ',],
    'hüseyin': ['hüseyin', 'hüseyn', 'hüseynim',],
    'ninni': ['ninni', 'ninna', 'nenni', 'ninnayı', 'ninneler', 'ninne',],
    'şinanay': ['şinanay',],
    'yayla': ['yayla', 'yaylaya', 'yaylada', 'yayladan', 'yaylası', 'yaylalar', 'yaylanın', 'yaylaları',
        'yaylasına', 'yaylasında', 'yaylamız', 'yaylalarda', 'yaylalara',],
    'tarla': ['tarla', 'tarlaya', 'tarlada', 'tarladan', 'tarlası', 'tarlalar', 'tarlanın', 'tarlaları',
        'tarlasına', 'tarlasında', 'tarlalarda',],
    'damla': ['damla', 'damlaya', 'damlada', 'damladan', 'damlası', 'damlalar',],
    'bahçe': ['bahçe', 'bahçede', 'bahçeden', 'bahçeye', 'bahçesi', 'bahçeler', 'bahçenin', 'bahçesinde',
        'bahçesine', 'bahçada', 'bahçaya', 'bahçalar', 'bahçasında', 'bahçenizde',],
    'meydan': ['meydan', 'meydanı', 'meydanda', 'meydandan', 'meydana', 'meydanlar', 'meydanında', 'meydanına',
        'meydanlarda',],
    'konak': ['konak', 'konağı', 'konaklar', 'konakta', 'konağa', 'konakları',],
    'mezar': ['mezar', 'mezara', 'mezarda', 'mezardan', 'mezarı', 'mezarıma', 'mezarım', 'mezarını', 'mezarımın',
        'mezarında', 'mezarlar',],
    'bekçi': ['bekçi', 'bekçisi', 'bekçinin', 'bekçiler', 'bekçiye',],
    'avcı': ['avcı', 'avcılar', 'avcının', 'avcıya',],
    'çoban': ['çoban', 'çobanı', 'çobana', 'çobanın', 'çobanlar',],
    'jandarma': ['jandarma', 'jandarmalar', 'candarma', 'candarmalar',],
    'doktor': ['doktor', 'doktorlar', 'doktora',],
    'asker': ['asker', 'askerler', 'askerin',],
    'fincan': ['fincan', 'fincanı', 'fincana', 'fincandan',],
    'deste': ['deste', 'destesi', 'desteler', 'destine',],
    'mektup': ['mektup', 'mektubu', 'mektuba', 'mektubum', 'mektubun', 'mektuplar', 'mektubunu',],
    'yastık': ['yastık', 'yastığa', 'yastıkta', 'yastığım', 'yastığı',],
    'türkü': ['türkü', 'türküsü', 'türküler', 'türkünün', 'türkülerimiz', 'türkülerde', 'türküleri', 'türküyü',],
    'yürek': ['yürek', 'yüreği', 'yüreğe', 'yürekler', 'yürekten', 'yürekte', 'yüreğim', 'yüreğin', 'yüreğime',
        'yüreğimde', 'yüreğimi', 'yüreğine', 'yüreğinde', 'yüreğimden', 'yüreğimiz', 'yüreciğim',],
    'çiçek': ['çiçek', 'çiçeği', 'çiçeğe', 'çiçekler', 'çiçekten', 'çiçekleri', 'çiçeğim', 'çiçeklerin',
        'çiçeklerden',],
    'toprak': ['toprak', 'toprağı', 'toprağa', 'topraklar', 'topraktan', 'toprakta', 'toprağın', 'toprağında',
        'toprağına', 'topraklara',],
    'yaprak': ['yaprak', 'yaprağı', 'yaprağa', 'yapraklar', 'yapraktan', 'yaprakları', 'yaprağın', 'yaprağını',],
    'kulak': ['kulak', 'kulağı', 'kulağa', 'kulaklar', 'kulağım', 'kulağın', 'kulağıma', 'kulağında', 'kulakları',],
    'ocak': ['ocak', 'ocağı', 'ocağa', 'ocaklar', 'ocakta', 'ocağın', 'ocağında', 'ocağına', 'ocağım', 'ocağını',],
    'derman': ['derman', 'dermanı', 'dermana', 'dermanın', 'dermanımı', 'dermanını',],
    'ferman': ['ferman', 'fermanı', 'fermana', 'fermanın', 'fermanım',],
    'devran': ['devran', 'devranı', 'devrana', 'devranın',],
    'kervan': ['kervan', 'kervanı', 'kervana', 'kervanın', 'kervanlar',],
}

# Bu kelimelerde ayni yazilis farkli kavramlara karsilik gelebilir.
# Ihtiyat: Stanza'nin lemmayi yeniden yorumlamasini zorlamiyoruz.
BELIRSIZ_KELIMELER = {
    "gelinlik", "gelinmez", "gelindi", "gelinleriz",
    "kınamayın", "kınalı", "kınan", "mahzun", "destine",
    "sunağım", "doktora",
    "yarem", "yareme", "yareler", "yarelerim",
}

# Benzer yazilisli 'yar' grubu sevgi / yara ayrimi nedeniyle
# tumden otomatik map edilmiyor; Stanza'nin mevcut sonucu saklaniyor.
BELIRSIZ_KOKLER = {"yar"}

HARITA = {}
for kok, kelimeler in FOLK_MAP_GROUPS.items():
    for kelime in kelimeler:
        if kelime in HARITA and HARITA[kelime] != kok:
            raise ValueError(f"Sozlukte celiskili esleme: {kelime}")
        HARITA[kelime] = kok


def uygun_sozcuk_turu(kok, upos):
    if kok in {"güzel", "deli"}:
        return upos in {"ADJ", "NOUN"}
    if kok == "yok":
        return upos in {"ADJ", "ADV", "PART", "INTJ"}
    if kok in {"ninni", "şinanay"}:
        return upos in {"NOUN", "INTJ"}
    return upos in {"NOUN", "PROPN"}


def oku_liste(deger, ad, satir):
    if pd.isna(deger):
        raise ValueError(f"{satir}. turku: {ad} bos")
    try:
        liste = json.loads(str(deger))
    except json.JSONDecodeError as hata:
        raise ValueError(f"{satir}. turku: {ad} JSON bozuk") from hata
    if not isinstance(liste, list):
        raise ValueError(f"{satir}. turku: {ad} liste degil")
    return liste


def main():
    parser = argparse.ArgumentParser(description="Guvenli lemma duzeltmesi")
    parser.add_argument("--girdi", type=Path, required=True)
    parser.add_argument("--cikti", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_excel(args.girdi, engine="openpyxl")
    gereken = ["Link", "Başlık", "Yöre", "Sözler", "Yalın Sözler (Temizlik Sonrası)",
               "Tokenler", "Lemmalar", "POS_Etiketleri"]
    eksik = set(gereken) - set(df.columns)
    if eksik:
        raise ValueError(f"04 girdisinde eksik sutun: {sorted(eksik)}")
    df = df[gereken].copy()
    kolonlar = ("Tokenler", "Lemmalar", "POS_Etiketleri")
    if not set(kolonlar).issubset(df.columns):
        raise ValueError(f"Eksik token sutunu: {set(kolonlar) - set(df.columns)}")

    degisen, belirsiz = Counter(), Counter()
    cikti_lemmalari = []
    for satir in range(1, len(df) + 1):
        veri = df.iloc[satir - 1]
        token, lemma, pos = [oku_liste(veri[k], k, satir) for k in kolonlar]
        # Bu uc listenin ayni indeksi ayni kelimeyi gostermeli.
        if len(token) != len(lemma) or len(lemma) != len(pos):
            raise ValueError(f"{satir}. turku: Token/Lemma/POS listeleri ayni uzunlukta degil")

        for i, (kelime, eski, tur) in enumerate(zip(token, lemma, pos)):
            hedef = HARITA.get(kelime)
            # Stanza zaten yanlis olarak baska kavrami vermisse de koru:
            # gelinlik -> gelin, kinamayin -> kina vb. ayrimlar.
            if kelime in BELIRSIZ_KELIMELER:
                if hedef and eski == hedef and kelime != hedef:
                    lemma[i] = kelime  # emin olmadigimizda yuzey bicimini tut
                    degisen[(kelime, eski, kelime)] += 1
                belirsiz[kelime] += 1
                continue
            if not hedef or hedef == eski:
                continue
            if hedef in BELIRSIZ_KOKLER or (POS_LEMMA_GUVENLIK_KONTROLU and not uygun_sozcuk_turu(hedef, tur)):
                belirsiz[kelime] += 1
                continue
            # Gelin ve kiz'da fiil etiketini kesinlikle seed lemmaya cevirme.
            if POS_LEMMA_GUVENLIK_KONTROLU and hedef in {"gelin", "kız"} and tur not in {"NOUN", "PROPN"}:
                belirsiz[kelime] += 1
                continue
            lemma[i] = hedef
            degisen[(kelime, eski, hedef)] += 1

        if any(not x or any(ch.isspace() for ch in x) for x in lemma):
            raise ValueError(f"{satir}. turku: bos/bosluklu lemma bulundu")
        cikti_lemmalari.append(lemma)

    df["Lemmalar"] = [json.dumps(x, ensure_ascii=False, separators=(",", ":"))
                     for x in cikti_lemmalari]
    if any(len(x) > 32767 for x in df["Lemmalar"]):
        raise ValueError("Excel hucre siniri asildi: lemmalar")

    if args.cikti.exists():
        raise FileExistsError(f"04 cikti zaten var: {args.cikti} (uzerine yazilmadi)")
    args.cikti.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.cikti, index=False, engine="openpyxl")
    print(f"04 TAMAM | {len(df)} turku | Duzeltilen token={sum(degisen.values())}")
    print(f"Kosullu/riski nedeniyle degistirilmeyen={sum(belirsiz.values())}")
    print("Ilk duzeltmeler:", degisen.most_common(12))
    print("Ilk korunan supheli kelimeler:", belirsiz.most_common(12))
    print(f"Cikti: {args.cikti}")


if __name__ == "__main__":
    main()
