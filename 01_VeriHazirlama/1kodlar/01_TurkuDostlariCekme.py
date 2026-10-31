# Bu kod zaten çekilmiş türküleri hafızada tutar ve tekrar çekme bozulur excel
# TEZ AŞAMASI: Türkü Dostları sitesinden ham türkü verilerinin toplanması.
# Bu dosyada NLP/lemmatizasyon yapılmaz; web sayfaları ayrıştırılıp Excel'e kaydedilir.

import os
import re
import sys
import time
import json
import random
import logging
import ctypes
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Set, Optional
import urllib.parse

import requests
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# ==========================================
# KONFİGÜRASYON & SABİTLER /////GÜNCELLENDİ YENİ KLASÖR YAPISINA GÖRE///////
# ==========================================
# Türkülerin alındığı kaynak web sitesi.
BASE_URL = "https://www.turkudostlari.net/"

# Yeni Dinamik Dosya Yolları
# Kod dosyasının bulunduğu klasörün üst klasörünü proje kökü kabul eder.
BASE_DIR = Path(__file__).resolve().parent.parent
# Ham türkü arşivinin ve işlem günlüklerinin kaydedileceği klasörler.
OUTPUT_DIR = BASE_DIR / "2_Veri" / "01_Ham_Veri"
LOG_DIR = BASE_DIR / "3_Raporlar_ve_Loglar"

# Klasörler yoksa oluşturulur; varsa olduğu gibi kullanılır.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# state.json: daha önce işlenen linkler ve tamamlanan harfler.
# Excel: ham türkü verileri. Log: çalışma ve hata mesajları.
STATE_FILE = LOG_DIR / "state.json"
EXCEL_FILE = OUTPUT_DIR / "turkudostlari_arsiv.xlsx"
LOG_FILE = LOG_DIR / "scraper.log"

# Bir URL için en fazla 3 istek denemesi; aynı anda en fazla 5 detay sayfası işlenir.
MAX_RETRIES = 3
MAX_WORKERS = 5  # Eşzamanlı çekilecek türkü sayısı

# Sitenin alfabetik listeleme sayfalarında kullandığı harf kimlikleri.
# Örnek: A harfi için abc=1, Ç için abc=4 kullanılır.
HARF_MAP = {
    'A': 1, 'B': 2, 'C': 3, 'Ç': 4, 'D': 5, 'E': 6, 'F': 7, 'G': 8, 'H': 10,
    'I': 11, 'İ': 12, 'J': 13, 'K': 14, 'L': 15, 'M': 16, 'N': 17, 'O': 18,
    'Ö': 19, 'P': 20, 'R': 21, 'S': 22, 'Ş': 23, 'T': 24, 'U': 25, 'Ü': 26,
    'V': 27, 'Y': 28, 'Z': 29
}

# Türkçe alfabetik sıralama önceliği
# Son Excel sıralamasında her harfin önceliği hesaplanır.
TR_ALFABE = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
HARF_SIRA = {harf: i for i, harf in enumerate(TR_ALFABE)}

# HTTP isteklerine tarayıcı kimliği ve tercih edilen dil eklenir.
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7'
}

# Excel'de her türkü için saklanan sekiz alan.
EXCEL_COLUMNS = ["Harf", "Liste Sayfası", "Başlık", "Yöre", "Eser Sahibi / Kaynak", "Derleyen", "Sözler", "Link"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# İşlem ve hata mesajları hem log dosyasına hem terminale yazılır.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TurboScraper")

# Thread-safe işlemler için Kilit (Lock)
# Eşzamanlı işçiler ortak kayıt tamponunu ve sayaçları aynı anda değiştirmesin diye kullanılır.
data_lock = threading.Lock()

# ==========================================
# WINDOWS UYKU ENGELLEYİCİ
# ==========================================
# Windows'ta işlem sürerken sistem/ekran uyku durumuna geçmesin diye yürütme durumu ayarlanır.
def prevent_sleep():
    if os.name == 'nt':
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
        except Exception:
            pass

# İşlem bitince normal uyku davranışına dönülür.
def allow_sleep():
    if os.name == 'nt':
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        except Exception:
            pass

# ==========================================
# STATE & EXCEL MANAGER
# ==========================================
# Önceki taramalardan kalan linkleri ve tamamlanan harfleri yönetir.
class ScraperState:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        # set: aynı bağlantı/harf birden fazla kez saklanmaz.
        self.scraped_links: Set[str] = set()
        self.completed_letters: Set[str] = set()
        self.load()

    # state.json varsa önceki tarama durumu belleğe alınır.
    def load(self):
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.scraped_links = set(data.get("scraped_links", []))
                    self.completed_letters = set(data.get("completed_letters", []))
                logger.info(f"State yüklendi: {len(self.scraped_links)} türkü hafızada. Zaten çekilenler anında atlanacak.")
            except Exception as e:
                logger.error(f"State okunamadı: {e}. Sıfırdan başlanıyor.")

    # Güncel bağlantılar, tamamlanan harfler ve zaman damgası JSON'a yazılır.
    # Önce geçici dosya yazılıp sonra asıl state.json dosyasının yerine geçirilir.
    def save(self):
        try:
            temp_path = self.filepath.with_suffix(".tmp")
            payload = {
                "scraped_links": list(self.scraped_links),
                "completed_letters": list(self.completed_letters),
                "last_updated": datetime.now().isoformat()
            }
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            temp_path.replace(self.filepath)
        except Exception as e:
            logger.error(f"State kaydedilemedi: {e}")

# Ham veri arşivini oluşturur, kayıt ekler, sonunda sıralar ve biçimlendirir.
class ExcelManager:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        # Yeni türkülerin diske yazılmadan önce tutulduğu geçici liste.
        self.data_buffer: List[Dict[str, Any]] = []
        self._ensure_file_exists()

    # Excel yoksa başlık satırı ve biçimlendirmeyle oluşturulur.
    # Varsa mevcut dosya korunur; bu aşamada üzerine yazılmaz.
    def _ensure_file_exists(self):
        if not self.filepath.exists():
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Turkuler"
            
            header_fill = PatternFill(start_color="8B4513", end_color="8B4513", fill_type="solid")
            for col_idx, h_text in enumerate(EXCEL_COLUMNS, start=1):
                c = ws.cell(row=1, column=col_idx, value=h_text)
                c.font = Font(name="Segoe UI", size=10.5, bold=True, color="FFFFFF")
                c.fill = header_fill
                c.alignment = Alignment(horizontal="center", vertical="center")
            
            col_widths = {'A': 8, 'B': 12, 'C': 32, 'D': 18, 'E': 26, 'F': 26, 'G': 85, 'H': 35}
            for col_l, w in col_widths.items():
                ws.column_dimensions[col_l].width = w

            ws.freeze_panes = "A2"
            wb.save(self.filepath)

    # Excel'deki Link sütununu okuyup mevcut bağlantıları kümeye dönüştürür.
    # Böylece state.json dışında Excel'de bulunan kayıtlar da yeniden çekilmez.
    def load_existing_links(self) -> Set[str]:
        if not self.filepath.exists():
            return set()
        try:
            df = pd.read_excel(self.filepath, engine="openpyxl")
            if "Link" in df.columns:
                return set(df["Link"].astype(str).str.strip().dropna().unique())
        except Exception:
            pass
        return set()

    # Ayrıştırılan tek bir türkü kaydını geçici belleğe ekler.
    def add_record(self, record: Dict[str, Any]):
        self.data_buffer.append(record)

    # Tampondaki türküleri mevcut Excel'in sonuna ekler.
    # Kaydetme başarılıysa tampon temizlenir; hata olursa False döner.
    def save_to_excel(self) -> bool:
        if not self.data_buffer:
            return False
        try:
            wb = openpyxl.load_workbook(self.filepath)
            ws = wb.active
            zebra_fill = PatternFill(start_color="FDF8F3", end_color="FDF8F3", fill_type="solid")
            border_thin = Border(left=Side(style='thin', color='E5D8CC'), right=Side(style='thin', color='E5D8CC'), 
                                 top=Side(style='thin', color='E5D8CC'), bottom=Side(style='thin', color='E5D8CC'))
            start_row = ws.max_row + 1
            
            # Her kayıt sekiz sütuna yerleştirilir; sözler satır kaydırmalı tutulur.
            for i, r_data in enumerate(self.data_buffer):
                current_row = start_row + i
                is_even = (current_row % 2 == 0)
                
                ws.cell(row=current_row, column=1, value=r_data.get("Harf")).alignment = Alignment(horizontal="center", vertical="top")
                ws.cell(row=current_row, column=2, value=r_data.get("Liste Sayfası")).alignment = Alignment(horizontal="center", vertical="top")
                ws.cell(row=current_row, column=3, value=r_data.get("Başlık")).alignment = Alignment(horizontal="left", vertical="top")
                ws.cell(row=current_row, column=4, value=r_data.get("Yöre")).alignment = Alignment(horizontal="center", vertical="top")
                ws.cell(row=current_row, column=5, value=r_data.get("Kaynak")).alignment = Alignment(horizontal="left", vertical="top")
                ws.cell(row=current_row, column=6, value=r_data.get("Derleyen")).alignment = Alignment(horizontal="left", vertical="top")
                
                soz_c = ws.cell(row=current_row, column=7, value=r_data.get("Sözler"))
                soz_c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
                
                link_c = ws.cell(row=current_row, column=8, value=r_data.get("Link"))
                link_c.alignment = Alignment(horizontal="left", vertical="top")
                link_c.font = Font(name="Segoe UI", size=9, color="0044CC", underline="single")

                # Alternatif satırlara açık renk ve tüm hücrelere kenarlık uygulanır.
                for c_idx in range(1, 9):
                    c = ws.cell(row=current_row, column=c_idx)
                    if is_even:
                        c.fill = zebra_fill
                    c.border = border_thin
                    if c_idx != 8:
                        c.font = Font(name="Segoe UI", size=9.5)

            wb.save(self.filepath)
            logger.info(f"Excel'e {len(self.data_buffer)} yeni kayıt yazıldı.")
            self.data_buffer.clear()
            return True
        except Exception as e:
            logger.error(f"Excel yazma hatası (Açık kalmış olabilir): {e}")
            return False

    # Son aşamada aynı Link'e sahip tekrarları kaldırır ve Excel'i yeniden düzenler.
    def sort_and_format_final_excel(self):
    
        if not self.filepath.exists():
            return
        
        logger.info("Excel dosyası alfabetik ve sayfa sırasına göre düzenleniyor...")
        try:
            df = pd.read_excel(self.filepath, engine="openpyxl")
            if df.empty:
                return
            
            # Tekrarlayan satır kontrolü ve temizliği
            # Aynı Link'e sahip kayıtlardan ilkini bırakır; söz benzerliğiyle tekilleştirme yapmaz.
            df = df.drop_duplicates(subset=["Link"]).reset_index(drop=True)

            # Türkçe harf önceliği sırası oluşturma
            # Harf bulunamazsa sıralama önceliği 99 kabul edilir.
            df["_harf_sira"] = df["Harf"].map(lambda x: HARF_SIRA.get(str(x).upper(), 99))
            df["Liste Sayfası"] = pd.to_numeric(df["Liste Sayfası"], errors="coerce").fillna(0)

            # Sıralama: Harf Sırası -> Liste Sayfası -> Başlık
            df = df.sort_values(by=["_harf_sira", "Liste Sayfası", "Başlık"]).reset_index(drop=True)
            df = df.drop(columns=["_harf_sira"])

            # Sıralanmış veriyi biçimlendirerek tekrar Excel'e kaydet
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Turkuler"

            header_fill = PatternFill(start_color="8B4513", end_color="8B4513", fill_type="solid")
            for col_idx, h_text in enumerate(EXCEL_COLUMNS, start=1):
                c = ws.cell(row=1, column=col_idx, value=h_text)
                c.font = Font(name="Segoe UI", size=10.5, bold=True, color="FFFFFF")
                c.fill = header_fill
                c.alignment = Alignment(horizontal="center", vertical="center")

            col_widths = {'A': 8, 'B': 12, 'C': 32, 'D': 18, 'E': 26, 'F': 26, 'G': 85, 'H': 35}
            for col_l, w in col_widths.items():
                ws.column_dimensions[col_l].width = w

            zebra_fill = PatternFill(start_color="FDF8F3", end_color="FDF8F3", fill_type="solid")
            border_thin = Border(left=Side(style='thin', color='E5D8CC'), right=Side(style='thin', color='E5D8CC'), 
                                 top=Side(style='thin', color='E5D8CC'), bottom=Side(style='thin', color='E5D8CC'))

            # Tekilleştirilmiş ve sıralanmış kayıtları yeni çalışma kitabına aktarır.
            for i, row in df.iterrows():
                current_row = i + 2
                is_even = (current_row % 2 == 0)

                ws.cell(row=current_row, column=1, value=row.get("Harf")).alignment = Alignment(horizontal="center", vertical="top")
                ws.cell(row=current_row, column=2, value=row.get("Liste Sayfası")).alignment = Alignment(horizontal="center", vertical="top")
                ws.cell(row=current_row, column=3, value=row.get("Başlık")).alignment = Alignment(horizontal="left", vertical="top")
                ws.cell(row=current_row, column=4, value=row.get("Yöre")).alignment = Alignment(horizontal="center", vertical="top")
                ws.cell(row=current_row, column=5, value=row.get("Eser Sahibi / Kaynak", row.get("Kaynak", ""))).alignment = Alignment(horizontal="left", vertical="top")
                ws.cell(row=current_row, column=6, value=row.get("Derleyen")).alignment = Alignment(horizontal="left", vertical="top")

                soz_c = ws.cell(row=current_row, column=7, value=row.get("Sözler"))
                soz_c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

                link_c = ws.cell(row=current_row, column=8, value=row.get("Link"))
                link_c.alignment = Alignment(horizontal="left", vertical="top")
                link_c.font = Font(name="Segoe UI", size=9, color="0044CC", underline="single")

                for c_idx in range(1, 9):
                    c = ws.cell(row=current_row, column=c_idx)
                    if is_even:
                        c.fill = zebra_fill
                    c.border = border_thin
                    if c_idx != 8:
                        c.font = Font(name="Segoe UI", size=9.5)

            ws.freeze_panes = "A2"
            wb.save(self.filepath)
            logger.info(f"Sıralama tamamlandı! Toplam {len(df)} türkü A-Z düzeninde kaydedildi.")
        except Exception as e:
            logger.error(f"Sıralama sırasında hata: {e}")

# ==========================================
# ANA SCRAPER MOTORU (MULTITHREADED)
# ==========================================
# Web sayfalarından linkleri bulur, türkü detaylarını çeker ve kaydeder.
class TurboScraper:
    def __init__(self):
        # Önce durum ve Excel yöneticileri başlatılır.
        self.state = ScraperState(STATE_FILE)
        self.excel_mgr = ExcelManager(EXCEL_FILE)
        self.completed_count = 0
        
        # Daha önce çekilenleri State'e senkronize et
        # Excel'deki bağlantılar da hafızaya eklenir; eski türkülerin yeniden istenmesi önlenir.
        existing_links = self.excel_mgr.load_existing_links()
        self.state.scraped_links.update(existing_links)

    # Verilen URL'nin HTML'ini indirir ve BeautifulSoup nesnesi döndürür.
    # Başarısız isteklerde en fazla 3 deneme yapılır; başarılı HTTP 200 aranır.
    def fetch_soup(self, url: str) -> Optional[BeautifulSoup]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Her isteğin öncesinde 0,1–0,5 saniye rastgele beklenir.
                time.sleep(random.uniform(0.1, 0.5)) 
                r = requests.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 200:
                    # Sayfa metni Türkçe Windows-1254 kodlamasıyla çözümlenir.
                    r.encoding = 'windows-1254'
                    return BeautifulSoup(r.text, 'html.parser')
            except requests.RequestException:
                time.sleep(2 * attempt)
        return None

    # Harfin ilk liste sayfasındaki toplam kayıt sayısını bulur.
    # Sayfa başına 50 kayıt kabul edilerek toplam sayfa sayısı hesaplanır.
    def calculate_max_pages(self, abc_id: int) -> int:
        url = f"{BASE_URL}liste.asp?x=1&abc={abc_id}&s=1"
        soup = self.fetch_soup(url)
        if not soup:
            return 1
        
        # HTML'de '(... Kayıt' biçimindeki sayı aranır.
        match = re.search(r'<li class="baslik"><h1>.*?\(([\d\.]+)\s*Kayıt', soup.decode(), re.IGNORECASE)
        if match:
            kayit = int(match.group(1).replace('.', ''))
            return (kayit // 50) + (1 if kayit % 50 > 0 else 0)
        return 1

    # Tek bir türkünün detay sayfasından başlık, söz, yöre, kaynak ve derleyen bilgisi çıkarılır.
    def parse_detail_page(self, link: str, harf: str, page_num: int) -> Optional[Dict[str, Any]]:
       
        soup = self.fetch_soup(link)
        if not soup:
            return None

        # Başlık, itemprop='name' HTML alanından alınır.
        baslik = ""
        name_elem = soup.select_one("span[itemprop='name']")
        if name_elem:
            baslik = name_elem.get_text(strip=True)

        # Türkü sözleri itemprop='articleBody' alanından alınır.
        sozler = ""
        body_elem = soup.select_one("span[itemprop='articleBody']")
        if body_elem:
            # HTML'deki <br> etiketleri gerçek satır sonuna dönüştürülür.
            for br in body_elem.find_all('br'):
                br.replace_with('\n')
            
            raw_lines = body_elem.get_text().splitlines()
            cleaned_lines = []
            
            # Her satırın başındaki/sonundaki boşluklar kaldırılır.
            # Boş satırların tamamı silinmez: art arda gelen boş satırlar teke indirilir.
            for l in raw_lines:
                line_str = l.strip()
                if not line_str:
                    if cleaned_lines and cleaned_lines[-1] != "":
                        cleaned_lines.append("")
                else:
                    cleaned_lines.append(line_str)
            # Satırlar birleştirilir; metnin başındaki ve sonundaki boşluklar temizlenir.
            sozler = "\n".join(cleaned_lines).strip()

        # Kaynak ve yöre, sayfadaki ilgili bağlantıların sırasına göre alınır.
        kaynak, yore, derleyen = "", "", ""
        ky_elems = soup.select("li.kaynakyore h2 a")
        if len(ky_elems) > 0:
            kaynak = ky_elems[0].get_text(strip=True)
        if len(ky_elems) > 1:
            yore = ky_elems[1].get_text(strip=True)

        # Dipnotta 'tarafından derlenmiştir' ifadesinin önündeki metin derleyen olarak alınır.
        dipnot_elem = soup.select_one("li.dipnot")
        if dipnot_elem:
            dip_text = dipnot_elem.get_text()
            m = re.search(r'(.*?)\s+tarafından derlenmiştir', dip_text, re.IGNORECASE)
            if m:
                derleyen = m.group(1).strip()
                
        # Sözleri boş olan türkü kayda alınmaz.
        if not sozler:
            return None

        # Başlık bulunamazsa URL'nin son '=' işaretinden sonraki kısmı kullanılır.
        if not baslik:
            baslik = link.split("=")[-1]

        # Bir türküye ait sekiz alanlık ham veri kaydı oluşturulur.
        return {
            "Harf": harf,
            "Liste Sayfası": page_num,
            "Başlık": baslik,
            "Yöre": yore if yore else "Belirtilmemiş",
            "Kaynak": kaynak,
            "Derleyen": derleyen,
            "Sözler": sozler,
            "Link": link
        }

    # Taramanın ana akışı: harf -> liste sayfası -> yeni linkler -> detaylar -> kayıt.
    def run(self):
        logger.info("=== TURBO KAZIMA (MULTITHREADING) BAŞLATILDI ===")
        prevent_sleep()

        try:
            # Harfler HARF_MAP sözlüğündeki sırayla işlenir.
            for harf, abc_id in HARF_MAP.items():
                # Tamamlandı olarak işaretlenen harf tekrar taranmaz.
                if harf in self.state.completed_letters:
                    logger.info(f"[{harf}] Harfi daha önce tamamlanmış, atlanıyor...")
                    continue

                max_page = self.calculate_max_pages(abc_id)
                logger.info(f"\n--- [{harf}] Harfi Taranıyor (Toplam: {max_page} Sayfa) ---")
                
                # İlgili harfin liste sayfaları tek tek ziyaret edilir.
                for page_num in range(1, max_page + 1):
                    url_list = f"{BASE_URL}liste.asp?x=1&abc={abc_id}&s={page_num}"
                    soup_list = self.fetch_soup(url_list)
                    
                    # Liste sayfası indirilemezse bu sayfa atlanır.
                    if not soup_list:
                        continue
                        
                    # Liste sayfasındaki türkü bağlantıları toplanır.
                    turku_links = []
                    for a_tag in soup_list.select("li.turkuadi a"):
                        href = a_tag.get("href")
                        if href:
                            # Göreli bağlantı tam URL'ye dönüştürülür.
                            turku_links.append(urllib.parse.urljoin(BASE_URL, href))
                    
                    # Ön eleme: Sadece hafızada OLMAYAN linkleri çekilecekler listesine al
                    links_to_scrape = [l for l in turku_links if l not in self.state.scraped_links]
                    
                    # Sayfadaki bütün linkler önceden kayıtlıysa detay sayfaları açılmaz.
                    if not links_to_scrape:
                        logger.info(f"[{harf}] - Sayfa {page_num}/{max_page} tamamen hafızada. Hızla geçiliyor...")
                        continue
                        
                    logger.info(f"[{harf}] - Sayfa {page_num}/{max_page} işleniyor... ({len(links_to_scrape)} yeni türkü)")

                    # 🚀 MULTITHREADING (Eşzamanlı Kazıma)
                    # En fazla beş işçi, farklı türkü detay sayfalarını eşzamanlı ayrıştırır.
                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = [executor.submit(self.parse_detail_page, link, harf, page_num) for link in links_to_scrape]
                        
                        # Sonuçlar listedeki sırayla değil, görevler tamamlandıkça alınır.
                        for future in concurrent.futures.as_completed(futures):
                            detay = future.result()
                            if detay:
                                # Başarılı kayıt tampona eklenir, link hafızaya alınır, sayaç artar.
                                with data_lock:
                                    self.excel_mgr.add_record(detay)
                                    self.state.scraped_links.add(detay["Link"])
                                    self.completed_count += 1
                                    
                                    # Her 10 yeni kayıtta ilerleme mesajı yazılır.
                                    if self.completed_count % 10 == 0:
                                        logger.info(f"  -> Son eklenen: {detay['Başlık']} | Yeni Çekilen Toplam: {self.completed_count}")

                    # Her sayfanın bitiminde verileri otomatik diske yaz
                    # DİKKAT: Excel'e yazma başarısı kontrol edilmeden state.json kaydedilir.
                    with data_lock:
                        if self.excel_mgr.data_buffer:
                            self.excel_mgr.save_to_excel()
                            self.state.save()

                # Bir harf tamamen bittiğinde
                # DİKKAT: Atlanan/indirilemeyen sayfalar olsa da harf tamamlandı olarak işaretlenir.
                with data_lock:
                    self.state.completed_letters.add(harf)
                    self.state.save()
                    logger.info(f"[{harf}] Harfi arşivlemesi %100 tamamlandı.")

        except KeyboardInterrupt:
            logger.info("Kullanıcı müdahalesi. Mevcut veriler kaydediliyor...")
        except Exception as e:
            logger.critical(f"Beklenmeyen hata: {e}", exc_info=True)
        finally:
            # Normal bitişte veya yakalanan kesintide kalan veriler kaydedilmeye çalışılır.
            with data_lock:
                if self.excel_mgr.data_buffer:
                    self.excel_mgr.save_to_excel()
                self.state.save()
            
            # Tüm işlem bittiğinde Excel dosyasını Türkçe A-Z ve sayfa sırasına göre düzenle
            # Aynı Link'e sahip tekrarlar kaldırılır; sıralama ve biçimlendirme yapılır.
            self.excel_mgr.sort_and_format_final_excel()
            
            allow_sleep()
            logger.info(f"Program sonlandı. Oturumda çekilen yeni türkü sayısı: {self.completed_count}")

# Dosya doğrudan çalıştırılırsa scraper nesnesi oluşturulur ve tarama başlar.
if __name__ == "__main__":
     scraper = TurboScraper()
     scraper.run()

# ==========================================
# VERİ NASIL ÇEKİLDİ? — KISA TEZ NOTU
# ==========================================
# 1. Türkü Dostları'nın alfabetik liste sayfaları requests ile indirildi.
# 2. BeautifulSoup ile türkü linkleri bulundu; daha önce kaydedilen linkler atlandı.
# 3. En fazla 5 işçiyle detay sayfalarından başlık, söz, yöre, kaynak ve derleyen alındı.
# 4. Türkü sözlerindeki <br> etiketleri satır sonuna çevrildi; satır başı/sonu boşlukları silindi.
# 5. Art arda gelen boş satırlar tek boş satıra indirildi; TÜM boş satırlar silinmedi.
# 6. Sözleri boş olan türkü kaydedilmedi; yöre boşsa 'Belirtilmemiş' yazıldı.
# 7. Kayıtlar Excel'e aktarıldı; aynı Link'e sahip tekrarlar son aşamada kaldırıldı.
# 8. Excel, Türkçe harf sırası, liste sayfası ve başlığa göre sıralandı.
# Bu kodda noktalama/stopword temizliği, lemmatizasyon veya anlamsal analiz YAPILMADI.
