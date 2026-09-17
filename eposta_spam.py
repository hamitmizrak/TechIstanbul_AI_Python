# -*- coding: utf-8 -*-

"""
===============================================================================
MINI MACHINE LEARNING PROJESI - E-POSTA SPAM TAHMINI
===============================================================================

Bu proje, kullanicinin disaridan bir CSV dosyasi secmesini ve bu veri uzerinde
console/terminal araciligiyla temel Machine Learning adimlarini uygulamasini
saglar.

PROJENIN AMACI
--------------
Bir e-postanin SPAM olup olmadigini tahmin eden bir Classification uygulamasi
olusturmaktir.

ORNEK CSV SUTUNLARI
-------------------
kelime_sayisi
link_sayisi
buyuk_harf_orani
supheli_kelime_sayisi
gonderici_puani
ek_var
spam

Ornek:
kelime_sayisi,link_sayisi,buyuk_harf_orani,supheli_kelime_sayisi,gonderici_puani,ek_var,spam
120,0,0.05,0,92,0,0
45,6,0.72,5,18,1,1

spam:
    0 -> Normal e-posta
    1 -> Spam e-posta

ONEMLI
------
Bu proje icin hedef sutun otomatik olarak 'spam' kabul edilir.
'spam' disindaki sutunlar feature olarak kullanilir.

Program sayisal ve kategorik sutunlari otomatik algilar.
"""

# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Asagidaki import bolumu, projenin ihtiyac duydugu kutuphaneleri programa
# dahil eder.
#
# os / pathlib:
#   Dosya ve klasor islemleri icin kullanilir.
#
# pandas:
#   CSV dosyasini okumak, tablo halinde incelemek ve temizlemek icin kullanilir.
#
# numpy:
#   Sayisal islemlerde ve veri tipleriyle calisirken kullanilir.
#
# matplotlib:
#   Confusion Matrix grafigini PNG olarak kaydetmek icin kullanilir.
#
# scikit-learn:
#   Veriyi train/test olarak ayirmak, on isleme yapmak, model egitmek ve
#   Accuracy, Precision, Recall, F1 gibi metrikleri hesaplamak icin kullanilir.
# -----------------------------------------------------------------------------
from pathlib import Path
from tkinter.ttk import Style
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from colorama import Fore
from numpy.random.mtrand import choice
from sklearn.pipeline import Pipeline


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# AppState sinifi program boyunca kullanilan verileri tek bir yerde tutar.
#
# Neden gereklidir?
# Console uygulamalarinda kullanici once CSV yukler, sonra temizleme yapar,
# sonra model egitir. Her adimda ayni veriyi tekrar tekrar okumak yerine
# programin mevcut durumunu burada sakliyoruz.
#
# raw_df:
#   CSV dosyasindan ilk okunan, dokunulmamis orijinal veri.
#
# df:
#   Temizleme ve analiz islemlerinde kullanilan aktif veri.
#
# target_column:
#   Tahmin edilmek istenen hedef sutun.
#
# feature_columns:
#   Modelin tahmin yaparken kullanacagi giris sutunlari.
#
# best_model:
#   Egitilen modeller arasinda F1 skoruna gore en basarili model.
# -----------------------------------------------------------------------------
class AppState:
    def __init__(self):
        self.csv_path: Optional[Path] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None

        self.target_column: Optional[str] = None
        self.feature_columns: List[str] = []

        self.best_model: Optional[Pipeline] = None
        self.best_model_name: Optional[str] = None

        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
        self.y_pred: Optional[np.ndarray] = None

        self.model_results: List[Dict[str, Any]] = []

        # Program ilk açıldığında yalnızca veri hazırlama adımları (1-6)
        # gösterilir. Veri temizleme başarıyla tamamlandığında ikinci aşama
        # yani Machine Learning seçenekleri açılır.
        self.preprocessing_completed: bool = False

        # Aktif console ekranini takip eder.
        # 1 = Veri Hazirlama, 2 = Machine Learning.
        self.current_step: int = 1

        # Aktif olarak hangi veriyle devam edildigini takip eder.
        # Degerler: 'original', 'cleaned' veya None
        self.active_data_source: Optional[str] = None
        self.cleaned_csv_path: Optional[Path] = None
        self.last_pdf_report_path: Optional[Path] = None


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# console ekranı daha okunabilir hale gelmesini sağlamak
# SOLID: Single Responsibility
def print_header(title:str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Menü kuallnıcının sonucu okyabilmesini için ENTER
def pause() -> None:
    input("\nDevam etmek için lütfen ENTER tuşuna basınız...")


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Menu yazısını farklı renklerde kullanmamızı sağlar
def print_menu_option(text:str) -> None:
    print(Fore.LIGHTCYAN_EX + text + Style.RESET_ALL)

# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# print_step_title fonskiyon STEP başlıklarını menu seçeneklerinden ayırmak için parlak camgöbeği renkte gösterir.
def print_step_title(text:str) -> None:
    print(Fore.CYAN + Style.BRIGHT+ text + Style.RESET_ALL)


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# CSV sutun adlarını daha düzenli hale getirmek
# Ornek:
# " KeliME Sayısi " -> "kelime_sayisi"
# Bu sayede sutun isimlerindeki boşluk, büyük/küçük harf farklarlarından kaynaklanan hataları azaltmak
def normalize_column_name(name:str) -> str:
    value = str(name).replace("\ufeff","").strip().lower()

    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        " ": "_",
        "-": "_",
        "/": "_",
        "\\": "_",
    }

    for old, new in replacements.items():
        value = value.replace(old,new)

    while "__" in value:
        value= value.replace("__","_")

    return value.strip("_")



# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# discover_csv_files fonskiyonu kullanicini dosya seçebilmesini için CSV dosyalarını tarar ve sadece görününe CSV dosyalarını eklemeye yani dinamik olarak csv dosyalarını seçmeye yarar.
def discover_csv_files() -> List[Path]:
    found: List[Path] = []

    search_dirs = [
        Path.cwd(),
        Path.cwd() / "data"
    ]

    for folder in search_dirs:
        if not folder.exists() or not folder.is_dir():
            continue

        for file_path in folder.glob("*.csv"):
            resolved = file_path.resolve()
            if resolved not in found:
                found.append(resolved)
    return sorted(found, key=lambda p: p.name.lower())


# -----------------------------------------------------------------------------
# BU KOD NE ISE YARAR?
# -----------------------------------------------------------------------------
# Manuel olarak (Copy/Pasce) olarak girilen yolu seçmek
def choose_csv_path() -> Optional[Path]:
    print_header("CSV DOSYASINI SEÇ")

    print_menu_option("0 -Ana menüye dön")
    print_menu_option("1 -Dosya yolunu manuel gir")
    print_menu_option("2 -Dosya yolunu dosya seçerek gir")

    choice =input("\nSeçiminiz: ").strip()

    if choice == "0":
        return None

    if choice == "1":
        raw_path = input(
            "\nCSV dosyasını tam yolunu giriniz: "
        ).strip().strip('"')

        if not raw_path:
            print("\nHATA: Dosya yolu boş bırakılamaz.")
            return None

        path = Path(raw_path).expanduser()

        if not path.exists():
            print("\nHATA Girilen dosya bulunamadı")
            return None

        if not path.is_file():
            print("\nHATA Girilen yol dosya değil")
            return None

        if not path.suffix.lower() != ".csv":
            print("\nHATA Girilen dosya CSV uzantılı değil")
            return None

        print(f"\nSeçilen CSV dosyasi:\n{path.resolve()}")
        return path.resolve()