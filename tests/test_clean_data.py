"""
test_clean_data.py
--------------------
src/clean_data.py içindeki temizleme mantığı için birim testleri.
Sentetik/küçük ölçekli örnek verilerle çalışır (gerçek veri setine bağımlı değil).
Çalıştırma: pytest tests/ -v
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clean_data import clean


@pytest.fixture
def sahte_pop():
    return pd.DataFrame({
        "il": ["TestİliA", "TestİliB"],
        "plaka": [1, 2],
        "bolge": ["Marmara", "Ege"],
        "nufus_2022": [1_000_000, 500_000],
        "yogunluk_kategori": ["yuksek", "orta"],
    })


@pytest.fixture
def sahte_cases():
    haftalar = pd.date_range("2022-01-03", periods=6, freq="W-MON")
    rows = []
    for il in ["TestİliA", "TestİliB"]:
        for grup in ["ÜSYE"]:
            for i, tarih in enumerate(haftalar):
                # Bilerek bir hafta eksik bırakıyoruz (TestİliA'nın 3. haftası) — interpolasyonu test etmek için
                if il == "TestİliA" and i == 2:
                    continue
                rows.append({
                    "il": il, "hastalik_grubu": grup, "tarih": tarih,
                    "vaka_100bin": 10.0 + i, "kaynak": "proxy_kalibreli",
                })
    return pd.DataFrame(rows)


class TestClean:
    def test_negatif_deger_varsa_hata_verir(self, sahte_cases, sahte_pop):
        bozuk = sahte_cases.copy()
        bozuk.loc[0, "vaka_100bin"] = -5.0
        with pytest.raises(AssertionError):
            clean(bozuk, sahte_pop)

    def test_tekrar_eden_satirlar_temizlenir(self, sahte_cases, sahte_pop):
        tekrarli = pd.concat([sahte_cases, sahte_cases.iloc[[0]]], ignore_index=True)
        sonuc = clean(tekrarli, sahte_pop)
        # her il+grup+tarih kombinasyonu tam olarak bir kez olmalı
        tekrar_sayisi = sonuc.duplicated(subset=["il", "hastalik_grubu", "tarih"]).sum()
        assert tekrar_sayisi == 0

    def test_eksik_hafta_interpole_edilir(self, sahte_cases, sahte_pop):
        sonuc = clean(sahte_cases, sahte_pop)
        # TestİliA orijinalde 5 satırdı (6 haftadan 1'i eksikti), temizleme sonrası 6 olmalı
        a_satirlari = sonuc[(sonuc["il"] == "TestİliA") & (sonuc["hastalik_grubu"] == "ÜSYE")]
        assert len(a_satirlari) == 6
        # interpole edilen satırın kaynak etiketi doğru olmalı
        assert "interpole" in a_satirlari["kaynak"].values

    def test_nufus_ile_birlestirme_dogru_calisir(self, sahte_cases, sahte_pop):
        sonuc = clean(sahte_cases, sahte_pop)
        assert "nufus_2022" in sonuc.columns
        assert sonuc[sonuc["il"] == "TestİliA"]["nufus_2022"].iloc[0] == 1_000_000

    def test_mutlak_vaka_sayisi_hesabi_dogru(self, sahte_cases, sahte_pop):
        sonuc = clean(sahte_cases, sahte_pop)
        satir = sonuc[(sonuc["il"] == "TestİliA") & (sonuc["vaka_100bin"] == 10.0)].iloc[0]
        beklenen = 10.0 / 100000 * 1_000_000
        assert satir["tahmini_vaka_sayisi"] == pytest.approx(beklenen, rel=0.01)

    def test_lag_ozellikleri_veri_sizintisi_yapmaz(self, sahte_cases, sahte_pop):
        # lag_1'in bir satırdaki değeri, BİR ÖNCEKİ satırın vaka_100bin'i olmalı — kendisi değil
        sonuc = clean(sahte_cases, sahte_pop).sort_values(["il", "hastalik_grubu", "tarih"])
        a = sonuc[(sonuc["il"] == "TestİliA") & (sonuc["hastalik_grubu"] == "ÜSYE")].reset_index(drop=True)
        for i in range(1, len(a)):
            assert a.loc[i, "lag_1"] == pytest.approx(a.loc[i - 1, "vaka_100bin"])

    def test_takvim_ozellikleri_dogru_uretilir(self, sahte_cases, sahte_pop):
        sonuc = clean(sahte_cases, sahte_pop)
        for sutun in ["yil", "hafta", "ay", "mevsim"]:
            assert sutun in sonuc.columns
        assert sonuc["mevsim"].isin(["Kış", "İlkbahar", "Yaz", "Sonbahar"]).all()
