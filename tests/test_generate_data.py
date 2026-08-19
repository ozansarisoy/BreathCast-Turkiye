"""
test_generate_data.py
----------------------
src/generate_data.py içindeki saf fonksiyonlar için birim testleri.
Çalıştırma: pytest tests/ -v
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_data import (
    yogunluk_carpani, mevsim_carpani, mevsim_carpani_ayarli,
    pandemi_dalga_carpani, build_population_table,
)


class TestYogunlukCarpani:
    def test_bilinen_kategoriler_pozitif_carpan_dondurur(self):
        for kategori in ["dusuk", "orta", "yuksek"]:
            assert yogunluk_carpani(kategori) > 0

    def test_yuksek_yogunluk_dusuk_yogunluktan_buyuk_olmali(self):
        # Kalabalık iller (yüksek yoğunluk) daha yüksek çarpana sahip olmalı —
        # bu, projenin temel varsayımı (yoğun iller daha yüksek insidans gösterir)
        assert yogunluk_carpani("yuksek") > yogunluk_carpani("orta") > yogunluk_carpani("dusuk")

    def test_bilinmeyen_kategori_hata_verir(self):
        with pytest.raises(KeyError):
            yogunluk_carpani("bilinmeyen_kategori")


class TestMevsimCarpani:
    def test_kis_ayi_yaz_ayindan_yuksek_olmali(self):
        # Solunum yolu enfeksiyonları kışın artar, yazın azalır — temel mevsimsellik varsayımı
        ocak = mevsim_carpani(pd.Timestamp("2021-01-15"))
        temmuz = mevsim_carpani(pd.Timestamp("2021-07-15"))
        assert ocak > temmuz

    def test_tum_aylar_pozitif_carpan_dondurur(self):
        for ay in range(1, 13):
            tarih = pd.Timestamp(f"2021-{ay:02d}-15")
            assert mevsim_carpani(tarih) > 0


class TestMevsimCarpaniAyarli:
    def test_guc_sifirken_carpan_bire_yakinsar(self):
        # gucu=0 -> mevsimsellik etkisi tamamen nötrlenmeli (çarpan ~1.0)
        tarih = pd.Timestamp("2021-01-15")  # normalde güçlü kış etkisi olan bir tarih
        sonuc = mevsim_carpani_ayarli(tarih, gucu=0.0)
        assert sonuc == pytest.approx(1.0, abs=0.01)

    def test_guc_birken_orijinal_carpanla_ayni(self):
        tarih = pd.Timestamp("2021-01-15")
        beklenen = mevsim_carpani(tarih)
        sonuc = mevsim_carpani_ayarli(tarih, gucu=1.0)
        assert sonuc == pytest.approx(beklenen, abs=0.001)

    def test_sonuc_hicbir_zaman_negatif_olmamali(self):
        # Üretilen vaka oranları negatif olamaz; bu çarpan da mantıksal olarak negatif dönmemeli
        for ay in range(1, 13):
            tarih = pd.Timestamp(f"2021-{ay:02d}-15")
            for gucu in [0.0, 0.5, 1.0, 1.5]:
                assert mevsim_carpani_ayarli(tarih, gucu) >= 0


class TestPandemiDalgaCarpani:
    def test_dalga_tarihinde_carpan_uzaktan_yuksek_olmali(self):
        # Bilinen bir COVID dalga zirvesi (Ocak 2022) civarındaki çarpan,
        # dalgadan çok uzak bir tarihten (örn. 2023 yazı) yüksek olmalı
        dalga_civari = pandemi_dalga_carpani(pd.Timestamp("2022-01-10"))
        dalgadan_uzak = pandemi_dalga_carpani(pd.Timestamp("2023-07-01"))
        assert dalga_civari > dalgadan_uzak

    def test_carpan_her_zaman_birden_buyuk_esittir(self):
        # weight = 1.0 + ... formülü gereği çarpan hiçbir zaman 1'in altına düşmemeli
        for ay_offset in range(0, 48, 3):
            tarih = pd.Timestamp("2020-01-06") + pd.Timedelta(weeks=ay_offset * 4)
            assert pandemi_dalga_carpani(tarih) >= 1.0


class TestBuildPopulationTable:
    def test_81_il_donmeli(self):
        pop = build_population_table()
        assert len(pop) == 81

    def test_beklenen_sutunlar_mevcut(self):
        pop = build_population_table()
        for sutun in ["il", "plaka", "bolge", "nufus_2022", "yogunluk_kategori"]:
            assert sutun in pop.columns

    def test_il_isimleri_tekil_olmali(self):
        pop = build_population_table()
        assert pop["il"].duplicated().sum() == 0

    def test_nufus_degerleri_pozitif_olmali(self):
        pop = build_population_table()
        assert (pop["nufus_2022"] > 0).all()

    def test_toplam_nufus_gercekci_araliktadir(self):
        # Türkiye'nin gerçek 2022 nüfusu ~85.3 milyon (TÜİK ADNKS) — makul bir toleransla kontrol
        pop = build_population_table()
        toplam = pop["nufus_2022"].sum()
        assert 80_000_000 < toplam < 90_000_000
