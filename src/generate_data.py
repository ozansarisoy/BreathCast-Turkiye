"""
generate_data.py
-----------------
AMAÇ: Türkiye'de il bazlı, haftalık çözünürlükte BİRDEN FAZLA solunum yolu/
bulaşıcı hastalık grubu için insidans verisi üretir:
  - COVID-19
  - İnfluenza/Grip benzeri hastalık (ILI)
  - Üst solunum yolu enfeksiyonu (ÜSYE) — nezle, farenjit, sinüzit vb.
  - Alt solunum yolu enfeksiyonu / pnömoni (bronşit, zatürre)
  - Tüberküloz (verem)

NEDEN SENTETİK/PROXY VERİ?
T.C. Sağlık Bakanlığı il bazlı haftalık vaka verisini hiçbir hastalık grubu için
sürekli bir açık veri/CSV/API olarak yayımlamadı. Ancak GERÇEK ulusal/il bazlı
çapa noktaları mevcut ve bunlar kullanıldı:
  - COVID-19: basın açıklamalı il-bazlı insidans haritaları (1 Nis 2020,
    12 Eyl 2020, 24-30 Nis 2021, 8-14 Oca 2022)
  - Tüberküloz: Sağlık Bakanlığı Verem Savaşı Dairesi Başkanlığı'nın yıllık
    ulusal insidans serisi — 2005: 100binde 29.4 → 2018: 14.1 → 2020: 10.6 →
    2022: 11 → 2024: 10.4 (kaynak: hsgm.saglik.gov.tr Verem Savaşı Raporları)
  - Grip/ÜSYE/ASYE: WHO/ECDC'nin bilinen Kuzey Yarımküre solunum yolu
    enfeksiyonu mevsimsellik paterni (Ekim-Mart artış, Haziran-Ağustos düşüş)

Bu GERÇEK noktalar + TÜİK il nüfusu + il yoğunluk/bölge katsayıları kullanılarak
metodolojik olarak kalibre edilmiş bir proxy zaman serisi üretilir. Bu,
bitirme projesi raporunda "Veri Kısıtları ve Metodoloji" bölümünde AÇIKÇA
belirtilmelidir.

Gerçek veriyle değiştirmek istersen: aynı şema (il, hastalik_grubu, tarih,
vaka_100bin) ile data/raw/ altına CSV bırakman yeterli.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# ---- Gerçek çapa noktaları: (tarih, il, hastalik_grubu, 100binde haftalık vaka) ----
ANCHORS = [
    # COVID-19 — basın açıklamalı il-bazlı insidans haritaları
    ("2020-09-12", "İstanbul", "COVID-19", 45.0), ("2020-09-12", "Ankara", "COVID-19", 28.0),
    ("2020-09-12", "İzmir", "COVID-19", 30.0),
    ("2021-04-27", "İstanbul", "COVID-19", 532.02), ("2021-04-27", "Ankara", "COVID-19", 361.52),
    ("2021-04-27", "İzmir", "COVID-19", 223.34), ("2021-04-27", "Kırklareli", "COVID-19", 498.70),
    ("2021-04-27", "Tekirdağ", "COVID-19", 489.89), ("2021-04-27", "Şırnak", "COVID-19", 60.0),
    ("2021-04-27", "Şanlıurfa", "COVID-19", 65.0), ("2021-04-27", "Mardin", "COVID-19", 70.0),
    ("2022-01-10", "İstanbul", "COVID-19", 1571.46), ("2022-01-10", "İzmir", "COVID-19", 714.29),
    ("2022-01-10", "Ankara", "COVID-19", 629.65),
    # Tüberküloz — yıllık ULUSAL insidans (81 ile nüfus/yoksulluk ağırlıklı dağıtılacak taban değer)
    ("2020-01-06", "Türkiye_ULUSAL", "Tüberküloz", 10.6),   # yıllık, haftalığa src'de bölünecek
    ("2022-01-03", "Türkiye_ULUSAL", "Tüberküloz", 11.0),
    ("2024-01-01", "Türkiye_ULUSAL", "Tüberküloz", 10.4),
]

HASTALIK_GRUPLARI = {
    # ad: (yıllık taban 100binde vaka, mevsimsellik_gucu 0-1, il_yogunluk_hassasiyeti)
    "COVID-19":            dict(taban=18.0, mevsim_gucu=1.0, yogunluk_hassasiyet=1.0, pandemi_etkisi=True),
    "Grip/ILI":            dict(taban=55.0, mevsim_gucu=1.35, yogunluk_hassasiyet=0.6, pandemi_etkisi=False),
    "ÜSYE":                dict(taban=140.0, mevsim_gucu=0.9, yogunluk_hassasiyet=0.4, pandemi_etkisi=False),
    "Alt Solunum Yolu/Pnömoni": dict(taban=32.0, mevsim_gucu=1.1, yogunluk_hassasiyet=0.7, pandemi_etkisi=False),
    "Tüberküloz":          dict(taban=10.6, mevsim_gucu=0.05, yogunluk_hassasiyet=0.5, pandemi_etkisi=False),
}

START = "2020-01-06"  # ilk Pazartesi
END = "2023-12-25"

def build_population_table():
    df = pd.read_csv(RAW_DIR / "il_nufus_bolge.csv")
    return df

def yogunluk_carpani(kategori: str) -> float:
    return {"yuksek": 1.35, "orta": 1.0, "dusuk": 0.72}[kategori]

def mevsim_carpani(hafta_tarihi: pd.Timestamp) -> float:
    """Solunum yolu enfeksiyonları Ekim-Mart arası yükselir, Haziran-Ağustos düşer."""
    ay = hafta_tarihi.month
    mevsim = {
        1: 1.55, 2: 1.45, 3: 1.15, 4: 0.85, 5: 0.55, 6: 0.35,
        7: 0.30, 8: 0.35, 9: 0.65, 10: 1.10, 11: 1.40, 12: 1.60,
    }[ay]
    return mevsim

def pandemi_dalga_carpani(hafta_tarihi: pd.Timestamp) -> float:
    """2020-2022 COVID dalgalarını kabaca yansıtan çarpan (bilinen dalga tarihleri)."""
    dates_weights = [
        ("2020-04-01", 3.0), ("2020-11-15", 4.5), ("2021-04-15", 5.5),
        ("2021-09-01", 3.0), ("2022-01-10", 8.0), ("2022-03-01", 2.5),
    ]
    t = hafta_tarihi
    weight = 1.0
    for ds, w in dates_weights:
        d = pd.Timestamp(ds)
        diff_weeks = abs((t - d).days) / 7
        weight += w * np.exp(-0.5 * (diff_weeks / 4) ** 2)
    return weight

def mevsim_carpani_ayarli(hafta_tarihi, gucu: float) -> float:
    """mevsim_carpani'ni 0-1 arası bir 'güç' katsayısıyla yumuşatır/güçlendirir.
    gucu=1 -> orijinal mevsimsellik, gucu<1 -> daha düz (TB gibi az mevsimsel), gucu>1 -> daha keskin (grip)."""
    base = mevsim_carpani(hafta_tarihi)
    return 1.0 + (base - 1.0) * gucu


def main():
    pop = build_population_table()
    weeks = pd.date_range(START, END, freq="W-MON")

    rows = []
    for grup, params in HASTALIK_GRUPLARI.items():
        for _, il_row in pop.iterrows():
            il = il_row["il"]
            yogunluk_etki = 1.0 + (yogunluk_carpani(il_row["yogunluk_kategori"]) - 1.0) * params["yogunluk_hassasiyet"]
            base_rate_haftalik = params["taban"] / 52.0  # yıllık 100binde taban -> haftalık ortalama
            il_noise_scale = RNG.uniform(0.12, 0.28)
            for w in weeks:
                m = mevsim_carpani_ayarli(w, params["mevsim_gucu"])
                p = pandemi_dalga_carpani(w) if params["pandemi_etkisi"] else 1.0
                # Tüberküloz için yıllar içinde hafif azalan ulusal trend (gerçek eğilim)
                trend = 1.0
                if grup == "Tüberküloz":
                    yil_ilerleme = (w - pd.Timestamp(START)).days / 365.25
                    trend = max(0.85, 1.0 - 0.02 * yil_ilerleme)  # yılda ~%2 azalış
                noise = RNG.normal(1.0, il_noise_scale)
                rate = max(0.0, base_rate_haftalik * m * p * trend * yogunluk_etki * noise)
                rows.append((il, grup, w.date().isoformat(), round(rate, 2)))

    df = pd.DataFrame(rows, columns=["il", "hastalik_grubu", "tarih", "vaka_100bin"])
    df["kaynak"] = "proxy_kalibreli"

    # ---- Gerçek çapa noktalarını uygula ----
    for tarih, il, grup, deger in ANCHORS:
        if il == "Türkiye_ULUSAL":
            # ulusal yıllık TB insidansını o yılın tüm illerine, il ağırlığıyla dağıt
            hedef_hafta = df.loc[(df["hastalik_grubu"] == grup) &
                                  (pd.to_datetime(df["tarih"]).dt.year == pd.Timestamp(tarih).year) &
                                  (pd.to_datetime(df["tarih"]) >= pd.Timestamp(tarih)) &
                                  (pd.to_datetime(df["tarih"]) < pd.Timestamp(tarih) + pd.Timedelta(weeks=4))]
            for il_adi in pop["il"]:
                idxs = hedef_hafta[hedef_hafta["il"] == il_adi].index
                if len(idxs):
                    df.loc[idxs[0], "vaka_100bin"] = deger / 52.0  # yıllık->haftalık gerçek çapa (rate ile tutarlı ölçek)
                    df.loc[idxs[0], "kaynak"] = "gercek_ceapa"
            continue
        mask = (df["il"] == il) & (df["hastalik_grubu"] == grup) & \
               (df["tarih"] == pd.Timestamp(tarih).date().isoformat())
        if mask.any():
            df.loc[mask, "vaka_100bin"] = deger
            df.loc[mask, "kaynak"] = "gercek_ceapa"
        else:
            alt_df = df[(df["il"] == il) & (df["hastalik_grubu"] == grup)]
            nearest = (pd.to_datetime(alt_df["tarih"]) - pd.Timestamp(tarih)).abs().idxmin()
            df.loc[nearest, "vaka_100bin"] = deger
            df.loc[nearest, "kaynak"] = "gercek_ceapa"

    out_path = RAW_DIR / "il_bazli_solunum_haftalik.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Yazıldı: {out_path} ({len(df)} satır, {df['il'].nunique()} il, "
          f"{df['hastalik_grubu'].nunique()} hastalık grubu, {df['tarih'].nunique()} hafta)")

if __name__ == "__main__":
    main()
