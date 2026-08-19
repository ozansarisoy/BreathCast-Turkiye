"""
clean_data.py
-------------
Ham il-bazlı haftalık veriyi okur, temizler, nüfus/bölge bilgisiyle birleştirir,
lag ve rolling özellikleri üretir. Çıktı: data/processed/model_ready.parquet
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_raw():
    cases = pd.read_csv(RAW_DIR / "il_bazli_solunum_haftalik.csv", parse_dates=["tarih"])
    pop = pd.read_csv(RAW_DIR / "il_nufus_bolge.csv")
    return cases, pop


def clean(cases: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    # 1) temel doğrulama
    assert cases["vaka_100bin"].ge(0).all(), "Negatif vaka oranı bulundu"
    cases = cases.drop_duplicates(subset=["il", "hastalik_grubu", "tarih"])

    # 2) eksik hafta kontrolü (il + hastalık grubu bazında tam takvim garantisi)
    full_weeks = pd.date_range(cases["tarih"].min(), cases["tarih"].max(), freq="W-MON")
    completed = []
    for (il, grup), g in cases.groupby(["il", "hastalik_grubu"]):
        g = g.set_index("tarih").reindex(full_weeks)
        g["il"] = il
        g["hastalik_grubu"] = grup
        g["vaka_100bin"] = g["vaka_100bin"].interpolate(limit_direction="both")
        g["kaynak"] = g["kaynak"].fillna("interpole")
        completed.append(g.reset_index().rename(columns={"index": "tarih"}))
    cases = pd.concat(completed, ignore_index=True)

    # 3) nüfus/bölge ile birleştir
    df = cases.merge(pop, on="il", how="left")

    # 4) mutlak vaka sayısı (100binlik oranı gerçek nüfusa ölçekle)
    df["tahmini_vaka_sayisi"] = (df["vaka_100bin"] / 100000 * df["nufus_2022"]).round(0)

    # 5) takvim özellikleri
    df["yil"] = df["tarih"].dt.year
    df["hafta"] = df["tarih"].dt.isocalendar().week.astype(int)
    df["ay"] = df["tarih"].dt.month
    df["mevsim"] = df["ay"].map({12: "Kış", 1: "Kış", 2: "Kış", 3: "İlkbahar", 4: "İlkbahar",
                                   5: "İlkbahar", 6: "Yaz", 7: "Yaz", 8: "Yaz", 9: "Sonbahar",
                                   10: "Sonbahar", 11: "Sonbahar"})

    # 6) lag ve rolling özellikleri (il + hastalık grubu bazında, veri sızıntısı olmadan)
    df = df.sort_values(["il", "hastalik_grubu", "tarih"])
    grp_key = ["il", "hastalik_grubu"]
    for lag in [1, 2, 4, 52]:
        df[f"lag_{lag}"] = df.groupby(grp_key)["vaka_100bin"].shift(lag)
    df["roll_mean_4"] = df.groupby(grp_key)["vaka_100bin"].transform(lambda s: s.shift(1).rolling(4).mean())
    df["roll_std_4"] = df.groupby(grp_key)["vaka_100bin"].transform(lambda s: s.shift(1).rolling(4).std())

    return df


def main():
    cases, pop = load_raw()
    df = clean(cases, pop)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "model_ready.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(PROCESSED_DIR / "model_ready.csv", index=False, encoding="utf-8-sig")
    print(f"Yazıldı: {out} ({df.shape[0]} satır, {df.shape[1]} sütun)")
    print(df.isna().sum()[df.isna().sum() > 0])


if __name__ == "__main__":
    main()
