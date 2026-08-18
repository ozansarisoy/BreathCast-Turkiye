"""
train_models.py
----------------
Türkiye geneli + seçili iller için 3 farklı yaklaşımla zaman serisi tahmini
kurar ve karşılaştırır: SARIMA (statsmodels), Prophet (Meta), LightGBM (lag-feature).
Test seti: son 12 hafta. Metrik: RMSE, MAE.
En iyi model models/ altına kaydedilir (Streamlit'in kullanması için).
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from lightgbm import LGBMRegressor

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
TEST_WEEKS = 12
FOCUS_ILLER = ["İstanbul", "Ankara", "İzmir", "Gaziantep", "Van"]  # rapor için örnek iller


def load_data():
    return pd.read_parquet(PROCESSED_DIR / "model_ready.parquet")


def turkiye_geneli_seri(df: pd.DataFrame, grup: str) -> pd.Series:
    """Nüfus-ağırlıklı Türkiye geneli haftalık oran (tek hastalık grubu için)."""
    sub = df[df["hastalik_grubu"] == grup]
    agg = sub.groupby("tarih").apply(
        lambda g: np.average(g["vaka_100bin"], weights=g["nufus_2022"])
    )
    agg.index = pd.to_datetime(agg.index)
    return agg.sort_index()


def evaluate(y_true, y_pred):
    return {
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
    }


def run_sarima(series: pd.Series):
    train, test = series[:-TEST_WEEKS], series[-TEST_WEEKS:]
    model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 52),
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    pred = fit.forecast(TEST_WEEKS)
    return fit, pred, evaluate(test.values, pred.values)


def run_prophet(series: pd.Series):
    from prophet import Prophet
    train, test = series[:-TEST_WEEKS], series[-TEST_WEEKS:]
    dfp = train.reset_index()
    dfp.columns = ["ds", "y"]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=TEST_WEEKS, freq="W-MON")
    fc = m.predict(future)
    pred = fc.set_index("ds")["yhat"].iloc[-TEST_WEEKS:]
    return m, pred, evaluate(test.values, pred.values)


def run_lightgbm(df: pd.DataFrame):
    """Tüm il + hastalık grubu kombinasyonları için ortak, lag-feature tabanlı
    tek bir model (havuzlanmış/pooled eğitim, hastalık grubu kategorik değişken)."""
    data = df.copy()
    data["hastalik_grubu_kod"] = data["hastalik_grubu"].astype("category").cat.codes
    feats = ["lag_1", "lag_2", "lag_4", "lag_52", "roll_mean_4", "roll_std_4",
              "ay", "hafta", "nufus_2022", "hastalik_grubu_kod"]
    data = data.dropna(subset=feats + ["vaka_100bin"]).copy()
    data = data.sort_values("tarih")
    cutoff = data["tarih"].max() - pd.Timedelta(weeks=TEST_WEEKS)
    train, test = data[data["tarih"] <= cutoff], data[data["tarih"] > cutoff]

    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, max_depth=6,
                            num_leaves=31, random_state=42, verbosity=-1)
    model.fit(train[feats], train["vaka_100bin"])
    pred = model.predict(test[feats])
    metrics = evaluate(test["vaka_100bin"].values, pred)

    # hastalık grubu -> kod eşlemesini kaydet (Streamlit'te kullanılacak)
    kod_map = dict(enumerate(data["hastalik_grubu"].astype("category").cat.categories))
    return model, feats, metrics, kod_map


def main():
    df = load_data()
    hastalik_gruplari = sorted(df["hastalik_grubu"].unique())

    print("== Hastalık grubu bazında model karşılaştırması (Türkiye geneli, nüfus ağırlıklı) ==")
    all_results = {}
    for grup in hastalik_gruplari:
        tr_series = turkiye_geneli_seri(df, grup)
        grup_sonuc = {}

        try:
            _, _, sarima_metrics = run_sarima(tr_series)
            grup_sonuc["SARIMA"] = sarima_metrics
        except Exception as e:
            print(f"[{grup}] SARIMA atlandı: {e}")

        try:
            _, _, prophet_metrics = run_prophet(tr_series)
            grup_sonuc["Prophet"] = prophet_metrics
        except Exception as e:
            print(f"[{grup}] Prophet atlandı: {e}")

        all_results[grup] = grup_sonuc
        print(f"[{grup}]", grup_sonuc)

    # Tek, havuzlanmış (pooled) LightGBM modeli — tüm hastalık gruplarını birlikte öğrenir
    lgbm_model, feats, lgbm_metrics, kod_map = run_lightgbm(df)
    print("\nLightGBM (havuzlanmış, tüm il+hastalık grupları):", lgbm_metrics)

    # Karşılaştırma tablosunu düzleştir
    rows = []
    for grup, sonuc in all_results.items():
        for model_adi, met in sonuc.items():
            rows.append({"hastalik_grubu": grup, "model": model_adi, **met})
    rows.append({"hastalik_grubu": "TÜMÜ (havuzlanmış)", "model": "LightGBM", **lgbm_metrics})
    cmp_df = pd.DataFrame(rows)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": lgbm_model, "features": feats, "hastalik_grubu_kod_map": kod_map},
                MODELS_DIR / "lightgbm_model.pkl")
    cmp_df.to_csv(MODELS_DIR / "model_karsilastirma.csv", index=False)
    print("\nModel karşılaştırma tablosu ve LightGBM modeli kaydedildi -> models/")
    print(cmp_df)


if __name__ == "__main__":
    main()
