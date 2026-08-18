import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Türkiye Solunum Yolu Enfeksiyon İzleme Paneli",
                    layout="wide", page_icon="🫁")


@st.cache_data
def load_data():
    df = pd.read_parquet(BASE / "data" / "processed" / "model_ready.parquet")
    df["tarih"] = pd.to_datetime(df["tarih"])
    return df


@st.cache_data
def load_model_comparison():
    return pd.read_csv(BASE / "models" / "model_karsilastirma.csv", index_col=0)


@st.cache_resource
def load_lgbm():
    obj = joblib.load(BASE / "models" / "lightgbm_model.pkl")
    return obj["model"], obj["features"], obj["hastalik_grubu_kod_map"]


df = load_data()
model, feats, kod_map = load_lgbm()
grup_to_kod = {v: k for k, v in kod_map.items()}

st.title("🫁 Türkiye Solunum Yolu ve Bulaşıcı Hastalık İzleme Paneli")
st.caption(
    "⚠️ Bu panelde kullanılan il bazlı zaman serisi, T.C. Sağlık Bakanlığı'nın "
    "yalnızca birkaç tarihte açıkladığı gerçek insidans değerleri ile TÜİK nüfus "
    "verisi kalibre edilerek üretilmiş bir **proxy veri setidir** (bkz. Metodoloji "
    "sekmesi). Gerçek zamanlı klinik karar için kullanılmamalıdır."
)

# ---------------- Sidebar filtreler ----------------
st.sidebar.header("Filtreler")
hastalik_listesi = sorted(df["hastalik_grubu"].unique())
secili_hastalik = st.sidebar.selectbox("Hastalık grubu", hastalik_listesi,
                                         index=hastalik_listesi.index("ÜSYE"))
il_listesi = sorted(df["il"].unique())
secili_il = st.sidebar.selectbox("İl seçin", il_listesi, index=il_listesi.index("İstanbul"))
tarih_min, tarih_max = df["tarih"].min(), df["tarih"].max()
tarih_araligi = st.sidebar.date_input("Tarih aralığı", [tarih_min, tarih_max],
                                        min_value=tarih_min, max_value=tarih_max)
bolge_secim = st.sidebar.multiselect("Bölge (harita için)", sorted(df["bolge"].unique()),
                                       default=sorted(df["bolge"].unique()))

df_h = df[df["hastalik_grubu"] == secili_hastalik]
il_df = df_h[df_h["il"] == secili_il].copy()
if len(tarih_araligi) == 2:
    il_df = il_df[(il_df["tarih"] >= pd.Timestamp(tarih_araligi[0])) &
                  (il_df["tarih"] <= pd.Timestamp(tarih_araligi[1]))]

tab1, tab2, tab3, tab4 = st.tabs(["📊 Genel Bakış", "🗺️ Coğrafi Dağılım", "🔮 Tahmin", "📎 Metodoloji"])

# ---------------- TAB 1: Genel Bakış ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Seçili il", secili_il)
    c2.metric("Son hafta 100binde vaka", f"{il_df['vaka_100bin'].iloc[-1]:.1f}")
    degisim = il_df['vaka_100bin'].iloc[-1] - il_df['vaka_100bin'].iloc[-5]
    c3.metric("4 haftalık değişim", f"{degisim:+.1f}")

    fig = px.line(il_df, x="tarih", y="vaka_100bin",
                   title=f"{secili_il} — {secili_hastalik} haftalık 100binde vaka oranı",
                   labels={"vaka_100bin": "100binde vaka", "tarih": "Tarih"})
    fig.add_scatter(x=il_df["tarih"], y=il_df["roll_mean_4"], mode="lines",
                     name="4 haftalık hareketli ortalama", line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mevsimsellik")
    mevsim_ort = il_df.groupby("mevsim")["vaka_100bin"].mean().reindex(
        ["Kış", "İlkbahar", "Yaz", "Sonbahar"])
    st.bar_chart(mevsim_ort)

    st.subheader(f"{secili_il} — Tüm hastalık gruplarının karşılaştırması")
    tum_gruplar_il = df[df["il"] == secili_il]
    fig_cmp = px.line(tum_gruplar_il, x="tarih", y="vaka_100bin", color="hastalik_grubu",
                        title="Hastalık grupları arası karşılaştırma (aynı il)")
    st.plotly_chart(fig_cmp, use_container_width=True)

# ---------------- TAB 2: Coğrafi Dağılım ----------------
with tab2:
    son_tarih = df_h["tarih"].max()
    harita_df = df_h[(df_h["tarih"] == son_tarih) & (df_h["bolge"].isin(bolge_secim))]
    st.subheader(f"{secili_hastalik} — Son hafta ({son_tarih.date()}) il bazlı 100binde vaka")
    fig2 = px.bar(harita_df.sort_values("vaka_100bin", ascending=False).head(20),
                   x="il", y="vaka_100bin", color="bolge",
                   title="En yüksek 20 il — 100binde vaka")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(harita_df[["il", "bolge", "vaka_100bin", "tahmini_vaka_sayisi"]]
                 .sort_values("vaka_100bin", ascending=False), use_container_width=True)
    st.caption("Not: gerçek coğrafi harita (choropleth) için Türkiye il sınırları "
               "GeoJSON dosyası eklenip `px.choropleth_mapbox` kullanılabilir; "
               "bu iskelette tablo/bar grafik ile gösterilmiştir.")

# ---------------- TAB 3: Tahmin ----------------
with tab3:
    st.subheader(f"{secili_il} — {secili_hastalik} için LightGBM tahmini")
    ufuk = st.slider("Kaç hafta ileri tahmin edilsin?", 1, 8, 4)
    grup_kod = grup_to_kod[secili_hastalik]

    tahminler = []
    calisma = df_h[df_h["il"] == secili_il].sort_values("tarih").copy()
    for i in range(ufuk):
        son_satir = calisma.iloc[-1]
        yeni_tarih = son_satir["tarih"] + pd.Timedelta(weeks=1)
        girdi = pd.DataFrame([{
            "lag_1": son_satir["vaka_100bin"],
            "lag_2": calisma.iloc[-2]["vaka_100bin"] if len(calisma) > 1 else son_satir["vaka_100bin"],
            "lag_4": calisma.iloc[-4]["vaka_100bin"] if len(calisma) > 3 else son_satir["vaka_100bin"],
            "lag_52": calisma.iloc[-52]["vaka_100bin"] if len(calisma) > 51 else son_satir["vaka_100bin"],
            "roll_mean_4": calisma["vaka_100bin"].tail(4).mean(),
            "roll_std_4": calisma["vaka_100bin"].tail(4).std(),
            "ay": yeni_tarih.month, "hafta": yeni_tarih.isocalendar().week,
            "nufus_2022": son_satir["nufus_2022"],
            "hastalik_grubu_kod": grup_kod,
        }])
        tahmin_deger = max(0, model.predict(girdi[feats])[0])
        tahminler.append({"tarih": yeni_tarih, "vaka_100bin": tahmin_deger, "tip": "Tahmin"})
        yeni_satir = son_satir.copy()
        yeni_satir["tarih"], yeni_satir["vaka_100bin"] = yeni_tarih, tahmin_deger
        calisma = pd.concat([calisma, pd.DataFrame([yeni_satir])], ignore_index=True)

    gecmis = il_df[["tarih", "vaka_100bin"]].tail(20).copy()
    gecmis["tip"] = "Gerçekleşen"
    birlesik = pd.concat([gecmis, pd.DataFrame(tahminler)], ignore_index=True)
    fig3 = px.line(birlesik, x="tarih", y="vaka_100bin", color="tip",
                    title=f"{secili_il} — Son 20 hafta + {ufuk} haftalık tahmin")
    st.plotly_chart(fig3, use_container_width=True)

    cmp = load_model_comparison()
    st.subheader("Model karşılaştırması (hastalık grubu bazında, Türkiye geneli, son 12 hafta test)")
    st.dataframe(cmp, use_container_width=True)
    st.caption("Her hastalık grubu için SARIMA ve Prophet ayrı ayrı denendi (klasik zaman serisi "
               "yaklaşımı). Üretim/tahmin sekmesinde ise tüm il+hastalık grubu kombinasyonlarını "
               "birlikte öğrenen tek bir havuzlanmış (pooled) LightGBM modeli kullanılıyor — "
               "bu sayede az veri olan il-hastalık kombinasyonları da diğerlerinden öğrenebiliyor.")

# ---------------- TAB 4: Metodoloji ----------------
with tab4:
    st.subheader("Veri ve Yöntem Notu")
    st.markdown("""
**Kapsanan hastalık grupları:** COVID-19, İnfluenza/Grip benzeri hastalık (ILI),
Üst Solunum Yolu Enfeksiyonu (ÜSYE), Alt Solunum Yolu Enfeksiyonu/Pnömoni,
Tüberküloz (verem).

**Veri kaynağı ve kısıt:** T.C. Sağlık Bakanlığı, il bazlı vaka verisini sürekli
bir açık veri/CSV/API olarak yayımlamamıştır. Ancak gerçek çapa noktaları mevcuttur:
- **COVID-19:** il-bazlı "100 binde vaka" basın açıklamaları (1 Nisan 2020,
  12 Eylül 2020, 24-30 Nisan 2021, 8-14 Ocak 2022)
- **Tüberküloz:** Sağlık Bakanlığı Verem Savaşı Dairesi Başkanlığı'nın yıllık
  ULUSAL insidans serisi (2020: 100binde 10.6 · 2022: 11.0 · 2024: 10.4) —
  hsgm.saglik.gov.tr Verem Savaşı Raporları
- **Grip/ÜSYE/Alt solunum yolu:** WHO/ECDC'nin bilinen Kuzey Yarımküre solunum
  yolu enfeksiyonu mevsimsellik paterni (kış artışı, yaz düşüşü) referans alındı

**Bu projede izlenen yöntem:**
1. TÜİK il nüfus verisi (2022) temel alındı.
2. Yukarıdaki gerçek noktalar **çapa** olarak sabitlendi; Tüberküloz için ulusal
   yıllık değer, il nüfus/yoğunluk ağırlığıyla il bazına dağıtıldı.
3. Her hastalık grubu için ayrı mevsimsellik gücü ve il-yoğunluk hassasiyeti
   katsayısı tanımlandı (örn. ÜSYE her ilde yaygındır → düşük hassasiyet;
   COVID-19 kalabalık illerde daha belirgin yayılmıştır → yüksek hassasiyet).
4. Üretilen veri `kaynak` sütununda `gercek_ceapa` / `proxy_kalibreli` /
   `interpole` olarak etiketlenmiştir — şeffaflık için.

**Gerçek veriyle değiştirme:** Aynı şema (`il, tarih, vaka_100bin`) ile
`data/raw/` altına CSV eklenip `src/clean_data.py` + `src/train_models.py`
yeniden çalıştırılarak tüm pipeline gerçek veriyle yeniden üretilebilir.
""")
    st.dataframe(df[["il", "tarih", "vaka_100bin", "kaynak"]].sample(10), use_container_width=True)
