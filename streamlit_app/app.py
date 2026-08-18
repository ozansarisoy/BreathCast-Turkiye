import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Türkiye Solunum Yolu Enfeksiyon İzleme Paneli / Turkey Respiratory Disease Dashboard",
                    layout="wide", page_icon="🫁")

# ============================================================
# ÇEVİRİLER / TRANSLATIONS
# ============================================================
TEXTS = {
    "tr": {
        "title": "🫁 Türkiye Solunum Yolu ve Bulaşıcı Hastalık İzleme Paneli",
        "warning": ("⚠️ Bu panelde kullanılan il bazlı zaman serisi, T.C. Sağlık Bakanlığı'nın "
                    "yalnızca birkaç tarihte açıkladığı gerçek insidans değerleri ile TÜİK nüfus "
                    "verisi kalibre edilerek üretilmiş bir **proxy veri setidir** (bkz. Metodoloji "
                    "sekmesi). Gerçek zamanlı klinik karar için kullanılmamalıdır."),
        "filters": "Filtreler",
        "disease_group": "Hastalık grubu",
        "select_province": "İl seçin",
        "date_range": "Tarih aralığı",
        "region_map": "Bölge (harita için)",
        "tab_overview": "📊 Genel Bakış",
        "tab_geo": "🗺️ Coğrafi Dağılım",
        "tab_forecast": "🔮 Tahmin",
        "tab_methodology": "📎 Metodoloji",
        "selected_province": "Seçili il",
        "last_week_rate": "Son hafta 100binde vaka",
        "change_4w": "4 haftalık değişim",
        "weekly_rate_title": "haftalık 100binde vaka oranı",
        "rolling_avg": "4 haftalık hareketli ortalama",
        "seasonality": "Mevsimsellik",
        "disease_profile": "Hastalık Profili (Radar)",
        "disease_profile_desc": "Seçili ilde her hastalık grubunun mevsim ortalamasına göre kıyaslanması",
        "disease_compare": "— Tüm hastalık gruplarının karşılaştırması",
        "compare_title": "Hastalık grupları arası karşılaştırma (aynı il)",
        "geo_subheader": "— Son hafta ({date}) il bazlı 100binde vaka",
        "geo_no_geojson": "GeoJSON dosyası bulunamadı, harita yerine bar grafik gösteriliyor.",
        "top20_title": "En yüksek 20 il — 100binde vaka",
        "download_csv": "📥 Bu tabloyu CSV olarak indir",
        "race_chart_title": "Aylık İl Sıralaması Animasyonu — En Yüksek 15 İl",
        "race_chart_desc": "Play tuşuna basarak zaman içinde illerin sıralamasının nasıl değiştiğini izleyin",
        "animated_map_title": "Animasyonlu Harita — Aylık Değişim ({year})",
        "animated_map_desc": "Play tuşuna basarak seçili yılda aylık coğrafi değişimi izleyin",
        "select_year": "Animasyon için yıl seçin",
        "forecast_subheader": "— için LightGBM tahmini",
        "forecast_horizon": "Kaç hafta ileri tahmin edilsin?",
        "actual": "Gerçekleşen",
        "forecast": "Tahmin",
        "forecast_chart_title": "Son 20 hafta + {n} haftalık tahmin",
        "model_comparison": "Model karşılaştırması (hastalık grubu bazında, Türkiye geneli, son 12 hafta test)",
        "model_comparison_caption": ("Her hastalık grubu için SARIMA ve Prophet ayrı ayrı denendi (klasik zaman serisi "
                    "yaklaşımı). Üretim/tahmin sekmesinde ise tüm il+hastalık grubu kombinasyonlarını "
                    "birlikte öğrenen tek bir havuzlanmış (pooled) LightGBM modeli kullanılıyor — "
                    "bu sayede az veri olan il-hastalık kombinasyonları da diğerlerinden öğrenebiliyor."),
        "shap_expander": "🔍 Model neye bakarak tahmin yapıyor? (SHAP özellik önemi)",
        "shap_title": "Özellik önemi (ortalama |SHAP| değeri)",
        "shap_caption": ("Çubuk ne kadar uzunsa, o özellik modelin tahminini o kadar güçlü etkiliyor. "
                    "Genelde lag_1 (bir önceki hafta) en baskın sinyal olur — bu, solunum yolu "
                    "enfeksiyonlarının hafta içinde yüksek otokorelasyona sahip olmasıyla tutarlıdır."),
        "shap_missing": "SHAP kütüphanesi yüklü değil. `pip install shap` ile kurup yeniden çalıştırabilirsiniz.",
        "methodology_header": "Veri ve Yöntem Notu",
        "methodology_md": """
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
""",
        "province": "il", "region": "bölge", "rate": "100binde vaka", "date": "tarih",
        "season_order": ["Kış", "İlkbahar", "Yaz", "Sonbahar"],
    },
    "en": {
        "title": "🫁 Turkey Respiratory & Infectious Disease Surveillance Dashboard",
        "warning": ("⚠️ The province-level time series used in this dashboard is a **calibrated proxy "
                    "dataset**, built from TurkStat population data and a handful of real incidence "
                    "values announced by Turkey's Ministry of Health (see the Methodology tab). "
                    "It should not be used for real-time clinical decisions."),
        "filters": "Filters",
        "disease_group": "Disease group",
        "select_province": "Select province",
        "date_range": "Date range",
        "region_map": "Region (for map)",
        "tab_overview": "📊 Overview",
        "tab_geo": "🗺️ Geographic Distribution",
        "tab_forecast": "🔮 Forecast",
        "tab_methodology": "📎 Methodology",
        "selected_province": "Selected province",
        "last_week_rate": "Last week per-100k rate",
        "change_4w": "4-week change",
        "weekly_rate_title": "weekly cases per 100,000",
        "rolling_avg": "4-week rolling average",
        "seasonality": "Seasonality",
        "disease_profile": "Disease Profile (Radar)",
        "disease_profile_desc": "Seasonal average comparison of every disease group for the selected province",
        "disease_compare": "— Comparison across all disease groups",
        "compare_title": "Comparison across disease groups (same province)",
        "geo_subheader": "— Last week ({date}) cases per 100k by province",
        "geo_no_geojson": "GeoJSON file not found, showing a bar chart instead of the map.",
        "top20_title": "Top 20 provinces — cases per 100k",
        "download_csv": "📥 Download this table as CSV",
        "race_chart_title": "Monthly Province Ranking Animation — Top 15 Provinces",
        "race_chart_desc": "Press play to watch how the province ranking changes over time",
        "animated_map_title": "Animated Map — Monthly Change ({year})",
        "animated_map_desc": "Press play to watch the monthly geographic evolution for the selected year",
        "select_year": "Select year for animation",
        "forecast_subheader": "— LightGBM forecast for",
        "forecast_horizon": "How many weeks ahead to forecast?",
        "actual": "Actual",
        "forecast": "Forecast",
        "forecast_chart_title": "Last 20 weeks + {n}-week forecast",
        "model_comparison": "Model comparison (by disease group, Turkey-wide, last-12-week test)",
        "model_comparison_caption": ("SARIMA and Prophet were tried separately for each disease group (classic "
                    "time series approach). The forecast tab uses a single pooled LightGBM model that "
                    "jointly learns all province + disease group combinations — this lets sparse "
                    "province-disease combinations borrow strength from others."),
        "shap_expander": "🔍 What is the model looking at? (SHAP feature importance)",
        "shap_title": "Feature importance (mean |SHAP| value)",
        "shap_caption": ("The longer the bar, the more strongly that feature drives the model's prediction. "
                    "lag_1 (previous week) is usually the dominant signal — consistent with respiratory "
                    "infections having high week-to-week autocorrelation."),
        "shap_missing": "The SHAP library isn't installed. Run `pip install shap` and restart to enable this.",
        "methodology_header": "Data & Methodology Note",
        "methodology_md": """
**Disease groups covered:** COVID-19, Influenza/Influenza-Like Illness (ILI),
Upper Respiratory Tract Infection (URTI), Lower Respiratory Tract Infection/Pneumonia,
Tuberculosis (TB).

**Data source & constraint:** Turkey's Ministry of Health has never published province-level
case data as continuous open data/CSV/API. However, real anchor points exist:
- **COVID-19:** province-level "cases per 100k" press releases (Apr 1 2020,
  Sep 12 2020, Apr 24-30 2021, Jan 8-14 2022)
- **Tuberculosis:** TB Control Department's annual NATIONAL incidence series
  (2020: 10.6 · 2022: 11.0 · 2024: 10.4 per 100k) — hsgm.saglik.gov.tr TB Reports
- **Influenza/URTI/Lower respiratory:** WHO/ECDC's known Northern Hemisphere
  respiratory infection seasonality pattern (winter rise, summer trough)

**Method followed in this project:**
1. TurkStat province population data (2022) was used as the base.
2. The real points above were fixed as **anchors**; for TB, the national annual
   value was distributed to provinces by population/density weighting.
3. A separate seasonality strength and province-density sensitivity coefficient
   was defined for each disease group (e.g. URTI is common everywhere → low
   sensitivity; COVID-19 spread more visibly in dense provinces → high sensitivity).
4. Generated data is tagged in the `source` column as `real_anchor` /
   `calibrated_proxy` / `interpolated` — for transparency.

**Substituting real data:** Using the same schema (`province, date, cases_per_100k`),
a CSV can be added under `data/raw/` and the pipeline (`src/clean_data.py` +
`src/train_models.py`) re-run to regenerate everything with real data.
""",
        "province": "province", "region": "region", "rate": "cases per 100k", "date": "date",
        "season_order": ["Winter", "Spring", "Summer", "Autumn"],
    },
}

DISEASE_LABELS_EN = {
    "COVID-19": "COVID-19",
    "Grip/ILI": "Influenza/ILI",
    "ÜSYE": "URTI",
    "Alt Solunum Yolu/Pnömoni": "Lower Resp. Tract/Pneumonia",
    "Tüberküloz": "Tuberculosis",
}
REGION_LABELS_EN = {
    "Marmara": "Marmara", "Ege": "Aegean", "Akdeniz": "Mediterranean",
    "İç Anadolu": "Central Anatolia", "Karadeniz": "Black Sea",
    "Doğu Anadolu": "Eastern Anatolia", "Güneydoğu Anadolu": "Southeastern Anatolia",
}
SEASON_LABELS_EN = {"Kış": "Winter", "İlkbahar": "Spring", "Yaz": "Summer", "Sonbahar": "Autumn"}


def disease_label(name, lang):
    return DISEASE_LABELS_EN.get(name, name) if lang == "en" else name


def region_label(name, lang):
    return REGION_LABELS_EN.get(name, name) if lang == "en" else name


def season_label(name, lang):
    return SEASON_LABELS_EN.get(name, name) if lang == "en" else name


# ============================================================
# VERİ YÜKLEME / DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_parquet(BASE / "data" / "processed" / "model_ready.parquet")
    df["tarih"] = pd.to_datetime(df["tarih"])
    return df


@st.cache_data
def load_model_comparison():
    return pd.read_csv(BASE / "models" / "model_karsilastirma.csv")


@st.cache_resource
def load_lgbm():
    obj = joblib.load(BASE / "models" / "lightgbm_model.pkl")
    return obj["model"], obj["features"], obj["hastalik_grubu_kod_map"]


@st.cache_data
def load_geojson():
    path = BASE / "data" / "raw" / "tr_cities.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        geo = json.load(f)
    for feat in geo["features"]:
        if feat["properties"]["name"] == "Afyon":
            feat["properties"]["name"] = "Afyonkarahisar"
    return geo


df = load_data()
model, feats, kod_map = load_lgbm()
grup_to_kod = {v: k for k, v in kod_map.items()}

# ============================================================
# SIDEBAR — DİL TOGGLE + FİLTRELER
# ============================================================
col_flag1, col_toggle, col_flag2 = st.sidebar.columns([1, 2, 1])
with col_flag1:
    st.markdown("<div style='font-weight:700; text-align:center; padding-top:6px; opacity:0.85;'>TR</div>",
                unsafe_allow_html=True)
with col_toggle:
    is_english = st.toggle("dil_toggle", value=False, label_visibility="collapsed", key="lang_toggle")
with col_flag2:
    st.markdown("<div style='font-weight:700; text-align:center; padding-top:6px; opacity:0.85;'>EN</div>",
                unsafe_allow_html=True)
lang = "en" if is_english else "tr"
T = TEXTS[lang]

st.title(T["title"])
st.caption(T["warning"])

st.sidebar.header(T["filters"])
hastalik_listesi = sorted(df["hastalik_grubu"].unique())
hastalik_display = {disease_label(h, lang): h for h in hastalik_listesi}
# key dile bağlı: dil değişince seçenek etiketleri değiştiği için widget'ın
# eski dildeki seçili değeri yeni seçeneklerle eşleşmeyebilir — bu yüzden
# dil değiştiğinde widget'ı sıfırdan oluşturuyoruz (aksi halde boş/hatalı görünür)
secili_hastalik_disp = st.sidebar.selectbox(
    T["disease_group"], list(hastalik_display.keys()),
    index=list(hastalik_display.keys()).index(disease_label("ÜSYE", lang))
    if disease_label("ÜSYE", lang) in hastalik_display else 0,
    key=f"disease_select_{lang}",
)
secili_hastalik = hastalik_display[secili_hastalik_disp]

il_listesi = sorted(df["il"].unique())
secili_il = st.sidebar.selectbox(T["select_province"], il_listesi,
                                   index=il_listesi.index("İstanbul"), key="province_select")

tarih_min, tarih_max = df["tarih"].min(), df["tarih"].max()
tarih_araligi = st.sidebar.date_input(T["date_range"], [tarih_min, tarih_max],
                                        min_value=tarih_min, max_value=tarih_max,
                                        key="date_range_select")

bolge_listesi = sorted(df["bolge"].unique())
bolge_display_map = {region_label(b, lang): b for b in bolge_listesi}
secili_bolge_disp = st.sidebar.multiselect(T["region_map"], list(bolge_display_map.keys()),
                                             default=list(bolge_display_map.keys()),
                                             key=f"region_multiselect_{lang}")
bolge_secim = [bolge_display_map[b] for b in secili_bolge_disp]
# Kullanıcı tüm bölge seçimini kaldırırsa boş kalmasın diye tüm bölgelere geri dön
if not bolge_secim:
    bolge_secim = bolge_listesi

df_h = df[df["hastalik_grubu"] == secili_hastalik]
il_df = df_h[df_h["il"] == secili_il].copy()
if len(tarih_araligi) == 2:
    il_df = il_df[(il_df["tarih"] >= pd.Timestamp(tarih_araligi[0])) &
                  (il_df["tarih"] <= pd.Timestamp(tarih_araligi[1]))]

# ---- Ek sidebar özellikleri ----
st.sidebar.divider()

# "En riskli il" artık seçili hastalık grubu VE seçili bölge filtresine göre hesaplanıyor
son_tarih_genel = df_h["tarih"].max()
df_h_bolge = df_h[df_h["bolge"].isin(bolge_secim)]
en_riskli_baslik = "En riskli il (bu hafta)" if lang == "tr" else "Highest-risk province (this week)"
if len(df_h_bolge):
    en_riskli = df_h_bolge[df_h_bolge["tarih"] == son_tarih_genel].nlargest(1, "vaka_100bin").iloc[0]
    rate_label = "100binde vaka" if lang == "tr" else "cases per 100k"
    st.sidebar.metric(en_riskli_baslik, en_riskli["il"])
    st.sidebar.caption(f"{rate_label}: **{en_riskli['vaka_100bin']:.2f}**")
else:
    st.sidebar.metric(en_riskli_baslik, "—")

esik_baslik = "Uyarı eşiği (100binde vaka)" if lang == "tr" else "Alert threshold (per 100k)"
# key hastalık grubuna bağlı: her hastalık grubunun değer aralığı çok farklı
# (COVID onlarca, ÜSYE binlerce olabilir) — aynı key kullanılırsa slider eski
# hastalığın aralığında/değerinde donuk kalır, bu yüzden grup değişince widget
# sıfırdan oluşturulup o grubun doğru varsayılan eşiğiyle (90. yüzdelik) başlar
esik_max = float(df_h["vaka_100bin"].max())
esik_varsayilan = float(df_h["vaka_100bin"].quantile(0.9))
esik_deger = st.sidebar.slider(esik_baslik, 0.0, esik_max, esik_varsayilan, step=max(esik_max / 100, 0.01),
                                 key=f"threshold_slider_{secili_hastalik}")
son_deger_secili_il = il_df["vaka_100bin"].iloc[-1] if len(il_df) else 0
if son_deger_secili_il >= esik_deger:
    uyari_msg = (f"⚠️ {secili_il}, eşiğin üzerinde ({son_deger_secili_il:.2f} ≥ {esik_deger:.2f})"
                 if lang == "tr" else
                 f"⚠️ {secili_il} is above the threshold ({son_deger_secili_il:.2f} ≥ {esik_deger:.2f})")
    st.sidebar.warning(uyari_msg)
else:
    tamam_msg = (f"✅ {secili_il}, eşiğin altında" if lang == "tr"
                 else f"✅ {secili_il} is below the threshold")
    st.sidebar.success(tamam_msg)

st.sidebar.divider()
if st.sidebar.button("🔄 " + ("Filtreleri sıfırla" if lang == "tr" else "Reset filters")):
    # Dinamik (dile/hastalığa bağlı) key'ler de dahil, ilgili tüm önekleri temizle
    onekler = ("disease_select_", "province_select", "date_range_select",
               "region_multiselect_", "threshold_slider_")
    for k in list(st.session_state.keys()):
        if k.startswith(onekler):
            del st.session_state[k]
    st.rerun()

with st.sidebar.expander("ℹ️ " + ("Hakkında" if lang == "tr" else "About")):
    if lang == "tr":
        st.markdown(
            "**BreathCast Türkiye**\n\n"
            "Veri Bilimi bitirme projesi — Türkiye'de il bazlı solunum yolu ve "
            "bulaşıcı hastalık sürveyansı ve tahminlemesi.\n\n"
            "[GitHub reposu](https://github.com/ozansarisoy/BreathCast-Turkiye)"
        )
    else:
        st.markdown(
            "**BreathCast Turkey**\n\n"
            "A data science capstone project — province-level respiratory & "
            "infectious disease surveillance and forecasting for Turkey.\n\n"
            "[GitHub repository](https://github.com/ozansarisoy/BreathCast-Turkiye)"
        )

tab1, tab2, tab3, tab4 = st.tabs([T["tab_overview"], T["tab_geo"], T["tab_forecast"], T["tab_methodology"]])

# ---------------- TAB 1: Genel Bakış / Overview ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric(T["selected_province"], secili_il)
    c2.metric(T["last_week_rate"], f"{il_df['vaka_100bin'].iloc[-1]:.1f}")
    degisim = il_df['vaka_100bin'].iloc[-1] - il_df['vaka_100bin'].iloc[-5]
    c3.metric(T["change_4w"], f"{degisim:+.1f}")

    fig = px.line(il_df, x="tarih", y="vaka_100bin",
                   title=f"{secili_il} — {secili_hastalik_disp} {T['weekly_rate_title']}",
                   labels={"vaka_100bin": T["rate"], "tarih": T["date"]})
    fig.add_scatter(x=il_df["tarih"], y=il_df["roll_mean_4"], mode="lines",
                     name=T["rolling_avg"], line=dict(dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader(T["seasonality"])
        mevsim_ort = il_df.groupby("mevsim")["vaka_100bin"].mean().reindex(
            ["Kış", "İlkbahar", "Yaz", "Sonbahar"])
        mevsim_ort.index = [season_label(m, lang) for m in mevsim_ort.index]
        st.bar_chart(mevsim_ort)

    with col_b:
        st.subheader(T["disease_profile"])
        st.caption(T["disease_profile_desc"])
        radar_data = df[df["il"] == secili_il].groupby("hastalik_grubu")["vaka_100bin"].mean()
        radar_data = radar_data.reindex(hastalik_listesi)
        radar_labels = [disease_label(h, lang) for h in hastalik_listesi]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_data.values, theta=radar_labels, fill="toself",
            name=secili_il, line_color="#0F6E6E",
        ))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                  showlegend=False, height=350,
                                  margin=dict(l=40, r=40, t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader(f"{secili_il} {T['disease_compare']}")
    tum_gruplar_il = df[df["il"] == secili_il].copy()
    tum_gruplar_il["hastalik_grubu_disp"] = tum_gruplar_il["hastalik_grubu"].apply(lambda h: disease_label(h, lang))
    fig_cmp = px.line(tum_gruplar_il, x="tarih", y="vaka_100bin", color="hastalik_grubu_disp",
                        title=T["compare_title"], labels={"vaka_100bin": T["rate"], "tarih": T["date"],
                                                             "hastalik_grubu_disp": T["disease_group"]})
    st.plotly_chart(fig_cmp, use_container_width=True)

# ---------------- TAB 2: Coğrafi Dağılım / Geographic ----------------
with tab2:
    son_tarih = df_h["tarih"].max()
    harita_df = df_h[(df_h["tarih"] == son_tarih) & (df_h["bolge"].isin(bolge_secim))].copy()
    harita_df["bolge_disp"] = harita_df["bolge"].apply(lambda b: region_label(b, lang))
    st.subheader(f"{secili_hastalik_disp} {T['geo_subheader'].format(date=son_tarih.date())}")

    geo = load_geojson()
    if geo is not None:
        fig_map = px.choropleth_map(
            harita_df, geojson=geo, locations="il", featureidkey="properties.name",
            color="vaka_100bin", color_continuous_scale="OrRd",
            hover_name="il", hover_data={"bolge_disp": True, "vaka_100bin": ":.2f"},
            labels={"vaka_100bin": T["rate"], "bolge_disp": T["region"]},
            map_style="white-bg", zoom=4.3, center={"lat": 39.0, "lon": 35.2}, opacity=0.9,
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=480)
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info(T["geo_no_geojson"])

    fig2 = px.bar(harita_df.sort_values("vaka_100bin", ascending=False).head(20),
                   x="il", y="vaka_100bin", color="bolge_disp",
                   title=T["top20_title"], labels={"vaka_100bin": T["rate"], "il": T["province"],
                                                     "bolge_disp": T["region"]})
    st.plotly_chart(fig2, use_container_width=True)

    tablo_df = harita_df[["il", "bolge_disp", "vaka_100bin", "tahmini_vaka_sayisi"]] \
        .rename(columns={"bolge_disp": T["region"]}) \
        .sort_values("vaka_100bin", ascending=False) \
        .reset_index(drop=True)
    tablo_df.index = tablo_df.index + 1

    tahmini_vaka_label = "Tahmini vaka sayısı" if lang == "tr" else "Estimated case count"
    st.dataframe(
        tablo_df,
        use_container_width=True,
        column_config={
            "il": st.column_config.TextColumn(T["province"].capitalize()),
            T["region"]: st.column_config.TextColumn(T["region"].capitalize()),
            "vaka_100bin": st.column_config.NumberColumn(
                T["rate"].capitalize(), format="%.2f"),
            "tahmini_vaka_sayisi": st.column_config.NumberColumn(
                tahmini_vaka_label, format="%d"),
        },
    )

    st.download_button(
        T["download_csv"],
        data=tablo_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{secili_hastalik}_{son_tarih.date()}_by_province.csv",
        mime="text/csv",
    )

    st.divider()

    # ---- Animasyonlu bar chart race: aylık il sıralaması ----
    st.subheader(T["race_chart_title"])
    st.caption(T["race_chart_desc"])

    aylik = df_h[df_h["bolge"].isin(bolge_secim)].copy()
    aylik["yil_ay"] = aylik["tarih"].dt.to_period("M").astype(str)
    aylik_agg = aylik.groupby(["yil_ay", "il", "bolge"], as_index=False)["vaka_100bin"].mean()
    aylik_agg["bolge_disp"] = aylik_agg["bolge"].apply(lambda b: region_label(b, lang))

    top_iller = set()
    for period in sorted(aylik_agg["yil_ay"].unique()):
        top_bu_ay = aylik_agg[aylik_agg["yil_ay"] == period].nlargest(15, "vaka_100bin")["il"]
        top_iller.update(top_bu_ay)
    race_df = aylik_agg[aylik_agg["il"].isin(top_iller)].sort_values(["yil_ay", "vaka_100bin"])

    fig_race = px.bar(
        race_df, x="vaka_100bin", y="il", color="bolge_disp", orientation="h",
        animation_frame="yil_ay", range_x=[0, race_df["vaka_100bin"].max() * 1.1],
        labels={"vaka_100bin": T["rate"], "il": T["province"], "bolge_disp": T["region"]},
        height=550,
    )
    fig_race.update_layout(yaxis=dict(categoryorder="total ascending"))
    if fig_race.layout.updatemenus:
        fig_race.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 600
    st.plotly_chart(fig_race, use_container_width=True)

    st.divider()

    # ---- Animasyonlu harita: aylık coğrafi değişim ----
    if geo is not None:
        st.subheader(T["select_year"])
        yil_listesi = sorted(df_h["tarih"].dt.year.unique())
        secili_yil = st.selectbox(T["select_year"], yil_listesi, index=len(yil_listesi) - 1,
                                    label_visibility="collapsed")

        st.subheader(T["animated_map_title"].format(year=secili_yil))
        st.caption(T["animated_map_desc"])

        yillik = df_h[df_h["tarih"].dt.year == secili_yil].copy()
        yillik["ay"] = yillik["tarih"].dt.month
        yillik_agg = yillik.groupby(["ay", "il"], as_index=False)["vaka_100bin"].mean()
        yillik_agg = yillik_agg.sort_values("ay")

        fig_anim_map = px.choropleth_map(
            yillik_agg, geojson=geo, locations="il", featureidkey="properties.name",
            color="vaka_100bin", color_continuous_scale="OrRd",
            animation_frame="ay", hover_name="il",
            range_color=[0, yillik_agg["vaka_100bin"].max()],
            labels={"vaka_100bin": T["rate"]},
            map_style="white-bg", zoom=4.3, center={"lat": 39.0, "lon": 35.2}, opacity=0.9,
        )
        fig_anim_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=500)
        if fig_anim_map.layout.updatemenus:
            fig_anim_map.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 700
        st.plotly_chart(fig_anim_map, use_container_width=True)

# ---------------- TAB 3: Tahmin / Forecast ----------------
with tab3:
    st.subheader(f"{secili_il} {T['forecast_subheader']} {secili_hastalik_disp}")
    ufuk = st.slider(T["forecast_horizon"], 1, 8, 4)
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
        tahminler.append({"tarih": yeni_tarih, "vaka_100bin": tahmin_deger, "tip": T["forecast"]})
        yeni_satir = son_satir.copy()
        yeni_satir["tarih"], yeni_satir["vaka_100bin"] = yeni_tarih, tahmin_deger
        calisma = pd.concat([calisma, pd.DataFrame([yeni_satir])], ignore_index=True)

    gecmis = il_df[["tarih", "vaka_100bin"]].tail(20).copy()
    gecmis["tip"] = T["actual"]
    birlesik = pd.concat([gecmis, pd.DataFrame(tahminler)], ignore_index=True)
    fig3 = px.line(birlesik, x="tarih", y="vaka_100bin", color="tip",
                    title=f"{secili_il} — {T['forecast_chart_title'].format(n=ufuk)}",
                    labels={"vaka_100bin": T["rate"], "tarih": T["date"], "tip": ""})
    st.plotly_chart(fig3, use_container_width=True)

    cmp = load_model_comparison()
    st.subheader(T["model_comparison"])
    cmp_disp = cmp.copy()
    cmp_disp["hastalik_grubu"] = cmp_disp["hastalik_grubu"].apply(lambda h: disease_label(h, lang))
    cmp_disp = cmp_disp.reset_index(drop=True)
    cmp_disp.index = cmp_disp.index + 1
    hastalik_grubu_baslik = "Hastalık grubu" if lang == "tr" else "Disease group"
    st.dataframe(
        cmp_disp,
        use_container_width=True,
        column_config={
            "hastalik_grubu": st.column_config.TextColumn(hastalik_grubu_baslik),
            "model": st.column_config.TextColumn("Model"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.3f"),
            "MAE": st.column_config.NumberColumn("MAE", format="%.3f"),
        },
    )
    st.caption(T["model_comparison_caption"])

    with st.expander(T["shap_expander"]):
        try:
            import shap
            df_shap = df.copy()
            df_shap["hastalik_grubu_kod"] = df_shap["hastalik_grubu"].map(grup_to_kod)
            ozellik_orneklemi = df_shap.dropna(subset=feats).sample(
                min(300, len(df_shap.dropna(subset=feats))), random_state=42)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(ozellik_orneklemi[feats])
            ortalama_etki = pd.DataFrame({
                "ozellik": feats,
                "ortalama_mutlak_shap": np.abs(shap_values).mean(axis=0),
            }).sort_values("ortalama_mutlak_shap", ascending=True)
            fig_shap = px.bar(ortalama_etki, x="ortalama_mutlak_shap", y="ozellik",
                                orientation="h", title=T["shap_title"])
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption(T["shap_caption"])
        except ImportError:
            st.info(T["shap_missing"])

# ---------------- TAB 4: Metodoloji / Methodology ----------------
with tab4:
    st.subheader(T["methodology_header"])
    st.markdown(T["methodology_md"])

    ornek_baslik = "Veri Örneği (10 rastgele satır)" if lang == "tr" else "Data Sample (10 random rows)"
    st.subheader(ornek_baslik)

    ornek_df = df[["il", "tarih", "vaka_100bin", "kaynak"]].sample(10, random_state=1) \
        .sort_values("tarih", ascending=False).reset_index(drop=True)
    ornek_df.index = ornek_df.index + 1

    kaynak_etiketleri = {
        "gercek_ceapa": "Gerçek çapa" if lang == "tr" else "Real anchor",
        "proxy_kalibreli": "Kalibreli proxy" if lang == "tr" else "Calibrated proxy",
        "interpole": "İnterpole" if lang == "tr" else "Interpolated",
    }
    ornek_df["kaynak"] = ornek_df["kaynak"].map(kaynak_etiketleri).fillna(ornek_df["kaynak"])

    st.dataframe(
        ornek_df,
        use_container_width=True,
        column_config={
            "il": st.column_config.TextColumn(T["province"].capitalize()),
            "tarih": st.column_config.DateColumn(T["date"].capitalize(), format="DD.MM.YYYY"),
            "vaka_100bin": st.column_config.NumberColumn(T["rate"].capitalize(), format="%.2f"),
            "kaynak": st.column_config.TextColumn("Kaynak" if lang == "tr" else "Source"),
        },
    )
