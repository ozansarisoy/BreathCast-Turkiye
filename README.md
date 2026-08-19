# Türkiye Solunum Yolu ve Bulaşıcı Hastalık İzleme Paneli

Veri Bilimi bitirme projesi — Türkiye'de il bazlı, **beş solunum yolu/bulaşıcı
hastalık grubunun** (yalnızca COVID-19 değil) zaman serisi analizi, tahmini ve
interaktif Streamlit paneli.

## 1. Proje Özeti

Bu proje, Türkiye'nin 81 ilinde aşağıdaki hastalık gruplarının haftalık
insidansını (100 binde vaka) modelleyen uçtan uca bir veri bilimi pipeline'ıdır:

- COVID-19
- İnfluenza / Grip benzeri hastalık (ILI)
- Üst Solunum Yolu Enfeksiyonu (ÜSYE)
- Alt Solunum Yolu Enfeksiyonu / Pnömoni (bronşit, zatürre)
- Tüberküloz (verem)

`veri toplama → temizleme → özellik mühendisliği → zaman serisi modelleme →
model karşılaştırma → Streamlit dashboard`

## 2. Veri ve Metodolojik Not (ÖNEMLİ — jüri sunumunda mutlaka açıklayın)

T.C. Sağlık Bakanlığı hiçbir hastalık grubu için il bazlı vaka verisini sürekli
açık veri olarak yayımlamamıştır. Proje, TÜİK il nüfus verisi + gerçek
çapa noktaları + bilinen mevsimsellik/dalga paternleri kullanılarak kalibre
edilmiş bir **proxy veri seti** ile çalışır:

| Hastalık grubu | Gerçek çapa kaynağı |
|---|---|
| COVID-19 | Sağlık Bakanlığı'nın il-bazlı basın açıklamalı insidans haritaları (1 Nis 2020, 12 Eyl 2020, 24-30 Nis 2021, 8-14 Oca 2022) |
| Tüberküloz | Verem Savaşı Dairesi Başkanlığı yıllık ULUSAL insidansı (2020: 10.6, 2022: 11.0, 2024: 10.4 / 100bin) — il'e nüfus/yoğunlukla dağıtıldı |
| Grip/ILI, ÜSYE, Alt Solunum Yolu | WHO/ECDC'nin bilinen Kuzey Yarımküre solunum yolu enfeksiyonu mevsimsellik paterni (kış zirvesi, yaz düşüşü) |

Bu, veri kısıtı altında gerçekçi bir metodolojik tercihtir ve
`data/raw/il_bazli_solunum_haftalik.csv` içindeki `kaynak` sütununda şeffaf
şekilde etiketlenmiştir (`gercek_ceapa` / `proxy_kalibreli` / `interpole`).

Gerçek veri elde edilirse (örn. bir kurumdan / TÜİK özel talep yoluyla), aynı
şema (`il, hastalik_grubu, tarih, vaka_100bin`) korunarak `data/raw/` altına
konup pipeline yeniden çalıştırılabilir.

## 3. Klasör Yapısı

```
turkiye-solunum-projesi/
├── data/
│   ├── raw/                  # ham veri (nüfus + üretilen zaman serisi)
│   └── processed/            # temizlenmiş, özellik mühendisliği yapılmış veri
├── src/
│   ├── generate_data.py      # proxy veri üretimi
│   ├── clean_data.py         # temizleme + lag/rolling özellikleri
│   └── train_models.py       # SARIMA / Prophet / LightGBM eğitim ve karşılaştırma
├── models/                    # eğitilmiş model + karşılaştırma tablosu
├── streamlit_app/
│   └── app.py                 # 4 sekmeli interaktif dashboard
├── requirements.txt
└── README.md
```

## 4. Kurulum ve Çalıştırma

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py     # ham veri üret
python src/clean_data.py        # temizle + özellik üret
python src/train_models.py      # modelleri eğit, karşılaştır, kaydet

streamlit run streamlit_app/app.py
```

## 5. Yöntem Detayları

**Özellik mühendisliği:** il + hastalık grubu bazında lag_1, lag_2, lag_4,
lag_52 (geçen yıl aynı hafta), 4 haftalık hareketli ortalama/std, takvim
özellikleri (ay, hafta, mevsim), nüfus, hastalık grubu kodu.

**Modeller ve karşılaştırma (hastalık grubu bazında, Türkiye geneli nüfus
ağırlıklı seri, son 12 hafta test — `models/model_karsilastirma.csv`):**

| Hastalık grubu | SARIMA RMSE | Prophet RMSE |
|---|---|---|
| COVID-19 | ~2.74 | ~4.38 |
| Grip/ILI | ~0.09 | ~0.13 |
| ÜSYE | ~0.17 | ~0.17 |
| Alt Solunum Yolu/Pnömoni | ~0.078 | ~0.075 |
| Tüberküloz | ~0.018 | ~0.012 |

Havuzlanmış (pooled) **LightGBM** — tüm il + hastalık grubu kombinasyonlarını
tek modelde birlikte öğrenir, hastalık grubu kategorik değişken olarak eklenir:
**RMSE ≈ 1.48, MAE ≈ 0.31** (tüm gruplar birlikte değerlendirildiğinde).
Üretim/tahmin sekmesinde bu tek model kullanılır çünkü (a) az örnekli
il-hastalık kombinasyonları diğerlerinden öğrenebilir, (b) tek bir model
dosyasıyla 405 (81 il × 5 grup) seri yönetilebilir.

**Neden bu üçü?** SARIMA klasik/yorumlanabilir baseline, Prophet mevsimsel
trend ayrıştırması güçlü ve az parametre ayarı gerektiren pratik bir seçenek,
LightGBM ise çoklu il+hastalık verisini ortak öğrenip genelleme yapabilen ve
dış değişken (nüfus, takvim, hastalık grubu) ekleyebilen esnek bir yaklaşım
olduğu için seçildi.

## 6. Streamlit Panel Yapısı

Sidebar'da **hastalık grubu** ve **il** filtreleri birlikte çalışır:

- **Genel Bakış:** seçili il+hastalık için zaman serisi + hareketli ortalama +
  mevsimsellik grafiği + aynı ilde 5 hastalık grubunun karşılaştırması
- **Coğrafi Dağılım:** seçili hastalık için son haftanın il/bölge bazlı sıralaması (bar + tablo)
- **Tahmin:** LightGBM ile seçili il+hastalık için 1-8 haftalık ileriye dönük
  tahmin, hastalık grubu bazında model karşılaştırma tablosu
- **Metodoloji:** her hastalık grubu için veri kısıtı ve kalibrasyon yönteminin şeffaf açıklaması

Performans için `@st.cache_data` (veri) ve `@st.cache_resource` (model) kullanıldı.

## 7. Geliştirme Fikirleri (rapora "gelecek çalışmalar" olarak eklenebilir)

- Gerçek il-bazlı veri temin edilirse choropleth harita (geopandas + GeoJSON)
- Isolation Forest ile anomali/erken uyarı sekmesi
- K-Means ile benzer hastalık profiline sahip illerin kümelenmesi
- Hava kirliliği / sıcaklık gibi dış değişkenlerin modele eklenmesi

## 8. Testler

```bash
pip install pytest
pytest tests/ -v
```

`tests/` klasörü, `src/generate_data.py` ve `src/clean_data.py` içindeki temel
fonksiyonlar için 22 birim testi içerir (mevsimsellik, yoğunluk çarpanları,
veri temizleme, lag özellikleri, nüfus birleştirme vb.). Bu testler bir kez
gerçek bir hatayı da yakaladı: `mevsim_carpani_ayarli()` yüksek mevsimsellik
gücü (gucu) değerlerinde teorik olarak negatif çarpan üretebiliyordu — testler
bunu bulup düzeltilmesini sağladı (bkz. Bölüm 10).

## 9. Deployment (Streamlit Community Cloud)

1. Bu klasörü GitHub reposu olarak push edin (bkz. aşağıdaki Git adımları)
2. https://share.streamlit.io adresinde GitHub hesabınızla giriş yapın
3. "New app" → repo/branch seçin → main file path: `streamlit_app/app.py`
4. Deploy — birkaç dakika içinde canlı URL alırsınız

```bash
git init
git add .
git commit -m "Türkiye solunum yolu enfeksiyon izleme paneli - ilk sürüm"
git branch -M main
git remote add origin <SENIN_GITHUB_REPO_URL>
git push -u origin main
```

## 9. Sınırlılıklar

- Veri proxy/kalibreli olduğu için mutlak sayılar gerçek epidemiyolojik
  değerler olarak sunulmamalı, yalnızca metodolojik gösterim amaçlı kabul
  edilmelidir.
- LightGBM tahminleri kısa ufuklarda (1-4 hafta) daha güvenilirdir; ufuk
  uzadıkça hata birikimi artar (özyinelemeli tahmin yapısı nedeniyle).
- Bu proxy veri "bildirilen/kayıtlı vaka" tarzı oranları taklit eder; ÜSYE gibi
  toplumda çok sık görülen ama nadiren sağlık kuruluşuna bildirilen
  hastalıkların gerçek epidemiyolojik insidansı (yüz binde on binler) çok
  daha yüksek olabilir. Model, sürveyans/bildirim ölçeğini yansıtır.

## 10. Bilinen Düzeltme Kaydı

İlk sürümde `src/generate_data.py` içinde bir birim dönüştürme hatası vardı:
`taban` değeri "yıllık 100binde vaka" olarak tanımlanmıştı ancak haftalık
orana çevrilirken 52 yerine 12'ye bölünüyordu (aylık ölçek), bu da
gerçek çapa noktaları dışındaki tüm proxy haftalarda oranları ~4 kat
şişiriyordu. Aynı hata, tüberküloz için ulusal yıllık değerin tek bir haftaya
çapalanmasında da vardı. Her ikisi de 52'ye bölünecek şekilde düzeltildi ve
tüm pipeline (`generate_data.py → clean_data.py → train_models.py`) yeniden
çalıştırılıp doğrulandı — bu README'deki tüm sayılar düzeltilmiş sürümü
yansıtır.
