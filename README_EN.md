# Turkey Respiratory & Infectious Disease Surveillance Dashboard

Data Science capstone project — province-level time series analysis,
forecasting, and an interactive Streamlit dashboard for **five respiratory /
infectious disease groups** in Turkey (not just COVID-19).

## 1. Project Overview

This project is an end-to-end data science pipeline that models the weekly
incidence (cases per 100,000 population) of the following disease groups
across Turkey's 81 provinces:

- COVID-19
- Influenza / Influenza-Like Illness (ILI)
- Upper Respiratory Tract Infection (URTI)
- Lower Respiratory Tract Infection / Pneumonia (bronchitis, pneumonia)
- Tuberculosis (TB)

`data collection → cleaning → feature engineering → time series modeling →
model comparison → Streamlit dashboard`

## 2. Data & Methodology Note (IMPORTANT — always explain this in a review/defense)

Turkey's Ministry of Health has never published province-level case data as
continuous open data for any disease group. This project works with a
**calibrated proxy dataset**, built from TurkStat province population data +
real anchor points + known seasonality/wave patterns:

| Disease group | Real anchor source |
|---|---|
| COVID-19 | Ministry of Health's press-released province-level incidence maps (Apr 1 2020, Sep 12 2020, Apr 24-30 2021, Jan 8-14 2022) |
| Tuberculosis | TB Control Department's annual NATIONAL incidence (2020: 10.6, 2022: 11.0, 2024: 10.4 per 100,000) — distributed to provinces by population/density |
| Influenza/ILI, URTI, Lower Respiratory Tract | WHO/ECDC's known Northern Hemisphere respiratory infection seasonality pattern (winter peak, summer trough) |

This is a realistic methodological choice given the data constraint, and is
transparently labeled in the `source` (`kaynak`) column of
`data/raw/il_bazli_solunum_haftalik.csv`. Note that the tag values themselves
are stored in Turkish regardless of the dashboard's display language:
`gercek_capa` (= real anchor), `proxy_kalibreli` (= calibrated proxy),
`interpole` (= interpolated).

If real data becomes available (e.g. via an institutional request to
TurkStat), it can be substituted — the pipeline expects the actual Turkish
column names (`il` = province, `hastalik_grubu` = disease group, `tarih` =
date, `vaka_100bin` = cases per 100k, `kaynak` = source). A CSV with those
exact column headers can be dropped into `data/raw/` and the pipeline
re-run.

## 3. Folder Structure

```
turkiye-solunum-projesi/
├── data/
│   ├── raw/                  # raw data (population + generated time series)
│   └── processed/            # cleaned data with engineered features
├── src/
│   ├── generate_data.py      # proxy data generation
│   ├── clean_data.py         # cleaning + lag/rolling features
│   └── train_models.py       # SARIMA / Prophet / LightGBM training & comparison
├── models/                    # trained model + comparison table
├── streamlit_app/
│   └── app.py                 # 4-tab interactive dashboard
├── tests/                     # unit tests (pytest)
├── requirements.txt
└── README.md
```

## 4. Setup & Running

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/generate_data.py     # generate raw data
python src/clean_data.py        # clean + engineer features
python src/train_models.py      # train models, compare, save

streamlit run streamlit_app/app.py
```

## 5. Methodology Details

**Feature engineering:** per province + disease group — lag_1, lag_2, lag_4,
lag_52 (same week last year), 4-week rolling mean/std, calendar features
(month, week, season), population, disease group code.

**Models & comparison (per disease group, Turkey-wide population-weighted
series, last-12-week test — see `models/model_karsilastirma.csv`):**

| Disease group | SARIMA RMSE | Prophet RMSE |
|---|---|---|
| COVID-19 | ~2.74 | ~4.38 |
| Influenza/ILI | ~0.09 | ~0.13 |
| URTI | ~0.17 | ~0.17 |
| Lower Respiratory Tract/Pneumonia | ~0.078 | ~0.075 |
| Tuberculosis | ~0.018 | ~0.012 |

A pooled **LightGBM** model — trained jointly on all province + disease
group combinations, with disease group added as a categorical feature:
**RMSE ≈ 1.48, MAE ≈ 0.31** (evaluated across all groups together). This
single model is used in the production/forecast tab because (a) sparse
province-disease combinations can borrow strength from others, and (b) one
model file can serve all 405 (81 provinces × 5 groups) series.

**Why these three models?** SARIMA is a classic, interpretable baseline;
Prophet offers strong seasonal decomposition with minimal tuning; LightGBM
can jointly learn from multi-province, multi-disease data and generalize
while incorporating exogenous features (population, calendar, disease
group).

## 6. Streamlit Dashboard Structure

The sidebar's **disease group** and **province** filters work together:

- **Summary:** Turkey-wide KPI cards (national average, 4-week change,
  highest-risk province, provinces above the 90th percentile), a 26-week
  national trend chart, a by-disease-group comparison, and a z-score-based
  anomaly table (trailing 12-week window)
- **Overview:** time series + rolling average + seasonality chart for the
  selected province+disease, an optional multi-province comparison overlay,
  plus a comparison of all 5 disease groups within the same province
- **Geographic Distribution:** last week's province/region ranking for the
  selected disease (choropleth map + bar chart + table), plus animated
  monthly ranking and geographic-change views
- **Forecast:** LightGBM-based 1-8 week forward forecast for the selected
  province+disease, a model comparison table by disease group, and a SHAP
  feature-importance explainer
- **Methodology:** transparent explanation of the data constraint and
  calibration method for each disease group

`@st.cache_data` (data) and `@st.cache_resource` (model) are used for
performance.

### AI-Powered Quick Summary

The **"🤖 Quick AI Summary"** button on the Summary tab generates a 2-3
sentence natural-language summary of the current filtered view (national
average, 4-week change, highest-risk province, number of provinces above
threshold, anomaly count). The operation is always fast because only a
handful of already-computed numeric metrics — not raw data — are sent to
the model.

- **If an API key is configured** (see below), a real LLM call is made to
  Anthropic Claude.
- **If no API key is configured, or the call fails**, the app automatically
  and seamlessly falls back to a rule-based (template) summary. This means
  the feature works even in a deployment with no AI API budget or key.

**To add your own API key** (optional):
1. Locally: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   and fill in your real key (this file is excluded from Git via
   `.gitignore` and is never committed).
2. On Streamlit Community Cloud: add the line
   `ANTHROPIC_API_KEY = "sk-ant-..."` under your app's **Settings → Secrets**.

## 7. Future Work Ideas (can be added to a report as "future work")

- An anomaly/early-warning tab using Isolation Forest
- Clustering provinces with similar disease profiles using K-Means
- Adding exogenous variables such as air pollution or temperature to the model

## 8. Tests

```bash
pip install pytest
pytest tests/ -v
```

The `tests/` folder contains 22 unit tests for the core functions in
`src/generate_data.py` and `src/clean_data.py` (seasonality, density
multipliers, data cleaning, lag features, population merging, etc.). These
tests once caught a real bug: `mevsim_carpani_ayarli()` could theoretically
produce a negative multiplier at high seasonality-strength values — the
tests caught it and it was fixed (see Section 11).

## 9. Deployment (Streamlit Community Cloud)

1. Push this folder to a GitHub repository (see Git steps below)
2. Sign in with your GitHub account at https://share.streamlit.io
3. "New app" → select repo/branch → main file path: `streamlit_app/app.py`
4. Deploy — you'll have a live URL within a few minutes

```bash
git init
git add .
git commit -m "Turkey respiratory disease surveillance dashboard - initial release"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## 10. Limitations

- Since the data is proxy/calibrated, absolute numbers should not be
  presented as real epidemiological figures — treat them as a methodology
  demonstration only.
- LightGBM forecasts are more reliable over short horizons (1-4 weeks);
  error accumulates as the horizon grows (due to the recursive forecasting
  structure).
- This proxy data mimics "reported/registered case" style rates; the true
  epidemiological incidence of conditions like URTI, which are very common
  in the population but rarely reported to a health facility, could be
  substantially higher (tens of thousands per 100,000). The model reflects
  the surveillance/reporting scale, not total community incidence.

## 11. Known Fix Log

The initial version had a unit-conversion bug in `src/generate_data.py`: the
`taban` (baseline) value was defined as "annual cases per 100,000" but was
divided by 12 instead of 52 when converting to a weekly rate (monthly
scale), inflating rates by ~4x for all proxy weeks outside the real anchor
points. The same bug also affected how tuberculosis's annual national value
was anchored to a single week. Both were fixed to divide by 52.

A second bug was later found by the unit test suite: `mevsim_carpani_ayarli()`
could return a negative multiplier at high seasonality-strength values
(never triggered by the disease groups currently in use, but unsafe for
future additions) — fixed by clamping the result to a minimum of 0.

A third, silent bug was found in the anomaly-detection logic on the Summary
tab: the trailing window used to compute each province's baseline mean/std
was accidentally using the *entire* multi-year history instead of the
intended trailing 12 weeks, which could misclassify normal seasonal peaks
as anomalies — fixed to properly window to the last 12 weeks.

The full pipeline (`generate_data.py → clean_data.py → train_models.py`)
was re-run and re-validated after each fix — all numbers in this README
reflect the corrected version.
