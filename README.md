# Retail Demand Forecasting & Market Basket Analysis

An end-to-end data science portfolio project analyzing historical wholesale and B2B e-commerce transactions from the [Online Retail II UCI Dataset](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) (2009–2011). This project combines **time-series demand forecasting**, **association rule mining (FP-Growth)**, and **commercial risk analytics** to support inventory planning, product bundling, and customer retention.

---

## Business Objectives

1. **Multi-Horizon Demand Forecasting:** Predict daily and weekly unit demand to optimize warehouse staffing schedules and macro inventory replenishment.
2. **Product Bundling & Cross-Selling:** Identify high-affinity item combinations within multi-item orders to design targeted product bundles.
3. **Commercial Risk Mitigation:** Quantify revenue leakage from product returns and identify high-value customer segments at risk of churn.

---

## Methodology & Pipeline

### 1. Data Cleaning & Preprocessing
* **Deduplication & Imputation:** Removed `34,355` exact duplicate records, imputed missing product descriptions via canonical `StockCode` mapping, and labeled missing `CustomerID` entries as `'Guest'`.
* **Transaction Separation:** Filtered out non-product operational codes (`POST`, `D`, `DOT`, `M`, `BANK CHARGES`, `AMAZONFEE`, `CRUK`, `C2`) and separated cancellations/returns (`Invoice` starting with `'C'` or `Quantity < 0`) from net positive sales.
* **Calendar Continuity:** Reindexed daily time series across the full date range, filling non-operational days (Saturday closures and year-end holidays) with `0` to preserve true temporal spacing.

### 2. Feature Engineering & EDA
* **Temporal & Autoregressive Features:** Engineered calendar indicators (`day_of_week`, `is_saturday`, `month`, `quarter`, `week_of_year`) alongside lag and rolling-window statistics (`qty_lag_7`, `qty_lag_14`, `qty_roll_mean_7`) validated via Autocorrelation (ACF) spikes at multiples of 7 days.
* **Operational Patterns:** Uncovered strong Q4 seasonal surges, complete Saturday shutdowns characteristic of B2B/wholesale operations, and right-skewed multi-item basket distributions.

### 3. Modeling & Evaluation
* **Time-Series Validation:** Evaluated models using 5-fold `TimeSeriesSplit` cross-validation and a chronological 80/20 holdout test split against a `Seasonal Naive` benchmark.
* **Algorithms Tested:** Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor, and SARIMAX.
* **Association Rule Mining:** Applied the **FP-Growth** algorithm (`mlxtend`) on UK multi-item baskets (`>= 2` unique SKUs per invoice) with `min_support = 0.02`, `lift >= 1.5`, and `confidence >= 0.30`.

---

## Key Results

### 1. Demand Forecasting Performance

#### Daily Forecast (Holdout Test Set)
| Model | 5-Fold CV RMSE | MAE | Test RMSE | wMAPE (%) | R² Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **9,155.92** | **5,824.65** | **9,576.73** | **29.34%** | **0.5051** |
| Ridge Regression | 9,532.62 | 6,510.56 | 9,868.37 | 32.79% | 0.4745 |
| Seasonal Naive (`lag_7`) | - | 6,403.95 | 10,091.25 | 32.26% | 0.4505 |
| Gradient Boosting | 10,640.21  | 6,416.31  | 11,204.06 | 32.32% | 0.3226 |
| SARIMAX (`s=7`) | - | 8,484.69 | 12,804.98 | 42.74% | 0.1152 |

#### Weekly Forecast (Holdout Test Set)
| Model | 5-Fold CV RMSE | MAE | Test RMSE | wMAPE (%) | R² Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Naive (`lag_1`)** | - | **23,724.14** | **33,110.12** | **17.28%** | **0.2811** |
| Ridge Regression | 38,112.01 | 27,397.46 | 36,126.75 | 19.96% | 0.1441 |
| Random Forest | 35,462.19 | 30,198.11 | 40,534.86 | 22.00% | -0.0775 |
| Gradient Boosting | 44,643.31 | 36,912.07 | 45,958.75 | 26.89% | -0.3851 |
| SARIMAX (`s=4`) | - | 58,915.82 | 71,700.55 | 42.91% | -2.3713 |

* **Insight:** **Random Forest** excels at daily forecasting by capturing weekly lag cycles and Saturday closures without overfitting (`CV RMSE: 9,155.92` vs. `Test RMSE: 9,576.73`). At the weekly level, aggregation reduces relative forecast error (`wMAPE`) from `29.34%` down to `17.28%`, where simpler autoregressive baselines (`lag_1`) and linear models outperform tree-based models due to the limited ~80-week training sample and Q4 extrapolation dynamics.

### 2. Market Basket Analysis (FP-Growth)
* **High-Affinity Product Line:** Variants of the *Regency Teacup and Saucer* (Pink, Green, and Roses) recorded **Lift ratios between 20.39x and 24.91x**.
* **Cross-Sell Conversion:** Customers purchasing both the *Pink* and *Roses* variants demonstrated a **90% confidence** of adding the *Green* variant to the same order.

### 3. Commercial Risk Analytics
* **Product Return Risk Matrix:** Mapped active high-volume SKUs (`Sold_Qty >= 500`) across return rate (`%`) and financial return loss (`£`) to isolate key drivers of revenue leakage.
* **Customer Risk Segmentation (RFM):** Classified registered wholesale buyers into five actionable tiers (*Champions*, *At-Risk High Value*, *Loyal / Potential*, *Recent / Promising*, and *Hibernating / Lost*) to highlight revenue concentration exposure.

---

## Strategic Recommendations

1. **Tiered Forecasting Workflow:** Deploy **Random Forest** for daily warehouse scheduling and dispatch operations, while relying on **Weekly Lag-1 / Ridge Regression** benchmarks for weekly procurement planning.
2. **Product Bundling:** Package the *Regency Teacup and Saucer* trio into a bundled SKU and implement automated *"Frequently Bought Together"* recommendations at checkout to lift Average Order Value (AOV).
3. **Return Loss Mitigation:** Audit top revenue-draining items in the *Product Return Risk Matrix* to resolve recurring bulk-order cancellations, packaging fragility, or catalog discrepancies.
4. **VIP Retention Outreach:** Prioritize account-manager intervention and tailored reorder incentives for the **At-Risk High Value** RFM segment to safeguard core wholesale revenue.

---

## Tech Stack & Repository Structure

* **Languages & Libraries:** Python, `pandas`, `NumPy`, `scikit-learn`, `statsmodels`, `mlxtend`, `Plotly`

```text
├── data/
│   └── online_retail_II.csv                 # Raw transactional dataset (UCI / Kaggle)
├── Retail Demand & Basket Analysis.ipynb    # Full end-to-end analysis, modeling & interactive Plotly visuals
└── README.md                                # Project documentation
