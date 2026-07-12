# Superstore Sales Forecasting & Demand Intelligence System

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Machine Learning](https://img.shields.io/badge/ML-Forecasting%20%7C%20Clustering-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A professional end-to-end analytics and forecasting project that transforms retail order data into an interactive **sales intelligence dashboard**. The system supports sales exploration, future demand forecasting, anomaly detection, and product demand segmentation using time-series modeling and machine learning.

The project is built around the Superstore sales dataset and includes a complete analytical workflow, saved model outputs, visual reports, and a deployable Streamlit web application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Streamlit App Pages](#streamlit-app-pages)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Modeling Approach](#modeling-approach)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Run the Dashboard](#run-the-dashboard)
- [Reproduce the Analysis](#reproduce-the-analysis)
- [Deployment](#deployment)
- [Business Insights](#business-insights)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Project Overview

Retail and e-commerce businesses rely on accurate demand forecasting to avoid two costly problems:

- **Overstocking**, which increases storage costs and ties up capital
- **Understocking**, which causes lost sales and poor customer experience

This project answers key business questions such as:

- How have total sales changed over time?
- Which regions and categories drive the most revenue?
- What will sales look like over the next 1 to 3 months?
- Which sales periods are unusual or anomalous?
- Which product sub-categories behave similarly in demand patterns?

The final deliverable is a four-page Streamlit dashboard designed for business users, analysts, and decision-makers.

---

## Key Features

- Interactive sales overview dashboard
- Yearly and monthly sales trend visualization
- Region and category-level filtering
- 1, 2, and 3-month sales forecasts
- Forecast accuracy metrics: MAE and RMSE
- Anomaly detection for unusual weekly sales behavior
- Product demand clustering using K-Means
- Clean output tables for anomaly dates and demand clusters
- Precomputed outputs for fast dashboard loading
- GitHub and Streamlit Cloud-ready project structure

---

## Streamlit App Pages

### Page 1: Sales Overview Dashboard

Includes:

- Total sales by year bar chart
- Monthly sales trend line chart
- Interactive sales filters by region and category
- Sales breakdown by region, category, and sub-category
- High-level KPI cards for revenue, orders, customers, and shipping delay

### Page 2: Forecast Explorer

Includes:

- Dropdown to select forecast input type: **Category** or **Region**
- Dropdown to choose a specific category or region
- Forecast horizon slider for **1, 2, or 3 months ahead**
- Forecast output from the selected forecasting model
- Forecast chart with historical and predicted sales
- MAE and RMSE displayed below the chart
- Forecast values in a readable table

### Page 3: Anomaly Report

Includes:

- Weekly anomaly chart from the anomaly detection task
- Isolation Forest anomaly markers
- Z-score anomaly markers
- Table of detected anomaly dates
- Table of anomaly sales values

### Page 4: Product Demand Segments

Includes:

- Cluster chart from the demand segmentation task
- Sub-category demand cluster assignments
- Cluster-level stocking strategy recommendations
- Business-friendly interpretation of demand groups

---

## Tech Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data Processing | pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web App | Streamlit |
| Forecasting | SARIMA, Prophet, XGBoost |
| Machine Learning | scikit-learn, K-Means, Isolation Forest |
| Statistical Analysis | statsmodels |
| Reporting | PDF report and saved chart outputs |

---

## Project Structure

```text
 deliverable/
 ├── app/
 │   └── app.py                         # Streamlit dashboard
 │
 ├── data/
 │   └── train.csv                      # Raw Superstore sales data
 │
 ├── Charts/
 │   ├── 01_eda_overview.png
 │   ├── 02_decomposition.png
 │   ├── 03_prophet_components.png
 │   ├── 04_model_comparison.png
 │   ├── 05_segment_forecasts.png
 │   ├── 06_anomalies.png
 │   ├── 07_elbow.png
 │   └── 08_clusters.png
 │
 ├── notebooks/
 │   ├── sales_forecasting.ipynb        # Main analytical notebook
 │   └── run_analysis.py
 │
 ├── outputs/
 │   ├── anomalies_iso.csv
 │   ├── anomalies_z.csv
 │   ├── clean_orders.csv
 │   ├── monthly_sales.csv
 │   ├── results.json
 │   ├── subcategory_clusters.csv
 │   └── weekly_sales.csv
 │
 ├── reports/
 │   └── summary.pdf                    # Executive summary report
 │
 ├── build_report.py
 ├── requirements.txt
 ├── run_analysis.py                    # Reproduces analysis outputs
 └── README.md
```

---

## Dataset

The project uses a Superstore-style retail dataset containing order-level sales records from **2015 to 2018**.

Important fields include:

- Order Date
- Ship Date
- Region
- Category
- Sub-Category
- Sales
- Customer ID
- Order ID
- Product information

Dataset summary:

| Metric | Value |
|---|---:|
| Rows | 9,800 |
| Time Period | 2015-01-03 to 2018-12-30 |
| Main Target | Sales |
| Granularity | Order-level transactions |

---

## Modeling Approach

### 1. Exploratory Data Analysis

The analysis begins with:

- Date parsing
- Time feature engineering
- Missing value checks
- Duplicate checks
- Weekly and monthly aggregation
- Revenue by category and region
- Seasonality exploration

### 2. Time-Series Analysis

The monthly sales series is analyzed using:

- Trend visualization
- Seasonal decomposition
- Stationarity testing with the Augmented Dickey-Fuller test
- Monthly seasonality comparison across years

### 3. Forecasting Models

Three forecasting approaches are compared:

#### SARIMA

A statistical time-series model used to capture autoregressive, moving average, and seasonal patterns.

#### Prophet

A robust forecasting model designed for trend and seasonality modeling.

#### XGBoost

A machine learning approach using engineered lag features, rolling means, and calendar-based features.

### 4. Anomaly Detection

Two anomaly detection methods are used:

- **Isolation Forest** for global sales outliers
- **Z-score method** for local deviations from rolling weekly behavior

### 5. Product Demand Clustering

K-Means clustering groups sub-categories using demand-related features such as:

- Total sales
- Sales volatility
- Average order value
- Year-over-year growth

---

## Model Performance

The models were evaluated using the last six months of sales as the test period.

| Model | MAE | RMSE | MAPE | Month 1 Forecast | Month 2 Forecast | Month 3 Forecast |
|---|---:|---:|---:|---:|---:|---:|
| **SARIMA** | 14,862 | **17,300** | 17.9% | 46,782 | 40,285 | 72,234 |
| Prophet | **14,501** | 19,156 | **17.8%** | 42,548 | 33,310 | 80,305 |
| XGBoost | 20,285 | 22,119 | 24.8% | 40,286 | 30,892 | 53,929 |

**Recommended model:** SARIMA  
SARIMA is selected as the best overall model because it achieved the lowest RMSE and provides native confidence intervals, making it suitable for planning and risk-aware decision-making.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/superstore-sales-forecasting.git
cd superstore-sales-forecasting
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Dashboard

From the project root, run:

```bash
python -m streamlit run app/app.py
```

Or, if Streamlit is available directly on your PATH:

```bash
streamlit run app/app.py
```

Then open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

---

## Reproduce the Analysis

To regenerate all processed outputs, model results, anomaly files, cluster files, and figures:

```bash
python run_analysis.py
```

This will update files inside:

- `outputs/`
- `figures/`

The Streamlit app reads these precomputed artifacts so that the dashboard loads quickly without retraining every model during each page visit.

---

## Deployment

This project can be deployed for free on **Streamlit Community Cloud**.

### Steps

1. Push the project to a public GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Click **New app**.
4. Select your GitHub repository.
5. Set the main file path to:

```text
app/app.py
```

6. Choose Python 3.11 or newer.
7. Click **Deploy**.

### Deployment Note

The app uses precomputed files from the `outputs/` folder. This keeps the deployed dashboard responsive and avoids expensive model training during app usage.

---

## Business Insights

Key findings from the project include:

- Sales show a clear upward trend across the four-year period.
- November, December, and September are consistently strong sales months.
- Technology generates the highest category revenue.
- The East region shows the most consistent sales growth.
- Weekly sales anomalies often occur around unusually high seasonal or promotional periods.
- Product sub-categories can be grouped into practical demand segments such as high-volume stable items, volatile items, growing demand items, and low-volume steady items.

---

## Future Improvements

Potential enhancements include:

- Add model retraining directly from the dashboard
- Add SKU-level or product-level forecasting
- Include profit and discount optimization
- Add inventory reorder point recommendations
- Add downloadable forecast reports
- Deploy a scheduled pipeline for automatic monthly updates
- Add authentication for business users
- Connect to a live database or cloud data warehouse

---

## Example Use Cases

This dashboard can help teams with:

- Monthly sales planning
- Inventory replenishment
- Regional demand comparison
- Product category forecasting
- Detecting abnormal sales spikes or drops
- Segmenting products by demand behavior
- Executive reporting

---

## License

This project is intended for educational, portfolio, and demonstration purposes. Add a license file if you plan to distribute or reuse it publicly.

---

## Author

Developed as an end-to-end sales forecasting and demand intelligence project using Python, machine learning, statistical modeling, and Streamlit.
