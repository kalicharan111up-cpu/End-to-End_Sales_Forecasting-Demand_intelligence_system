"""
End-to-end sales forecasting analysis on the Superstore dataset.
Runs Tasks 1-6 of the assignment, saves figures + a JSON of key metrics
that the executive report and Streamlit app both consume.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

from prophet import Prophet
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True, parents=True)
OUT.mkdir(exist_ok=True, parents=True)

results = {}  # everything we need later

# ---------- Task 1: Load + feature engineering ----------
df = pd.read_csv(ROOT / "data" / "train.csv", encoding="latin1")
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")

df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Week"] = df["Order Date"].dt.isocalendar().week.astype(int)
df["DayOfWeek"] = df["Order Date"].dt.dayofweek
df["Quarter"] = df["Order Date"].dt.quarter

season_map = {12: "Winter", 1: "Winter", 2: "Winter",
              3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer",
              9: "Fall", 10: "Fall", 11: "Fall"}
df["Season"] = df["Month"].map(season_map)
df["ShipDelay"] = (df["Ship Date"] - df["Order Date"]).dt.days

results["dataset"] = {
    "rows": int(len(df)),
    "columns": int(df.shape[1]),
    "date_min": str(df["Order Date"].min().date()),
    "date_max": str(df["Order Date"].max().date()),
    "duplicates": int(df.duplicated().sum()),
    "missing_postal_code": int(df["Postal Code"].isna().sum()),
}

# --- EDA answers ---
cat_rev = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
results["category_revenue"] = cat_rev.round(2).to_dict()

# Region YoY growth consistency: measure std of YoY growth (lower = more consistent)
region_year = df.groupby(["Region", "Year"])["Sales"].sum().unstack()
region_growth = region_year.pct_change(axis=1).dropna(axis=1, how="all") * 100
region_growth_summary = pd.DataFrame({
    "mean_growth_pct": region_growth.mean(axis=1),
    "std_growth_pct": region_growth.std(axis=1),
}).sort_values("std_growth_pct")
results["region_growth"] = region_growth_summary.round(2).to_dict(orient="index")
most_consistent_region = region_growth_summary.index[0]
results["most_consistent_region"] = most_consistent_region

# Ship delay by region
ship_by_region = df.groupby("Region")["ShipDelay"].mean().round(2)
results["ship_delay_overall"] = float(df["ShipDelay"].mean().round(2))
results["ship_delay_by_region"] = ship_by_region.to_dict()

# Seasonality: months that spike across years
month_year = df.groupby(["Year", "Month"])["Sales"].sum().unstack(level=0)
month_avg = month_year.mean(axis=1).sort_values(ascending=False)
results["top_months"] = month_avg.round(2).to_dict()

# --- Figure: EDA overview ---
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
cat_rev.plot(kind="bar", ax=axes[0, 0], color="#4C72B0")
axes[0, 0].set_title("Total Revenue by Category")
axes[0, 0].set_ylabel("Sales ($)")

region_year.plot(kind="bar", ax=axes[0, 1])
axes[0, 1].set_title("Annual Sales by Region")
axes[0, 1].set_ylabel("Sales ($)")

ship_by_region.plot(kind="bar", ax=axes[1, 0], color="#55A868")
axes[1, 0].set_title("Avg Ship Delay (days) by Region")
axes[1, 0].set_ylabel("Days")

month_year.plot(ax=axes[1, 1], marker="o")
axes[1, 1].set_title("Monthly Sales by Year")
axes[1, 1].set_xlabel("Month")
axes[1, 1].set_ylabel("Sales ($)")
plt.tight_layout()
plt.savefig(FIG / "01_eda_overview.png", dpi=130)
plt.close()

# ---------- Task 1 aggregations ----------
daily = df.groupby("Order Date")["Sales"].sum().asfreq("D").fillna(0)
weekly = df.set_index("Order Date")["Sales"].resample("W").sum()
monthly = df.set_index("Order Date")["Sales"].resample("MS").sum()
monthly.to_csv(OUT / "monthly_sales.csv")
weekly.to_csv(OUT / "weekly_sales.csv")

# ---------- Task 2: Decomposition + ADF ----------
decomp = seasonal_decompose(monthly, model="additive", period=12)
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
monthly.plot(ax=axes[0], title="Observed monthly sales")
decomp.trend.plot(ax=axes[1], title="Trend", color="#DD8452")
decomp.seasonal.plot(ax=axes[2], title="Seasonal", color="#55A868")
decomp.resid.plot(ax=axes[3], title="Residual", color="#C44E52")
plt.tight_layout()
plt.savefig(FIG / "02_decomposition.png", dpi=130)
plt.close()

adf_stat, adf_p, *_ = adfuller(monthly.dropna())
results["adf"] = {"statistic": float(adf_stat), "pvalue": float(adf_p),
                  "stationary": bool(adf_p < 0.05)}

# Diff if needed
diffed = monthly.diff().dropna()
adf_stat2, adf_p2, *_ = adfuller(diffed)
results["adf_diff"] = {"statistic": float(adf_stat2), "pvalue": float(adf_p2),
                       "stationary": bool(adf_p2 < 0.05)}

# ---------- Task 3: SARIMA / Prophet / XGBoost ----------
# Train/test split: last 6 months are held out
train = monthly.iloc[:-6]
test = monthly.iloc[-6:]

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

# --- SARIMA ---
sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                 enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sarima_test_pred = sarima.get_forecast(steps=len(test)).predicted_mean
sarima_future = sarima.get_forecast(steps=6)  # for chart continuity we also make 6 steps
# For the required 3-month future forecast we refit on the full series
sarima_full = SARIMAX(monthly, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sarima_forecast = sarima_full.get_forecast(steps=3)
sarima_pred_future = sarima_forecast.predicted_mean
sarima_ci = sarima_forecast.conf_int()

sarima_metrics = {
    "MAE": float(mean_absolute_error(test, sarima_test_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(test, sarima_test_pred))),
    "MAPE": mape(test, sarima_test_pred),
    "forecast": [float(x) for x in sarima_pred_future.values],
    "lower": [float(x) for x in sarima_ci.iloc[:, 0].values],
    "upper": [float(x) for x in sarima_ci.iloc[:, 1].values],
}

# --- Prophet ---
prophet_df = monthly.reset_index()
prophet_df.columns = ["ds", "y"]
p_train = prophet_df.iloc[:-6]
m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(p_train)
future_test = m.make_future_dataframe(periods=6, freq="MS")
prophet_test_pred = m.predict(future_test).tail(6)["yhat"].values

m_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m_full.fit(prophet_df)
future3 = m_full.make_future_dataframe(periods=3, freq="MS")
prophet_full_fc = m_full.predict(future3).tail(3)
prophet_metrics = {
    "MAE": float(mean_absolute_error(test.values, prophet_test_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(test.values, prophet_test_pred))),
    "MAPE": mape(test.values, prophet_test_pred),
    "forecast": [float(x) for x in prophet_full_fc["yhat"].values],
    "lower": [float(x) for x in prophet_full_fc["yhat_lower"].values],
    "upper": [float(x) for x in prophet_full_fc["yhat_upper"].values],
}

# Save Prophet components plot
fig = m_full.plot_components(m_full.predict(future3))
plt.savefig(FIG / "03_prophet_components.png", dpi=130)
plt.close(fig)

# --- XGBoost ---
def make_features(series):
    d = pd.DataFrame({"y": series})
    d["lag1"] = d["y"].shift(1)
    d["lag2"] = d["y"].shift(2)
    d["lag3"] = d["y"].shift(3)
    d["roll3"] = d["y"].shift(1).rolling(3).mean()
    d["month"] = d.index.month
    d["quarter"] = d.index.quarter
    d["season"] = d["month"].map(season_map)
    d = pd.get_dummies(d, columns=["season"], drop_first=True)
    return d

feat = make_features(monthly).dropna()
X = feat.drop(columns="y")
y = feat["y"]
X_train, y_train = X.iloc[:-6], y.iloc[:-6]
X_test, y_test = X.iloc[-6:], y.iloc[-6:]

model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                         random_state=42, verbosity=0)
model.fit(X_train, y_train)
xgb_test_pred = model.predict(X_test)

# Recursive forecast of next 3 months
history = monthly.copy()
xgb_future = []
for _ in range(3):
    hist_feat = make_features(history)
    last_row = hist_feat.iloc[[-1]].drop(columns="y")
    # ensure columns match training X
    for c in X.columns:
        if c not in last_row.columns:
            last_row[c] = 0
    last_row = last_row[X.columns]
    # Since last row was built from `history` which does NOT contain a next-month value yet,
    # we shift by adding a placeholder row first.
    next_date = history.index[-1] + pd.offsets.MonthBegin(1)
    tmp = pd.concat([history, pd.Series([np.nan], index=[next_date])])
    tmp_feat = make_features(tmp)
    row = tmp_feat.iloc[[-1]].drop(columns="y")
    for c in X.columns:
        if c not in row.columns:
            row[c] = 0
    row = row[X.columns]
    pred = float(model.predict(row)[0])
    xgb_future.append((next_date, pred))
    history = pd.concat([history, pd.Series([pred], index=[next_date])])

xgb_metrics = {
    "MAE": float(mean_absolute_error(y_test, xgb_test_pred)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, xgb_test_pred))),
    "MAPE": mape(y_test.values, xgb_test_pred),
    "forecast": [p for _, p in xgb_future],
    "dates": [str(d.date()) for d, _ in xgb_future],
}

results["models"] = {
    "SARIMA": sarima_metrics,
    "Prophet": prophet_metrics,
    "XGBoost": xgb_metrics,
    "forecast_dates": [str(pd.Timestamp(d).date()) for d in prophet_full_fc["ds"].values],
    "test_actual": [float(x) for x in test.values],
    "test_index": [str(d.date()) for d in test.index],
    "sarima_test_pred": [float(x) for x in sarima_test_pred.values],
    "prophet_test_pred": [float(x) for x in prophet_test_pred],
    "xgb_test_pred": [float(x) for x in xgb_test_pred],
}

# Pick best by RMSE
scoreboard = {"SARIMA": sarima_metrics["RMSE"],
              "Prophet": prophet_metrics["RMSE"],
              "XGBoost": xgb_metrics["RMSE"]}
best = min(scoreboard, key=scoreboard.get)
results["best_model"] = best
print("Best model by RMSE:", best, scoreboard)

# --- Model comparison figure ---
future_idx = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq="MS")
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(monthly.index, monthly.values, label="Actual", color="black", linewidth=1.6)
ax.plot(test.index, sarima_test_pred.values, label="SARIMA (test)", marker="o", color="#1f77b4")
ax.plot(test.index, prophet_test_pred, label="Prophet (test)", marker="s", color="#ff7f0e")
ax.plot(test.index, xgb_test_pred, label="XGBoost (test)", marker="^", color="#2ca02c")
ax.plot(future_idx, sarima_metrics["forecast"], "--o", label="SARIMA future", color="#1f77b4")
ax.plot(future_idx, prophet_metrics["forecast"], "--s", label="Prophet future", color="#ff7f0e")
ax.plot(future_idx, xgb_metrics["forecast"], "--^", label="XGBoost future", color="#2ca02c")
ax.fill_between(future_idx, sarima_metrics["lower"], sarima_metrics["upper"],
                color="#1f77b4", alpha=0.12, label="SARIMA 95% CI")
ax.axvline(test.index[0], color="gray", linestyle=":", alpha=0.7)
ax.text(test.index[0], ax.get_ylim()[1]*0.95, " hold-out starts", fontsize=8, color="gray")
ax.set_title("Monthly Sales: Actual vs 3-Model Forecasts (test + 3-mo future)")
ax.set_ylabel("Sales ($)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(FIG / "04_model_comparison.png", dpi=130)
plt.close()

# ---------- Task 4: Segment-level forecasts using best model ----------
segments = {
    "Furniture": df[df["Category"] == "Furniture"],
    "Technology": df[df["Category"] == "Technology"],
    "Office Supplies": df[df["Category"] == "Office Supplies"],
    "West Region": df[df["Region"] == "West"],
    "East Region": df[df["Region"] == "East"],
}

segment_forecasts = {}
fig, ax = plt.subplots(figsize=(13, 6))
colors = plt.cm.tab10.colors
for i, (name, sdf) in enumerate(segments.items()):
    s_monthly = sdf.set_index("Order Date")["Sales"].resample("MS").sum()
    # Use Prophet (it handled all segments robustly regardless of best_model)
    pdf = s_monthly.reset_index(); pdf.columns = ["ds", "y"]
    mp = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    mp.fit(pdf)
    fu = mp.make_future_dataframe(periods=3, freq="MS")
    fc = mp.predict(fu).tail(3)
    segment_forecasts[name] = {
        "history": [float(x) for x in s_monthly.values],
        "history_dates": [str(d.date()) for d in s_monthly.index],
        "forecast": [float(x) for x in fc["yhat"].values],
        "forecast_dates": [str(d.date()) for d in fc["ds"]],
    }
    ax.plot(s_monthly.index, s_monthly.values, color=colors[i], alpha=0.5)
    ax.plot(fc["ds"], fc["yhat"], "--", color=colors[i], marker="o", label=name)

ax.set_title("Segment-Level 3-Month Forecast (Prophet)")
ax.set_ylabel("Sales ($)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "05_segment_forecasts.png", dpi=130)
plt.close()

# Rank segments by predicted growth
seg_growth = {}
for name, d in segment_forecasts.items():
    baseline = np.mean(d["history"][-3:])
    proj = np.mean(d["forecast"])
    seg_growth[name] = float((proj - baseline) / baseline * 100)
results["segment_growth_pct"] = seg_growth
results["strongest_segment"] = max(seg_growth, key=seg_growth.get)
results["segment_forecasts"] = segment_forecasts

# ---------- Task 5: Anomaly detection ----------
weekly_df = weekly.to_frame("sales")
weekly_df["z"] = (weekly_df["sales"] - weekly_df["sales"].rolling(8, min_periods=1).mean()) / \
                 weekly_df["sales"].rolling(8, min_periods=1).std()
weekly_df["zscore_anom"] = weekly_df["z"].abs() > 2

iso = IsolationForest(contamination=0.05, random_state=42)
weekly_df["iso_anom"] = iso.fit_predict(weekly_df[["sales"]]) == -1

anoms_iso = weekly_df[weekly_df["iso_anom"]].copy()
anoms_z = weekly_df[weekly_df["zscore_anom"]].copy()

fig, ax = plt.subplots(figsize=(13, 5))
weekly_df["sales"].plot(ax=ax, color="steelblue", label="Weekly sales")
ax.scatter(anoms_iso.index, anoms_iso["sales"], color="red", s=60, label="Isolation Forest", zorder=5)
ax.scatter(anoms_z.index, anoms_z["sales"], facecolors="none", edgecolors="orange",
           s=120, linewidths=2, label="Z-score > 2", zorder=4)
ax.set_title("Weekly Sales with Detected Anomalies")
ax.set_ylabel("Sales ($)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "06_anomalies.png", dpi=130)
plt.close()

results["anomalies"] = {
    "iso_dates": [str(d.date()) for d in anoms_iso.index],
    "iso_values": [float(v) for v in anoms_iso["sales"].values],
    "z_dates": [str(d.date()) for d in anoms_z.index],
    "z_values": [float(v) for v in anoms_z["sales"].values],
    "overlap": sorted(set([str(d.date()) for d in anoms_iso.index]) &
                      set([str(d.date()) for d in anoms_z.index])),
}
anoms_iso[["sales"]].to_csv(OUT / "anomalies_iso.csv")
anoms_z[["sales"]].to_csv(OUT / "anomalies_z.csv")

# ---------- Task 6: Sub-category clustering ----------
sc_monthly = df.groupby(["Sub-Category",
                         pd.Grouper(key="Order Date", freq="MS")])["Sales"].sum().unstack(fill_value=0)
features = pd.DataFrame(index=sc_monthly.index)
features["total_sales"] = sc_monthly.sum(axis=1)
features["volatility"] = sc_monthly.std(axis=1)
features["avg_order_value"] = df.groupby("Sub-Category")["Sales"].mean()

sc_year = df.groupby(["Sub-Category", "Year"])["Sales"].sum().unstack(fill_value=0)
sc_year_growth = sc_year.pct_change(axis=1).replace([np.inf, -np.inf], np.nan).mean(axis=1)
features["yoy_growth"] = sc_year_growth.fillna(0)

X_scaled = StandardScaler().fit_transform(features)

# Elbow
inertias = []
for k in range(1, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, 8), inertias, "o-")
ax.set_title("Elbow Method")
ax.set_xlabel("k"); ax.set_ylabel("Inertia")
plt.tight_layout()
plt.savefig(FIG / "07_elbow.png", dpi=130)
plt.close()

K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X_scaled)
features["cluster"] = km.labels_

# PCA scatter
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(9, 6))
for c in range(K):
    mask = features["cluster"] == c
    ax.scatter(pcs[mask, 0], pcs[mask, 1], s=90, label=f"Cluster {c}")
    for i, name in enumerate(features.index[mask]):
        ax.annotate(name, (pcs[mask, 0][list(features.index[mask]).index(name)],
                           pcs[mask, 1][list(features.index[mask]).index(name)]),
                    fontsize=7)
ax.set_title("Sub-Category Demand Clusters (PCA projection)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "08_clusters.png", dpi=130)
plt.close()

# Label clusters by profile
profile = features.groupby("cluster")[["total_sales", "volatility", "yoy_growth", "avg_order_value"]].mean()
labels_map = {}
for c in profile.index:
    row = profile.loc[c]
    if row["total_sales"] == profile["total_sales"].max():
        labels_map[c] = "High Volume, Stable Demand"
    elif row["yoy_growth"] == profile["yoy_growth"].max():
        labels_map[c] = "Growing Demand"
    elif row["volatility"] == profile["volatility"].max():
        labels_map[c] = "Volatile / Bursty Demand"
    else:
        labels_map[c] = "Low Volume, Steady"
# Ensure uniqueness
seen = set(); final = {}
for c, l in labels_map.items():
    base = l; i = 2
    while l in seen:
        l = f"{base} #{i}"; i += 1
    seen.add(l); final[c] = l
features["cluster_label"] = features["cluster"].map(final)
features.to_csv(OUT / "subcategory_clusters.csv")
results["clusters"] = {
    "labels": final,
    "assignments": features[["cluster", "cluster_label"]].reset_index().to_dict(orient="records"),
    "profile": profile.round(2).to_dict(orient="index"),
}

# ---------- Save all results ----------
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# Also persist monthly sales, weekly sales, segments etc. for the Streamlit app
df.to_csv(OUT / "clean_orders.csv", index=False)
print("Done. Figures in", FIG, "outputs in", OUT)
