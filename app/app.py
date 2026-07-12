"""
Superstore Sales Forecasting Dashboard
Run locally:  streamlit run app/app.py
Deploy free : Streamlit Community Cloud → point to this file.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:  # Streamlit can still run with the fallback forecaster.
    SARIMAX = None

st.set_page_config(page_title="Superstore Forecasting Dashboard",
                   page_icon="📈", layout="wide")

# ---------- Load pre-computed artefacts ----------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

@st.cache_data(show_spinner=False)
def load_data():
    orders = pd.read_csv(OUT / "clean_orders.csv", parse_dates=["Order Date", "Ship Date"])
    with open(OUT / "results.json") as f:
        res = json.load(f)
    monthly = pd.read_csv(OUT / "monthly_sales.csv",
                          parse_dates=["Order Date"]).rename(columns={"Order Date": "date"})
    weekly = pd.read_csv(OUT / "weekly_sales.csv",
                         parse_dates=["Order Date"]).rename(columns={"Order Date": "date"})
    clusters = pd.read_csv(OUT / "subcategory_clusters.csv")
    return orders, res, monthly, weekly, clusters

orders, res, monthly_df, weekly_df, clusters = load_data()


@st.cache_data(show_spinner=False)
def forecast_for_segment(segment_type: str, segment_value: str, horizon: int):
    """Fit the notebook's best model style, SARIMA, for a selected category/region.

    If statsmodels is unavailable or a segment is too short, use a transparent
    seasonal-naive fallback so the app remains fully functional.
    """
    if segment_type == "Category":
        dff = orders[orders["Category"] == segment_value]
    else:
        dff = orders[orders["Region"] == segment_value]

    ts = dff.set_index("Order Date")["Sales"].resample("MS").sum().asfreq("MS").fillna(0)
    test_size = min(6, max(1, len(ts) // 5))
    train, test = ts.iloc[:-test_size], ts.iloc[-test_size:]

    def fallback_forecast(series, steps):
        vals = []
        for i in range(steps):
            if len(series) >= 12:
                vals.append(float(series.iloc[-12 + i % min(12, len(series))]))
            else:
                vals.append(float(series.tail(3).mean()))
        return vals

    model_name = "SARIMA"
    try:
        if SARIMAX is None or len(train) < 24:
            raise RuntimeError("SARIMA unavailable or not enough observations")
        fitted = SARIMAX(
            train,
            order=(1, 0, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        test_pred = fitted.get_forecast(steps=len(test)).predicted_mean

        final_model = SARIMAX(
            ts,
            order=(1, 0, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        future_obj = final_model.get_forecast(steps=horizon)
        future = future_obj.predicted_mean
        conf = future_obj.conf_int()
        forecast_values = [max(0.0, float(v)) for v in future]
        lower = [max(0.0, float(v)) for v in conf.iloc[:, 0]]
        upper = [max(0.0, float(v)) for v in conf.iloc[:, 1]]
    except Exception:
        model_name = "Seasonal naive fallback"
        test_vals = fallback_forecast(train, len(test))
        test_pred = pd.Series(test_vals, index=test.index)
        forecast_values = fallback_forecast(ts, horizon)
        lower = [max(0.0, v * 0.8) for v in forecast_values]
        upper = [v * 1.2 for v in forecast_values]

    mae = float(np.mean(np.abs(test.values - np.asarray(test_pred))))
    rmse = float(np.sqrt(np.mean((test.values - np.asarray(test_pred)) ** 2)))
    forecast_dates = pd.date_range(ts.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return {
        "model_name": model_name,
        "history_dates": ts.index,
        "history": ts.values,
        "forecast_dates": forecast_dates,
        "forecast": forecast_values,
        "lower": lower,
        "upper": upper,
        "mae": mae,
        "rmse": rmse,
    }

st.sidebar.title("📈 Superstore Forecaster")
page = st.sidebar.radio("Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Data: {res['dataset']['date_min']} → {res['dataset']['date_max']}  \n"
                   f"Rows: {res['dataset']['rows']:,}")

# ==================== PAGE 1 ====================
if page == "Sales Overview":
    st.title("🛒 Sales Overview")
    total = orders["Sales"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total/1e6:.2f}M")
    c2.metric("Orders", f"{orders['Order ID'].nunique():,}")
    c3.metric("Customers", f"{orders['Customer ID'].nunique():,}")
    c4.metric("Avg Ship Delay", f"{res['ship_delay_overall']} days")

    with st.expander("Filter data", expanded=True):
        regions = st.multiselect("Region", sorted(orders["Region"].unique()),
                                 default=sorted(orders["Region"].unique()))
        cats = st.multiselect("Category", sorted(orders["Category"].unique()),
                              default=sorted(orders["Category"].unique()))
    dff = orders[(orders["Region"].isin(regions)) & (orders["Category"].isin(cats))]

    col1, col2 = st.columns(2)
    yearly = dff.groupby(dff["Order Date"].dt.year)["Sales"].sum().reset_index()
    yearly.columns = ["Year", "Sales"]
    col1.plotly_chart(px.bar(yearly, x="Year", y="Sales", title="Total Sales by Year",
                             text_auto=".2s"), use_container_width=True)
    m = dff.set_index("Order Date")["Sales"].resample("MS").sum().reset_index()
    col2.plotly_chart(px.line(m, x="Order Date", y="Sales",
                              title="Monthly Sales Trend", markers=True),
                      use_container_width=True)

    col3, col4 = st.columns(2)
    reg_cat = dff.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    col3.plotly_chart(px.bar(reg_cat, x="Region", y="Sales", color="Category",
                             title="Sales by Region × Category", barmode="group"),
                      use_container_width=True)
    subc = dff.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=True).reset_index()
    col4.plotly_chart(px.bar(subc, x="Sales", y="Sub-Category", orientation="h",
                             title="Sales by Sub-Category"), use_container_width=True)

# ==================== PAGE 2 ====================
elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.caption(f"Notebook benchmark best model: **{res['best_model']}**. The selected segment is forecast with SARIMA when available.")

    csel1, csel2 = st.columns([1, 2])
    segment_type = csel1.selectbox("Select input type", ["Category", "Region"])
    options = sorted(orders[segment_type].dropna().unique())
    segment_value = csel2.selectbox(f"Select {segment_type}", options)
    horizon = st.slider("Date range / forecast horizon", min_value=1, max_value=3, value=3,
                        format="%d month(s) ahead")

    s = forecast_for_segment(segment_type, segment_value, horizon)
    fc_dates = list(s["forecast_dates"])
    fc_vals = list(s["forecast"])
    lower = list(s["lower"])
    upper = list(s["upper"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s["history_dates"], y=s["history"], name="Historical sales",
                             line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_vals, name=f"{s['model_name']} forecast",
                             line=dict(color="crimson", dash="dash"), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=fc_dates + fc_dates[::-1], y=upper + lower[::-1],
                             fill="toself", fillcolor="rgba(220,20,60,0.15)",
                             line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
                             showlegend=True, name="Forecast interval"))
    fig.update_layout(title=f"{segment_type}: {segment_value} — {horizon}-Month Forecast",
                      xaxis_title="Date", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Model used", s["model_name"])
    c2.metric("MAE", f"${s['mae']:,.0f}")
    c3.metric("RMSE", f"${s['rmse']:,.0f}")

    st.subheader("Forecast output")
    st.dataframe(pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in fc_dates],
        "Forecast Sales": [f"${v:,.0f}" for v in fc_vals],
        "Lower Bound": [f"${v:,.0f}" for v in lower],
        "Upper Bound": [f"${v:,.0f}" for v in upper],
    }), hide_index=True, use_container_width=True)

# ==================== PAGE 3 ====================
elif page == "Anomaly Report":
    st.title("🚨 Anomaly Report")
    st.caption("Two methods run in parallel: **Isolation Forest** (global outliers) "
               "and **Z-score > 2** (local outliers versus 8-week rolling mean).")

    iso_dates = pd.to_datetime(res["anomalies"]["iso_dates"])
    z_dates = pd.to_datetime(res["anomalies"]["z_dates"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly_df["date"], y=weekly_df["Sales"],
                             mode="lines", name="Weekly sales", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=iso_dates, y=res["anomalies"]["iso_values"],
                             mode="markers", name="Isolation Forest",
                             marker=dict(color="red", size=10)))
    fig.add_trace(go.Scatter(x=z_dates, y=res["anomalies"]["z_values"],
                             mode="markers", name="Z-score > 2",
                             marker=dict(color="orange", size=14,
                                         symbol="circle-open", line=dict(width=2))))
    fig.update_layout(title="Weekly Sales with Detected Anomalies",
                      xaxis_title="Date", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Iso-Forest anomalies", len(iso_dates))
    c2.metric("Z-score anomalies", len(z_dates))
    c3.metric("Overlap", len(res["anomalies"]["overlap"]))

    tab1, tab2 = st.tabs(["Isolation Forest", "Z-score"])
    with tab1:
        st.dataframe(pd.DataFrame({
            "Week": [str(d.date()) for d in iso_dates],
            "Sales ($)": [f"{v:,.0f}" for v in res["anomalies"]["iso_values"]]
        }), hide_index=True)
    with tab2:
        st.dataframe(pd.DataFrame({
            "Week": [str(d.date()) for d in z_dates],
            "Sales ($)": [f"{v:,.0f}" for v in res["anomalies"]["z_values"]]
        }), hide_index=True)

# ==================== PAGE 4 ====================
elif page == "Product Demand Segments":
    st.title("🧩 Product Demand Segments")
    st.caption("K-Means clustering on sub-category features (total sales, YoY growth, volatility, avg order value).")

    fig = px.scatter(clusters, x="total_sales", y="volatility",
                     size="avg_order_value", color="cluster_label",
                     hover_name="Sub-Category",
                     title="Sub-Category Demand Clusters",
                     labels={"total_sales": "Total Sales ($)",
                             "volatility": "Volatility (std of monthly sales)"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster assignments")
    st.dataframe(clusters[["Sub-Category", "total_sales", "volatility",
                           "yoy_growth", "avg_order_value", "cluster_label"]]
                 .sort_values("cluster_label").reset_index(drop=True),
                 hide_index=True, use_container_width=True)

    st.subheader("Recommended stocking strategy per cluster")
    st.markdown("""
| Cluster | Description | Action |
|---|---|---|
| **High Volume, Stable** | Consistent core-catalogue items (Binders, Paper, Storage). | Continuous replenishment. Safety stock ≥ 2 weeks. Never stock-out. |
| **Volatile / Bursty** | Big-ticket items with sharp swings (Copiers, Machines, Tables). | Order-to-forecast. Add safety margin around Q4. Watch weekly. |
| **Growing Demand** | Rising sub-categories (Accessories, Phones). | Increase base stock 15–25% quarterly. Enroll in vendor-managed replenishment. |
| **Low Volume, Steady** | Long-tail items (Envelopes, Fasteners, Labels). | Make-to-order / drop-ship. Do not tie up capital. |
""")
