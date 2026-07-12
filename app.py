import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from common import (
    load_raw_data, load_overview, apply_chart_style, kpi_card_css,
    CATEGORY_COLORS, REGION_COLORS, PRIMARY,
)

st.set_page_config(page_title="Sales Overview | Superstore Dashboard", page_icon="📊", layout="wide")
kpi_card_css()

df = load_raw_data()
overview = load_overview()

st.title("Superstore Sales Dashboard")
st.caption("Sales performance, forecasts, anomalies, and demand segments for the Superstore dataset (2015–2018).")

st.markdown("### Sales Overview")

# ---------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------
total_sales = df["Sales"].sum()
total_orders = df["Order ID"].nunique()
avg_order_value = df.groupby("Order ID")["Sales"].sum().mean()
years_covered = f"{df['Year'].min()}–{df['Year'].max()}"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Sales", f"${total_sales:,.0f}")
c2.metric("Total Orders", f"{total_orders:,}")
c3.metric("Avg. Order Value", f"${avg_order_value:,.0f}")
c4.metric("Years Covered", years_covered)

st.divider()

# ---------------------------------------------------------------------
# Total sales by year (bar) + Monthly sales trend (line)
# ---------------------------------------------------------------------
col1, col2 = st.columns([1, 1.6])

with col1:
    st.markdown("**Total sales by year**")
    by_year = overview["by_year"]
    fig = go.Figure(go.Bar(
        x=[str(r["Year"]) for r in by_year],
        y=[r["Sales"] for r in by_year],
        marker_color=PRIMARY,
        text=[f"${r['Sales']:,.0f}" for r in by_year],
        textposition="outside",
    ))
    fig = apply_chart_style(fig, y_title="Sales ($)", show_legend=False)
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Monthly sales trend**")
    by_month = overview["by_month"]
    import pandas as pd
    m = pd.DataFrame(by_month)
    m["ds"] = pd.to_datetime(m["ds"])
    fig = go.Figure(go.Scatter(
        x=m["ds"], y=m["Sales"], mode="lines", line=dict(color=PRIMARY, width=2),
        fill="tozeroy", fillcolor="rgba(44,110,107,0.08)",
    ))
    fig = apply_chart_style(fig, y_title="Sales ($)", show_legend=False)
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------
# Sales by region and category, with interactive filters
# ---------------------------------------------------------------------
st.markdown("**Sales by region and category**")

fcol1, fcol2, fcol3 = st.columns([1, 1, 1])
with fcol1:
    regions = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
with fcol2:
    categories = st.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))
with fcol3:
    year_range = st.select_slider(
        "Year range",
        options=sorted(df["Year"].unique()),
        value=(df["Year"].min(), df["Year"].max()),
    )

filtered = df[
    df["Region"].isin(regions)
    & df["Category"].isin(categories)
    & df["Year"].between(year_range[0], year_range[1])
]

if filtered.empty:
    st.info("No data for the selected filters. Try widening your selection.")
else:
    grouped = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    fig = px.bar(
        grouped, x="Region", y="Sales", color="Category", barmode="group",
        color_discrete_map=CATEGORY_COLORS,
        category_orders={"Region": sorted(df["Region"].unique())},
    )
    fig = apply_chart_style(fig, y_title="Sales ($)")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View underlying totals"):
        pivot = grouped.pivot(index="Region", columns="Category", values="Sales").fillna(0).round(2)
        st.dataframe(pivot.style.format("${:,.0f}"), use_container_width=True)

st.divider()
st.caption("Use the sidebar to navigate to Forecast Explorer, Anomaly Report, and Product Demand Segments.")
