import streamlit as st
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# -------------------
# PAGE CONFIG + STYLE
# -------------------
st.set_page_config(
    page_title="Revenue Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Revenue Intelligence Dashboard")

# -------------------
# SIDEBAR CONTROLS
# -------------------
st.sidebar.title("📊 Controls")

show_data = st.sidebar.checkbox("Show Raw Data", False)
show_forecast_table = st.sidebar.checkbox("Show Forecast Table", True)

# -------------------
# LOAD DATA
# -------------------
df = pd.read_csv("data/sales.csv")

df.columns = ["Month", "Revenue"]
df["Month"] = pd.to_datetime(df["Month"])
df = df.set_index("Month")

# -------------------
# KPI CALCULATIONS
# -------------------
growth_rate = (df["Revenue"].iloc[-1] - df["Revenue"].iloc[0]) / df["Revenue"].iloc[0]

# -------------------
# KPIs UI
# -------------------
st.markdown("## 📌 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Revenue", f"{df['Revenue'].sum():,.0f}")
col2.metric("📊 Avg Monthly Revenue", f"{df['Revenue'].mean():,.0f}")
col3.metric("📈 Growth Rate", f"{growth_rate:.2%}")

st.divider()

# -------------------
# HISTORICAL DATA
# -------------------
st.markdown("## 📈 Revenue Trends")

st.line_chart(df, use_container_width=True)

# -------------------
# MODEL
# -------------------
model = ARIMA(df["Revenue"], order=(5,1,0))
model_fit = model.fit()

# -------------------
# FORECAST
# -------------------
forecast = model_fit.forecast(steps=12)

forecast_df = pd.DataFrame({
    "Forecast": forecast.values
})

forecast_dates = pd.date_range(
    start=df.index[-1] + pd.DateOffset(months=1),
    periods=12,
    freq="MS"
)

forecast_df.index = forecast_dates

# -------------------
# COMBINED VIEW
# -------------------
combined = pd.concat([df, forecast_df])

# -------------------
# FORECAST UI
# -------------------
st.markdown("## 🔮 12-Month Forecast")

st.line_chart(combined, use_container_width=True)

# -------------------
# MODEL ACCURACY
# -------------------
st.markdown("## 📉 Model Performance")

fitted_values = model_fit.fittedvalues
mae = mean_absolute_error(df["Revenue"][1:], fitted_values[1:])

st.metric("Mean Absolute Error (MAE)", f"{mae:,.2f}")

st.divider()

# -------------------
# TABLES (EXPANDABLE UI)
# -------------------
if show_forecast_table:
    with st.expander("📄 Forecast Data"):
        st.dataframe(forecast_df, use_container_width=True)

if show_data:
    with st.expander("📂 Raw Data"):
        st.dataframe(df, use_container_width=True)