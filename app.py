import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Inventory AI", layout="wide")
st.title("📦 Smart Inventory Analytics Dashboard")

# ---------------- LOAD DATA (FINAL FIX) ----------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file, encoding='latin1')  # ✅ correct for your dataset


file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = load_data(file)

    # -------- DATA PREVIEW --------
    st.subheader("📊 Data Preview (Original Format)")
    st.dataframe(df.head(), use_container_width=True)

    # -------- CLEAN REQUIRED COLUMNS --------
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors='coerce')
    df["QUANTITYORDERED"] = pd.to_numeric(df["QUANTITYORDERED"], errors='coerce')

    df = df.dropna(subset=["ORDERDATE", "QUANTITYORDERED", "PRODUCTCODE"])
    df = df[df["QUANTITYORDERED"] > 0]

    if df.empty:
        st.error("❌ Data invalid after cleaning")
        st.stop()

    st.success("✅ Dataset Loaded Correctly (No Column Issues)")

    # -------- SALES TREND --------
    st.subheader("📈 Daily Sales (Units)")

    daily_df = df.groupby("ORDERDATE")["QUANTITYORDERED"].sum().reset_index()
    daily_df.columns = ["Date", "Units Sold"]

    st.bar_chart(daily_df.set_index("Date"))

    # -------- TOP PRODUCTS --------
    st.subheader("🏆 Top 10 Products")

    top_df = df.groupby("PRODUCTCODE")["QUANTITYORDERED"].sum().sort_values(ascending=False).head(10)
    top_df = top_df.reset_index()
    top_df.columns = ["Product", "Units Sold"]

    st.bar_chart(top_df.set_index("Product"))

    # -------- CATEGORY ANALYSIS --------
    st.subheader("🏷️ Category Sales")

    cat_df = df.groupby("PRODUCTLINE")["QUANTITYORDERED"].sum().reset_index()
    cat_df.columns = ["Category", "Units Sold"]

    st.bar_chart(cat_df.set_index("Category"))

    # -------- INVENTORY INTELLIGENCE --------
    st.subheader("🧠 Inventory Intelligence")

    lead = st.slider("Lead Time (days)", 1, 30, 5)
    service = st.slider("Service Level (%)", 80, 99, 95)

    avg = daily_df["Units Sold"].mean()
    std = daily_df["Units Sold"].std()

    z = 1.65 if service >= 95 else 1.28
    safety = z * std
    reorder = (avg * lead) + safety

    st.write(f"📦 Safety Stock: {int(safety)} units")
    st.write(f"📦 Reorder Point: {int(reorder)} units")

    # -------- FORECAST --------
    st.subheader("🔮 Demand Forecast (Next 7 Days)")

    temp = daily_df.copy()
    temp["day"] = (temp["Date"] - temp["Date"].min()).dt.days

    X = temp[["day"]]
    y = temp["Units Sold"]

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(temp["day"].max()+1, temp["day"].max()+8).reshape(-1,1)
    pred = model.predict(future_days)

    future_dates = pd.date_range(temp["Date"].max(), periods=7)

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Predicted Units": pred.astype(int)
    })

    st.bar_chart(forecast.set_index("Date"))
    st.dataframe(forecast)

    # -------- FINAL INVENTORY DECISIONS --------
    st.subheader("📄 Final Inventory Decisions")

    all_products = df.groupby("PRODUCTCODE")["QUANTITYORDERED"].sum()

    report = []
    for p in all_products.index:
        sales = all_products[p]
        stock = sales * 0.5  # simulated

        if stock < reorder:
            action = "🔴 Reorder"
        elif sales < all_products.mean() * 0.3:
            action = "⚫ Stop Ordering"
        else:
            action = "🟡 Reduce Stock"

        report.append([p, int(sales), int(stock), int(reorder), action])

    final_df = pd.DataFrame(report, columns=[
        "Product", "Sales (units)", "Stock (units)", "Reorder Point", "Action"
    ])

    st.dataframe(final_df)

    # -------- DECISION GRAPH --------
    st.subheader("📊 Decision Summary")

    decision_df = final_df["Action"].value_counts().reset_index()
    decision_df.columns = ["Decision", "Count"]

    st.bar_chart(decision_df.set_index("Decision"))

    # -------- DOWNLOAD --------
    csv = final_df.to_csv(index=False).encode()
    st.download_button("📥 Download Report", csv, "inventory_report.csv")

    st.success("🚀 System Fully Working — Clean Columns + Correct Data")
