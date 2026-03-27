import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Inventory AI", layout="wide")

st.title("📦 Smart Inventory Analytics Dashboard")

# -------- CACHE DATA --------
@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file, encoding='utf-8')
    except:
        return pd.read_csv(file, encoding='latin1')

# -------- CACHE MODEL --------
@st.cache_resource
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = load_data(file)

    # -------- LIMIT DATA (FAST) --------
    df = df.tail(5000)

    # -------- COLUMN INFO --------
    st.subheader("📌 Dataset Columns")
    st.write(df.columns.tolist())

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # -------- SELECT COLUMNS --------
    st.subheader("⚙️ Configure Data")

    col1, col2, col3, col4 = st.columns(4)
    date_col = col1.selectbox("Date Column", df.columns)
    product_col = col2.selectbox("Product Column", df.columns)
    quantity_col = col3.selectbox("Quantity Column", df.columns)
    category_col = col4.selectbox("Category Column (Optional)", [None] + list(df.columns))

    # -------- CLEAN DATA --------
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[quantity_col] = pd.to_numeric(df[quantity_col], errors='coerce')

    df = df.dropna(subset=[date_col, product_col, quantity_col])
    df = df[df[quantity_col] > 0]

    if df.empty:
        st.error("❌ Data empty after cleaning")
        st.stop()

    st.success("✅ Data Ready")

    # -------- DAILY SALES --------
    st.subheader("📈 Sales Trend")

    daily = df.groupby(date_col)[quantity_col].sum()
    daily_df = daily.to_frame(name="Sales").reset_index()
    daily_df.rename(columns={date_col: "Date"}, inplace=True)

    st.line_chart(daily_df.set_index("Date"))

    # -------- TOP PRODUCTS --------
    st.subheader("🏆 Top 10 Products")

    product_sales = df.groupby(product_col)[quantity_col].sum().sort_values(ascending=False).head(10)

    product_df = product_sales.reset_index()
    product_df.columns = ["Product", "Sales"]

    st.bar_chart(product_df.set_index("Product"))

    # -------- TOP CATEGORIES --------
    if category_col:
        st.subheader("🏷️ Top 10 Categories")

        category_sales = df.groupby(category_col)[quantity_col].sum().sort_values(ascending=False).head(10)

        cat_df = category_sales.reset_index()
        cat_df.columns = ["Category", "Sales"]

        st.bar_chart(cat_df.set_index("Category"))

    # -------- INVENTORY INPUT --------
    st.subheader("🧠 Inventory Parameters")

    lead = st.slider("Lead Time (days)", 1, 30, 5)
    service = st.slider("Service Level (%)", 80, 99, 95)

    avg_daily = daily_df['Sales'].mean()
    std = daily_df['Sales'].std()

    z = 1.65 if service >= 95 else 1.28
    safety = z * std
    reorder = (avg_daily * lead) + safety

    st.write(f"📌 Safety Stock: {int(safety)}")
    st.write(f"📌 Reorder Point: {int(reorder)}")

    # -------- ML FORECAST --------
    st.subheader("🔮 Future Demand (ML)")

    temp = daily_df.copy()
    temp['day'] = (temp['Date'] - temp['Date'].min()).dt.days

    X = temp[['day']]
    y = temp['Sales']

    model = train_model(X, y)

    future_days = np.arange(temp['day'].max()+1, temp['day'].max()+8).reshape(-1,1)
    pred = model.predict(future_days)

    future_dates = pd.date_range(temp['Date'].max(), periods=7)

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Predicted Demand": pred.astype(int)
    })

    st.line_chart(forecast.set_index("Date"))
    st.dataframe(forecast)

    st.info("📌 X-axis = Date | Y-axis = Predicted Demand (ML Linear Regression)")

    # -------- FINAL REPORT --------
    st.subheader("📄 Final Intelligent Report")

    report_data = []

    for p in product_df["Product"]:
        sales_val = product_df[product_df["Product"] == p]["Sales"].values[0]
        stock = sales_val * 0.5

        if stock < reorder:
            action = "Reorder"
        elif sales_val < product_df["Sales"].mean() * 0.3:
            action = "Stop Ordering"
        else:
            action = "Reduce Stock"

        report_data.append([
            p,
            int(sales_val),
            int(stock),
            int(reorder),
            action
        ])

    final_report = pd.DataFrame(report_data, columns=[
        "Product",
        "Sales",
        "Current Stock",
        "Reorder Point",
        "Action"
    ])

    st.dataframe(final_report)

    # -------- DECISION GRAPH --------
    st.subheader("📊 Decision Overview")

    decision_counts = final_report["Action"].value_counts()

    decision_df = decision_counts.reset_index()
    decision_df.columns = ["Decision", "Count"]

    st.bar_chart(decision_df.set_index("Decision"))

    # -------- DOWNLOAD --------
    csv = final_report.to_csv(index=False).encode()
    st.download_button("📥 Download Report", csv, "inventory_report.csv")

    st.success("🚀 System Ready")