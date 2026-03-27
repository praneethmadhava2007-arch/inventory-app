import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Inventory AI", layout="wide")

st.title("📦 Smart Inventory Analytics Dashboard")

@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file, encoding='utf-8')
    except:
        return pd.read_csv(file, encoding='latin1')

@st.cache_resource
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = load_data(file)
    df = df.tail(5000)

    # -------- DATA PREVIEW --------
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # -------- COLUMN SELECT --------
    st.subheader("⚙️ Configure Data")

    col1, col2, col3, col4 = st.columns(4)
    date_col = col1.selectbox("📅 Date Column", df.columns)
    product_col = col2.selectbox("📦 Product Column", df.columns)
    quantity_col = col3.selectbox("🔢 Quantity Column", df.columns)
    category_col = col4.selectbox("🏷️ Category Column (Optional)", [None] + list(df.columns))

    # -------- CLEAN --------
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[quantity_col] = pd.to_numeric(df[quantity_col], errors='coerce')

    df = df.dropna(subset=[date_col, product_col, quantity_col])
    df = df[df[quantity_col] > 0]

    if df.empty:
        st.error("❌ Data invalid")
        st.stop()

    st.success("✅ Data Ready")

    # -------- SALES TREND --------
    st.subheader("📈 Sales Trend")

    daily_df = df.groupby(date_col)[quantity_col].sum().reset_index()
    daily_df.columns = ["Date", "Sales"]

    st.line_chart(daily_df.set_index("Date"))

    # -------- TOP 10 PRODUCTS --------
    st.subheader("🏆 Top 10 Products")

    top_products = df.groupby(product_col)[quantity_col].sum().sort_values(ascending=False).head(10)
    top_products_df = top_products.reset_index()
    top_products_df.columns = ["Product", "Sales"]

    st.bar_chart(top_products_df.set_index("Product"))

    # -------- TOP 10 CATEGORIES --------
    if category_col:
        st.subheader("🏷️ Top 10 Categories")

        top_cat = df.groupby(category_col)[quantity_col].sum().sort_values(ascending=False).head(10)
        top_cat_df = top_cat.reset_index()
        top_cat_df.columns = ["Category", "Sales"]

        st.bar_chart(top_cat_df.set_index("Category"))

    # -------- INVENTORY --------
    st.subheader("🧠 Inventory Intelligence")

    lead = st.slider("Lead Time (days)", 1, 30, 5)
    service = st.slider("Service Level (%)", 80, 99, 95)

    avg = daily_df["Sales"].mean()
    std = daily_df["Sales"].std()

    z = 1.65 if service >= 95 else 1.28
    safety = z * std
    reorder = (avg * lead) + safety

    st.write(f"📌 Safety Stock: {int(safety)}")
    st.write(f"📌 Reorder Point: {int(reorder)}")

    # -------- ML FORECAST --------
    st.subheader("🔮 Future Demand (ML)")

    temp = daily_df.copy()
    temp["day"] = (temp["Date"] - temp["Date"].min()).dt.days

    X = temp[["day"]]
    y = temp["Sales"]

    model = train_model(X, y)

    future_days = np.arange(temp["day"].max()+1, temp["day"].max()+8).reshape(-1,1)
    pred = model.predict(future_days)

    future_dates = pd.date_range(temp["Date"].max(), periods=7)

    forecast = pd.DataFrame({
        "Date": future_dates,
        "Predicted Demand": pred.astype(int)
    })

    st.line_chart(forecast.set_index("Date"))
    st.dataframe(forecast)

    st.info("📌 X-axis = Date | Y-axis = Demand predicted using ML")

    # -------- FINAL REPORT (ALL PRODUCTS) --------
    st.subheader("📄 Final Inventory Decisions")

    all_products = df.groupby(product_col)[quantity_col].sum()

    report = []

    for p in all_products.index:
        sales = all_products[p]
        stock = sales * 0.5

        if stock < reorder:
            action = "Reorder"
        elif sales < all_products.mean() * 0.3:
            action = "Stop Ordering"
        else:
            action = "Reduce Stock"

        report.append([p, int(sales), int(stock), int(reorder), action])

    final_df = pd.DataFrame(report, columns=[
        "Product", "Sales", "Stock", "Reorder Point", "Action"
    ])

    st.dataframe(final_df)

    # -------- DECISION GRAPH --------
    st.subheader("📊 Decision Overview")

    decision = final_df["Action"].value_counts()
    decision_df = decision.reset_index()
    decision_df.columns = ["Decision", "Count"]

    st.bar_chart(decision_df.set_index("Decision"))

    # -------- DOWNLOAD --------
    csv = final_df.to_csv(index=False).encode()
    st.download_button("📥 Download Report", csv, "inventory_report.csv")

    st.success("🚀 System Ready")
