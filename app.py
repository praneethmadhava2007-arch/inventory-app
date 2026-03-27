import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Inventory AI", layout="wide")

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;'>📦 Welcome to Smart Inventory Dashboard</h1>
<hr>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file, encoding='latin1')

file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    df = load_data(file)

    # ---------------- CLEAN DATA ----------------
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors='coerce')
    df["QUANTITYORDERED"] = pd.to_numeric(df["QUANTITYORDERED"], errors='coerce')

    df = df.dropna(subset=["ORDERDATE", "QUANTITYORDERED", "PRODUCTCODE"])
    df = df[df["QUANTITYORDERED"] > 0]

    # ---------------- NAVIGATION ----------------
    option = st.selectbox(
        "📊 Select Preview Type",
        [
            "Data Preview",
            "Daily Sales",
            "Top Products",
            "Category Sales",
            "Inventory Insights",
            "Forecast",
            "Final Decisions"
        ]
    )

    # ---------------- DATA PREVIEW ----------------
    if option == "Data Preview":
        st.subheader("📊 Data Preview")
        st.write("Rows = records | Columns = dataset features")
        st.dataframe(df.head(), use_container_width=True)

    # ---------------- DAILY SALES ----------------
    elif option == "Daily Sales":
        st.subheader("📈 Daily Sales (Units)")
        st.write("X-axis = Date | Y-axis = Units Sold")

        daily_df = df.groupby("ORDERDATE")["QUANTITYORDERED"].sum().reset_index()
        daily_df.columns = ["Date", "Units Sold"]

        st.bar_chart(daily_df.set_index("Date"))

    # ---------------- TOP PRODUCTS ----------------
    elif option == "Top Products":
        st.subheader("🏆 Top 10 Products")
        st.write("X-axis = Product | Y-axis = Units Sold")

        top_df = df.groupby("PRODUCTCODE")["QUANTITYORDERED"].sum().sort_values(ascending=False).head(10)
        top_df = top_df.reset_index()
        top_df.columns = ["Product", "Units Sold"]

        st.bar_chart(top_df.set_index("Product"))

    # ---------------- CATEGORY SALES ----------------
    elif option == "Category Sales":
        st.subheader("🏷️ Category Sales")
        st.write("X-axis = Category | Y-axis = Units Sold")

        cat_df = df.groupby("PRODUCTLINE")["QUANTITYORDERED"].sum().reset_index()
        cat_df.columns = ["Category", "Units Sold"]

        st.bar_chart(cat_df.set_index("Category"))

    # ---------------- INVENTORY ----------------
    elif option == "Inventory Insights":
        st.subheader("🧠 Inventory Intelligence")

        daily_df = df.groupby("ORDERDATE")["QUANTITYORDERED"].sum().reset_index()
        avg = daily_df["QUANTITYORDERED"].mean()
        std = daily_df["QUANTITYORDERED"].std()

        lead = st.slider("Lead Time (days)", 1, 30, 5)
        service = st.slider("Service Level (%)", 80, 99, 95)

        z = 1.65 if service >= 95 else 1.28
        safety = z * std
        reorder = (avg * lead) + safety

        st.write(f"📦 Avg Daily Demand: {int(avg)} units/day")
        st.write(f"📦 Safety Stock: {int(safety)} units")
        st.write(f"📦 Reorder Point: {int(reorder)} units")

    # ---------------- FORECAST ----------------
    elif option == "Forecast":
        st.subheader("🔮 Demand Forecast (Next 7 Days)")
        st.write("X-axis = Date | Y-axis = Predicted Units")

        daily_df = df.groupby("ORDERDATE")["QUANTITYORDERED"].sum().reset_index()
        daily_df.columns = ["Date", "Units Sold"]

        daily_df["day"] = (daily_df["Date"] - daily_df["Date"].min()).dt.days

        X = daily_df[["day"]]
        y = daily_df["Units Sold"]

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.arange(daily_df["day"].max()+1, daily_df["day"].max()+8).reshape(-1,1)
        pred = model.predict(future_days)

        future_dates = pd.date_range(daily_df["Date"].max(), periods=7)

        forecast = pd.DataFrame({
            "Date": future_dates,
            "Predicted Units": pred.astype(int)
        })

        st.bar_chart(forecast.set_index("Date"))
        st.dataframe(forecast)

    # ---------------- FINAL DECISIONS ----------------
    elif option == "Final Decisions":
        st.subheader("📄 Final Inventory Decisions")

        product_sales = df.groupby("PRODUCTCODE")["QUANTITYORDERED"].sum()
        avg_sales = product_sales.mean()

        report = []

        for p in product_sales.index:
            sales = product_sales[p]
            stock = sales * 0.6

            # Improved logic
            if sales > avg_sales * 1.2:
                movement = "Fast Moving"
                action = "Increase Stock"
            elif sales > avg_sales * 0.6:
                movement = "Moderate"
                action = "Maintain"
            elif sales > avg_sales * 0.3:
                movement = "Slow Moving"
                action = "Reduce Ordering"
            else:
                movement = "Dead Stock"
                action = "Stop Ordering"

            report.append([p, int(sales), int(stock), movement, action])

        final_df = pd.DataFrame(report, columns=[
            "Product", "Sales (units)", "Stock (units)", "Category", "Action"
        ])

        st.dataframe(final_df)

        # -------- CATEGORY TABLES --------
        st.subheader("📊 Stock Classification")

        fast = final_df[final_df["Category"] == "Fast Moving"]
        slow = final_df[final_df["Category"] == "Slow Moving"]
        dead = final_df[final_df["Category"] == "Dead Stock"]

        col1, col2, col3 = st.columns(3)

        col1.write("🚀 Fast Moving")
        col1.dataframe(fast)

        col2.write("🐢 Slow Moving")
        col2.dataframe(slow)

        col3.write("❌ Dead Stock")
        col3.dataframe(dead)

        # -------- DECISION GRAPH --------
        st.subheader("📊 Decision Summary")
        st.write("X-axis = Decision | Y-axis = Number of Products")

        decision_df = final_df["Action"].value_counts().reset_index()
        decision_df.columns = ["Decision", "Count"]

        st.bar_chart(decision_df.set_index("Decision"))

        # -------- DOWNLOAD --------
        csv = final_df.to_csv(index=False).encode()
        st.download_button("📥 Download Report", csv, "inventory_report.csv")

        st.success("🚀 Intelligent System Running Successfully")
