# 📦 Smart Inventory AI Dashboard

## Overview

Smart Inventory AI Dashboard is a Streamlit-based inventory management and analytics application that helps businesses analyze sales data, optimize stock levels, forecast future demand, and make intelligent inventory decisions.

The application uses data analysis and machine learning techniques to provide insights into product performance, sales trends, inventory requirements, and demand forecasting.

---

## Features

### 📊 Data Preview

* Upload sales data in CSV format
* View and inspect dataset records

### 📈 Daily Sales Analysis

* Analyze daily sales trends
* Visualize units sold over time

### 🏆 Top Products Analysis

* Identify top 10 best-selling products
* Compare product performance

### 🏷️ Category Sales Analysis

* Analyze sales by product categories
* Understand category-wise demand

### 🧠 Inventory Intelligence

* Calculate Average Daily Demand
* Calculate Safety Stock
* Calculate Reorder Point
* Adjustable Lead Time and Service Level

### 🔮 Demand Forecasting

* Predict demand for the next 7 days
* Machine Learning using Linear Regression

### 📄 Inventory Decision Support

Automatically classify products as:

* Fast Moving
* Moderate
* Slow Moving
* Dead Stock

Recommended actions:

* Increase Stock
* Maintain Stock
* Reduce Ordering
* Stop Ordering

### 📥 Report Generation

* Download inventory reports as CSV files

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Linear Regression

---

## Dataset Requirements

The uploaded CSV file should contain the following columns:

| Column Name     | Description        |
| --------------- | ------------------ |
| ORDERDATE       | Order date         |
| PRODUCTCODE     | Product identifier |
| PRODUCTLINE     | Product category   |
| QUANTITYORDERED | Quantity sold      |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/smart-inventory-ai.git
cd smart-inventory-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Project Workflow

1. Upload CSV Dataset
2. Clean and Validate Data
3. Analyze Sales Trends
4. Calculate Inventory Metrics
5. Forecast Future Demand
6. Generate Inventory Decisions
7. Download Reports

---

## Machine Learning Model

The forecasting module uses Linear Regression from Scikit-Learn to predict future inventory demand based on historical sales data.

---

## Inventory Metrics

### Safety Stock

Safety Stock = Z × Standard Deviation

### Reorder Point

Reorder Point = (Average Daily Demand × Lead Time) + Safety Stock

These metrics help businesses maintain optimal inventory levels and reduce stock shortages.

---

## Future Enhancements

* Product-Level Forecasting
* Advanced Forecasting Models (ARIMA, Prophet, LSTM)
* Real-Time Inventory Monitoring
* Database Integration
* User Authentication
* Cloud Deployment

---

## Author

Praneeth

Aspiring Data Analyst | Machine Learning Enthusiast

---

## License

This project is developed for educational and portfolio purposes.
