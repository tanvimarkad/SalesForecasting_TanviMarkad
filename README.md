# 📊 End-to-End Sales Forecasting & Demand Intelligence System

A complete Data Science and Machine Learning project that analyzes historical retail sales, forecasts future demand, detects anomalies, and segments products using clustering techniques. The project includes an interactive Streamlit dashboard for business users.

---

## 🚀 Project Overview

This project helps retail businesses:

- Analyze historical sales trends
- Forecast future sales using multiple models
- Detect unusual sales patterns (anomalies)
- Segment products based on demand
- Support inventory and supply chain decision-making

---

## 📌 Features

- 📈 Interactive Sales Dashboard
- 📅 Monthly Sales Trend Analysis
- 🔮 Sales Forecasting (SARIMA, Prophet & XGBoost)
- 🚨 Anomaly Detection
  - Z-Score
  - Isolation Forest
- 📦 Product Demand Segmentation using K-Means Clustering
- 📊 Interactive Plotly Visualizations
- 📥 Downloadable Reports
- 🎨 Professional Streamlit UI

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Plotly
- Scikit-learn
- XGBoost
- Prophet
- Statsmodels
- Seaborn

---

## 📂 Project Structure

```
SalesForecasting_TanviMarkad/
│
├── app.py
├── train.csv
├── style.css
├── requirements.txt
├── analysis.ipynb
├── Charts/
│   ├── Monthly Sales Trend.png
│   ├── Product Demand Segmentation.png
│   ├── Furniture Forecast.png
│   ├── Isolation Forest Anomaly Detection.png
│   ├── Z-Score Anomaly Detection.png
│   └── ...
└── README.md
```

---

## 📊 Dashboard Modules

### 📈 Sales Dashboard

- Total Sales
- Average Sales
- Order Analysis
- Regional Performance

### 🔮 Forecast Explorer

- SARIMA Forecast
- Prophet Forecast
- XGBoost Forecast
- Actual vs Predicted Comparison

### 🚨 Anomaly Report

- Z-Score Detection
- Isolation Forest Detection
- Anomaly Visualization

### 📦 Demand Segmentation

- K-Means Clustering
- PCA Visualization
- Cluster Summary
- Product Segmentation

---

## 📈 Machine Learning Models

| Model | Purpose |
|--------|----------|
| SARIMA | Time Series Forecasting |
| Prophet | Trend & Seasonality Forecasting |
| XGBoost | Machine Learning Forecasting |
| Isolation Forest | Anomaly Detection |
| K-Means | Product Demand Segmentation |

---

## 📊 Model Performance

| Model | MAE | RMSE | MAPE |
|-------|-------:|-------:|-------:|
| SARIMA | 18031.40 | 19009.18 | 18.97% |
| Prophet | 20250.79 | 22318.41 | 21.86% |
| **XGBoost** | **14443.46** | **17069.09** | **14.45%** |

**Best Model:** XGBoost

---

## 📷 Project Screenshots

Include screenshots from the `Charts` folder.

- Sales Dashboard
- Monthly Sales Trend
- Forecast Results
- Product Demand Segmentation
- Anomaly Detection

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/tanvimarkad/SalesForecasting_TanviMarkad.git
```

Move into the project folder

```bash
cd SalesForecasting_TanviMarkad
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📁 Dataset

Dataset: Superstore Sales Dataset

Contains

- Order Date
- Ship Date
- Sales
- Category
- Sub-Category
- Region
- Customer Details
- Product Details

---

## 📈 Business Insights

- Technology category generated the highest revenue.
- West region showed the most consistent sales growth.
- November and December had the highest seasonal demand.
- XGBoost achieved the best forecasting performance.
- High-demand product clusters require frequent inventory replenishment.

---

## 🎯 Future Improvements

- Real-time forecasting
- Cloud deployment
- Automated report generation
- Deep Learning (LSTM)
- Power BI Integration
- Live inventory monitoring

---

## 👩‍💻 Author

**Tanvi Markad**

AI & Data Science Student

Graphic Designer | Video Editor | Machine Learning Enthusiast

GitHub: https://github.com/tanvimarkad

---

## ⭐ If you like this project

Please ⭐ Star this repository and share your feedback!
