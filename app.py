import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)
st.title("End-to-End Sales Forecasting & Demand Intelligence System")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Sales Dashboard",
        "Forecast Explorer",
        "Anomaly Report",
        "Demand Segmentation"
    ]
)
import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("train.csv", encoding="latin1")

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        # Convert Order Date if it exists
        if "Order Date" in df.columns:
            df["Order Date"] = pd.to_datetime(
                df["Order Date"],
                dayfirst=True,
                errors="coerce"
            )

        # Convert Ship Date if it exists
        if "Ship Date" in df.columns:
            df["Ship Date"] = pd.to_datetime(
                df["Ship Date"],
                dayfirst=True,
                errors="coerce"
            )

        # Remove invalid date rows
        if "Order Date" in df.columns:
            df = df.dropna(subset=["Order Date"])

        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
df = load_data()

if df is None:
    st.stop()
# ==========================
# SALES DASHBOARD
# ==========================

if page == "Sales Dashboard":
    st.title("📊 Sales Dashboard")
    total_sales = df["Sales"].sum() if "Sales" in df.columns else 0
    total_profit = df["Profit"].sum() if "Profit" in df.columns else None
    total_orders = len(df)
    avg_sales = df["Sales"].mean() if "Sales" in df.columns else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "💰 Total Sales",
        f"${total_sales:,.2f}"
    )
    if total_profit is not None:
        c2.metric(
            "📈 Total Profit",
            f"${total_profit:,.2f}"
        )
    else:
        c2.metric(
            "📈 Total Profit",
            "N/A"
        )

    c3.metric(
        "🛒 Total Orders",
        total_orders
    )

    c4.metric(
        "📦 Average Sales",
        f"${avg_sales:,.2f}"
    )

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    with st.expander("Dataset Columns"):
        st.write(df.columns.tolist())
    
##################################################
# YEARLY SALES
##################################################

    yearly = df.groupby(
        df["Order Date"].dt.year
    )["Sales"].sum().reset_index()
    yearly.columns=["Year","Sales"]
    fig = px.bar( yearly, x="Year", y="Sales", color="Sales", title="Yearly Sales")
    st.plotly_chart(fig,use_container_width=True)

##################################################
# MONTHLY SALES
##################################################

    monthly = df.groupby(
        pd.Grouper(key="Order Date",freq="ME")
    )["Sales"].sum().reset_index()
    fig = px.line(monthly,x="Order Date",y="Sales",markers=True,title="Monthly Sales Trend")
    st.plotly_chart(fig,use_container_width=True)

##################################################
# REGION FILTER
##################################################

    st.subheader("Region Analysis")
    region = st.selectbox("Choose Region",sorted(df["Region"].unique()))
    region_df = df[
        df["Region"]==region

    ]

##################################################
# CATEGORY SALES
##################################################

    category_sales = region_df.groupby("Category")["Sales"].sum().reset_index()
    fig = px.bar(category_sales,x="Category",y="Sales",color="Category",title="Category Wise Sales")
    st.plotly_chart(fig,use_container_width=True)

##################################################
# SUB CATEGORY
##################################################

    sub = region_df.groupby("Sub-Category")["Sales"].sum().reset_index()
    fig = px.pie(sub,names="Sub-Category",values="Sales",title="Sub Category Distribution")
    st.plotly_chart(fig,use_container_width=True)

##################################################
# STATE SALES
##################################################

    state = region_df.groupby("State")["Sales"].sum().reset_index()
    fig = px.bar(state,x="State",y="Sales",color="Sales",title="State Wise Sales")
    st.plotly_chart(fig,use_container_width=True)

##################################################
# TOP CUSTOMERS
##################################################

    st.subheader("Top Customers")
    top_customer = df.groupby("Customer Name")["Sales"].sum().nlargest(10).reset_index()
    st.dataframe(top_customer)

##################################################
# TOP PRODUCTS
##################################################

    st.subheader("Top Products")
    top_product = df.groupby(
        "Product Name"
    )["Sales"].sum().nlargest(10).reset_index()
    st.dataframe(top_product)

##################################################
# DATASET
##################################################

    if st.checkbox("Show Dataset"):
        st.dataframe(df)

##################################################
# DOWNLOAD
##################################################

    csv = df.to_csv(index=False)
    st.download_button("Download Dataset",csv,"sales.csv","text/csv" )

#############################################
# FORECAST EXPLORER
#############################################

if page=="Forecast Explorer":
    st.header("📈 Sales Forecast")
    option = st.selectbox(
        "Forecast By",
        ["Category","Region"]
    )
    if option=="Category":
        value = st.selectbox(
            "Select Category",
            df["Category"].unique()
        )
        temp = df[df["Category"]==value]
    else:
        value = st.selectbox(
            "Select Region",
            df["Region"].unique()
        )
        temp = df[df["Region"]==value]
    sales = temp.groupby(
        pd.Grouper(
            key="Order Date",
            freq="ME"
        )
    )["Sales"].sum()

    model = SARIMAX(
        sales,
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    )
    result = model.fit(disp=False)
    forecast = result.forecast(3)
    future = pd.date_range(
        sales.index[-1],
        periods=4,
        freq="ME"
    )[1:]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=sales.index, y=sales.values, mode="lines", name="Actual" )
    )
    fig.add_trace(
        go.Scatter( x=future, y=forecast, mode="lines+markers", name="Forecast")
    )
    st.plotly_chart(fig, use_container_width=True)
    forecast_df = pd.DataFrame({ "Date":future, "Forecast Sales":forecast})
    st.subheader("Forecast Table")
    st.dataframe(forecast_df)

########################################################
# ANOMALY REPORT
########################################################

if page=="Anomaly Report":
    st.header("Sales Anomaly Detection")
    weekly = df.groupby(
        pd.Grouper(key="Order Date",freq="W")
    )["Sales"].sum().reset_index()
    iso = IsolationForest(contamination=0.03,random_state=42)
    weekly["Anomaly"] = iso.fit_predict(
        weekly[["Sales"]]
    )

    anomaly = weekly[
        weekly["Anomaly"]==-1
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter( x=weekly["Order Date"], y=weekly["Sales"], mode="lines", name="Weekly Sales")
    )

    fig.add_trace(
        go.Scatter(x=anomaly["Order Date"],y=anomaly["Sales"],mode="markers",marker=dict( color="red", size=10 ),
            name="Anomaly"
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Detected Anomalies")
    st.dataframe(anomaly)
    csv = anomaly.to_csv(index=False)
    st.download_button(
        "Download Anomaly Report",
        csv,
        "anomaly_report.csv",
        "text/csv"
    )
# ==========================================
# DEMAND SEGMENTATION
# ==========================================

if page == "Demand Segmentation":

    st.header("Product Demand Segmentation")
    cluster = df.groupby("Sub-Category").agg({
        "Sales": "sum"
    }).reset_index()
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = cluster[["Sales"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )
    cluster["Cluster"] = model.fit_predict(X_scaled)
    pca = PCA(n_components=1)
    pca_data = pca.fit_transform(X_scaled)

    cluster["PCA1"] = pca_data[:, 0]
    cluster["PCA2"] = 0   
    fig = px.scatter(
        cluster,
        x="PCA1",
        y="PCA2",
        color=cluster["Cluster"].astype(str),
        hover_name="Sub-Category",
        size="Sales",
        title="Product Demand Segmentation"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Cluster Details")
    st.dataframe(cluster)
    summary = cluster.groupby("Cluster").agg({
        "Sales": ["sum", "mean", "count"]
    })
    st.subheader("Cluster Summary")
    st.dataframe(summary)
    csv = cluster.to_csv(index=False)
    st.download_button(
        "Download Cluster Report",
        csv,
        "cluster_report.csv",
        "text/csv"
    )

########################################################
# SIDEBAR FOOTER
########################################################

st.sidebar.markdown("---")
st.sidebar.success(
    "AI Internship Project\n\n"
    "End-to-End Sales Forecasting\n\n"
    "Developed by\n"
    "Tanvi Vinod Markad"
)

########################################################
# FOOTER
########################################################

st.markdown("---")
st.caption(
    "End-to-End Sales Forecasting & Demand Intelligence System | "
    "AI & ML Internship Project"
)

 
import streamlit as st

def load_css():
    with open("style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()