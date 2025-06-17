import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import warnings

# --- IMPORTS FOR MACHINE LEARNING ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Superstore Analytics Dashboard",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING AND CACHING ---
@st.cache_data
def load_data():
    df = pd.read_csv('data/superstore-us.csv', encoding='latin1')
    df.drop(columns=["Postal Code", "Country"], inplace=True, errors='ignore')
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# --- ML MODEL TRAINING (Cached for performance) ---
@st.cache_resource
def train_profit_model(data):
    features = ['Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount']
    target = 'Profit'

    X = data[features]
    y = data[target]

    # Create a preprocessor for categorical and numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['Sales', 'Quantity', 'Discount']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category', 'Sub-Category'])
        ])

    # Create a pipeline that preprocesses the data then trains the model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=50, random_state=42))])
    
    model.fit(X, y)
    return model

profit_model = train_profit_model(df)


# --- SIDEBAR / NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
    (
        "📈 Executive Dashboard",
        "🗺️ Geospatial Analysis", 
        "💡 Profit Simulator", 
        "📊 Deep-Dive Explorer"
    )
)

st.sidebar.markdown("---")
st.sidebar.header("About")
about_text = """
<div style="
    text-align: justify;
    text-justify: inter-word;
    color: #cdd6f4;
    font-size: 14px;
    background-color: #292b3d;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #89b4fa;
">
This advanced analytics dashboard provides deep insights into the Superstore dataset, 
covering critical areas like sales performance, overall profitability, and detailed 
geospatial trends across the United States.
</div>
"""
st.sidebar.markdown(about_text, unsafe_allow_html=True)


# --- PAGE 1: EXECUTIVE DASHBOARD ---
if page == "📈 Executive Dashboard":
    st.title("📈 Executive Sales & Profit Dashboard")
    st.markdown("This page provides a high-level overview of key performance indicators (KPIs).")

    # --- KPIs ---
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    profit_margin = (total_profit / total_sales) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${total_sales:,.2f}", delta_color="off")
    col2.metric("Total Profit", f"${total_profit:,.2f}", delta=f"{profit_margin:.2f}% Margin")
    col3.metric("Total Orders", f"{len(df)}", delta_color="off")

    st.markdown("---")

    # --- MAIN CHARTS ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sales by Region")
        sales_by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
        fig_sales_region = px.bar(sales_by_region, x=sales_by_region.index, y='Sales', title=None, labels={'Sales': 'Total Sales ($)', 'Region': 'Region'}, color=sales_by_region.index)
        st.plotly_chart(fig_sales_region, use_container_width=True)

    with col2:
        st.subheader("Profit by Category")
        profit_by_cat = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
        fig_profit_cat = px.bar(profit_by_cat, x=profit_by_cat.index, y='Profit', title=None, labels={'Profit': 'Total Profit ($)', 'Category': 'Product Category'}, color=profit_by_cat.index)
        st.plotly_chart(fig_profit_cat, use_container_width=True)
    
    st.subheader("Sales vs. Profit Scatter Plot")
    fig_scatter = px.scatter(df, x='Sales', y='Profit', color='Category', 
                             hover_data=['Sub-Category', 'State', 'Profit', 'Sales'], 
                             title='Sales vs. Profit Across Product Categories', log_x=True)
    st.plotly_chart(fig_scatter, use_container_width=True)


# --- NEW PAGE 2: PROFIT SIMULATOR ---
elif page == "💡 Profit Simulator":
    st.title("💡 Predictive Analytics")
    st.markdown("Use the controls below to simulate an order and predict its profitability.")

    with st.expander("How does this work?"):
        st.info("A Random Forest machine learning model was trained on the historical order data to predict profit based on the inputs you select. This allows you to explore how factors like discounts and product category affect profitability.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Order Details")
        category = st.selectbox("Product Category", df['Category'].unique())
        sub_category = st.selectbox("Product Sub-Category", df[df['Category'] == category]['Sub-Category'].unique())
        quantity = st.number_input("Quantity", min_value=1, max_value=20, value=2)

    with col2:
        st.subheader("Financial Details")
        sales = st.slider("Total Sales ($)", min_value=1.0, max_value=5000.0, value=250.0, step=10.0)
        discount = st.slider("Discount (%)", min_value=0.0, max_value=80.0, value=10.0, step=5.0) / 100.0

    # Create a dataframe from the user's inputs for prediction
    input_data = pd.DataFrame({
        'Category': [category],
        'Sub-Category': [sub_category],
        'Sales': [sales],
        'Quantity': [quantity],
        'Discount': [discount]
    })
    
    # Use the trained model to predict profit
    predicted_profit = profit_model.predict(input_data)[0]

    st.markdown("---")
    st.header("Predicted Profit")

    delta_color = "normal" if predicted_profit >= 0 else "inverse"
    st.metric(
        label="Predicted Profit for this Order", 
        value=f"${predicted_profit:,.2f}",
        delta=f"{predicted_profit/sales*100 if sales > 0 else 0:.1f}% Margin",
        delta_color=delta_color
    )
    if predicted_profit < 0:
        st.warning("Warning: This configuration is predicted to result in a loss.")
    else:
        st.success("This configuration is predicted to be profitable.")


# --- PAGE 3: GEOSPATIAL ANALYSIS ---
elif page == "🗺️ Geospatial Analysis":
    st.title("🗺️ Geospatial Profit & Sales Analysis")
    st.markdown("Visualize sales and profit data across the United States.")

    us_state_to_abbrev = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
        "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
        "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
        "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
        "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
        "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
        "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
    }
    
    state_analysis = df.groupby('State').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
    state_analysis['State_Abbr'] = state_analysis['State'].map(us_state_to_abbrev)
    
    metric_to_plot = st.selectbox("Select Metric to Visualize", ["Sales", "Profit"])

    fig_map = px.choropleth(
        state_analysis,
        locations='State_Abbr',
        locationmode='USA-states',
        color=metric_to_plot,
        scope='usa',
        color_continuous_scale='Viridis',
        hover_name='State',
        hover_data={'Sales': ':.2f', 'Profit': ':.2f', 'State_Abbr': False},
        title=f'Total {metric_to_plot} by State'
    )
    fig_map.update_layout(height=600)
    st.plotly_chart(fig_map, use_container_width=True)


# --- PAGE 4: DEEP-DIVE EXPLORER ---
elif page == "📊 Deep-Dive Explorer":
    st.title("📊 Deep-Dive Data Explorer")
    st.markdown("Use the filters below to explore the dataset in detail.")

    st.sidebar.markdown("---")
    st.sidebar.header("Explorer Filters")
    
    region_filter = st.sidebar.multiselect("Filter by Region", df['Region'].unique(), default=df['Region'].unique())
    category_filter = st.sidebar.multiselect("Filter by Category", df['Category'].unique(), default=df['Category'].unique())
    
    filtered_df = df[
        (df['Region'].isin(region_filter)) &
        (df['Category'].isin(category_filter))
    ]

    st.subheader(f"Displaying {len(filtered_df)} Orders")
    st.dataframe(filtered_df)

    st.markdown("---")
    st.subheader("Filtered Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        profit_by_subcat = filtered_df.groupby('Sub-Category')['Profit'].sum().sort_values()
        fig_subcat = px.bar(profit_by_subcat, x='Profit', y=profit_by_subcat.index, orientation='h', title='Profit by Sub-Category')
        st.plotly_chart(fig_subcat, use_container_width=True)

    with col2:
        sales_by_segment = filtered_df.groupby('Segment')['Sales'].sum().sort_values()
        fig_segment_pie = px.pie(sales_by_segment, names=sales_by_segment.index, values='Sales', title='Sales by Customer Segment', hole=0.3)
        st.plotly_chart(fig_segment_pie, use_container_width=True)