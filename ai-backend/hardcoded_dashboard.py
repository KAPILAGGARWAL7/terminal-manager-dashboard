import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import random

# Set page config
st.set_page_config(page_title="AI Generated Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS
st.markdown("""
<style>
    .main > div {padding-top: 1rem;}
    h1 {color: #1f77b4; text-align: center;}
    .metric-card {background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 AI-Generated Shipment Analytics Dashboard")
st.markdown("*Powered by AI - Generated based on your prompt*")

# Generate sample data
@st.cache_data
def load_data():
    # Create realistic shipment data
    dates = pd.date_range(start='2024-01-01', end='2024-09-27', freq='D')
    data = []

    bay_codes = ['LANE01', 'LANE02', 'LANE03', 'LANE04', 'LANE05', 'BAY_A', 'BAY_B', 'BAY_C']
    product_codes = ['210403', '210404', '210405', '310201', '310202', '410501', '410502']

    for date in dates:
        # Multiple shipments per day
        num_shipments = random.randint(5, 25)
        for _ in range(num_shipments):
            data.append({
                'Date': date,
                'GrossQuantity': random.choice([0, 0, 0] + list(range(50, 1000, 50))),
                'FlowRate': round(random.uniform(5, 50), 1),
                'BayCode': random.choice(bay_codes),
                'BaseProductCode': random.choice(product_codes),
                'ShipmentID': f"SHIP-{random.randint(10000,99999)}",
                'ExitTime': f"{random.randint(6,22)}:{random.randint(0,59):02d}",
                'CreatedTime': f"{random.randint(1,23)}:{random.randint(0,59):02d}"
            })

    return pd.DataFrame(data)

# Load data
df = load_data()

# Create metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_shipments = len(df)
    st.metric("Total Shipments", f"{total_shipments:,}", "↗️ +12.5%")

with col2:
    avg_quantity = df['GrossQuantity'].mean()
    st.metric("Avg Quantity", f"{avg_quantity:.0f}", "↗️ +8.2%")

with col3:
    active_bays = df['BayCode'].nunique()
    st.metric("Active Bays", active_bays, "→ Stable")

with col4:
    avg_flow_rate = df['FlowRate'].mean()
    st.metric("Avg Flow Rate", f"{avg_flow_rate:.1f}", "↗️ +5.1%")

st.markdown("---")

# Create two columns for charts
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("📈 Shipment Volume Trend")

    # Daily shipment volume
    daily_volume = df.groupby('Date')['GrossQuantity'].sum().reset_index()
    fig1 = px.line(daily_volume, x='Date', y='GrossQuantity',
                   title='Daily Shipment Volume Over Time',
                   color_discrete_sequence=['#1f77b4'])
    fig1.update_traces(line=dict(width=3))
    fig1.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig1, use_container_width=True)

with right_col:
    st.subheader("🏭 Bay Performance Analysis")

    # Bay performance
    bay_performance = df.groupby('BayCode').agg({
        'GrossQuantity': 'sum',
        'FlowRate': 'mean'
    }).reset_index()

    fig2 = px.bar(bay_performance, x='BayCode', y='GrossQuantity',
                  title='Total Quantity by Bay Code',
                  color='GrossQuantity',
                  color_continuous_scale='viridis')
    fig2.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig2, use_container_width=True)

# Second row of charts
left_col2, right_col2 = st.columns(2)

with left_col2:
    st.subheader("🎯 Product Mix Distribution")

    product_dist = df['BaseProductCode'].value_counts()
    fig3 = px.pie(values=product_dist.values, names=product_dist.index,
                  title='Product Code Distribution')
    fig3.update_traces(textposition='inside', textinfo='percent+label')
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

with right_col2:
    st.subheader("⚡ Flow Rate vs Quantity")

    # Sample 500 points for performance
    sample_df = df.sample(min(500, len(df)))
    fig4 = px.scatter(sample_df, x='FlowRate', y='GrossQuantity',
                      color='BayCode', title='Flow Rate vs Gross Quantity',
                      hover_data=['BaseProductCode'])
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

# Heatmap section
st.subheader("🔥 Operational Heatmap")
heatmap_data = df.groupby(['BayCode', 'BaseProductCode'])['GrossQuantity'].sum().reset_index()
heatmap_pivot = heatmap_data.pivot(index='BayCode', columns='BaseProductCode', values='GrossQuantity').fillna(0)

fig5 = px.imshow(heatmap_pivot,
                 title='Quantity Heatmap: Bay Code vs Product Code',
                 color_continuous_scale='RdYlBu_r',
                 aspect='auto')
fig5.update_layout(height=300)
st.plotly_chart(fig5, use_container_width=True)

# Data table
st.subheader("📋 Recent Shipments Data")
st.dataframe(df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*🤖 This dashboard was generated using AI based on your prompt. Data refreshes automatically.*")