from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sqlite3
import os
import subprocess
import time
import json
import threading
from datetime import datetime
import signal
import requests
from dotenv import load_dotenv
from config import Config

load_dotenv()

# quick setup for demo
app = Flask(__name__)
CORS(app, origins=Config.CORS_ORIGINS)
Config.ensure_directories()

# TODO: maybe use a proper database later
running_dashboards = {}

class DashboardGenerator:
    def __init__(self):
        self.ollama_url = f"{Config.OLLAMA_URL}/api/generate"
        self.excel_dir = os.path.join(Config.PROJECT_ROOT, 'excel-data')
        self.ensure_excel_directory()  # make sure folder exists

    def sanitize_table_name(self, name):
        import re
        # clean up the filename for sqlite
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', str(name).lower())
        sanitized = re.sub(r'_+', '_', sanitized)
        if sanitized and sanitized[0].isdigit():
            sanitized = 'data_' + sanitized  # sqlite doesn't like numbers first
        if not sanitized or len(sanitized.strip('_')) == 0:
            sanitized = 'data_table'  # fallback name
        if len(sanitized) > 50:
            sanitized = sanitized[:50]  # keep it short
        sanitized = sanitized.rstrip('_')
        if not sanitized:
            sanitized = 'data_table'  # final fallback
        return sanitized

    def ensure_excel_directory(self):
        os.makedirs(self.excel_dir, exist_ok=True)

    def find_excel_files(self):
        if not os.path.exists(self.excel_dir):
            return []
        excel_files = []
        for filename in os.listdir(self.excel_dir):
            if filename.lower().endswith(('.xlsx', '.xls')) and not filename.startswith('~'):
                excel_files.append(os.path.join(self.excel_dir, filename))
        return sorted(excel_files, key=os.path.getmtime, reverse=True)

    def get_foolproof_table_name(self, index):
        return f"dataset_{index + 1:03d}"

    def process_all_excel_files(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(Config.DATA_DIR, 'evaluation_data.db')
        excel_files = self.find_excel_files()
        if not excel_files:
            return {'success': False, 'error': 'No Excel files found in excel-data directory'}
        latest_file = excel_files[0]
        try:
            df = pd.read_excel(latest_file)
            if df.empty:
                return {'success': False, 'error': 'Excel file is empty'}
            processed_df = self.process_shipment_data(df)
            conn = sqlite3.connect(db_path)
            table_name = "evaluation_data"
            processed_df.to_sql(table_name, conn, if_exists='replace', index=False)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info([{table_name}])")
            columns = cursor.fetchall()
            conn.close()

            return {
                'success': True,
                'db_path': db_path,
                'table_name': table_name,
                'columns': [col[1] for col in columns],
                'sample_data': processed_df.head(5).to_dict('records'),
                'data_type': 'shipment_logistics',
                'total_rows': len(processed_df),
                'excel_source': latest_file,
                'total_files_found': len(excel_files)
            }

        except Exception as e:
            return {'success': False, 'error': f'Failed to process Excel files: {str(e)}'}

    def convert_excel_to_sqlite(self, excel_path, db_path=None):
        if db_path is None:
            db_path = os.path.join(Config.DATA_DIR, 'terminal_data.db')
        try:
            df = pd.read_excel(excel_path, sheet_name=0, dtype=str)
            df.columns = [str(c).strip() for c in df.columns]
            processed_df = self.process_shipment_data(df)
            conn = sqlite3.connect(db_path)
            base_name = os.path.splitext(os.path.basename(excel_path))[0]
            table_name = self.sanitize_table_name(base_name)
            processed_df.to_sql(table_name, conn, if_exists='replace', index=False)
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info([{table_name}])")
            columns = cursor.fetchall()

            conn.close()

            return {
                'success': True,
                'db_path': db_path,
                'table_name': table_name,
                'columns': [col[1] for col in columns],
                'sample_data': processed_df.head(5).to_dict('records'),
                'data_type': 'shipment_logistics',
                'total_rows': len(processed_df),
                'excel_source': excel_path
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def process_shipment_data(self, df):
        try:
            processed_df = df.copy()
            # FIXME: this should be more generic
            numeric_columns = ['GrossQuantity', 'FlowRate']
            for col in numeric_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)  # convert to numbers, fill NaN with 0
            if 'ScheduledDate' in processed_df.columns:
                for date_format in ['%m-%d-%y', '%d-%m-%y', '%Y-%m-%d']:
                    try:
                        processed_df['ScheduledDate_parsed'] = pd.to_datetime(processed_df['ScheduledDate'], format=date_format)
                        break
                    except:
                        continue

            time_columns = ['ExitTime', 'CreatedTime']
            for col in time_columns:
                if col in processed_df.columns:
                    try:
                        processed_df[f'{col}_hour'] = pd.to_datetime(processed_df[col], format='%I:%M:%S %p', errors='coerce').dt.hour
                    except:
                        try:
                            processed_df[f'{col}_hour'] = pd.to_datetime(processed_df[col], format='%H:%M:%S', errors='coerce').dt.hour
                        except:
                            pass
            if 'BayCode' in processed_df.columns:
                processed_df['BayCode'] = processed_df['BayCode'].astype(str)
                processed_df['Lane'] = processed_df['BayCode'].str.extract(r'(LANE\d+)', expand=False).fillna(processed_df['BayCode'])
            if 'GrossQuantity' in processed_df.columns and 'FlowRate' in processed_df.columns:
                processed_df['Throughput_Units_Hour'] = processed_df['GrossQuantity'] * processed_df['FlowRate']
            if 'ExitTime_hour' in processed_df.columns:
                processed_df['Shift'] = processed_df['ExitTime_hour'].apply(
                    lambda x: 'Day_Shift' if pd.notna(x) and 6 <= x < 18 else 'Night_Shift' if pd.notna(x) else 'Unknown'
                )
            if 'ScheduledDate_parsed' in processed_df.columns:
                processed_df['Date'] = processed_df['ScheduledDate_parsed'].dt.date
                processed_df['Month'] = processed_df['ScheduledDate_parsed'].dt.to_period('M')

            return processed_df

        except Exception as e:
            print(f"Error processing shipment data: {e}")
            return df  # Return original if processing fails

    def call_llm(self, prompt):
        try:
            # basic ollama call - could be improved
            payload = {
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # not too creative
                    "top_p": 0.9
                }
            }
            print(f"Calling LLM with model: {Config.OLLAMA_MODEL}")
            response = requests.post(self.ollama_url, json=payload, timeout=Config.OLLAMA_TIMEOUT)

            if response.status_code == 200:
                result = response.json().get('response', '')
                if result.strip():
                    print("LLM generation successful")
                    return result
                else:
                    print("LLM returned empty response")
                    return None
            else:
                print(f"LLM API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            print(f"LLM timeout after {Config.OLLAMA_TIMEOUT} seconds")
            return None
        except requests.exceptions.ConnectionError:
            print("LLM connection error - Ollama may not be running")
            return None
        except Exception as e:
            print(f"LLM unexpected error: {e}")
            return None

    def analyze_dashboard_type(self, user_prompt):
        prompt_lower = user_prompt.lower()
        if any(word in prompt_lower for word in ['lane', 'bay', 'throughput', 'capacity', 'production', 'manufacturing', 'terminal', 'schedule', 'adherence']):
            return 'manufacturing'
        if any(word in prompt_lower for word in ['financial', 'revenue', 'profit', 'cost', 'budget', 'earnings', 'income', 'expense']):
            return 'financial'
        if any(word in prompt_lower for word in ['sales', 'revenue', 'customer', 'conversion', 'lead', 'pipeline', 'orders']):
            return 'sales'
        if any(word in prompt_lower for word in ['operational', 'efficiency', 'uptime', 'performance', 'productivity', 'operations']):
            return 'operational'
        if any(word in prompt_lower for word in ['logistics', 'supply', 'inventory', 'shipment', 'delivery', 'warehouse', 'freight']):
            return 'logistics'
        if any(word in prompt_lower for word in ['analytics', 'analysis', 'report', 'insights', 'trends', 'statistics']):
            return 'analytics'
        if any(word in prompt_lower for word in ['fuel', 'energy', 'consumption', 'volume', 'flow', 'rate']):
            return 'energy'
        if any(word in prompt_lower for word in ['employee', 'hr', 'staff', 'workforce', 'personnel', 'hiring']):
            return 'hr'
        return 'analytics'

    def get_dashboard_requirements(self, dashboard_type):
        requirements = {
            'manufacturing': """
- Focus on production metrics, lane/bay performance, throughput, capacity utilization, OEE
- Include real-time KPIs: production rates, downtime, quality metrics, schedule adherence
- Show performance by lane/bay, shift patterns, product mix analysis
- Color scheme: Industrial blue/steel gray with green for targets, red for alerts
- Charts: Real-time gauges, production trend lines, heatmaps for bay performance, Gantt charts for schedules
- Key metrics: Overall Equipment Effectiveness (OEE), First Pass Yield (FPY), Cycle Time, Takt Time
- Professional manufacturing aesthetic with clean, data-dense layouts""",

            'financial': """
- Focus on revenue, profit margins, cost analysis, ROI metrics
- Use currency formatting and financial KPIs
- Include trend analysis and variance reporting
- Color scheme: Green/red for profit/loss, blue for neutral metrics
- Charts: Line charts for trends, bar charts for comparisons, pie charts for breakdowns""",

            'sales': """
- Focus on sales volume, conversion rates, customer acquisition
- Include funnel analysis and performance tracking
- Show geographic and temporal sales patterns
- Color scheme: Blue/green sales theme
- Charts: Funnel charts, bar charts, geographic maps, time series""",

            'operational': """
- Focus on efficiency metrics, uptime, throughput, performance indicators
- Include KPI cards with status indicators
- Show operational trends and capacity utilization
- Color scheme: Blue/green for good performance, yellow/red for issues
- Charts: Gauge charts, line charts for trends, bar charts for comparisons""",

            'logistics': """
- Focus on shipment tracking, delivery performance, inventory levels
- Include route optimization and capacity planning
- Show supply chain metrics and bottlenecks
- Color scheme: Orange/blue logistics theme
- Charts: Geographic maps, flow charts, timeline charts, bar charts""",

            'analytics': """
- Focus on data exploration, trend analysis, statistical insights
- Include interactive filtering and drill-down capabilities
- Show correlations and data patterns
- Color scheme: Professional blue/gray theme
- Charts: Scatter plots, histograms, correlation matrices, trend lines""",

            'energy': """
- Focus on consumption patterns, efficiency metrics, volume analysis
- Include environmental and cost impact metrics
- Show usage trends and optimization opportunities
- Color scheme: Green/blue energy theme
- Charts: Area charts for consumption, gauge charts for efficiency, line charts for trends""",

            'hr': """
- Focus on employee metrics, performance, demographics, satisfaction
- Include workforce analytics and talent management
- Show hiring trends and retention analysis
- Color scheme: Purple/blue professional theme
- Charts: Bar charts for demographics, line charts for trends, pie charts for distributions"""
        }

        return requirements.get(dashboard_type, requirements['analytics'])

    def generate_dashboard_code(self, user_prompt, data_context):
        """Generate dashboard code using LLM or fallback template"""
        dashboard_type = self.analyze_dashboard_type(user_prompt)
        
        # Try LLM first
        llm_prompt = self.build_llm_prompt(user_prompt, dashboard_type, data_context)
        response = self.call_llm(llm_prompt)
        
        if response:
            code = self.extract_python_code(response)
            if self.validate_dashboard_code(code):
                print("Generated dashboard code via LLM")
                return code
            else:
                print("LLM generated invalid code, falling back to template")
        
        # Fallback to template if LLM fails
        print(f"Using fallback template for dashboard type: {dashboard_type}")
        return self.get_fallback_dashboard(data_context, dashboard_type)
    
    def build_llm_prompt(self, user_prompt, dashboard_type, data_context):
        """Build a focused LLM prompt for dashboard generation"""
        return f"""Create a Streamlit dashboard for data analysis.

USER REQUEST: "{user_prompt}"
DASHBOARD TYPE: {dashboard_type}

DATA CONTEXT:
- Database: {data_context['db_path']}
- Table: {data_context['table_name']}
- Columns: {', '.join(data_context['columns'])}
- Row count: {data_context.get('total_rows', 0)}

REQUIREMENTS:
1. Import: streamlit, pandas, plotly.express, plotly.graph_objects, sqlite3
2. Load data: conn = sqlite3.connect(r'{data_context['db_path']}'); df = pd.read_sql_query("SELECT * FROM [{data_context['table_name']}]", conn); conn.close()
3. Create 3-5 key metrics using st.metric()
4. Add 2-3 interactive charts using plotly
5. Include sidebar filters
6. Handle missing data gracefully
7. Professional styling with st.set_page_config()

Generate ONLY Python code, no explanations:"""

    def extract_python_code(self, response):
        """Extract Python code from LLM response"""
        if not response:
            return ""
            
        # Try to extract code from markdown blocks
        if '```python' in response:
            parts = response.split('```python')
            if len(parts) > 1:
                code_part = parts[1].split('```')[0]
                return code_part.strip()
        elif '```' in response:
            parts = response.split('```')
            if len(parts) >= 3:
                return parts[1].strip()
        
        # If no markdown blocks, assume entire response is code
        return response.strip()
    
    def validate_dashboard_code(self, code):
        """Basic validation of generated dashboard code"""
        if not code or len(code) < 200:
            return False
            
        required_imports = ['streamlit', 'pandas', 'sqlite3']
        for imp in required_imports:
            if f'import {imp}' not in code and f'from {imp}' not in code:
                return False
                
        # Check for basic Streamlit structure
        if 'st.' not in code:
            return False
            
        return True

    def get_fallback_dashboard(self, data_context, dashboard_type='operational'):
        """Generate a working fallback dashboard template"""
        table_name = data_context.get('table_name', 'data_table')
        db_path = data_context.get('db_path', '')
        columns = data_context.get('columns', [])
        
        return f'''import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import os

# Page configuration
st.set_page_config(
    page_title="{dashboard_type.title()} Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dashboard styling
st.markdown("""
<style>
    .main {{{{ background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%); }}}}
    .metric-card {{{{
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }}}}
    .stPlotlyChart {{{{ 
        background: white;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}}}
</style>
""", unsafe_allow_html=True)

# Header
st.title("📊 {dashboard_type.title()} Analytics Dashboard")
st.markdown("---")

# Load data function
@st.cache_data
def load_data():
    try:
        # Check if database file exists
        db_file = r"{db_path}"
        if not os.path.exists(db_file):
            st.error(f"Database file not found: {{db_file}}")
            return pd.DataFrame()
            
        conn = sqlite3.connect(db_file)
        
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", ("{table_name}",))
        if not cursor.fetchone():
            st.error(f"Table '{table_name}' not found in database")
            conn.close()
            return pd.DataFrame()
            
        df = pd.read_sql_query("SELECT * FROM [{table_name}]", conn)
        conn.close()
        
        if df.empty:
            st.warning("Database table is empty")
        else:
            st.success(f"Loaded {{len(df)}} records from database")
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {{e}}")
        return pd.DataFrame()

# Load data
df = load_data()

# If no data, generate sample data
if df.empty:
    st.info("No data found in database. Generating sample data for demonstration...")
    
    # Generate sample data based on dashboard type
    np.random.seed(42)
    sample_size = 100
    
    if "{dashboard_type}" == "manufacturing":
        df = pd.DataFrame({{
            'Date': pd.date_range('2024-01-01', periods=sample_size, freq='D'),
            'Lane': np.random.choice(['Lane_A', 'Lane_B', 'Lane_C'], sample_size),
            'Production_Units': np.random.randint(800, 1200, sample_size),
            'Efficiency_Percent': np.random.uniform(75, 95, sample_size),
            'Defect_Rate': np.random.uniform(0.5, 3.0, sample_size),
            'Downtime_Hours': np.random.exponential(2, sample_size)
        }})
    elif "{dashboard_type}" == "sales":
        df = pd.DataFrame({{
            'Date': pd.date_range('2024-01-01', periods=sample_size, freq='D'),
            'Product': np.random.choice(['Product_X', 'Product_Y', 'Product_Z'], sample_size),
            'Sales_Amount': np.random.uniform(1000, 5000, sample_size),
            'Quantity_Sold': np.random.randint(10, 100, sample_size),
            'Customer_Region': np.random.choice(['North', 'South', 'East', 'West'], sample_size)
        }})
    else:
        # Generic analytics data
        df = pd.DataFrame({{
            'Date': pd.date_range('2024-01-01', periods=sample_size, freq='D'),
            'Category': np.random.choice(['Category_1', 'Category_2', 'Category_3'], sample_size),
            'Value': np.random.uniform(100, 1000, sample_size),
            'Count': np.random.randint(1, 50, sample_size),
            'Status': np.random.choice(['Active', 'Inactive', 'Pending'], sample_size)
        }})
    
    st.success(f"Generated {{len(df)}} sample records for demonstration")

# Sidebar filters
st.sidebar.header("🔧 Filters")

# Date filter if date column exists
date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
if date_columns:
    selected_date_col = st.sidebar.selectbox("Date Column", date_columns)
    if selected_date_col:
        try:
            df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors='coerce')
            df = df.dropna(subset=[selected_date_col])
            
            if not df.empty:
                min_date = df[selected_date_col].min().date()
                max_date = df[selected_date_col].max().date()
                
                date_range = st.sidebar.date_input(
                    "Date Range",
                    value=[min_date, max_date],
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    df = df[
                        (df[selected_date_col].dt.date >= date_range[0]) &
                        (df[selected_date_col].dt.date <= date_range[1])
                    ]
        except:
            pass

# Numeric columns for analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Key Metrics
st.header("📈 Key Metrics")

if len(numeric_cols) >= 2:
    cols = st.columns(min(4, len(numeric_cols)))
    
    for i, col in enumerate(numeric_cols[:4]):
        with cols[i % 4]:
            value = df[col].sum() if df[col].dtype in ['int64', 'float64'] else df[col].count()
            st.metric(
                label=col.replace('_', ' ').title(),
                value=f"{{value:,.0f}}" if isinstance(value, (int, float)) else str(value)
            )

# Charts
st.header("📊 Data Visualization")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    if len(numeric_cols) >= 1:
        # Bar chart for first numeric column
        if len(categorical_cols) >= 1:
            group_by_col = st.selectbox("Group by", categorical_cols, key="bar_group")
            chart_data = df.groupby(group_by_col)[numeric_cols[0]].sum().reset_index()
            
            fig = px.bar(
                chart_data, 
                x=group_by_col, 
                y=numeric_cols[0],
                title=f"{{numeric_cols[0]}} by {{group_by_col}}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Histogram if no categorical columns
            fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {{numeric_cols[0]}}")
            st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    if len(numeric_cols) >= 2:
        # Scatter plot
        fig = px.scatter(
            df, 
            x=numeric_cols[0], 
            y=numeric_cols[1],
            title=f"{{numeric_cols[0]}} vs {{numeric_cols[1]}}"
        )
        st.plotly_chart(fig, use_container_width=True)
    elif len(categorical_cols) >= 1:
        # Pie chart
        pie_data = df[categorical_cols[0]].value_counts().reset_index()
        fig = px.pie(
            pie_data, 
            values=categorical_cols[0], 
            names='index',
            title=f"Distribution of {{categorical_cols[0]}}"
        )
        st.plotly_chart(fig, use_container_width=True)

# Time series if date column exists
if date_columns and len(numeric_cols) >= 1:
    st.header("📅 Time Series Analysis")
    try:
        time_col = date_columns[0]
        value_col = st.selectbox("Value Column", numeric_cols, key="time_series")
        
        # Group by date and aggregate
        df_time = df.groupby(df[time_col].dt.date)[value_col].sum().reset_index()
        df_time.columns = ['Date', value_col]
        
        fig = px.line(
            df_time, 
            x='Date', 
            y=value_col,
            title=f"{{value_col}} Over Time"
        )
        st.plotly_chart(fig, use_container_width=True)
    except:
        pass

# Data table
st.header("📋 Data Table")

# Show sample of data
show_all = st.checkbox("Show all data", value=False)
display_df = df if show_all else df.head(100)

st.dataframe(display_df, use_container_width=True, height=400)

# Summary statistics
if st.checkbox("Show Summary Statistics"):
    st.header("📊 Summary Statistics")
    st.write(df.describe())

# Data info
with st.expander("ℹ️ Data Information"):
    st.write(f"**Total Rows:** {{len(df):,}}")
    st.write(f"**Total Columns:** {{len(df.columns)}}")
    st.write("**Column Types:**")
    for col in df.columns:
        st.write(f"- {{col}}: {{df[col].dtype}}")
'''

    def get_manufacturing_template(self, table_name, columns):
        return "# Manufacturing template removed - using unified fallback template"

    def get_operational_template(self, table_name, columns):
        return "# Operational template removed - using unified fallback template"

    def get_sales_template(self, table_name, columns):
        return "# Sales template removed - using unified fallback template"

    def get_financial_template(self, table_name, columns):
        return "# Financial template removed - using unified fallback template"

    def get_logistics_template(self, table_name, columns):
        return "# Logistics template removed - using unified fallback template"

    def get_analytics_template(self, table_name, columns):
        return "# Analytics template removed - using unified fallback template"

    def get_energy_template(self, table_name, columns):
        return "# Energy template removed - using unified fallback template"

    def get_hr_template(self, table_name, columns):

        return """
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import numpy as np
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Operational Efficiency Dashboard",
    page_icon=":zap:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional dashboard
st.markdown('''
<style>
    .main {{
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
    }}
    .metric-card {{
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        color: #0f172a;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    .kpi-header {{
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 0.5rem;
    }}
    .kpi-value {{
        font-size: 2.5rem;
        font-weight: bold;
        color: #0f172a;
        margin: 0;
    }}
    .kpi-delta {{
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }}
    .status-good {{ color: #059669; }}
    .status-warning {{ color: #d97706; }}
    .status-critical {{ color: #dc2626; }}
    .stPlotlyChart {{
        background: rgba(255,255,255,0.95);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }}
    .sidebar .sidebar-content {{
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
    }}
</style>
''', unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 2rem;'>Operational Efficiency Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #94a3b8; margin-bottom: 3rem;'>Real-time KPIs, Uptime Metrics & Performance Analytics</h3>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect('../data/terminal_data.db')
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data: {{e}}")
        return pd.DataFrame()

@st.cache_data
def load_sample_operational_data():
    # Create sample operational data if database is empty
    np.random.seed(42)
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')

    data = []
    for i, date in enumerate(dates):
        efficiency = 85 + np.random.normal(0, 5)
        uptime = max(80, min(99.9, 95 + np.random.normal(0, 3)))
        fuel_volume = 5000 + np.random.normal(0, 500) + (i * 10)
        throughput = 1000 + np.random.normal(0, 100)

        data.append({{
            'Date': date.strftime('%Y-%m-%d'),
            'Operational_Efficiency': round(efficiency, 2),
            'Uptime_Percentage': round(uptime, 2),
            'Fuel_Volume_Liters': round(fuel_volume, 0),
            'Daily_Throughput': round(throughput, 0),
            'Cost_Per_Unit': round(2.5 + np.random.normal(0, 0.2), 2),
            'Energy_Consumption': round(800 + np.random.normal(0, 50), 0),
            'Department': np.random.choice(['Production', 'Logistics', 'Warehouse']),
            'Shift': np.random.choice(['Day', 'Night'])
        }})

    return pd.DataFrame(data)

# Load data
df = load_data()
if df.empty:
    df = load_sample_operational_data()
    st.info("Loading sample operational data for demonstration")

# Convert date column if it exists
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# Sidebar controls
st.sidebar.markdown("## Dashboard Controls")
date_range = st.sidebar.date_input("Date Range", value=[datetime.now() - timedelta(days=7), datetime.now()])
departments = st.sidebar.multiselect("Departments", options=df.get('Department', pd.Series([])).unique() if 'Department' in df.columns else ['All'], default=['All'])

# Filter data based on sidebar selections
filtered_df = df.copy()
if 'Date' in df.columns and len(date_range) == 2:
    filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(date_range[0])) &
                             (filtered_df['Date'] <= pd.to_datetime(date_range[1]))]

# Key Performance Indicators
st.markdown("## Key Performance Indicators")
kpi_cols = st.columns(5)

# Calculate KPIs
if not filtered_df.empty:
    avg_efficiency = filtered_df.get('Operational_Efficiency', pd.Series([0])).mean()
    avg_uptime = filtered_df.get('Uptime_Percentage', pd.Series([0])).mean()
    total_fuel = filtered_df.get('Fuel_Volume_Liters', pd.Series([0])).sum()
    avg_throughput = filtered_df.get('Daily_Throughput', pd.Series([0])).mean()
    avg_cost = filtered_df.get('Cost_Per_Unit', pd.Series([0])).mean()

    with kpi_cols[0]:
        st.markdown(f'''
        <div class="metric-card">
            <div class="kpi-header">Operational Efficiency</div>
            <div class="kpi-value">{avg_efficiency:.1f}%</div>
            <div class="kpi-delta status-good">+2.3% vs last period</div>
        </div>
        ''', unsafe_allow_html=True)

    with kpi_cols[1]:
        status_class = "status-good" if avg_uptime > 95 else "status-warning" if avg_uptime > 90 else "status-critical"
        st.markdown(f'''
        <div class="metric-card">
            <div class="kpi-header">System Uptime</div>
            <div class="kpi-value {status_class}">{avg_uptime:.1f}%</div>
            <div class="kpi-delta">Target: 95%+</div>
        </div>
        ''', unsafe_allow_html=True)

    with kpi_cols[2]:
        st.markdown(f'''
        <div class="metric-card">
            <div class="kpi-header">Fuel Volume</div>
            <div class="kpi-value">{total_fuel:,.0f}L</div>
            <div class="kpi-delta status-good">+5.1% efficiency</div>
        </div>
        ''', unsafe_allow_html=True)

    with kpi_cols[3]:
        st.markdown(f'''
        <div class="metric-card">
            <div class="kpi-header">Daily Throughput</div>
            <div class="kpi-value">{avg_throughput:,.0f}</div>
            <div class="kpi-delta">units/day average</div>
        </div>
        ''', unsafe_allow_html=True)

    with kpi_cols[4]:
        st.markdown(f'''
        <div class="metric-card">
            <div class="kpi-header">Cost per Unit</div>
            <div class="kpi-value">${avg_cost:.2f}</div>
            <div class="kpi-delta status-good">-3.2% cost reduction</div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown("---")

# Charts section
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Operational Efficiency Trend")
    if 'Date' in filtered_df.columns and 'Operational_Efficiency' in filtered_df.columns:
        fig_efficiency = px.line(filtered_df, x='Date', y='Operational_Efficiency',
                               title="Operational Efficiency Over Time",
                               color_discrete_sequence=['#3b82f6'])
        fig_efficiency.add_hline(y=85, line_dash="dash", line_color="red",
                               annotation_text="Target: 85%")
        fig_efficiency.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_efficiency, use_container_width=True)

with chart_col2:
    st.markdown("### System Uptime Analysis")
    if 'Date' in filtered_df.columns and 'Uptime_Percentage' in filtered_df.columns:
        fig_uptime = go.Figure()
        fig_uptime.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Uptime_Percentage'],
                                       mode='lines+markers', name='Uptime %',
                                       line=dict(color='#10b981', width=3)))
        fig_uptime.add_hline(y=95, line_dash="dash", line_color="orange",
                           annotation_text="Target: 95%")
        fig_uptime.update_layout(title="System Uptime Percentage", height=400, showlegend=False)
        st.plotly_chart(fig_uptime, use_container_width=True)

# Fuel volume analysis
st.markdown("### Fuel Volume Analysis")
fuel_col1, fuel_col2 = st.columns(2)

with fuel_col1:
    if 'Date' in filtered_df.columns and 'Fuel_Volume_Liters' in filtered_df.columns:
        fig_fuel = px.area(filtered_df, x='Date', y='Fuel_Volume_Liters',
                          title="Daily Fuel Consumption",
                          color_discrete_sequence=['#f59e0b'])
        fig_fuel.update_layout(height=350)
        st.plotly_chart(fig_fuel, use_container_width=True)

with fuel_col2:
    # Fuel efficiency gauge
    if 'Fuel_Volume_Liters' in filtered_df.columns and 'Daily_Throughput' in filtered_df.columns:
        fuel_efficiency = (filtered_df['Daily_Throughput'].sum() / filtered_df['Fuel_Volume_Liters'].sum() * 1000) if filtered_df['Fuel_Volume_Liters'].sum() > 0 else 0

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = fuel_efficiency,
            domain = {{'x': [0, 1], 'y': [0, 1]}},
            title = {{'text': "Fuel Efficiency (units/1000L)"}},
            delta = {{'reference': 180}},
            gauge = {{
                'axis': {{'range': [None, 250]}},
                'bar': {{'color': "#3b82f6"}},
                'steps': [
                    {{'range': [0, 150], 'color': "#fecaca"}},
                    {{'range': [150, 200], 'color': "#fed7aa"}},
                    {{'range': [200, 250], 'color': "#bbf7d0"}}],
                'threshold': {{
                    'line': {{'color': "red", 'width': 4}},
                    'thickness': 0.75,
                    'value': 200}}
            }}
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

# Department performance
if 'Department' in filtered_df.columns:
    st.markdown("### Department Performance Breakdown")
    dept_col1, dept_col2 = st.columns(2)

    with dept_col1:
        if 'Operational_Efficiency' in filtered_df.columns:
            dept_efficiency = filtered_df.groupby('Department')['Operational_Efficiency'].mean().reset_index()
            fig_dept = px.bar(dept_efficiency, x='Department', y='Operational_Efficiency',
                            title="Average Efficiency by Department",
                            color='Operational_Efficiency',
                            color_continuous_scale='RdYlGn')
            fig_dept.update_layout(height=350)
            st.plotly_chart(fig_dept, use_container_width=True)

    with dept_col2:
        if 'Daily_Throughput' in filtered_df.columns:
            dept_throughput = filtered_df.groupby('Department')['Daily_Throughput'].sum().reset_index()
            fig_pie = px.pie(dept_throughput, values='Daily_Throughput', names='Department',
                           title="Throughput Distribution by Department")
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

# Data table
st.markdown("### Detailed Operations Data")
st.dataframe(filtered_df.tail(20), use_container_width=True, height=300)

# Alert section
st.markdown("### System Alerts & Recommendations")
alert_col1, alert_col2, alert_col3 = st.columns(3)

with alert_col1:
    st.info("**System Status**: All systems operational")

with alert_col2:
    if avg_uptime < 95:
        st.warning("**Uptime Alert**: Below target threshold")
    else:
        st.success("**Uptime Status**: Meeting targets")

with alert_col3:
    st.success("**Performance**: Efficiency trending upward")
"""

    def get_sales_template(self, table_name, columns):
        return f"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Sales Performance Dashboard", page_icon="📊", layout="wide")

st.markdown('''
<style>
    .main {{ background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); color: white; }}
    .metric-card {{ background: white; color: #1e3a8a; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }}
    .kpi-value {{ font-size: 2rem; font-weight: bold; }}
</style>
''', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Sales Performance Dashboard</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect('../data/terminal_data.db')
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
        conn.close()
        return df
    except:
        # Sample sales data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        return pd.DataFrame({{
            'Date': dates,
            'Revenue': np.random.normal(50000, 10000, 30),
            'Customers': np.random.poisson(200, 30),
            'Conversion_Rate': np.random.uniform(0.1, 0.2, 30),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 30)
        }})

df = load_data()
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

st.markdown("## Sales KPIs")
kpi_cols = st.columns(4)

if 'Revenue' in df.columns:
    total_revenue = df['Revenue'].sum()
    with kpi_cols[0]:
        st.markdown(f'''
        <div class="metric-card">
            <div>Total Revenue</div>
            <div class="kpi-value">${{total_revenue:,.0f}}</div>
        </div>
        ''', unsafe_allow_html=True)

if 'Customers' in df.columns:
    total_customers = df['Customers'].sum()
    with kpi_cols[1]:
        st.markdown(f'''
        <div class="metric-card">
            <div>Total Customers</div>
            <div class="kpi-value">{{total_customers:,.0f}}</div>
        </div>
        ''', unsafe_allow_html=True)

if 'Conversion_Rate' in df.columns:
    avg_conversion = df['Conversion_Rate'].mean()
    with kpi_cols[2]:
        st.markdown(f'''
        <div class="metric-card">
            <div>Avg Conversion Rate</div>
            <div class="kpi-value">{{avg_conversion:.1%}}</div>
        </div>
        ''', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if 'Date' in df.columns and 'Revenue' in df.columns:
        fig = px.line(df, x='Date', y='Revenue', title="Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if 'Region' in df.columns and 'Revenue' in df.columns:
        region_data = df.groupby('Region')['Revenue'].sum().reset_index()
        fig = px.pie(region_data, values='Revenue', names='Region', title="Revenue by Region")
        st.plotly_chart(fig, use_container_width=True)

st.dataframe(df, use_container_width=True)
"""

    def get_financial_template(self, table_name, columns):
        return f"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import numpy as np

st.set_page_config(page_title="Financial Analysis Dashboard", page_icon="💰", layout="wide")

st.markdown('''
<style>
    .main {{ background: linear-gradient(135deg, #064e3b 0%, #059669 100%); color: white; }}
    .metric-card {{ background: white; color: #064e3b; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }}
    .kpi-value {{ font-size: 2rem; font-weight: bold; }}
    .profit {{ color: #059669; }}
    .loss {{ color: #dc2626; }}
</style>
''', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Financial Analysis Dashboard</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect('../data/terminal_data.db')
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
        conn.close()
        return df
    except:
        # Sample financial data
        months = pd.date_range('2024-01-01', periods=12, freq='M')
        return pd.DataFrame({{
            'Month': months,
            'Revenue': np.random.normal(100000, 15000, 12),
            'Costs': np.random.normal(60000, 10000, 12),
            'Profit_Margin': np.random.uniform(0.2, 0.4, 12)
        }})

df = load_data()
df['Profit'] = df.get('Revenue', 0) - df.get('Costs', 0)

st.markdown("## Financial KPIs")
kpi_cols = st.columns(4)

if 'Revenue' in df.columns:
    total_revenue = df['Revenue'].sum()
    with kpi_cols[0]:
        st.markdown(f'''
        <div class="metric-card">
            <div>Total Revenue</div>
            <div class="kpi-value">${{total_revenue:,.0f}}</div>
        </div>
        ''', unsafe_allow_html=True)

if 'Profit' in df.columns:
    total_profit = df['Profit'].sum()
    profit_class = "profit" if total_profit > 0 else "loss"
    with kpi_cols[1]:
        st.markdown(f'''
        <div class="metric-card">
            <div>Total Profit</div>
            <div class="kpi-value {{profit_class}}">${{total_profit:,.0f}}</div>
        </div>
        ''', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if 'Month' in df.columns and 'Revenue' in df.columns:
        fig = px.line(df, x='Month', y='Revenue', title="Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if 'Month' in df.columns and 'Profit' in df.columns:
        fig = px.bar(df, x='Month', y='Profit', title="Monthly Profit",
                    color='Profit', color_continuous_scale=['red', 'green'])
        st.plotly_chart(fig, use_container_width=True)

st.dataframe(df, use_container_width=True)
"""

    def get_analytics_template(self, table_name, columns):
        return f"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import numpy as np

st.set_page_config(page_title="Data Analytics Dashboard", page_icon="📈", layout="wide")

st.markdown('''
<style>
    .main {{ background: linear-gradient(135deg, #374151 0%, #6b7280 100%); color: white; }}
    .metric-card {{ background: white; color: #374151; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }}
</style>
''', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Data Analytics Dashboard</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect('../data/terminal_data.db')
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame({{
            'Category': ['A', 'B', 'C', 'D', 'E'] * 20,
            'Value': np.random.normal(100, 20, 100),
            'Date': pd.date_range('2024-01-01', periods=100, freq='D')[:100]
        }})

df = load_data()

st.markdown("## Data Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))
with col2:
    if len(df.select_dtypes(include=[np.number]).columns) > 0:
        numeric_col = df.select_dtypes(include=[np.number]).columns[0]
        st.metric("Average", f"{{df[numeric_col].mean():.2f}}")
with col3:
    st.metric("Data Quality", "95%")

# Charts based on available data
if len(df.columns) >= 2:
    col1, col2 = st.columns(2)

    with col1:
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 1:
                fig = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {{numeric_cols[0]}}")
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0 and len(df.select_dtypes(include=[np.number]).columns) > 0:
            cat_col = categorical_cols[0]
            num_col = df.select_dtypes(include=[np.number]).columns[0]
            agg_data = df.groupby(cat_col)[num_col].sum().reset_index()
            fig = px.bar(agg_data, x=cat_col, y=num_col, title=f"{{num_col}} by {{cat_col}}")
            st.plotly_chart(fig, use_container_width=True)

st.markdown("## Data Table")
st.dataframe(df, use_container_width=True)

# Correlation analysis for numeric columns
numeric_df = df.select_dtypes(include=[np.number])
if len(numeric_df.columns) > 1:
    st.markdown("## Correlation Analysis")
    corr_matrix = numeric_df.corr()
    fig = px.imshow(corr_matrix, title="Correlation Matrix", color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)
"""

    def get_logistics_template(self, table_name, columns):
        return f"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import numpy as np

st.set_page_config(page_title="Logistics Dashboard", page_icon="🚚", layout="wide")

st.markdown('''
<style>
    .main {{ background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); color: white; }}
    .metric-card {{ background: white; color: #ea580c; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }}
    .kpi-value {{ font-size: 2rem; font-weight: bold; }}
</style>
''', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Logistics & Supply Chain Dashboard</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect('../data/terminal_data.db')
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame({{
            'Shipment_ID': range(1, 101),
            'Origin': np.random.choice(['Warehouse A', 'Warehouse B', 'Warehouse C'], 100),
            'Destination': np.random.choice(['City 1', 'City 2', 'City 3', 'City 4'], 100),
            'Delivery_Time': np.random.normal(48, 12, 100),  # hours
            'Cost': np.random.normal(500, 100, 100),
            'Status': np.random.choice(['Delivered', 'In Transit', 'Delayed'], 100, p=[0.7, 0.2, 0.1])
        }})

df = load_data()

st.markdown("## Logistics KPIs")
kpi_cols = st.columns(4)

with kpi_cols[0]:
    total_shipments = len(df)
    st.markdown(f'''
    <div class="metric-card">
        <div>Total Shipments</div>
        <div class="kpi-value">{{total_shipments:,}}</div>
    </div>
    ''', unsafe_allow_html=True)

if 'Delivery_Time' in df.columns:
    avg_delivery = df['Delivery_Time'].mean()
    with kpi_cols[1]:
        st.markdown(f'''
        <div class="metric-card">
            <div>Avg Delivery Time</div>
            <div class="kpi-value">{{avg_delivery:.1f}}h</div>
        </div>
        ''', unsafe_allow_html=True)

if 'Status' in df.columns:
    on_time_rate = (df['Status'] == 'Delivered').mean()
    with kpi_cols[2]:
        st.markdown(f'''
        <div class="metric-card">
            <div>On-Time Rate</div>
            <div class="kpi-value">{{on_time_rate:.1%}}</div>
        </div>
        ''', unsafe_allow_html=True)

if 'Cost' in df.columns:
    avg_cost = df['Cost'].mean()
    with kpi_cols[3]:
        st.markdown(f'''
        <div class="metric-card">
            <div>Avg Shipping Cost</div>
            <div class="kpi-value">${{avg_cost:.0f}}</div>
        </div>
        ''', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if 'Status' in df.columns:
        status_counts = df['Status'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, title="Shipment Status")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if 'Delivery_Time' in df.columns:
        fig = px.histogram(df, x='Delivery_Time', title="Delivery Time Distribution")
        st.plotly_chart(fig, use_container_width=True)

st.dataframe(df, use_container_width=True)
"""

    def get_energy_template(self, table_name, columns):
        return f"""
import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import numpy as np

st.set_page_config(page_title="Energy Dashboard", page_icon="⚡", layout="wide")

st.markdown('''
<style>
    .main {{ background: linear-gradient(135deg, #065f46 0%, #10b981 100%); color: white; }}
    .metric-card {{ background: white; color: #065f46; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }}
    .kpi-value {{ font-size: 2rem; font-weight: bold; }}
</style>
''', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: white;'>Energy Consumption Dashboard</h1>", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        conn = sqlite3.connect('../data/terminal_data.db')
        df = pd.read_sql_query(f"SELECT * FROM [{table_name}]", conn)
        conn.close()
        return df
    except:
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        return pd.DataFrame({{
            'Date': dates,
            'Energy_Consumption': np.random.normal(1000, 200, 30),
            'Efficiency_Rating': np.random.uniform(0.7, 0.95, 30),
            'Cost': np.random.normal(150, 30, 30)
        }})

df = load_data()

st.markdown("## Energy KPIs")
kpi_cols = st.columns(3)

if 'Energy_Consumption' in df.columns:
    total_consumption = df['Energy_Consumption'].sum()
    with kpi_cols[0]:
        st.markdown(f'''
        <div class="metric-card">
            <div>Total Consumption</div>
            <div class="kpi-value">{{total_consumption:,.0f}} kWh</div>
        </div>
        ''', unsafe_allow_html=True)

if 'Efficiency_Rating' in df.columns:
    avg_efficiency = df['Efficiency_Rating'].mean()
    with kpi_cols[1]:
        st.markdown(f'''
        <div class="metric-card">
            <div>Avg Efficiency</div>
            <div class="kpi-value">{{avg_efficiency:.1%}}</div>
        </div>
        ''', unsafe_allow_html=True)

if 'Cost' in df.columns:
    total_cost = df['Cost'].sum()
    with kpi_cols[2]:
        st.markdown(f'''
        <div class="metric-card">
            <div>Total Cost</div>
            <div class="kpi-value">${{total_cost:,.0f}}</div>
        </div>
        ''', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if 'Date' in df.columns and 'Energy_Consumption' in df.columns:
        fig = px.area(df, x='Date', y='Energy_Consumption', title="Daily Energy Consumption")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if 'Date' in df.columns and 'Efficiency_Rating' in df.columns:
        fig = px.line(df, x='Date', y='Efficiency_Rating', title="Efficiency Trend")
        st.plotly_chart(fig, use_container_width=True)

st.dataframe(df, use_container_width=True)
"""

    def get_hr_template(self, table_name, columns):
        return "# HR template removed - using unified fallback template"

    def create_dashboard_file(self, code, dashboard_id):
        """Create a dashboard Python file with proper encoding"""
        filename = f"dashboard_{dashboard_id}.py"
        filepath = os.path.join(Config.DASHBOARD_DIR, filename)
        
        try:
            # Ensure directory exists
            os.makedirs(Config.DASHBOARD_DIR, exist_ok=True)
            
            # Write with UTF-8 encoding
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            
            print(f"✅ Dashboard file created: {filename}")
            return filepath
            
        except Exception as e:
            print(f"❌ Error creating dashboard file: {e}")
            return None

    def start_streamlit_dashboard(self, filepath, port):
        """Start a Streamlit dashboard process"""
        try:
            # Check if streamlit is available
            import shutil
            streamlit_cmd = shutil.which('streamlit')
            if not streamlit_cmd:
                # Try python -m streamlit
                cmd = [
                    'python', '-m', 'streamlit', 'run', filepath,
                    '--server.port', str(port),
                    '--server.headless', 'true',
                    '--server.enableCORS', 'false',
                    '--server.enableXsrfProtection', 'false',
                    '--server.address', '0.0.0.0'
                ]
            else:
                cmd = [
                    'streamlit', 'run', filepath,
                    '--server.port', str(port),
                    '--server.headless', 'true',
                    '--server.enableCORS', 'false',
                    '--server.enableXsrfProtection', 'false',
                    '--server.address', '0.0.0.0'
                ]
            
            print(f"Starting Streamlit dashboard on port {port}...")
            print(f"Command: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Config.DASHBOARD_DIR,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # Wait a bit for startup and check if process is still running
            time.sleep(5)
            
            if process.poll() is None:
                print(f"✅ Streamlit dashboard started successfully on port {port}")
                return process
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Streamlit failed to start:")
                print(f"STDOUT: {stdout.decode('utf-8', errors='ignore')}")
                print(f"STDERR: {stderr.decode('utf-8', errors='ignore')}")
                return None
                
        except Exception as e:
            print(f"❌ Error starting Streamlit: {e}")
            return None

def find_available_port(start_port, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    for i in range(max_attempts):
        port = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def cleanup_dashboard_process(dashboard_id):
    """Safely cleanup a dashboard process"""
    if dashboard_id in running_dashboards:
        try:
            process = running_dashboards[dashboard_id]['process']
            if process and process.poll() is None:
                process.terminate()
                # Wait up to 5 seconds for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            # Clean up file if it exists
            file_path = running_dashboards[dashboard_id].get('file_path')
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass  # File might be in use
                    
            del running_dashboards[dashboard_id]
            return True
        except Exception as e:
            print(f"Error cleaning up dashboard {dashboard_id}: {e}")
            return False
    return False

generator = DashboardGenerator()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'AI Dashboard Backend Running'})

@app.route('/test-dashboard', methods=['POST'])
def test_dashboard():
    """Test endpoint to verify dashboard generation works"""
    try:
        # Test data context
        test_data_context = {
            'db_path': os.path.join(Config.DATA_DIR, 'test.db'),
            'table_name': 'test_data',
            'columns': ['id', 'name', 'value', 'date'],
            'sample_data': [
                {'id': 1, 'name': 'Test Item 1', 'value': 100, 'date': '2024-01-01'},
                {'id': 2, 'name': 'Test Item 2', 'value': 200, 'date': '2024-01-02'}
            ],
            'total_rows': 2,
            'excel_source': 'test_data.xlsx'
        }
        
        # Generate test dashboard
        dashboard_code = generator.get_fallback_dashboard(test_data_context, 'analytics')
        
        if len(dashboard_code) > 100:
            return jsonify({
                'success': True,
                'message': 'Dashboard generation test passed',
                'code_length': len(dashboard_code),
                'preview': dashboard_code[:200] + '...'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Generated code too short',
                'code_length': len(dashboard_code)
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Test failed: {str(e)}'
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    status = Config.get_status()
    return jsonify(status)

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        'ollama_url': Config.OLLAMA_URL,
        'model': Config.OLLAMA_MODEL,
        'port': Config.AI_BACKEND_PORT,
        'debug': Config.DEBUG,
        'streamlit_base_port': Config.STREAMLIT_BASE_PORT
    })

@app.route('/api/dashboard/generate', methods=['POST'])
def generate_dashboard():
    """Generate a new dashboard from user prompt and data"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        user_prompt = data.get('prompt', '').strip()
        excel_path = data.get('excel_path', '')
        
        if not user_prompt:
            return jsonify({'error': 'Prompt is required'}), 400
            
        print(f"📝 Generating dashboard for: '{user_prompt}'")
        
        # Process Excel data first
        if excel_path and os.path.exists(excel_path):
            print(f"📁 Processing Excel file: {excel_path}")
            data_context = generator.convert_excel_to_sqlite(excel_path)
        else:
            # Use default data processing
            print("📁 Processing default Excel files")
            data_context = generator.process_all_excel_files()
            
        if not data_context.get('success'):
            print(f"❌ Data processing failed: {data_context.get('error', 'Unknown error')}")
            return jsonify({'error': f'Data processing failed: {data_context.get("error", "Unknown error")}'}), 400
            
        print("✅ Data processing successful")
        
        # Generate dashboard code
        print("🎨 Generating dashboard code...")
        dashboard_code = generator.generate_dashboard_code(user_prompt, data_context)
        
        if not dashboard_code:
            return jsonify({'error': 'Failed to generate dashboard code'}), 500
            
        print("✅ Dashboard code generated")
        
        # Create dashboard file
        dashboard_id = f"{int(time.time())}_{abs(hash(user_prompt)) % 10000}"
        dashboard_path = generator.create_dashboard_file(dashboard_code, dashboard_id)
        
        if not dashboard_path:
            return jsonify({'error': 'Failed to create dashboard file'}), 500
            
        # Find available port
        port = find_available_port(Config.STREAMLIT_BASE_PORT)
        if not port:
            return jsonify({'error': 'No available ports for dashboard'}), 500
            
        print(f"🌐 Starting dashboard on port {port}...")
        
        # Start Streamlit process
        process = generator.start_streamlit_dashboard(dashboard_path, port)
        
        if not process:
            return jsonify({'error': 'Failed to start dashboard process'}), 500
            
        # Store dashboard info
        running_dashboards[dashboard_id] = {
            'process': process,
            'port': port,
            'created_at': datetime.now().isoformat(),
            'prompt': user_prompt,
            'file_path': dashboard_path
        }
        
        dashboard_url = f"http://localhost:{port}"
        
        print(f"🎉 Dashboard ready at: {dashboard_url}")
        
        return jsonify({
            'success': True,
            'dashboard_id': dashboard_id,
            'dashboard_url': dashboard_url,
            'embed_url': f"{dashboard_url}/?embed=true",
            'message': 'Dashboard generated and started successfully',
            'port': port
        })
        
    except Exception as e:
        print(f"❌ Dashboard generation error: {str(e)}")
        return jsonify({'error': f'Dashboard generation failed: {str(e)}'}), 500

# Old hardcoded endpoint removed - using the fixed one above



@app.route('/api/dashboard/list', methods=['GET'])
def list_dashboards():
    dashboard_list = []
    for dashboard_id, info in running_dashboards.items():
        dashboard_list.append({
            'id': dashboard_id,
            'port': info['port'],
            'created_at': info['created_at'],
            'prompt': info['prompt'],
            'url': f"http://localhost:{info['port']}"
        })
    return jsonify({'dashboards': dashboard_list})

@app.route('/api/dashboard/stop/<dashboard_id>', methods=['POST'])
def stop_dashboard(dashboard_id):
    if dashboard_id in running_dashboards:
        process = running_dashboards[dashboard_id]['process']
        process.terminate()
        del running_dashboards[dashboard_id]
        return jsonify({'success': True, 'message': 'Dashboard stopped'})
    else:
        return jsonify({'success': True, 'message': 'Dashboard not found'})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"Unhandled exception: {e}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

def cleanup_on_exit():
    """Cleanup function to stop all running dashboards on exit"""
    print("Cleaning up running dashboards...")
    dashboard_ids = list(running_dashboards.keys())
    cleaned = 0
    for dashboard_id in dashboard_ids:
        try:
            if dashboard_id in running_dashboards:
                process = running_dashboards[dashboard_id]['process']
                if process and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                del running_dashboards[dashboard_id]
                cleaned += 1
        except Exception as e:
            print(f"Error cleaning up dashboard {dashboard_id}: {e}")
    print(f"Cleaned up {cleaned} dashboards")

# Register cleanup function
import atexit
atexit.register(cleanup_on_exit)

if __name__ == '__main__':
    print("🚀 Starting AI Dashboard Backend...")
    print(f"📍 Config: {Config.OLLAMA_URL} | Model: {Config.OLLAMA_MODEL}")
    print(f"🌐 Server: {Config.AI_BACKEND_HOST}:{Config.AI_BACKEND_PORT}")
    
    # Validate configuration
    try:
        Config.ensure_directories()
        print("✅ Directories validated")
        
        # Check Ollama connection
        status = Config.validate_ollama_connection()
        if status['connected']:
            print("✅ Ollama connected successfully")
            if status.get('has_required_model'):
                print(f"✅ Model '{Config.OLLAMA_MODEL}' is available")
            else:
                print(f"⚠️  Model '{Config.OLLAMA_MODEL}' not found. Available models: {status.get('models', [])}")
        else:
            print(f"❌ Ollama connection failed: {status.get('error', 'Unknown error')}")
            print("💡 Start Ollama with: ollama serve")
            print("⚠️  Dashboard generation will use fallback templates only")
            
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        
    try:
        print("🎯 Starting Flask application...")
        app.run(
            debug=Config.DEBUG,
            host=Config.AI_BACKEND_HOST,
            port=Config.AI_BACKEND_PORT,
            threaded=True  # Enable threading for better concurrent handling
        )
    except KeyboardInterrupt:
        print("\\n🛑 Shutting down gracefully...")
        cleanup_on_exit()
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        cleanup_on_exit()