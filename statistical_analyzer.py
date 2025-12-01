import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import io
import json

# Constants
DEFAULT_NUM_POINTS = 100
DEFAULT_SEED = 42
DEFAULT_BINS = 20
FIGURE_SIZE = (10, 6)
ALPHA_VALUE = 0.5

# Set page configuration
st.set_page_config(
    page_title="Statistical Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for defaults
if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = 0
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'Generate Data'

# Data generation function with caching
@st.cache_data
def generate_data(num_points, random_seed, distribution_type):
    """
    Generate random data based on user specifications.

    Args:
        num_points: Number of data points to generate
        random_seed: Random seed for reproducibility
        distribution_type: Type of distribution

    Returns:
        pandas DataFrame with columns A, B, C
    """
    np.random.seed(random_seed)

    if distribution_type == 'Normal':
        data = np.random.randn(num_points, 3)
    elif distribution_type == 'Uniform':
        data = np.random.uniform(-3, 3, (num_points, 3))
    elif distribution_type == 'Exponential':
        data = np.random.exponential(1, (num_points, 3))
    elif distribution_type == 'Poisson':
        data = np.random.poisson(lam=5, size=(num_points, 3))
    elif distribution_type == 'Binomial':
        data = np.random.binomial(n=10, p=0.5, size=(num_points, 3))
    elif distribution_type == 'Log-Normal':
        data = np.random.lognormal(mean=0, sigma=1, size=(num_points, 3))
    elif distribution_type == 'Chi-Square':
        data = np.random.chisquare(df=2, size=(num_points, 3))
    elif distribution_type == 'Beta':
        data = np.random.beta(a=2, b=5, size=(num_points, 3))
    elif distribution_type == 'Gamma':
        data = np.random.gamma(shape=2, scale=2, size=(num_points, 3))
    else:
        data = np.random.randn(num_points, 3)

    return pd.DataFrame(data, columns=['A', 'B', 'C'])

# Function to export data in different formats
def export_data(df, format_type):
    """
    Export DataFrame to different file formats.

    Args:
        df: pandas DataFrame
        format_type: 'CSV', 'Excel', 'JSON', or 'Parquet'

    Returns:
        Bytes buffer of the exported data
    """
    buffer = io.BytesIO()

    if format_type == 'CSV':
        csv_data = df.to_csv(index=False)
        return csv_data.encode('utf-8')
    elif format_type == 'Excel':
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        return buffer.getvalue()
    elif format_type == 'JSON':
        json_data = df.to_json(orient='records', indent=2)
        return json_data.encode('utf-8')
    elif format_type == 'Parquet':
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()

    return None

# Add a title and some information about the app
st.title("Statistical Analysis Dashboard")
st.markdown("An interactive data visualization application with advanced features.")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Reset button
if st.sidebar.button("Reset to Defaults"):
    st.session_state.reset_trigger += 1
    st.session_state.uploaded_data = None
    st.session_state.data_source = 'Generate Data'
    st.rerun()

# Data source selection
data_source = st.sidebar.radio(
    "Data Source:",
    ['Generate Data', 'Upload File'],
    key=f"data_source_{st.session_state.reset_trigger}"
)

chart_data = None

if data_source == 'Upload File':
    st.sidebar.subheader("Upload Data File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        key=f"file_upload_{st.session_state.reset_trigger}"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                chart_data = pd.read_csv(uploaded_file)
            else:
                chart_data = pd.read_excel(uploaded_file)

            st.sidebar.success(f"File uploaded successfully! Shape: {chart_data.shape}")

            # Ensure numeric columns
            numeric_cols = chart_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 3:
                st.sidebar.warning("File should have at least 3 numeric columns. Using first 3 available columns.")
                chart_data = chart_data.iloc[:, :3]
            else:
                # Use first 3 numeric columns and rename them
                chart_data = chart_data[numeric_cols[:3]].copy()

            # Rename columns to A, B, C for consistency
            chart_data.columns = ['A', 'B', 'C']

        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            chart_data = None
    else:
        st.sidebar.info("Please upload a CSV or Excel file")

else:  # Generate Data
    st.sidebar.subheader("Data Generation Settings")

    # Add some interactive widgets to the sidebar
    distribution_type = st.sidebar.selectbox(
        'Select distribution type:',
        ['Normal', 'Uniform', 'Exponential', 'Poisson', 'Binomial', 'Log-Normal', 'Chi-Square', 'Beta', 'Gamma'],
        key=f"dist_type_{st.session_state.reset_trigger}"
    )

    num_points = st.sidebar.slider(
        'Number of data points:',
        10, 1000, DEFAULT_NUM_POINTS,
        key=f"num_points_{st.session_state.reset_trigger}"
    )

    random_seed = st.sidebar.number_input(
        'Random seed:',
        0, 1000, DEFAULT_SEED,
        key=f"seed_{st.session_state.reset_trigger}"
    )

    # Generate data using cached function
    with st.spinner('Generating data...'):
        chart_data = generate_data(num_points, random_seed, distribution_type)

# Visualization options (common for both data sources)
if chart_data is not None:
    st.sidebar.subheader("Visualization Settings")
    option = st.sidebar.selectbox(
        'Select a visualization type:',
        ['Line Chart', 'Bar Chart', 'Scatter Plot', 'Histogram', '3D Scatter', 'Box Plot'],
        key=f"viz_type_{st.session_state.reset_trigger}"
    )

# Main content area
if chart_data is not None:
    st.header("Data Visualization")
    data_description = f"Displaying a {option} with {len(chart_data)} data points"
    if data_source == 'Generate Data':
        data_description += f" using {distribution_type} distribution"
    else:
        data_description += " from uploaded file"
    st.write(data_description)

    # Display different visualizations based on user selection with error handling
    try:
        if option == 'Line Chart':
            fig = px.line(chart_data, y=['A', 'B', 'C'], title='Line Chart')
            fig.update_layout(height=500, xaxis_title='Index', yaxis_title='Values')
            st.plotly_chart(fig, use_container_width=True)

        elif option == 'Bar Chart':
            fig = px.bar(chart_data, y=['A', 'B', 'C'], title='Bar Chart', barmode='group')
            fig.update_layout(height=500, xaxis_title='Index', yaxis_title='Values')
            st.plotly_chart(fig, use_container_width=True)

        elif option == 'Scatter Plot':
            fig = px.scatter(chart_data, x='A', y='B', color='C',
                           title='Scatter Plot: A vs B (colored by C)',
                           color_continuous_scale='Viridis')
            fig.update_layout(height=500)
            fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))
            st.plotly_chart(fig, use_container_width=True)

        elif option == 'Histogram':
            fig = make_subplots(rows=1, cols=3, subplot_titles=('A Distribution', 'B Distribution', 'C Distribution'))
            fig.add_trace(go.Histogram(x=chart_data['A'], name='A', marker_color='blue', opacity=0.7), row=1, col=1)
            fig.add_trace(go.Histogram(x=chart_data['B'], name='B', marker_color='red', opacity=0.7), row=1, col=2)
            fig.add_trace(go.Histogram(x=chart_data['C'], name='C', marker_color='green', opacity=0.7), row=1, col=3)
            fig.update_layout(height=400, title_text='Histograms of All Variables', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        elif option == '3D Scatter':
            fig = px.scatter_3d(chart_data, x='A', y='B', z='C',
                              title='3D Scatter Plot',
                              color='C',
                              color_continuous_scale='Viridis')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        elif option == 'Box Plot':
            fig = go.Figure()
            fig.add_trace(go.Box(y=chart_data['A'], name='A', marker_color='blue'))
            fig.add_trace(go.Box(y=chart_data['B'], name='B', marker_color='red'))
            fig.add_trace(go.Box(y=chart_data['C'], name='C', marker_color='green'))
            fig.update_layout(title='Box Plot Comparison', yaxis_title='Values', height=500)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        st.info("Please try adjusting the parameters or selecting a different visualization type.")
else:
    st.info("Please generate data or upload a file to begin visualization.")

# Add a data table with the raw data
if chart_data is not None:
    st.subheader("Raw Data")
    st.dataframe(chart_data)

    # Multi-format export section
    st.subheader("Export Data")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        csv_data = export_data(chart_data, 'CSV')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="chart_data.csv",
            mime="text/csv",
        )

    with col2:
        excel_data = export_data(chart_data, 'Excel')
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name="chart_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with col3:
        json_data = export_data(chart_data, 'JSON')
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="chart_data.json",
            mime="application/json",
        )

    with col4:
        parquet_data = export_data(chart_data, 'Parquet')
        st.download_button(
            label="Download Parquet",
            data=parquet_data,
            file_name="chart_data.parquet",
            mime="application/octet-stream",
        )

# Add enhanced metrics
if chart_data is not None:
    st.subheader("Data Summary Statistics")

    # Display metrics in a 3x4 grid (added row for advanced stats)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Mean of A", value=round(chart_data['A'].mean(), 3))
        st.metric(label="Std Dev of A", value=round(chart_data['A'].std(), 3))
        st.metric(label="Median of A", value=round(chart_data['A'].median(), 3))
        st.metric(label="Skewness of A", value=round(stats.skew(chart_data['A']), 3))
        st.metric(label="Kurtosis of A", value=round(stats.kurtosis(chart_data['A']), 3))
    with col2:
        st.metric(label="Mean of B", value=round(chart_data['B'].mean(), 3))
        st.metric(label="Std Dev of B", value=round(chart_data['B'].std(), 3))
        st.metric(label="Median of B", value=round(chart_data['B'].median(), 3))
        st.metric(label="Skewness of B", value=round(stats.skew(chart_data['B']), 3))
        st.metric(label="Kurtosis of B", value=round(stats.kurtosis(chart_data['B']), 3))
    with col3:
        st.metric(label="Mean of C", value=round(chart_data['C'].mean(), 3))
        st.metric(label="Std Dev of C", value=round(chart_data['C'].std(), 3))
        st.metric(label="Median of C", value=round(chart_data['C'].median(), 3))
        st.metric(label="Skewness of C", value=round(stats.skew(chart_data['C']), 3))
        st.metric(label="Kurtosis of C", value=round(stats.kurtosis(chart_data['C']), 3))

    # Percentiles section
    st.subheader("Percentile Analysis")
    percentile_df = pd.DataFrame({
        'A': [
            chart_data['A'].quantile(0.25),
            chart_data['A'].quantile(0.50),
            chart_data['A'].quantile(0.75),
            chart_data['A'].quantile(0.90),
            chart_data['A'].quantile(0.95)
        ],
        'B': [
            chart_data['B'].quantile(0.25),
            chart_data['B'].quantile(0.50),
            chart_data['B'].quantile(0.75),
            chart_data['B'].quantile(0.90),
            chart_data['B'].quantile(0.95)
        ],
        'C': [
            chart_data['C'].quantile(0.25),
            chart_data['C'].quantile(0.50),
            chart_data['C'].quantile(0.75),
            chart_data['C'].quantile(0.90),
            chart_data['C'].quantile(0.95)
        ]
    }, index=['25th Percentile', '50th Percentile (Median)', '75th Percentile', '90th Percentile', '95th Percentile'])

    st.dataframe(percentile_df.style.format("{:.3f}").background_gradient(cmap='YlOrRd'))

# Add correlation matrix
if chart_data is not None:
    st.subheader("Correlation Matrix")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(chart_data.corr().style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format("{:.3f}"))

    with col2:
        # Use Plotly instead of matplotlib for interactive heatmap
        correlation = chart_data.corr()
        fig = px.imshow(correlation,
                       labels=dict(color="Correlation"),
                       x=correlation.columns,
                       y=correlation.columns,
                       color_continuous_scale='RdBu_r',
                       aspect="auto",
                       title='Interactive Correlation Heatmap',
                       zmin=-1, zmax=1)
        fig.update_traces(text=correlation.values.round(2), texttemplate='%{text}')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Add additional statistics
    st.subheader("Additional Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Min Value (all)", value=round(chart_data.min().min(), 3))
    with col2:
        st.metric(label="Max Value (all)", value=round(chart_data.max().max(), 3))
    with col3:
        st.metric(label="Range (all)", value=round(chart_data.max().max() - chart_data.min().min(), 3))
    with col4:
        st.metric(label="Total Points", value=len(chart_data))

# Add an expandable section with more information
with st.expander("See explanation and features"):
    st.write("""
        **This enhanced app demonstrates advanced Streamlit capabilities:**

        - **Data Source Options**:
            - Generate synthetic data with 9 different statistical distributions
            - Upload your own CSV or Excel files for analysis
        - **Interactive Plotly visualizations** with zoom, pan, and hover capabilities:
            - Line Chart, Bar Chart, Scatter Plot
            - Histogram, 3D Scatter Plot, Box Plot
        - **Comprehensive statistical analysis**:
            - Mean, median, standard deviation
            - Skewness and kurtosis (distribution shape)
            - Percentile analysis (25th, 50th, 75th, 90th, 95th)
            - Interactive correlation matrix and heatmap
            - Min/max values and range
        - **Multi-format data export**:
            - CSV, Excel, JSON, Parquet formats
        - **Professional features**:
            - Data caching for improved performance
            - Error handling for robust operation
            - Reset button to restore default parameters
            - Session state management for better UX

        **How to use:**
        1. Choose between generating data or uploading your own file
        2. If generating: select distribution type, adjust parameters
        3. If uploading: select a CSV or Excel file with numeric data
        4. Choose a visualization type from 6 available options
        5. Explore comprehensive statistics and correlations
        6. Download data in your preferred format (CSV, Excel, JSON, Parquet)
        7. Click "Reset to Defaults" to start over
    """)

# Footer
if chart_data is not None:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Created with Streamlit & Plotly**")
    with col2:
        st.markdown(f"**Data Points:** {len(chart_data)}")
    with col3:
        if data_source == 'Generate Data':
            st.markdown(f"**Distribution:** {distribution_type}")
        else:
            st.markdown(f"**Source:** Uploaded File")