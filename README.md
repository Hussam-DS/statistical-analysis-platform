# Statistical Analysis Platform

A professional web-based data visualization and statistical analysis application built with Streamlit and Plotly.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

### Data Sources
- **Generate Synthetic Data**: Create datasets using 9 different statistical distributions
- **Upload Files**: Import your own CSV or Excel files for analysis

### Interactive Visualizations
- Line Chart
- Bar Chart
- Scatter Plot (2D with color mapping)
- Histogram (side-by-side distributions)
- 3D Scatter Plot
- Box Plot

### Comprehensive Statistical Analysis
- **Descriptive Statistics**: Mean, median, standard deviation
- **Distribution Metrics**: Skewness and kurtosis
- **Percentile Analysis**: 25th, 50th, 75th, 90th, and 95th percentiles
- **Correlation Analysis**: Interactive heatmap and correlation matrix

### Data Export
Export your data in multiple formats:
- CSV
- Excel (.xlsx)
- JSON
- Parquet

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd statistical-analysis-platform
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit pandas numpy matplotlib plotly scipy openpyxl pyarrow
```

## Usage

### Running the Application

Start the Streamlit server:
```bash
streamlit run statistical_analyzer.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Choose Data Source**:
   - Select "Generate Data" to create synthetic datasets
   - Select "Upload File" to analyze your own CSV/Excel files

2. **Configure Parameters** (for generated data):
   - Select distribution type (Normal, Uniform, Exponential, etc.)
   - Adjust number of data points (10-1000)
   - Set random seed for reproducibility

3. **Select Visualization**:
   - Choose from 6 different chart types
   - Interactive plots support zoom, pan, and hover

4. **Explore Statistics**:
   - View summary statistics for each variable
   - Analyze percentile distributions
   - Examine correlation patterns

5. **Export Data**:
   - Download processed data in your preferred format

6. **Reset**:
   - Click "Reset to Defaults" to start fresh

## Supported Distributions

When generating synthetic data, choose from:
- **Normal**: Standard normal distribution
- **Uniform**: Uniform distribution between -3 and 3
- **Exponential**: Exponential distribution (λ=1)
- **Poisson**: Poisson distribution (λ=5)
- **Binomial**: Binomial distribution (n=10, p=0.5)
- **Log-Normal**: Log-normal distribution (μ=0, σ=1)
- **Chi-Square**: Chi-square distribution (df=2)
- **Beta**: Beta distribution (α=2, β=5)
- **Gamma**: Gamma distribution (shape=2, scale=2)

## File Upload Requirements

When uploading your own data:
- **Supported Formats**: CSV (.csv), Excel (.xlsx, .xls)
- **Data Requirements**: At least 3 numeric columns
- **Processing**: First 3 numeric columns are automatically selected and renamed to A, B, C

## Project Structure

```
statistical-analysis-platform/
├── statistical_analyzer.py    # Main application file
├── requirements.txt           # Python dependencies
├── CLAUDE.md                 # Developer documentation
└── README.md                 # This file
```

## Technical Details

### Architecture
- **Framework**: Streamlit for web interface
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas and NumPy
- **Statistics**: SciPy for advanced statistical functions

### Key Features
- **Session State Management**: Maintains state across user interactions
- **Data Caching**: Optimized performance with `@st.cache_data`
- **Error Handling**: Robust error handling for file uploads and visualizations
- **Responsive Layout**: Wide layout with multi-column displays

## Dependencies

| Package | Purpose |
|---------|---------|
| streamlit | Web application framework |
| pandas | Data manipulation and analysis |
| numpy | Numerical computing |
| matplotlib | Basic plotting (legacy support) |
| plotly | Interactive visualizations |
| scipy | Advanced statistical functions |
| openpyxl | Excel file support |
| pyarrow | Parquet file support |

## Contributing

Contributions are welcome! Areas for enhancement:
- Additional distribution types
- More visualization options
- Advanced statistical tests
- Custom theming
- Data transformation features

## License

This project is licensed under the MIT License.

## Support

For issues or questions:
1. Check the expandable "See explanation and features" section in the app
2. Review the CLAUDE.md file for developer documentation
3. Open an issue in the repository

## Screenshots

The application provides:
- **Sidebar Controls**: All parameters and settings in an organized sidebar
- **Main Dashboard**: Large visualization area with interactive charts
- **Statistics Panel**: Comprehensive metrics displayed in organized columns
- **Data Table**: Raw data preview with all values
- **Export Section**: One-click downloads in multiple formats

## Performance

- Supports datasets up to 1000 points (configurable)
- Cached data generation for improved performance
- Optimized Plotly visualizations for smooth interactions
- Efficient file export using in-memory buffers

---

**Built with Streamlit & Plotly** | Made for data analysis and visualization
