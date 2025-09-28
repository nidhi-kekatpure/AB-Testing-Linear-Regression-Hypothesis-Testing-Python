import streamlit as st
import pandas as pd
import numpy as np

# Set page config first - must be the first Streamlit command
st.set_page_config(
    page_title="Facebook vs AdWords Analysis",
    page_icon="📊",
    layout="wide"
)

# Debug: Show app is starting
st.write("🚀 App starting...")

# Try importing optional libraries with debug info
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    st.write("✅ Plotly imported successfully")
except ImportError as e:
    PLOTLY_AVAILABLE = False
    st.warning(f"⚠️ Plotly not available: {str(e)}")

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
    st.write("✅ SciPy imported successfully")
except ImportError as e:
    SCIPY_AVAILABLE = False
    st.warning(f"⚠️ SciPy not available: {str(e)}")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
    st.write("✅ Scikit-learn imported successfully")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    st.warning(f"⚠️ Scikit-learn not available: {str(e)}")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the marketing campaign data"""
    st.write("📂 Starting data load...")
    try:
        df = pd.read_csv('marketing_campaign.csv')
        st.write(f"✅ CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        st.write("✅ Date column converted")
        
        # Clean cost columns (remove $ and convert to float)
        cost_columns = ['Cost per Facebook Ad', 'Cost per AdWords Ad', 
                       'Facebook Cost per Click (Ad Cost / Clicks)', 
                       'AdWords Cost per Click (Ad Cost / Clicks)']
        
        for col in cost_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').astype(float)
        st.write("✅ Cost columns cleaned")
        
        # Clean percentage columns
        percentage_columns = ['Facebook Click-Through Rate (Clicks / View)',
                             'Facebook Conversion Rate (Conversions / Clicks)',
                             'AdWords Click-Through Rate (Clicks / View)',
                             'AdWords Conversion Rate (Conversions / Click)']
        
        for col in percentage_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100
        st.write("✅ Percentage columns cleaned")
        
        # Add derived columns
        df['Month'] = df['Date'].dt.month
        df['Day_of_Week'] = df['Date'].dt.day_name()
        st.write("✅ Derived columns added")
        
        return df
    except FileNotFoundError:
        st.error("❌ File 'marketing_campaign.csv' not found in repository")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def create_summary_metrics(df):
    """Create summary metrics for both platforms"""
    if df is None:
        return None, None
        
    try:
        fb_metrics = {
            'Total Views': df['Facebook Ad Views'].sum(),
            'Total Clicks': df['Facebook Ad Clicks'].sum(),
            'Total Conversions': df['Facebook Ad Conversions'].sum(),
            'Total Cost': df['Cost per Facebook Ad'].sum(),
            'Avg Daily Conversions': df['Facebook Ad Conversions'].mean(),
            'Conversion Rate': (df['Facebook Ad Conversions'].sum() / df['Facebook Ad Clicks'].sum()) * 100
        }
        
        aw_metrics = {
            'Total Views': df['AdWords Ad Views'].sum(),
            'Total Clicks': df['AdWords Ad Clicks'].sum(),
            'Total Conversions': df['AdWords Ad Conversions'].sum(),
            'Total Cost': df['Cost per AdWords Ad'].sum(),
            'Avg Daily Conversions': df['AdWords Ad Conversions'].mean(),
            'Conversion Rate': (df['AdWords Ad Conversions'].sum() / df['AdWords Ad Clicks'].sum()) * 100
        }
        
        st.write("✅ Metrics calculated successfully")
        return fb_metrics, aw_metrics
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")
        return None, None

def main():
    st.write("🎯 Starting main function...")
    
    # Header
    st.markdown('<h1 class="main-header">📊 Facebook vs AdWords Campaign Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    st.write("📊 Loading campaign data...")
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check if 'marketing_campaign.csv' exists in the repository.")
        st.stop()
    
    st.success(f"✅ Data loaded successfully! {len(df)} records found.")
    
    # Sidebar
    st.sidebar.header("📋 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["📈 Overview & Metrics", "📊 Performance Comparison", "🔬 Statistical Analysis"]
    )
    
    st.write(f"🔍 Selected analysis: {analysis_type}")
    
    # Main content based on selection
    if analysis_type == "📈 Overview & Metrics":
        st.header("Campaign Overview & Key Metrics")
        
        # Summary metrics
        st.write("📊 Calculating summary metrics...")
        fb_metrics, aw_metrics = create_summary_metrics(df)
        
        if fb_metrics and aw_metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔵 Facebook Campaign")
                st.metric("Total Views", f"{fb_metrics['Total Views']:,}")
                st.metric("Total Clicks", f"{fb_metrics['Total Clicks']:,}")
                st.metric("Total Conversions", f"{fb_metrics['Total Conversions']:,}")
                st.metric("Total Cost", f"${fb_metrics['Total Cost']:,.2f}")
                st.metric("Avg Daily Conversions", f"{fb_metrics['Avg Daily Conversions']:.2f}")
                st.metric("Overall Conversion Rate", f"{fb_metrics['Conversion Rate']:.2f}%")
            
            with col2:
                st.subheader("🟠 AdWords Campaign")
                st.metric("Total Views", f"{aw_metrics['Total Views']:,}")
                st.metric("Total Clicks", f"{aw_metrics['Total Clicks']:,}")
                st.metric("Total Conversions", f"{aw_metrics['Total Conversions']:,}")
                st.metric("Total Cost", f"${aw_metrics['Total Cost']:,.2f}")
                st.metric("Avg Daily Conversions", f"{aw_metrics['Avg Daily Conversions']:.2f}")
                st.metric("Overall Conversion Rate", f"{aw_metrics['Conversion Rate']:.2f}%")
        
        # Key insights
        st.markdown("""
        ### 🎯 Key Insights
        - **Facebook outperforms AdWords** in daily conversions (11.74 vs 5.98 average)
        - **Facebook has higher conversion rates** despite lower click volumes
        - **AdWords generates more clicks** but with lower conversion efficiency
        - **Facebook shows better ROI** with nearly double the conversion rate
        """)
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))
    
    elif analysis_type == "📊 Performance Comparison":
        st.header("Performance Comparison Analysis")
        
        if PLOTLY_AVAILABLE:
            st.write("📈 Creating Plotly charts...")
            try:
                # Create simple comparison chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Date'], 
                    y=df['Facebook Ad Conversions'],
                    name='Facebook',
                    line=dict(color='#1f77b4')
                ))
                fig.add_trace(go.Scatter(
                    x=df['Date'], 
                    y=df['AdWords Ad Conversions'],
                    name='AdWords',
                    line=dict(color='#ff7f0e')
                ))
                fig.update_layout(
                    title='Daily Conversions Over Time',
                    xaxis_title='Date',
                    yaxis_title='Conversions'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.write("✅ Plotly chart created successfully")
            except Exception as e:
                st.error(f"❌ Error creating Plotly chart: {str(e)}")
        else:
            st.write("📈 Creating basic chart...")
            # Fallback to basic chart
            chart_data = df[['Date', 'Facebook Ad Conversions', 'AdWords Ad Conversions']].set_index('Date')
            st.line_chart(chart_data)
            st.write("✅ Basic chart created successfully")
    
    elif analysis_type == "🔬 Statistical Analysis":
        st.header("Statistical Analysis")
        
        if SCIPY_AVAILABLE:
            st.write("📊 Performing statistical tests...")
            try:
                # Perform basic t-test
                fb_conversions = df['Facebook Ad Conversions']
                aw_conversions = df['AdWords Ad Conversions']
                
                t_stat, p_value = stats.ttest_ind(fb_conversions, aw_conversions)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Test Results")
                    st.metric("T-Statistic", f"{t_stat:.4f}")
                    st.metric("P-Value", f"{p_value:.4f}")
                    
                    if p_value < 0.05:
                        st.success("✅ **Statistically Significant**: Difference detected")
                    else:
                        st.warning("⚠️ **Not Statistically Significant**: No significant difference")
                
                with col2:
                    st.subheader("📈 Descriptive Statistics")
                    st.write("**Facebook Conversions:**")
                    st.write(f"Mean: {fb_conversions.mean():.2f}")
                    st.write(f"Std Dev: {fb_conversions.std():.2f}")
                    
                    st.write("**AdWords Conversions:**")
                    st.write(f"Mean: {aw_conversions.mean():.2f}")
                    st.write(f"Std Dev: {aw_conversions.std():.2f}")
                
                st.write("✅ Statistical analysis completed")
            except Exception as e:
                st.error(f"❌ Error in statistical analysis: {str(e)}")
        else:
            st.warning("Statistical analysis requires SciPy. Please install it.")
    
    # Footer
    st.markdown("---")
    st.markdown("📊 Facebook vs AdWords Campaign Analysis Dashboard")
    st.write("🏁 App completed successfully!")

if __name__ == "__main__":
    st.write("🎬 Starting application...")
    main()
    st.write("✅ Application finished!")
