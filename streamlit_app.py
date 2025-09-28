import streamlit as st
import pandas as pd
import numpy as np

# Set page config first - must be the first Streamlit command
st.set_page_config(
    page_title="Facebook vs AdWords Analysis",
    page_icon="📊",
    layout="wide"
)

# Try importing optional libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the marketing campaign data"""
    try:
        df = pd.read_csv('marketing_campaign.csv')
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Clean cost columns (remove $ and convert to float)
        cost_columns = ['Cost per Facebook Ad', 'Cost per AdWords Ad', 
                       'Facebook Cost per Click (Ad Cost / Clicks)', 
                       'AdWords Cost per Click (Ad Cost / Clicks)']
        
        for col in cost_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').astype(float)
        
        # Clean percentage columns
        percentage_columns = ['Facebook Click-Through Rate (Clicks / View)',
                             'Facebook Conversion Rate (Conversions / Clicks)',
                             'AdWords Click-Through Rate (Clicks / View)',
                             'AdWords Conversion Rate (Conversions / Click)']
        
        for col in percentage_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100
        
        # Add derived columns
        df['Month'] = df['Date'].dt.month
        df['Day_of_Week'] = df['Date'].dt.day_name()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_summary_metrics(df):
    """Create summary metrics for both platforms"""
    if df is None:
        return None, None
        
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
    
    return fb_metrics, aw_metrics

def plot_conversion_comparison(df):
    """Create conversion comparison plots"""
    if not PLOTLY_AVAILABLE:
        return None
        
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Conversions Over Time', 'Conversion Distribution', 
                       'Clicks vs Conversions', 'Monthly Conversion Trends'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Daily conversions over time
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Facebook Ad Conversions'], 
                  name='Facebook', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['AdWords Ad Conversions'], 
                  name='AdWords', line=dict(color='#ff7f0e')),
        row=1, col=1
    )
    
    # Conversion distribution
    fig.add_trace(
        go.Histogram(x=df['Facebook Ad Conversions'], name='Facebook', 
                    opacity=0.7, nbinsx=20, marker_color='#1f77b4'),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=df['AdWords Ad Conversions'], name='AdWords', 
                    opacity=0.7, nbinsx=20, marker_color='#ff7f0e'),
        row=1, col=2
    )
    
    # Clicks vs Conversions scatter
    fig.add_trace(
        go.Scatter(x=df['Facebook Ad Clicks'], y=df['Facebook Ad Conversions'],
                  mode='markers', name='Facebook', marker=dict(color='#1f77b4')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['AdWords Ad Clicks'], y=df['AdWords Ad Conversions'],
                  mode='markers', name='AdWords', marker=dict(color='#ff7f0e')),
        row=2, col=1
    )
    
    # Monthly trends
    monthly_fb = df.groupby('Month')['Facebook Ad Conversions'].mean()
    monthly_aw = df.groupby('Month')['AdWords Ad Conversions'].mean()
    
    fig.add_trace(
        go.Scatter(x=monthly_fb.index, y=monthly_fb.values, 
                  name='Facebook Monthly Avg', line=dict(color='#1f77b4')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=monthly_aw.index, y=monthly_aw.values, 
                  name='AdWords Monthly Avg', line=dict(color='#ff7f0e')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Campaign Performance Analysis")
    return fig

def perform_hypothesis_test(df):
    """Perform hypothesis test comparing Facebook vs AdWords conversions"""
    if not SCIPY_AVAILABLE:
        return None
        
    fb_conversions = df['Facebook Ad Conversions']
    aw_conversions = df['AdWords Ad Conversions']
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(fb_conversions, aw_conversions, alternative='greater')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(fb_conversions) - 1) * fb_conversions.var() + 
                         (len(aw_conversions) - 1) * aw_conversions.var()) / 
                        (len(fb_conversions) + len(aw_conversions) - 2))
    cohens_d = (fb_conversions.mean() - aw_conversions.mean()) / pooled_std
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'fb_mean': fb_conversions.mean(),
        'aw_mean': aw_conversions.mean(),
        'fb_std': fb_conversions.std(),
        'aw_std': aw_conversions.std()
    }

def create_regression_analysis(df):
    """Create regression analysis for Facebook clicks vs conversions"""
    if not SKLEARN_AVAILABLE:
        return None, None, None, None
        
    X = df[['Facebook Ad Clicks']]
    y = df['Facebook Ad Conversions']
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Create prediction plot
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df['Facebook Ad Clicks'], 
        y=df['Facebook Ad Conversions'],
        mode='markers',
        name='Actual Data',
        marker=dict(color='#1f77b4', size=8)
    ))
    
    # Regression line
    fig.add_trace(go.Scatter(
        x=df['Facebook Ad Clicks'], 
        y=y_pred,
        mode='lines',
        name=f'Regression Line (R² = {r2:.3f})',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Facebook Clicks vs Conversions - Linear Regression',
        xaxis_title='Facebook Ad Clicks',
        yaxis_title='Facebook Ad Conversions',
        height=500
    )
    
    return fig, model, r2, mse

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Facebook vs AdWords Campaign Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading campaign data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if 'marketing_campaign.csv' exists in the repository.")
        return
    
    st.success(f"✅ Data loaded successfully! {len(df)} records found.")
    
    # Sidebar
    st.sidebar.header("📋 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["📈 Overview & Metrics", "📊 Performance Comparison", "🔬 Statistical Analysis", "📉 Regression Analysis"]
    )
    
    # Main content based on selection
    if analysis_type == "📈 Overview & Metrics":
        st.header("Campaign Overview & Key Metrics")
        
        # Summary metrics
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
        <div class="insight-box">
        <h3>🎯 Key Insights</h3>
        <ul>
        <li><strong>Facebook outperforms AdWords</strong> in daily conversions (11.74 vs 5.98 average)</li>
        <li><strong>Facebook has higher conversion rates</strong> despite lower click volumes</li>
        <li><strong>AdWords generates more clicks</strong> but with lower conversion efficiency</li>
        <li><strong>Facebook shows better ROI</strong> with nearly double the conversion rate</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))
    
    elif analysis_type == "📊 Performance Comparison":
        st.header("Performance Comparison Analysis")
        
        if PLOTLY_AVAILABLE:
            # Interactive plots
            fig = plot_conversion_comparison(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to basic chart
            chart_data = df[['Date', 'Facebook Ad Conversions', 'AdWords Ad Conversions']].set_index('Date')
            st.line_chart(chart_data)
        
        # Weekly performance
        st.subheader("📅 Weekly Performance Analysis")
        weekly_data = df.groupby('Day_of_Week').agg({
            'Facebook Ad Conversions': 'mean',
            'AdWords Ad Conversions': 'mean'
        }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        if PLOTLY_AVAILABLE:
            fig_weekly = go.Figure()
            fig_weekly.add_trace(go.Bar(
                x=weekly_data.index,
                y=weekly_data['Facebook Ad Conversions'],
                name='Facebook',
                marker_color='#1f77b4'
            ))
            fig_weekly.add_trace(go.Bar(
                x=weekly_data.index,
                y=weekly_data['AdWords Ad Conversions'],
                name='AdWords',
                marker_color='#ff7f0e'
            ))
            fig_weekly.update_layout(
                title='Average Daily Conversions by Day of Week',
                xaxis_title='Day of Week',
                yaxis_title='Average Conversions',
                barmode='group'
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
        else:
            st.bar_chart(weekly_data)
    
    elif analysis_type == "🔬 Statistical Analysis":
        st.header("Statistical Hypothesis Testing")
        
        # Perform hypothesis test
        test_results = perform_hypothesis_test(df)
        
        if test_results:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Test Results")
                st.metric("T-Statistic", f"{test_results['t_statistic']:.4f}")
                st.metric("P-Value", f"{test_results['p_value']:.2e}")
                st.metric("Cohen's d (Effect Size)", f"{test_results['cohens_d']:.4f}")
                
                if test_results['p_value'] < 0.05:
                    st.success("✅ **Statistically Significant**: Facebook significantly outperforms AdWords")
                else:
                    st.warning("⚠️ **Not Statistically Significant**: No significant difference found")
            
            with col2:
                st.subheader("📈 Descriptive Statistics")
                st.write("**Facebook Conversions:**")
                st.write(f"Mean: {test_results['fb_mean']:.2f}")
                st.write(f"Std Dev: {test_results['fb_std']:.2f}")
                
                st.write("**AdWords Conversions:**")
                st.write(f"Mean: {test_results['aw_mean']:.2f}")
                st.write(f"Std Dev: {test_results['aw_std']:.2f}")
            
            # Hypothesis explanation
            st.markdown("""
            <div class="insight-box">
            <h3>🧪 Hypothesis Test Details</h3>
            <p><strong>H₀:</strong> μ_Facebook ≤ μ_AdWords (Facebook conversions are less than or equal to AdWords)</p>
            <p><strong>H₁:</strong> μ_Facebook > μ_AdWords (Facebook conversions are greater than AdWords)</p>
            <p><strong>Test Type:</strong> One-tailed independent t-test</p>
            <p><strong>Significance Level:</strong> α = 0.05</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Statistical analysis requires SciPy. Please install it.")
    
    elif analysis_type == "📉 Regression Analysis":
        st.header("Regression Analysis: Facebook Clicks → Conversions")
        
        # Create regression analysis
        fig_reg, model, r2, mse = create_regression_analysis(df)
        
        if fig_reg:
            st.plotly_chart(fig_reg, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Model Performance")
                st.metric("R² Score", f"{r2:.4f}")
                st.metric("Mean Squared Error", f"{mse:.4f}")
                st.metric("Model Coefficient", f"{model.coef_[0]:.4f}")
                st.metric("Model Intercept", f"{model.intercept_:.4f}")
            
            with col2:
                st.subheader("🔮 Conversion Predictor")
                clicks_input = st.number_input("Enter number of clicks:", min_value=1, max_value=100, value=50)
                predicted_conversions = model.predict([[clicks_input]])[0]
                st.metric("Predicted Conversions", f"{predicted_conversions:.1f}")
                
                # Prediction examples
                st.write("**Example Predictions:**")
                for clicks in [30, 50, 70]:
                    pred = model.predict([[clicks]])[0]
                    st.write(f"{clicks} clicks → {pred:.1f} conversions")
        else:
            st.warning("Regression analysis requires Scikit-learn. Please install it.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>📊 Facebook vs AdWords Campaign Analysis Dashboard</p>
    <p>Built with Streamlit • Data covers full year 2019 (365 days)</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
