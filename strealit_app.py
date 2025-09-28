import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Facebook vs AdWords Campaign Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    df = pd.read_csv('marketing_campaign.csv')
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Clean cost columns (remove $ and convert to float)
    cost_columns = ['Cost per Facebook Ad', 'Cost per AdWords Ad', 
                   'Facebook Cost per Click (Ad Cost / Clicks)', 
                   'AdWords Cost per Click (Ad Cost / Clicks)']
    
    for col in cost_columns:
        df[col] = df[col].str.replace('$', '').astype(float)
    
    # Clean percentage columns
    percentage_columns = ['Facebook Click-Through Rate (Clicks / View)',
                         'Facebook Conversion Rate (Conversions / Clicks)',
                         'AdWords Click-Through Rate (Clicks / View)',
                         'AdWords Conversion Rate (Conversions / Click)']
    
    for col in percentage_columns:
        df[col] = df[col].str.replace('%', '').astype(float) / 100
    
    # Add derived columns
    df['Month'] = df['Date'].dt.month
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Week'] = df['Date'].dt.isocalendar().week
    
    return df

def create_summary_metrics(df):
    """Create summary metrics for both platforms"""
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
    df = load_data()
    
    # Sidebar
    st.sidebar.header("📋 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["📈 Overview & Metrics", "📊 Performance Comparison", "🔬 Statistical Analysis", 
         "📉 Regression Analysis", "💰 Cost Analysis", "📅 Time Series Analysis"]
    )
    
    # Main content based on selection
    if analysis_type == "📈 Overview & Metrics":
        st.header("Campaign Overview & Key Metrics")
        
        # Summary metrics
        fb_metrics, aw_metrics = create_summary_metrics(df)
        
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
        
        # Interactive plots
        fig = plot_conversion_comparison(df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly performance
        st.subheader("📅 Weekly Performance Analysis")
        weekly_data = df.groupby('Day_of_Week').agg({
            'Facebook Ad Conversions': 'mean',
            'AdWords Ad Conversions': 'mean'
        }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
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
    
    elif analysis_type == "🔬 Statistical Analysis":
        st.header("Statistical Hypothesis Testing")
        
        # Perform hypothesis test
        test_results = perform_hypothesis_test(df)
        
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
    
    elif analysis_type == "📉 Regression Analysis":
        st.header("Regression Analysis: Facebook Clicks → Conversions")
        
        # Create regression analysis
        fig_reg, model, r2, mse = create_regression_analysis(df)
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
    
    elif analysis_type == "💰 Cost Analysis":
        st.header("Cost Effectiveness Analysis")
        
        # Cost per conversion analysis
        df['FB_Cost_per_Conversion'] = df['Cost per Facebook Ad'] / df['Facebook Ad Conversions']
        df['AW_Cost_per_Conversion'] = df['Cost per AdWords Ad'] / df['AdWords Ad Conversions']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💵 Cost Metrics")
            fb_avg_cost_conv = df['FB_Cost_per_Conversion'].mean()
            aw_avg_cost_conv = df['AW_Cost_per_Conversion'].mean()
            
            st.metric("Facebook Avg Cost per Conversion", f"${fb_avg_cost_conv:.2f}")
            st.metric("AdWords Avg Cost per Conversion", f"${aw_avg_cost_conv:.2f}")
            
            savings = aw_avg_cost_conv - fb_avg_cost_conv
            st.metric("Savings per Conversion (Facebook)", f"${savings:.2f}")
        
        with col2:
            # Cost comparison chart
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Box(
                y=df['FB_Cost_per_Conversion'],
                name='Facebook',
                marker_color='#1f77b4'
            ))
            fig_cost.add_trace(go.Box(
                y=df['AW_Cost_per_Conversion'],
                name='AdWords',
                marker_color='#ff7f0e'
            ))
            fig_cost.update_layout(
                title='Cost per Conversion Distribution',
                yaxis_title='Cost per Conversion ($)'
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        
        # Monthly cost trends
        st.subheader("📈 Monthly Cost Trends")
        monthly_costs = df.groupby('Month').agg({
            'FB_Cost_per_Conversion': 'mean',
            'AW_Cost_per_Conversion': 'mean'
        })
        
        fig_monthly_cost = go.Figure()
        fig_monthly_cost.add_trace(go.Scatter(
            x=monthly_costs.index,
            y=monthly_costs['FB_Cost_per_Conversion'],
            name='Facebook',
            line=dict(color='#1f77b4')
        ))
        fig_monthly_cost.add_trace(go.Scatter(
            x=monthly_costs.index,
            y=monthly_costs['AW_Cost_per_Conversion'],
            name='AdWords',
            line=dict(color='#ff7f0e')
        ))
        fig_monthly_cost.update_layout(
            title='Monthly Average Cost per Conversion',
            xaxis_title='Month',
            yaxis_title='Cost per Conversion ($)'
        )
        st.plotly_chart(fig_monthly_cost, use_container_width=True)
    
    elif analysis_type == "📅 Time Series Analysis":
        st.header("Time Series Analysis")
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Facebook correlation
            fb_corr = df['Facebook Ad Clicks'].corr(df['Facebook Ad Conversions'])
            st.metric("Facebook Clicks-Conversions Correlation", f"{fb_corr:.4f}")
            
            # AdWords correlation
            aw_corr = df['AdWords Ad Clicks'].corr(df['AdWords Ad Conversions'])
            st.metric("AdWords Clicks-Conversions Correlation", f"{aw_corr:.4f}")
        
        with col2:
            # Correlation heatmap
            corr_data = df[['Facebook Ad Clicks', 'Facebook Ad Conversions', 
                           'AdWords Ad Clicks', 'AdWords Ad Conversions']].corr()
            
            fig_heatmap = px.imshow(corr_data, 
                                   text_auto=True, 
                                   aspect="auto",
                                   title="Correlation Matrix")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Time series decomposition visualization
        st.subheader("📊 Conversion Trends Over Time")
        
        # Create rolling averages
        df['FB_Rolling_7'] = df['Facebook Ad Conversions'].rolling(window=7).mean()
        df['AW_Rolling_7'] = df['AdWords Ad Conversions'].rolling(window=7).mean()
        
        fig_trends = go.Figure()
        
        # Original data
        fig_trends.add_trace(go.Scatter(
            x=df['Date'], y=df['Facebook Ad Conversions'],
            name='Facebook (Daily)', opacity=0.3, line=dict(color='#1f77b4')
        ))
        fig_trends.add_trace(go.Scatter(
            x=df['Date'], y=df['AdWords Ad Conversions'],
            name='AdWords (Daily)', opacity=0.3, line=dict(color='#ff7f0e')
        ))
        
        # Rolling averages
        fig_trends.add_trace(go.Scatter(
            x=df['Date'], y=df['FB_Rolling_7'],
            name='Facebook (7-day avg)', line=dict(color='#1f77b4', width=3)
        ))
        fig_trends.add_trace(go.Scatter(
            x=df['Date'], y=df['AW_Rolling_7'],
            name='AdWords (7-day avg)', line=dict(color='#ff7f0e', width=3)
        ))
        
        fig_trends.update_layout(
            title='Daily Conversions with 7-Day Moving Average',
            xaxis_title='Date',
            yaxis_title='Conversions',
            height=500
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    
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
