import streamlit as st
import pandas as pd

st.set_page_config(page_title="Campaign Analysis", page_icon="📊")

st.title("📊 Facebook vs AdWords Analysis")

try:
    df = pd.read_csv('marketing_campaign.csv')
    st.success(f"Data loaded: {len(df)} records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Facebook")
        st.metric("Total Conversions", df['Facebook Ad Conversions'].sum())
        st.metric("Avg Daily", f"{df['Facebook Ad Conversions'].mean():.1f}")
    
    with col2:
        st.subheader("AdWords") 
        st.metric("Total Conversions", df['AdWords Ad Conversions'].sum())
        st.metric("Avg Daily", f"{df['AdWords Ad Conversions'].mean():.1f}")
    
    st.subheader("Data Preview")
    st.dataframe(df[['Date', 'Facebook Ad Conversions', 'AdWords Ad Conversions']].head())
    
except Exception as e:
    st.error(f"Error: {e}")
    st.write("Please ensure marketing_campaign.csv is in the repository.")
