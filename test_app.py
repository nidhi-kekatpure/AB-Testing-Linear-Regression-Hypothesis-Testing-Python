import streamlit as st

st.set_page_config(page_title="Test App", page_icon="🧪")

st.title("🧪 Streamlit Test App")
st.write("If you can see this, Streamlit is working!")

st.success("✅ Basic Streamlit functionality confirmed")

# Test pandas
try:
    import pandas as pd
    st.success("✅ Pandas imported successfully")
    
    # Test file reading
    try:
        df = pd.read_csv('marketing_campaign.csv')
        st.success(f"✅ CSV file loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(3))
    except Exception as e:
        st.error(f"❌ CSV loading failed: {str(e)}")
        
except Exception as e:
    st.error(f"❌ Pandas import failed: {str(e)}")

st.write("🔍 Debug complete!")
