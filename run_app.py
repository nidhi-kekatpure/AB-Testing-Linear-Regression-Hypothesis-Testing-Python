#!/usr/bin/env python3
"""
Simple launcher script for the Streamlit app
Run this file to start the application locally
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    try:
        # Check if streamlit is installed
        subprocess.run([sys.executable, "-c", "import streamlit"], check=True, capture_output=True)
        
        # Launch the app
        print("🚀 Starting Facebook vs AdWords Campaign Analysis Dashboard...")
        print("📊 Opening in your default browser...")
        print("🔗 Local URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        
    except subprocess.CalledProcessError:
        print("❌ Error: Streamlit is not installed.")
        print("📦 Please install requirements first:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: streamlit_app.py not found in current directory")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
