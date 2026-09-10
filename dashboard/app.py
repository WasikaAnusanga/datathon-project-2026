"""
Urban Flow Analytics - Streamlit Interactive Dashboard
Team: DataCraft (Datathon 2026)
"""

import streamlit as st

st.set_page_config(
    page_title="UrbanFlow Analytics | DataCraft",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🚕 UrbanFlow Analytics Dashboard")
    st.markdown("### Team DataCraft - Datathon 2026")
    st.info("Welcome to the UrbanFlow Analytics platform. Dashboard modules will be loaded here.")

if __name__ == "__main__":
    main()
