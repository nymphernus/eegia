import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st

st.set_page_config(
    page_title="EEG Insights Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 EEG Insights Agent")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📂 Данные", use_container_width=True):
        st.switch_page("pages/1_Data.py")
        
with col2:
    if st.button("🧠 Модели", use_container_width=True):
        st.switch_page("pages/2_Models.py")
        
with col3:
    if st.button("⚙️ Pipeline", use_container_width=True):
        st.switch_page("pages/3_Pipeline.py")
        
with col4:
    if st.button("📊 Benchmark", use_container_width=True):
        st.switch_page("pages/4_Benchmark.py")
        
with col5:
    if st.button("🔧 Настройки", use_container_width=True):
        st.switch_page("pages/5_Settings.py")

st.header("🖥️ Характеристики системы")
    
import platform
st.caption(f"ОС: {platform.system()} {platform.release()}")
st.caption(f"Python: {platform.python_version()}")
st.caption(f"Streamlit: {st.__version__}")
st.caption("Версия приложения 1.0.0 | © 2025")