import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st

st.set_page_config(
    page_title="EEG Insights Agent",
    page_icon="🧠",
    layout="wide"
)

st.sidebar.header("🖥️ Система")
import platform
def get_os_version():
    if platform.system() == "Windows":
        build_number = int(platform.version().split('.')[-1])
        return "Windows 11" if build_number >= 22000 else "Windows 10"
    return f"{platform.system()} {platform.release()}"
st.sidebar.caption(f"ОС: {get_os_version()}")
st.sidebar.caption(f"Python: {platform.python_version()}")
st.sidebar.caption(f"Streamlit: {st.__version__}")
st.sidebar.caption("Версия приложения 1.0.0 | © 2025")

st.title("🧠 EEG Insights Agent")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📂 Загрузить данные", use_container_width=True):
        st.switch_page("pages/1_Load_Datasets.py")
        
with col2:
    if st.button("🧠 Подключить модели", use_container_width=True):
        st.switch_page("pages/2_Load_Models.py")
    
with col3:
    if st.button("🧠 Обработанные данные", use_container_width=True):
        st.switch_page("pages/4_Processed_Data.py")

col1, col2 = st.columns(2)

with col1:
    if st.button("⚙️ Pipeline", use_container_width=True):
        st.switch_page("pages/3_Pipeline.py")
        
with col2:
    if st.button("📊 Benchmark", use_container_width=True):
        st.switch_page("pages/5_Benchmark.py")