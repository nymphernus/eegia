import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st

st.set_page_config(
    page_title="EEG Model Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 EEG Model Agent")
st.write("Добро пожаловать! Это прототип для анализа и сравнения моделей ЭЭГ.")

st.markdown("""
### Навигация:
- 📂 Данные: загрузка .edf / .csv
- 🧠 Models: выбор и запуск моделей
- ⚙️ Pipeline: предобработка сигналов
- 📊 Benchmark: сравнение моделей
- 🔧 Settings: настройки среды
""")
