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
- ⚙️ Pipeline: предобработка сигналов
- 🧠 Models: выбор и запуск моделей
- 📊 Benchmark: сравнение моделей
- 🔧 Settings: настройки среды
""")
