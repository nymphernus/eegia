from __init__ import setup_project_paths
setup_project_paths()
import streamlit as st
import os

if 'page_initialized' not in st.session_state:
    st.session_state.page_initialized = True
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded"
    )

st.title("🧠 Models")

# Кэширование модели
@st.cache_resource
def load_model_cached(model_type: str, model_name: str, model_path: str):
    from core.models.registry import get_model
    # НЕ ПЕРЕДАЁМ device - пусть модели сами решают
    model = get_model(model_type, model_name)
    model.load(model_path)
    return model

# Информация о доступных устройствах
st.sidebar.header("🖥️ Устройства")
try:
    import torch
    if torch.cuda.is_available():
        st.sidebar.success(f"✅ CUDA доступна ({torch.cuda.device_count()} устройств)")
        st.sidebar.info(f"Текущее: {torch.cuda.get_device_name()}")
    else:
        st.sidebar.warning("⚠️ CUDA недоступна")
except ImportError:
    st.sidebar.error("❌ PyTorch не установлен")

# Загрузка файлов моделей
st.subheader("📥 Загрузка модели")
uploaded_model = st.file_uploader("Загрузите файл модели (.h5, .pt, .bin)", 
                                 type=["h5", "pt", "bin", "pth", "ckpt"])

model_path_to_use = None
if uploaded_model is not None:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_model.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_model.getvalue())
        model_path_to_use = tmp_file.name
    st.success(f"Файл загружен: {uploaded_model.name}")

# Выбор модели
model_type = st.selectbox("Выберите тип модели", ["transformers", "pytorch", "tensorflow"])
model_name = st.text_input("Имя модели", "MyModel")

# Путь к модели
default_path = "cardiffnlp/twitter-roberta-base-sentiment-latest" if model_type == "transformers" else "models/tensorflow/eeg_v4.h5"
model_path = st.text_input("Путь к модели", model_path_to_use or default_path)

if st.button("🚀 Загрузить модель"):
    try:
        # Проверяем существование файла (только для локальных моделей)
        if model_type in ["pytorch", "tensorflow"] and not os.path.exists(model_path):
            st.error(f"❌ Файл не найден: {model_path}")
            st.info("💡 Используйте загрузчик файлов выше или укажите правильный путь.")
            st.stop()
        
        with st.spinner("Загрузка модели..."):
            model = load_model_cached(model_type, model_name, model_path)
            st.session_state["current_model"] = model
            st.session_state["current_model_type"] = model_type
            st.success(f"✅ Модель {model_name} ({model_type}) загружена!")
            
            # Показываем информацию о модели
            info = model.get_info()
            st.json(info)
            
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")

if "current_model" in st.session_state:
    st.subheader("🔮 Инференс")
    model = st.session_state["current_model"]
    model_type = st.session_state["current_model_type"]

    if model_type == "transformers":
        text_input = st.text_area("Введите текст", "Пример текста для анализа", height=100)
        if st.button("Анализировать текст"):
            try:
                with st.spinner("Выполняем анализ..."):
                    preds = model.predict([text_input])
                    st.write("📊 Результаты:")
                    st.write(preds)
            except Exception as e:
                st.error(f"❌ Ошибка инференса: {e}")
    else:
        st.info("")