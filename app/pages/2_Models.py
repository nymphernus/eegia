from __init__ import setup_project_paths
setup_project_paths()

import os
import streamlit as st
import tempfile
from core.models.models_manager import ModelsManager

if 'page_initialized' not in st.session_state:
    st.session_state.page_initialized = True
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title("🧠 Models")
manager = ModelsManager()

with st.sidebar:
    st.header("🖥️ Устройства")
    try:
        import torch
        if torch.cuda.is_available():
            st.success(f"✅ CUDA ({torch.cuda.device_count()} устройств)")
            st.info(torch.cuda.get_device_name())
        else:
            st.warning("⚠️ CUDA недоступна")
    except ImportError:
        st.error("❌ PyTorch не установлен")

st.subheader("📥 Добавить модель")

source_type = st.radio("Источник", ["Файл", "HuggingFace Hub"], horizontal=True)

model_type = "transformers" if source_type == "HuggingFace Hub" else st.selectbox("Тип модели", ["tensorflow", "pytorch"])
model_path = None

if source_type == "Файл":
    uploaded = st.file_uploader("Файл модели", type=["h5", "pt", "bin", "pth", "ckpt"])
    if uploaded:
        tmp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getvalue())
        model_path = tmp_path
        st.success(f"✅ Загружен: {uploaded.name}")
else:
    repo_id = st.text_input("HuggingFace repo_id", placeholder="org/model")
    if repo_id.strip():
        model_path = repo_id.strip()
        st.info(f"📡 Будет подключена: {repo_id}")

if st.button("🚀 Добавить в базу") and model_path:
    try:
        model_id = manager.add_model(
            name=None,
            model_type=model_type,
            file_path=model_path,
            metadata={"source": source_type}
        )
        st.success(f"✅ Модель сохранена (ID={model_id})")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")

st.subheader("📂 Доступные модели")
models = manager.list_models()

if not models:
    st.info("📥 Пока нет сохранённых моделей")
else:
    for m in models:
        with st.container(border=True):
            cols = st.columns([3, 2, 2, 1])
            
            with cols[0]:
                st.markdown(f"**{m['name']}**")
                st.caption(m["file_path"])
            
            with cols[1]:
                st.text(f"Тип: {m['model_type']}")
                st.text(f"Добавлена: {m.get('created_at_formatted', '—')}")
            
            with cols[2]:
                is_active = st.session_state.get("current_model_id") == m['id']
                if is_active:
                    if st.button("⏸ Деактивировать", key=f"deact_{m['id']}"):
                        st.session_state.pop("current_model", None)
                        st.session_state.pop("current_model_id", None)
                        st.rerun()
                else:
                    if st.button("▶ Активировать", key=f"act_{m['id']}"):
                        try:
                            with st.spinner("Загрузка..."):
                                model = manager.load_model(m['id'])
                            st.session_state["current_model"] = model
                            st.session_state["current_model_id"] = m['id']
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {e}")
            
            with cols[3]:
                if st.button("🗑", key=f"del_{m['id']}"):
                    manager.delete_model(m['id'])
                    st.rerun()

st.subheader("🔮 Инференс")
if "current_model" in st.session_state:
    model = st.session_state["current_model"]
    info = manager.get_model_info(st.session_state["current_model_id"])
    
    st.markdown(f"**Активна:** `{info['name']}`")
    try:
        st.json(model.get_info())
    except:
        pass

    if info["model_type"] == "transformers":
        text = st.text_area("Текст для анализа", "Пример текста", height=100)
        if st.button("Анализировать"):
            try:
                with st.spinner("Анализ..."):
                    result = model.predict([text])
                st.write("📊 Результаты:")
                st.write(result)
            except Exception as e:
                st.error(f"❌ {e}")
    else:
        st.info("⚡ Инференс для этой модели будет добавлен позже")
else:
    st.info("📭 Нет активной модели")