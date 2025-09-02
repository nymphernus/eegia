from __init__ import setup_project_paths
setup_project_paths()

import os
import streamlit as st
import tempfile

from core.models.models_manager import ModelsManager

if 'page_initialized' not in st.session_state:
    st.session_state.page_initialized = True
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded"
    )

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Models")

manager = ModelsManager()

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

st.subheader("📥 Добавить модель")

source_type = st.radio("Источник", ["Файл", "HuggingFace Hub"], horizontal=True)

if source_type == "HuggingFace Hub":
    model_type = "transformers"
else:
    model_type = st.selectbox("Тип модели", ["tensorflow", "pytorch"])

model_path = None
if source_type == "Файл":
    uploaded_model = st.file_uploader(
        "Файл модели (.h5, .pt, .bin, .pth, .ckpt)",
        type=["h5", "pt", "bin", "pth", "ckpt"]
    )
    if uploaded_model is not None:
        tmp_dir = tempfile.gettempdir()
        model_path = os.path.join(tmp_dir, uploaded_model.name)
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getvalue())

        st.success(f"Файл загружен: {uploaded_model.name}")
else:
    repo_id = st.text_input("HuggingFace repo_id", placeholder="org/model или model")
    if repo_id.strip():
        model_path = repo_id.strip()
        st.info(f"Будет подключена модель из HuggingFace: {repo_id}")

if st.button("🚀 Добавить в базу"):
    if not model_path:
        st.error("❌ Укажите файл или repo_id")
    else:
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
            st.error(f"❌ Ошибка добавления: {e}")

st.subheader("📂 Доступные модели")
models = manager.list_models()
if not models:
    st.info("Пока нет сохранённых моделей")
else:
    for m in models:
        cols = st.columns([3, 3, 2, 2])
        with cols[0]:
            st.markdown(f"**{m['name']}**")
            st.caption(m["file_path"])
        with cols[1]:
            st.text(f"type: {m['model_type']}")
            st.text(f"added: {m.get('created_at_formatted', m.get('created_at',''))}")
        with cols[2]:
            if "current_model_id" in st.session_state and st.session_state["current_model_id"] == m['id']:
                if st.button("Деактивировать", key=f"deact_{m['id']}"):
                    st.session_state.pop("current_model", None)
                    st.session_state.pop("current_model_id", None)
                    st.success(f"Модель {m['name']} деактивирована")
                    st.rerun()
            else:
                if st.button("Активировать", key=f"act_{m['id']}"):
                    try:
                        with st.spinner("Загрузка модели..."):
                            model = manager.load_model(m['id'])
                        st.session_state["current_model"] = model
                        st.session_state["current_model_id"] = m['id']
                        st.success(f"Активирована: {m['name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка: {e}")
        with cols[3]:
            if st.button("Удалить", key=f"del_{m['id']}"):
                ok = manager.delete_model(m['id'])
                if ok:
                    st.success("Удалено")
                    st.rerun()
                else:
                    st.error("Не удалось удалить")

if "current_model" in st.session_state:
    current_id = st.session_state.get("current_model_id")
    info = manager.get_model_info(current_id) if current_id else None
    st.subheader(f"🔮 Инференс (активна: {info['name'] if info else '—'})")

    model = st.session_state["current_model"]

    try:
        st.json(model.get_info())
    except Exception:
        pass

    if info and info["model_type"] == "transformers":
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
        st.info("Инференс демо")
else:
    st.subheader("🔮 Инференс")
    st.info("Нет активной модели. Активируйте модель, чтобы запустить анализ.")

