from __init__ import setup_project_paths
setup_project_paths()

import os
import streamlit as st
import tempfile
import numpy as np
import pandas as pd
from core.models.models_manager import ModelsManager

st.set_page_config(
        layout="wide",
        page_title="EEG Insights Agent",
        page_icon="🧬"
    )

if 'page_initialized' not in st.session_state:
    st.session_state.page_initialized = True

st.title("🧠 Подключение моделей")
manager = ModelsManager()

with st.sidebar:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    st.header("🖥️ Устройства")
    try:
        import torch
        st.markdown("**PyTorch:** " + torch.__version__)

        if torch.cuda.is_available():
            st.success(f"✅ CUDA: {torch.cuda.device_count()} устройств")
            st.caption(f"GPU: {torch.cuda.get_device_name()}")
            st.caption(f"GPU RAM: {torch.cuda.get_device_properties(0).total_memory // 1024 // 1024} MB")
        else:
            st.warning("⚠️ CUDA недоступна")
            st.caption("Используется CPU")

    except ImportError:
        st.error("❌ PyTorch не установлен")
    try:
        import tensorflow as tf
        st.markdown("**TensorFlow:** " + tf.__version__)
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            st.success(f"✅ TF GPU: {len(gpu_devices)} устройств")
    except ImportError:
        pass
    try:
        import psutil
        st.markdown("**Система:**")
        st.caption(f"CPU: {psutil.cpu_count()} ядер")
        st.caption(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB")
    except ImportError:
        pass

st.subheader("📥 Добавить модель")

source_type = st.radio("Источник", ["Файл", "HuggingFace Hub"], horizontal=True)

if source_type == "Файл":
    model_type = st.selectbox(
        "Тип модели",
        ["tensorflow", "pytorch", "lightgbm", "eegnet"]
    )
else:
    model_type = "transformers"

model_path = None

if source_type == "Файл":
    uploaded = st.file_uploader("Файл модели", type=["h5", "pt", "bin", "pth", "ckpt", "pkl"])
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

    model_type = info.get("model_type", "")

    # LightGBM инференс
    if model_type == "lightgbm":
        if "loaded_features" not in st.session_state:
            st.warning("⚠️ Загрузите фичи")
        else:
            X, y = st.session_state["loaded_features"]
            if st.button("🧠 Предсказать (LightGBM)"):
                try:
                    preds = model.predict(X)
                    df = pd.DataFrame({"Предсказание": preds})
                    if y is not None:
                        df["Истинное"] = y
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"❌ Ошибка LightGBM: {e}")

    # EEGNet инференс
    elif model_type == "eegnet":
        if "loaded_features" not in st.session_state:
            st.warning("⚠️ Загрузите данные (сырые или обработанные)")
        else:
            X, y = st.session_state["loaded_features"]
            if st.button("🧠 Предсказать (EEGNet)"):
                try:
                    preds = model.predict(X)
                    df = pd.DataFrame({"Предсказание": preds})
                    if y is not None:
                        df["Истинное"] = y
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"❌ Ошибка EEGNet: {e}")

    # Transformers
    elif model_type == "transformers":
        st.info("⚠️ Инференс для Transformers уже реализован выше (текст/тайм-серии)")

    # PyTorch
    elif model_type == "pytorch":
        st.info("⚠️ Используйте блок инференса PyTorch выше")

    # TensorFlow
    elif model_type == "tensorflow":
        st.info("⚠️ Используйте блок инференса TensorFlow выше")

else:
    st.info("📭 Нет активной модели")
