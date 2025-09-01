import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import os
import pandas as pd
import hashlib

from __init__ import setup_project_paths
setup_project_paths()

from core.data.loader import load_edf, load_csv
from core.data.manager import DataManager

st.title("📂 Данные")

manager = DataManager()

# ==============================
# 🔹 Функция для вычисления hash файла (для уникальности)
# ==============================
def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# ==============================
# 🔹 Блок загрузки нового файла
# ==============================
st.header("➕ Загрузить новый файл")

uploaded_file = st.file_uploader("Загрузите EEG файл (.edf или .csv)", type=["edf", "csv"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.name.split(".")[-1]
    fhash = file_hash(file_bytes)

    # Проверяем, есть ли уже такой файл в каталоге
    existing = [s for s in manager.list_samples() if s.get("hash") == fhash]
    if existing:
        st.warning(f"⚠️ Файл `{uploaded_file.name}` уже загружен (ID={existing[0]['id']})")
    else:
        # Сохраняем временно и грузим
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
            tmp_file.write(file_bytes)
            tmp_file_path = tmp_file.name

        try:
            if file_type == "edf":
                sample = load_edf(tmp_file_path)
            elif file_type == "csv":
                df_temp = pd.read_csv(tmp_file_path, nrows=1)
                has_labels_flag = 'label' in [col.lower() for col in df_temp.columns]
                sample = load_csv(tmp_file_path, sfreq=256, has_labels=has_labels_flag)
            else:
                st.error("Неподдерживаемый формат")
                st.stop()
        finally:
            os.unlink(tmp_file_path)

        # Сохраняем с hash
        sample_id = manager.add_sample(sample, filename=uploaded_file.name)
        manager.catalog[sample_id]["hash"] = fhash
        manager._save_catalog()

        st.success(f"Файл {uploaded_file.name} успешно загружен (ID={sample_id})")

# ==============================
# 🔹 Каталог загруженных данных
# ==============================
st.header("📋 Каталог загруженных данных")

catalog = manager.list_samples()
if not catalog:
    st.info("Каталог пуст. Загрузите хотя бы один файл.")
    st.stop()

df_catalog = pd.DataFrame(catalog)
df_catalog_display = df_catalog[["id", "filename", "sfreq", "n_channels", "n_samples"]]

st.dataframe(df_catalog_display, use_container_width=True, hide_index=True)

# ==============================
# 🔹 Действия по каждой записи
# ==============================
st.subheader("⚡ Управление записями")

for sample in catalog:
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write(f"**{sample['filename']}** (ID={sample['id']}) — {sample['n_channels']} каналов, {sample['sfreq']} Hz")
    with col2:
        if st.button("📊 Показать", key=f"show_{sample['id']}"):
            st.session_state["selected_id"] = sample["id"]
            st.rerun()   # ✅ новый API вместо experimental_rerun
    with col3:
        if st.button("🗑 Удалить", key=f"del_{sample['id']}"):
            manager.delete_sample(sample["id"])
            st.rerun()

# ==============================
# 🔹 Визуализация выбранной записи
# ==============================
if "selected_id" in st.session_state:
    selected_id = st.session_state["selected_id"]
    data, meta = manager.load_sample(selected_id)

    st.subheader(f"🔎 Запись {selected_id}")
    st.write(f"Файл: {meta['filename']}")
    st.write(f"Частота дискретизации: {meta['sfreq']} Hz")
    st.write(f"Каналы: {meta['channels'][:5]}... ({meta['n_channels']} всего)")
    st.write(f"Размер данных: {data.shape}")
    st.write(f"Диапазон значений в первом канале: [{data[0].min():.6f}, {data[0].max():.6f}]")

    # ==============================
    # Настройки графика
    # ==============================
    st.sidebar.header("⚙️ Настройки графика")

    amplitude_scale = st.sidebar.slider(
        "Масштаб амплитуды", 
        min_value=0.5, max_value=10000.0, value=10.0, step=0.1
    )

    time_samples = st.sidebar.slider(
        "Количество отсчётов", 
        min_value=100, max_value=50000, value=5000, step=100
    )

    vertical_spacing = st.sidebar.slider(
        "Вертикальное расстояние между каналами", 
        min_value=0.5, max_value=1000.0, value=10.0, step=0.5
    )

    channels_to_show = st.sidebar.slider(
        "Количество каналов", 
        min_value=1, max_value=64, value=32, step=1
    )

    # ==============================
    # Построение графика
    # ==============================
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    time_limit = min(time_samples, data.shape[1])
    channel_limit = min(channels_to_show, data.shape[0])
    
    for i in range(channel_limit):
        data_segment = data[i][:time_limit] * amplitude_scale
        ax.plot(data_segment + i * vertical_spacing, linewidth=0.8)
    
    ax.set_xlabel('Время (отсчёты)')
    ax.set_ylabel('Амплитуда (условные единицы)')
    ax.set_title(f'EEG сигналы (масштаб: {amplitude_scale}x, каналы: {channel_limit}, отсчёты: {time_limit})')
    
    st.pyplot(fig)
