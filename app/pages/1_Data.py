import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import os
import pandas as pd

from __init__ import setup_project_paths
setup_project_paths()

from core.data.loader import load_edf, load_csv
from core.data.manager import DataManager

st.title("📂 Данные")

st.sidebar.header("⚙️ Настройки графика")

amplitude_scale = st.sidebar.slider(
    "Масштаб амплитуды", 
    min_value=0.5, 
    max_value=10000.0, 
    value=10.0, 
    step=0.000001,
    format="%.1f"
)

time_samples = st.sidebar.slider(
    "Количество отсчётов", 
    min_value=100, 
    max_value=50000, 
    value=5000, 
    step=100
)

vertical_spacing = st.sidebar.slider(
    "Вертикальное расстояние между каналами", 
    min_value=0.5, 
    max_value=1000.0, 
    value=10.0, 
    step=0.00001,
    format="%.1f"
)

channels_to_show = st.sidebar.slider(
    "Количество каналов", 
    min_value=1, 
    max_value=64, 
    value=32, 
    step=1
)

uploaded_file = st.file_uploader("Загрузите EEG файл (.edf или .csv)", type=["edf", "csv"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
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

    st.success(f"Файл {uploaded_file.name} успешно загружен!")
    st.write(f"Частота дискретизации: {sample.sfreq} Hz")
    st.write(f"Каналы: {sample.ch_names[:5]}... ({len(sample.ch_names)} всего)")
    st.write(f"Размер данных: {sample.data.shape}")

    st.write(f"Диапазон значений в первом канале: [{sample.data[0].min():.6f}, {sample.data[0].max():.6f}]")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    time_limit = min(time_samples, sample.data.shape[1])
    channel_limit = min(channels_to_show, sample.data.shape[0])
    
    for i in range(channel_limit):
        data_segment = sample.data[i][:time_limit] * amplitude_scale
        ax.plot(data_segment + i * vertical_spacing, linewidth=0.8)
    
    ax.set_xlabel('Время (отсчёты)')
    ax.set_ylabel('Амплитуда (условные единицы)')
    ax.set_title(f'EEG сигналы (масштаб: {amplitude_scale}x, каналы: {channel_limit}, отсчёты: {time_limit})')
    
    st.pyplot(fig)