from __init__ import setup_project_paths
setup_project_paths()

import streamlit as st
import matplotlib.pyplot as plt

from core.data.manager import DataManager

if 'page_initialized' not in st.session_state:
    st.session_state.page_initialized = True
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
        page_title="Processed EEG Data",
        page_icon="📊"
    )

st.title("📊 Обработанные данные")


manager = DataManager()

st.sidebar.header("⚙️ Настройки")
amplitude_scale = st.sidebar.slider("Масштаб амплитуды", min_value=0.5, max_value=10000.0, value=10.0, step=0.000001, format="%.1f")
vertical_spacing = st.sidebar.slider("Вертикальное расстояние между каналами", min_value=0.5, max_value=1000.0, value=10.0, step=0.00001, format="%.1f")
channels_to_show = st.sidebar.slider("Количество каналов", min_value=1, max_value=64, value=32, step=1)
time_samples = st.sidebar.slider("Количество отсчётов", min_value=100, max_value=50000, value=5000, step=100)

processed_data = manager.list_all_processed()

if not processed_data:
    st.info("📥 Нет обработанных данных. Перейдите на страницу 'Pipeline' для обработки файлов.")
else:
    from collections import defaultdict
    grouped_data = defaultdict(list)
    for item in processed_data:
        grouped_data[item['parent_id']].append(item)
    
    source_files = {}
    for parent_id in grouped_data.keys():
        source_info = manager.get_sample_info(parent_id)
        if source_info:
            source_files[parent_id] = source_info
    
    for parent_id, processed_list in grouped_data.items():
        source_info = source_files.get(parent_id, {})
        source_name = source_info.get('filename', f'Unknown ({parent_id[:8]})')
        
        with st.expander(f"📁 {source_name} ({len(processed_list)} обработок)", expanded=False):
            if source_info:
                st.markdown(f"""
                **Исходный файл:** `{source_name}`  
                **Частота:** {source_info.get('sfreq', 'N/A')} Hz  
                **Каналов:** {source_info.get('n_channels', 'N/A')}  
                **Отсчётов:** {source_info.get('n_samples', 'N/A'):,}
                """)
            
            st.divider()
            
            for item in processed_list:
                proc_id = item['id']
                short_id = proc_id[:8]
                
                card_class = "data-card"
                if 'selected_processed_id' in st.session_state and st.session_state.selected_processed_id == proc_id:
                    card_class += " highlight-card"
                
                
                created_at = item.get('created_at_formatted', item.get('created_at', ''))
                st.markdown(f"**Обработка ID:** `{short_id}` • **Создано:** {created_at}")
                
                st.markdown(f"""
                **Параметры:** {item.get('sfreq', 'N/A')} Hz • 
                {item.get('n_channels', 'N/A')} каналов • 
                {item.get('n_samples', 'N/A'):,} отсчётов
                """)
                
                proc_info = manager.get_processed_info(proc_id)
                if proc_info and proc_info.get('pipeline_config'):
                    pipeline_cfg = proc_info['pipeline_config']
                    steps = pipeline_cfg.get('steps', [])
                    if steps:
                        st.markdown("**Пайплайн обработки:**")
                        with st.expander("📋 Подробности", expanded=False):
                            for i, step in enumerate(steps, 1):
                                step_name = step.get('name', 'Unknown')
                                step_params = step.get('params', {})
                                st.markdown(f"{i}. **{step_name}**: {step_params}")
                
                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
                
                with btn_col1:
                    if st.button("👁 Просмотр", key=f"view_{proc_id}", use_container_width=True):
                        st.session_state.selected_processed_id = proc_id
                        st.rerun()
                
                with btn_col2:
                    if st.button("🗑 Удалить", key=f"delete_{proc_id}", use_container_width=True):
                        if manager.delete_processed_sample(proc_id):
                            if 'selected_processed_id' in st.session_state and st.session_state.selected_processed_id == proc_id:
                                del st.session_state.selected_processed_id
                            st.success(f"✅ Обработка {short_id} удалена")
                            st.rerun()
                        else:
                            st.error(f"❌ Ошибка удаления {short_id}")
                
                if 'selected_processed_id' in st.session_state and st.session_state.selected_processed_id == proc_id:
                    st.divider()
                    st.subheader(f"👁 Визуализация обработки {short_id}")
                    
                    try:
                        proc_sample = manager.get_processed_sample(proc_id)
                        if proc_sample is not None and proc_sample.data is not None:
                            data_shape = proc_sample.data.shape
                            n_channels = data_shape[0]
                            n_samples = data_shape[1] if len(data_shape) == 2 else data_shape[2]
                            
                            viz_col1, viz_col2, viz_col3 = st.columns(3)
                            with viz_col1:
                                st.metric("Каналов", n_channels)
                            with viz_col2:
                                st.metric("Отсчётов", f"{n_samples:,}")
                            with viz_col3:
                                st.metric("Частота", f"{proc_sample.sfreq} Hz")
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            display_channels = min(channels_to_show, n_channels)
                            display_samples = min(time_samples, n_samples)
                            
                            if len(data_shape) == 3:
                                data_to_plot = proc_sample.data[:, 0, :]
                            else:
                                data_to_plot = proc_sample.data
                            
                            for i in range(display_channels):
                                channel_data = data_to_plot[i][:display_samples] * amplitude_scale
                                ax.plot(channel_data + i * vertical_spacing, 
                                       linewidth=0.8, alpha=0.8)
                            
                            ax.set_xlabel('Время (отсчёты)')
                            ax.set_ylabel('Амплитуда')
                            ax.set_title(f'Обработанные данные (ID: {short_id})')
                            ax.grid(True, alpha=0.3)
                            
                            st.pyplot(fig)
                            
                            if st.button("❌ Закрыть визуализацию", key=f"close_viz_{proc_id}"):
                                del st.session_state.selected_processed_id
                                st.rerun()
                        else:
                            st.error("❌ Ошибка загрузки данных для визуализации")
                    except Exception as e:
                        st.error(f"❌ Ошибка визуализации: {str(e)}")
                
                st.divider()