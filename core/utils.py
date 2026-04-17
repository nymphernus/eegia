import numpy as np
import matplotlib.pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from mne import events_from_annotations, find_events
from .epochs import epochs_from_raw
from mlflow import log_artifact
from os import path


'''
Визуализация матрицы ошибок и созранение её в артефактах MLflow
'''

def show_confusion_matrix(y_encoded, y_pred, display_labels, name, dataset_id, base_artifact_path):
    plt.figure(figsize=(6, 4))
    heatmap(confusion_matrix(y_encoded, y_pred), annot=True, fmt='d', cmap='Blues',
                yticklabels=display_labels)
    plt.title(f"Модель: {name} | Датасет: {dataset_id}")
    plt.xlabel('Predicted')
    plt.ylabel('True')

    filename = f"cm_{name}_ds{dataset_id}.png"
    full_path = path.join(base_artifact_path, filename)
    plt.savefig(full_path)

    log_artifact(full_path)

    plt.close()


'''
Подготовка размеченных данных

Сырые данные содержат множество технических меток (триггеров), не все из которых релевантны задаче классификации.
- Для MNE Sample мы объединяем разные типы звуковых и визуальных стимулов в два чистых макро-класса (Audio/Visual).
- Для ERP Core мы выделяем только те условия, которые отражают когнитивную нагрузку (Easy/Hard), отсеивая артефакты нажатия кнопок.
'''

def prepare_labeled_data(raw, dataset_id):
    if dataset_id == 2:
        events = find_events(raw, stim_channel='STI 014', verbose=False)
    else:
        events, _ = events_from_annotations(raw, verbose=False)
        
    # Оставляем только нужные триггеры, игнорируя технические события
    if dataset_id == 2:
        mask = np.isin(events[:, -1], [1, 2, 3, 4])
        events = events[mask]
        events[np.isin(events[:, -1], [1, 2]), -1] = 101
        events[np.isin(events[:, -1], [3, 4]), -1] = 102
        
    elif dataset_id == 3:
        mask = np.isin(events[:, -1], [3, 4, 5, 6])
        events = events[mask]
        events[np.isin(events[:, -1], [3, 4]), -1] = 201
        events[np.isin(events[:, -1], [5, 6]), -1] = 202
    
    return epochs_from_raw(raw, events=events)