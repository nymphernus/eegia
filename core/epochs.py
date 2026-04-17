from mne import events_from_annotations, find_events, Epochs
from os import getenv
from .ica import apply_ica
import numpy as np

'''
Функция разделения на эпохи

Нарезаем непрерывную запись на короткие эпохи вокруг событий. В ЭЭГ нас интересует реакция на конкретное событие.
Мы берем окно (например, от -0.2 до 0.8 сек), где 0 — момент стимула.
Baseline correction вычитает среднее значение "пред-стимульного" периода, чтобы скомпенсировать возможный сдвиг напряжения в момент записи.
'''

def epochs_from_raw(raw, tmin=-0.2, tmax=0.8, event_id=None, picks='data', events=None):
    sfreq = int(getenv("SFREQ"))
    if events is None:
        events, _ = events_from_annotations(raw, verbose=False)
        if len(events) == 0:
            events = find_events(raw, stim_channel='STI 014', verbose=False)

    unique, counts = np.unique(events[:, -1], return_counts=True)
    valid_classes = unique[counts > 5]
    events = events[np.isin(events[:, -1], valid_classes)]
    
    raw_proc = raw.copy()
    raw_proc.filter(1, 40, fir_design='firwin', picks='data', verbose=False)
    if raw_proc.info['sfreq'] > sfreq:
        raw_proc.resample(sfreq, verbose=False)
    raw_proc.set_eeg_reference('average', projection=False, verbose=False)

    raw_clean = apply_ica(raw_proc)
    
    epochs = Epochs(
        raw_clean, 
        events=events, 
        event_id=event_id,
        tmin=tmin, 
        tmax=tmax,
        baseline=(None, 0),
        preload=True,
        reject=dict(eeg=150e-6),
        flat=dict(eeg=1e-6),
        verbose=False
    )

    X = epochs.get_data(picks=picks) * 1e6
    y = epochs.events[:, -1]

    print(f"Всего эпох после фильтрации: {len(epochs)}")
    print(f"Классы: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y