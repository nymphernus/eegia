from os import getenv
from mne.datasets import eegbci, sample, erp_core
from mne.io import read_raw_edf, read_raw_fif
from mne import concatenate_raws

'''
Загрузка ДАТАСЕТА

Подгружаем сырые данные (Raw). Разные датасеты имеют свою специфику:
    EEG BCI фокусируется на моторном воображении (представлении движений),
    MNE Sample — на первичных сенсорных ответах (слух/зрение),
    а ERP Core — на когнитивном контроле (задача Фланкера).
Переименование каналов и установка стандартного монтажа (10-20) необходимы для корректной пространственной локализации электродов
'''

n_sub = int(getenv("N_SUB"))

def load_selected_dataset(choice):
    match choice:
        case 1:
            raw = eeg_bci_load()
        case 2:
            raw = mne_sample_load()
        case 3:
            raw = erp_core_load()
        case _:
            raise ValueError("Не существует")
    
    return raw

def eeg_bci_load():
    print("Загрузка EEG BCI")
    subjects = range(1, n_sub + 1)
    runs = [3, 7, 11]
    all_raws = []
    for s in subjects:
        fnames = eegbci.load_data(s, runs)
        all_raws.extend([read_raw_edf(f, preload=True) for f in fnames])
    raw = concatenate_raws(all_raws)
    raw.rename_channels(lambda x: x.strip('.').replace('Z', 'z').replace('FP', 'Fp'))
    raw.set_montage('standard_1020', on_missing='ignore')
    return raw

def mne_sample_load():
    print("Загрузка MNE Sample")
    data_path = sample.data_path()
    raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
    raw = read_raw_fif(raw_fname, preload=True)
    raw.pick(['eeg', 'stim'])
    return raw

def erp_core_load():
    print("Загрузка ERP Core")
    raw_fname = erp_core.data_path() / f'ERP-CORE_Subject-001_Task-Flankers_eeg.fif'
    raw = read_raw_fif(raw_fname, preload=True)
    raw.pick(['eeg', 'stim'])
    return raw