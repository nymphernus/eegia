from mne.preprocessing import ICA

'''
Функция ICA

Разбиваем многоканальный сигнал на статистически независимые источники и удаляем те, что похожи на моргание глаз. Глазные яблоки — это диполи.
При моргании они создают мощный электрический импульс, который "забивает" сигналы мозга.
ICA (Independent Component Analysis) позволяет отделить сигнал "моргания" от "мыслей", не удаляя при этом полезные данные из самих каналов.
'''

def apply_ica(raw, n_components=20, random_state=42, threshold=2.5):
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    n_channels = len(raw_for_ica.ch_names)
    n_components = min(n_components, n_channels - 1)
    ica = ICA(
        n_components=n_components, 
        random_state=random_state, 
        method='fastica',
        max_iter='auto'
    )
    ica.fit(raw_for_ica, verbose=False)
    # Список каналов с потенциальным морганием
    candidate_eog = ['Fp1', 'Fp2', 'FP1', 'FP2', 'EEG 001']
    # Находим существующие в датасете
    eog_ch = [ch for ch in candidate_eog if ch in raw_for_ica.ch_names]
    if eog_ch:
        # Используем первый найденный лобный канал как референс для поиска морганий
        eog_indices, eog_scores = ica.find_bads_eog(
            raw_for_ica, 
            ch_name=eog_ch[0], 
            threshold=threshold,
            verbose=False
        )
        ica.exclude = eog_indices
    else:
        ica.exclude = []
    raw_clean = ica.apply(raw.copy(), verbose=False)
    return raw_clean