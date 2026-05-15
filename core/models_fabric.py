from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sktime.classification.kernel_based import RocketClassifier

from core.converter import SktimeConverter
from core.model_hybrid import EEGHybridExtractor

def get_models(sfreq, resample_settings, dataset_id):
    if resample_settings[dataset_id] != False:
        sfreq_r = resample_settings[dataset_id]
    else:
        sfreq_r = sfreq
    models = {
        "Hybrid": Pipeline([
            ('extractor', EEGHybridExtractor(sfreq=sfreq_r)),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ]),
        "Rocket": Pipeline([
            ('convertor', SktimeConverter()),
            ('rocket_classifier', RocketClassifier(random_state=42))
        ]),
        "CSP_LDA": Pipeline([
            ('csp', CSP(log=True, norm_trace=False)),
            ('lda', LinearDiscriminantAnalysis())
        ]),
        "CSP_Forest": Pipeline([
            ('csp', CSP(log=True, norm_trace=False)),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])
    }
    return models