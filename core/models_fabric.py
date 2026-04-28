from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sktime.classification.kernel_based import RocketClassifier

from core.converter import SktimeConverter
from core.model_hybrid import EEGHybridExtractor

def get_models(sfreq):
    models = {
        "Hybrid": Pipeline([
            ('extractor', EEGHybridExtractor(sfreq=sfreq)),
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