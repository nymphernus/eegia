from .converter import SktimeConverter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import Rocket
from mne.time_frequency import psd_array_welch
import numpy as np

'''
Гибридный экстрактор
Объединяет спектральную мощность (частотный домен) и признаки ROCKET (временной домен)
'''

class EEGHybridExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, num_kernels=200, sfreq=None):
        self.num_kernels = num_kernels
        self.sfreq = sfreq
        self.rocket = Rocket(num_kernels=self.num_kernels, random_state=42)
        self.scaler = StandardScaler()
        self.converter = SktimeConverter()

    def _get_band_power(self, X):
        # Расчет мощности классических ритмов ЭЭГ через метод Велча
        nyquist = self.sfreq / 2
        f_limit = min(40, nyquist - 1)
        
        n_fft = min(256, X.shape[-1])
        psds, freqs = psd_array_welch(
            X, sfreq=self.sfreq, fmin=1, fmax=f_limit, 
            n_fft=n_fft, n_overlap=n_fft//2, verbose=False
        )
        
        bands = {"delta": (1, 4), "theta": (4, 7), "alpha": (7, 14), 
                 "beta": (14, 30), "gamma": (30, f_limit)}
        
        out = []
        for band, (fmin, fmax) in bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            power = psds[:, :, idx].mean(axis=-1) 
            power = 10 * np.log10(psds[:, :, idx].mean(axis=-1) + 1e-20)
            out.append(power)
        
        return np.concatenate(out, axis=1)
    def _combine_features(self, X):
        X_df = self.converter.transform(X)
        X_rocket = self.rocket.transform(X_df)
        X_bp = self._get_band_power(X)
        return np.concatenate([X_bp, np.asarray(X_rocket)], axis=1)
        
    def fit(self, X, y=None):
        X_df = self.converter.transform(X)
        self.rocket.fit(X_df)
        X_combined = self._combine_features(X)
        self.scaler.fit(X_combined)
        return self
        
    def transform(self, X):
        X_combined = self._combine_features(X)
        return self.scaler.transform(X_combined)