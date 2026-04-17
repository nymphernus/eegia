from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

'''
Конвертер данных для sktime.
Rонвертирует 3D массив (n_epochs, n_channels, n_times) в формат pd.DataFrame для sktime
'''

class SktimeConverter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        n_epochs, n_channels, n_times = X.shape
        cols = [f"ch{i}" for i in range(n_channels)]
        return pd.DataFrame({
            cols[i]: [pd.Series(X[j, i, :]) for j in range(n_epochs)]
            for i in range(n_channels)
        })