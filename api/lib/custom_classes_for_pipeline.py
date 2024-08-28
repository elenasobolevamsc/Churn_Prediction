import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
class LogData(BaseEstimator, TransformerMixin):

    def __init__(self, col_to_log):
        self.col_to_log = col_to_log

    def transform(self, X):
        for col in X.columns:
            if col in self.col_to_log:
                X.loc[:, col] = np.log(1 + X.loc[:, col])
        return X

    def fit(self, X, y=None):
        return self