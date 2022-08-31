"""
Class wrapping different regression techniques
"""
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class Regressor:
    @staticmethod
    def distance(y_pred: np.ndarray, y_true: np.ndarray, method: str = 'l1', pairwise: bool = False):
        if method == 'l1':
            dist = np.abs(y_pred - y_true)
        if method == 'l2':
            dist = np.sqrt(np.square(y_pred - y_true))
        if pairwise:
            return dist
        else:
            return np.mean(dist)

    def __init__(self, type: str, disable_scaler: bool = False, **kwargs):
        if type == 'svr':
            if disable_scaler:
                self.model = SVR(**kwargs)
            else:
                self.model = make_pipeline(StandardScaler(),
                                           SVR(**kwargs))
        if type == 'sgd':
            if disable_scaler:
                self.model = SGDRegressor(**kwargs)
            else:
                self.model = make_pipeline(StandardScaler(),
                                           SGDRegressor(**kwargs))

    def fit(self, data, target):
        self.model.fit(data, target)

    def predict(self, data):
        y_pred = self.model.predict(data)
        return y_pred


