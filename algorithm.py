from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import math


class Algorithm(ABC):
    def __init__(self, X_train, y_train, target_feature_size):
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=40)
        self.target_feature_size = target_feature_size
        self.selected_indices = []
        self.model = None

    def fit(self):
        self.model, self.selected_indices = self.get_selected_indices()
        return self.selected_indices

    def transform(self, X):
        if len(self.selected_indices) != 0:
            return X[:,self.selected_indices]
        return self.model.transform(X)

    @abstractmethod
    def get_selected_indices(self):
        pass

    def predict(self, X, y):
        if hasattr(self.model, "predict_it"):
            y_pred = self.model.predict_it(X)
            r2 = round(r2_score(y, y_pred), 2)
            r2 = max(0.0, r2)
            rmse = round(math.sqrt(mean_squared_error(y, y_pred)), 2)
            rmse = max(0.0,rmse)
            return r2, rmse
        return -1,-1

    def get_details(self):
        return self.__class__.__name__
