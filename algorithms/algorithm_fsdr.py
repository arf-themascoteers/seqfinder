from algorithm import Algorithm
from algorithms.fsdr.fsdr import FSDR


class AlgorithmFSDR(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        fsdr = FSDR(self.X_train.shape[0], self.X_train.shape[1], self.target_feature_size)
        fsdr.fit(self.X_train, self.y_train, self.X_validation, self.y_validation)
        return fsdr, fsdr.get_indices()