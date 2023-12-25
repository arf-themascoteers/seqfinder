from algorithm import Algorithm
from algorithms.fsdr.fsdr import FSDR


class AlgorithmFSDR(Algorithm):
    def __init__(self, X_train, y_train, target_feature_size,
                 seq_size,
                 distance, learnable_distance, learnable_distances,
                 embedding_method, embedding_size):
        super().__init__(X_train, y_train, target_feature_size)

    def get_selected_indices(self):
        fsdr = FSDR(self.X_train.shape[0], self.X_train.shape[1], self.target_feature_size, seq=True, mode="linear_multi")
        fsdr.fit(self.X_train, self.y_train, self.X_validation, self.y_validation)
        return fsdr, fsdr.get_indices()

    def get_details(self):
        # size filter
        # distance
        # larnable_distance?
        # larnable_distances?
        # embedding_size
        # embedding_method - asis, embedding (size),