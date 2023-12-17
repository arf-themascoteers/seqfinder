from sklearn.neural_network import MLPRegressor
from ds_manager import DSManager
from algorithms.algorithm_pca import AlgorithmPCA
import my_utils
import numpy as np

ds = DSManager(reduced_features=False,reduced_rows=False)
train_X, train_y, test_X, test_y = ds.get_train_test_X_y()
alg = AlgorithmPCA(train_X, train_y, 10)
pca, features = alg.get_selected_indices()
explained_variance_ratios = pca.explained_variance_ratio_
print(np.cumsum(explained_variance_ratios))
first_principal_component = pca.components_[0, :]
print(first_principal_component)