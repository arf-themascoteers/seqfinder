import numpy as np
from ds_manager import DSManager
from algorithms.algorithm_pcat95 import AlgorithmPCAT95
from sklearn.ensemble import RandomForestRegressor

ds = DSManager(reduced_features=True,reduced_rows=True)
train_X, train_y, test_X, test_y = ds.get_train_test_X_y()

rf = RandomForestRegressor()
rf.fit(train_X, train_y)
print(rf.score(test_X, test_y))

pcat = AlgorithmPCAT95(train_X, train_y, target_feature_size=2)
model, selectors = pcat.get_selected_indices()

train_X_reduced = model.transform(train_X)
test_X_reduced = model.transform(test_X)

rf = RandomForestRegressor()
rf.fit(train_X_reduced, train_y)
print(rf.score(train_X_reduced, train_y))
print(rf.score(test_X_reduced, test_y))
