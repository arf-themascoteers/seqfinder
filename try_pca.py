from sklearn.decomposition import PCA
import numpy as np
from ds_manager import DSManager

ds = DSManager(reduced_features=False,reduced_rows=False)
train_X, train_y, test_X, test_y = ds.get_train_test_X_y()

pca = PCA()
X_pca = pca.fit_transform(train_X)
explained_variance_ratios = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratios)
num_components_95_percent = np.argmax(cumulative_explained_variance >= 0.95) + 1
print("Number of components explaining at least 95% variance:", num_components_95_percent)

print(cumulative_explained_variance)
