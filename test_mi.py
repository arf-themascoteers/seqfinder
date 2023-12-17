import numpy as np
from ds_manager import DSManager
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

d = DSManager(reduced_features=True, reduced_rows=True)
mlr = LinearRegression()
trainx, trainy, testx, testy = d.get_train_test_X_y()

mi_scores = mutual_info_regression(trainx, trainy)
k = 5
selected_feature_indices = np.argsort(mi_scores)[::-1][:k]
print("\nSelected feature names:", selected_feature_indices)

X_train_selected = trainx[:, selected_feature_indices]
X_test_selected = testx[:, selected_feature_indices]

linear_model = LinearRegression()
linear_model.fit(X_train_selected, trainy)

y_pred = linear_model.predict(X_test_selected)

mse = r2_score(testy, y_pred)
print("\nr2_score:", mse)
