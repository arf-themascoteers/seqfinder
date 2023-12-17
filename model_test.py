from sklearn.neural_network import MLPRegressor
from ds_manager import DSManager
from algorithms.algorithm_pca import AlgorithmPCA
import my_utils


for reduced_size in [10, 100, 200, 300, 1000, 2000]:
    ds = DSManager(reduced_features=False,reduced_rows=False)
    train_X, train_y, test_X, test_y = ds.get_train_test_X_y()
    alg = AlgorithmPCA(train_X, train_y, reduced_size)
    model, features = alg.get_selected_indices()
    metrics_evaluator = my_utils.get_metric_evaluator_for_traditional(reduced_size)

    train_X = model.transform(train_X)
    test_X = model.transform(test_X)

    metrics_evaluator.fit(train_X, train_y)
    print(metrics_evaluator.score(test_X, test_y))
