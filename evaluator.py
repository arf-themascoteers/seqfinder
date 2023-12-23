from sklearn.linear_model import LinearRegression
from model_ann import ModelANN
from ds_manager import DSManager
from datetime import datetime
import os
from algorithm_creator import AlgorithmCreator
from sklearn.metrics import r2_score, mean_squared_error
import math
import pandas as pd
import my_utils


class Evaluator:
    def __init__(self, tasks):
        self.tasks = tasks
        self.filename = os.path.join("results","results.csv")
        if not os.path.exists(self.filename):
            with open(self.filename, 'w') as file:
                file.write("algorithm,dataset,rows,columns,time,target_size,final_size,"
                           "r2_original,r2_train,r2_test,"
                           "rmse_original,rmse_train,rmse_test,"
                           "selected_features\n")

    def evaluate(self):
        for task in self.tasks:
            print(task)
            feature = task["feature"]
            sample = task["sample"]
            target_feature_size = task["target_feature_size"]
            algorithm_name = task["algorithm"]
            dataset = DSManager(feature, sample)

            elapsed_time, r2_original, rmse_original, \
                r2_reduced_train, rmse_reduced_train, \
                r2_reduced_test, rmse_reduced_test, \
                final_indices, selected_features = \
                self.do_algorithm(algorithm_name, dataset, target_feature_size)


            with open(self.filename, 'a') as file:
                file.write(
                    f"{algorithm_name},{dataset},{dataset.count_rows()},"
                    f"{dataset.count_features()},{round(elapsed_time,2)},{target_feature_size},{final_indices},"
                    f"{r2_original},{r2_reduced_train},{r2_reduced_test},"
                    f"{rmse_original},{rmse_reduced_train},{rmse_reduced_test},"
                    f"{';'.join(str(i) for i in selected_features)}\n")

    def is_done(self,algorithm_name,dataset,target_feature_size):
        df = pd.read_csv(self.filename)
        if len(df) == 0:
            return False
        rows = df.loc[
            (df['algorithm'] == algorithm_name) &
            (df['rows'] == dataset.count_rows()) &
            (df['columns'] == dataset.count_features()) &
            (df['target_size'] == target_feature_size)
        ]
        return len(rows) != 0

    def do_algorithm(self, algorithm_name, dataset, target_feature_size):
        X_train, y_train, X_test, y_test = dataset.get_train_test_X_y()
        print(f"{algorithm_name}:X_train,X_test: {X_train.shape} {X_test.shape}")
        r2_original, rmse_original = Evaluator.get_metrics(X_train, y_train, X_test, y_test)
        algorithm = AlgorithmCreator.create(algorithm_name, X_train, y_train, target_feature_size)
        start_time = datetime.now()
        selected_features = algorithm.fit()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        X_train_reduced = algorithm.transform(X_train)
        X_test_reduced = algorithm.transform(X_test)
        r2_reduced, rmse_reduced = Evaluator.get_metrics(X_train_reduced, y_train, X_test_reduced, y_test)
        r2_embedded, rmse_embedded = algorithm.predict_it(X_test, y_test)
        results = {"r2_original": r2_original, "rmse_original": rmse_original,
                   "r2_reduced": r2_reduced, "rmse_reduced": rmse_reduced,
                   "r2_embedded": r2_embedded, "rmse_embedded": rmse_embedded,
                   "elapsed_time": elapsed_time,
                   "final_size": X_test_reduced.shape[1],
                   "selected_features": selected_features}
        return results

    @staticmethod
    def get_metrics(X_train, y_train, X_test, y_test):
        metric_evaluator = LinearRegression()
        metric_evaluator.fit(X_train, y_train)
        y_pred = metric_evaluator.predict(X_test)
        r2_test = round(r2_score(y_test, y_pred), 2)
        rmse_test = round(math.sqrt(mean_squared_error(y_test, y_pred)), 2)
        return r2_test, rmse_test
