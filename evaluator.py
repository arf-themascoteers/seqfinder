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
                file.write("algorithm,rows,columns,time,target_size,final_size,"
                           "r2_original,r2_train,r2_test,"
                           "rmse_original,rmse_train,rmse_test,"
                           "selected_features\n")

    def evaluate(self):
        for task in self.tasks:
            print(task)
            reduced_features = task["reduced_features"]
            reduced_rows = task["reduced_rows"]
            target_feature_size = task["target_feature_size"]
            algorithm_name = task["algorithm"]
            dataset = DSManager(reduced_features=reduced_features, reduced_rows=reduced_rows)
            # if self.is_done(algorithm_name, dataset, target_feature_size):
            #     print("Done already. Skipping.")
            #     continue
            elapsed_time, r2_original, rmse_original, \
                r2_reduced_train, rmse_reduced_train, \
                r2_reduced_test, rmse_reduced_test, \
                final_indices, selected_features = \
                self.do_algorithm(algorithm_name, dataset, target_feature_size)


            with open(self.filename, 'a') as file:
                file.write(
                    f"{algorithm_name},{dataset.count_rows()},"
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
        print(f"X_train,X_test: {X_train.shape} {X_test.shape}")
        _, _, r2_original, rmse_original = Evaluator.get_metrics(algorithm_name, X_train, y_train, X_test, y_test)
        algorithm = AlgorithmCreator.create(algorithm_name, X_train, y_train, target_feature_size)
        start_time = datetime.now()
        selected_features = algorithm.fit()
        elapsed_time = (datetime.now() - start_time).total_seconds()
        X_train_reduced = algorithm.transform(X_train)
        X_test_reduced = algorithm.transform(X_test)
        r2_reduced_train, rmse_reduced_train, r2_reduced_test, rmse_reduced_test = \
            Evaluator.get_metrics(algorithm_name, X_train_reduced, y_train, X_test_reduced, y_test)
        return elapsed_time, r2_original, rmse_original, \
            r2_reduced_train, rmse_reduced_train, \
            r2_reduced_test, rmse_reduced_test, X_test_reduced.shape[1], selected_features

    @staticmethod
    def get_metrics(algorithm_name, X_train, y_train, X_test, y_test):
        metric_evaluator = my_utils.get_metric_evaluator_for(algorithm_name, X_train)
        metric_evaluator.fit(X_train, y_train)

        y_pred = metric_evaluator.predict(X_train)
        r2_train = round(r2_score(y_train, y_pred), 2)
        rmse_train = round(math.sqrt(mean_squared_error(y_train, y_pred)), 2)

        y_pred = metric_evaluator.predict(X_test)
        r2_test = round(r2_score(y_test, y_pred), 2)
        rmse_test = round(math.sqrt(mean_squared_error(y_test, y_pred)), 2)

        print(f"r2 train {r2_train}")
        print(f"r2 test {r2_test}")

        return r2_train, rmse_train, r2_test, rmse_test



    # def do_pca(self,X_train, y_train, target_feature_size):
    #     pca = PCA(n_components=target_feature_size)
    #     pca.fit(X_train)
    #     return pca,[]
    #
    # def do_pls(self,X_train, y_train, target_feature_size):
    #     pls = PLSRegression(n_components=target_feature_size)
    #     pls.fit(X_train, y_train)
    #     return pls,[]
    #
    # def do_rfe(self,X_train, y_train, target_feature_size):
    #     rfe = RFE(LinearRegression(), n_features_to_select=target_feature_size)
    #     rfe.fit(X_train, y_train)
    #     indices = np.where(rfe.get_support())[0]
    #     return rfe, indices
    #
    # def do_kbest(self,X_train, y_train, target_feature_size):
    #     kbest = SelectKBest(score_func=f_regression, k=target_feature_size)
    #     kbest.fit(X_train, y_train)
    #     indices = np.where(kbest.get_support())[0]
    #     return kbest, indices
    #
    # def do_frommodel(self,X_train, y_train, target_feature_size):
    #     selector = SelectFromModel(Evaluator.get_internal_model(), threshold='median', max_features=target_feature_size)
    #     selector.fit(X_train, y_train)
    #     indices = np.where(selector.get_support())[0]
    #     return selector, indices


