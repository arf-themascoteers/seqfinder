from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    x = [(False, False), (True, False), (False, True)]
    for reduced_features, reduced_rows in x:
        for algorithm in ["mi","sfs","lasso","fsdr"]:
            for size in [2, 5, 10, 15, 20]:
                tasks.append(
                    {
                        "reduced_features":reduced_features,
                        "reduced_rows":reduced_rows,
                        "target_feature_size": size,
                        "algorithm": algorithm
                    }
                )
    ev = Evaluator(tasks)
    ev.evaluate()