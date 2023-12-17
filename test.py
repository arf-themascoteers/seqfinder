from evaluator import Evaluator

if __name__ == '__main__':
    tasks = []
    for algorithm in ["pcat95"]:
        for size in [10]:
            tasks.append(
                {
                    "reduced_features":False,
                    "reduced_rows":False,
                    "target_feature_size": size,
                    "algorithm": algorithm
                }
            )
    ev = Evaluator(tasks)
    ev.evaluate()