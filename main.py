from evaluator import Evaluator

if __name__ == '__main__':
    algorithms = ["mi","sfs","lasso","fsdr"]
    algorithms = ["fsdr"]
    datasets = ["original",
                "downscaled_525",
                "downscaled_66",
                "truncated_4200",
                "truncated_525"
                ]
    datasets = ["truncated_525"]
    sizes = [2, 5, 10, 15, 20]
    sizes = [5]
    tasks = []
    for dataset in datasets:
        for algorithm in algorithms:
            for size in sizes:
                tasks.append(
                    {
                        "dataset": dataset,
                        "target_feature_size": size,
                        "algorithm": algorithm
                    }
                )
    ev = Evaluator(tasks)
    ev.evaluate()