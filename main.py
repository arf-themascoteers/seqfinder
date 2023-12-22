from evaluator import Evaluator
from ds_manager import DSManager

if __name__ == '__main__':
    algorithms = ["fsdr"]
    features = [525]
    samples = [21782]
    sizes = [5]
    tasks = []
    for feature in features:
        for sample in samples:
            for algorithm in algorithms:
                for size in sizes:
                    tasks.append(
                        {
                            "dataset": DSManager(features, samples),
                            "target_feature_size": size,
                            "algorithm": algorithm
                        }
                    )
    ev = Evaluator(tasks)
    ev.evaluate()