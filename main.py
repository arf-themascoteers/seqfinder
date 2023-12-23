from evaluator import Evaluator
from ds_manager import DSManager

if __name__ == '__main__':
    algorithms = ["fsdr"]
    features = [525]
    samples = [21782]
    #samples = [871]
    sizes = [20]
    tasks = []
    for feature in features:
        for sample in samples:
            for algorithm in algorithms:
                for size in sizes:
                    tasks.append(
                        {
                            "feature": feature,
                            "sample": sample,
                            "target_feature_size": size,
                            "algorithm": algorithm
                        }
                    )
    ev = Evaluator(tasks)
    ev.evaluate()