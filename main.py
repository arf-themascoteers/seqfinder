from evaluator import Evaluator

if __name__ == '__main__':
    #n filters
    #size filter
    #distance
    #larnable?
    #embedding_size
    #embedding_method - asis, embedding (size),

    algorithms = ["fsdr"]
    features = [66]
    samples = [871]
    sizes = [5]

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