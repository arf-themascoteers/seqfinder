from evaluator import Evaluator

def fsdr_confs(seq_sizes, distances, distance_learnabilities, embedding_methods, embedding_sizes, outs):
    for seq_size in seq_sizes:
        for distance in distances:
            for distance_learnability in distance_learnabilities:
                for embedding_method in embedding_methods:
                    for embedding_size in embedding_sizes:
                        for out in outs:
                            yield seq_size, distance, distance_learnability, embedding_method, embedding_size, out

if __name__ == '__main__':
    algorithms = ["fsdr"]
    features = [66]
    samples = [871]
    sizes = [5]

    seq_sizes = [1]
    distances = [0]
    distance_learnabilities = [None]  # "none"(None),"uniform","diverse"
    embedding_methods = ["identity"]  # single_layer, two_layers,
    embedding_sizes = [1]
    outs = ["output"]  # output, concat_input, concat_mid_layer

    tasks = []
    for feature in features:
        for sample in samples:
                for size in sizes:
                    for algorithm in algorithms:
                        if algorithm == "fsdr":
                            for seq_size in seq_sizes:
                                for seq_size, distance, distance_learnability, embedding_method, embedding_size, out in fsdr_confs(seq_sizes, distances, distance_learnabilities, embedding_methods, embedding_sizes, outs)
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