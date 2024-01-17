from evaluator import Evaluator

def confs(algorithms, features, samples, 
          filters, seq_sizes,  distances,
          distance_learnabilities, embedding_methods, embedding_sizes,
          outs):
    for algorithm in algorithms:
        for feature in features:
            for sample in samples:
                for filter in filters:
                    for seq_size in seq_sizes:
                        for distance in distances:
                            for distance_learnability in distance_learnabilities:
                                for embedding_method in embedding_methods:
                                    for embedding_size in embedding_sizes:
                                        for out in outs:
                                            yield\
                                                algorithm,\
                                                feature,\
                                                sample,\
                                                filter,\
                                                seq_size,\
                                                distance,\
                                                distance_learnability,\
                                                embedding_method,\
                                                embedding_size,\
                                                out

if __name__ == '__main__':
    algorithms = ["fsdr"]
    features = [66]
    samples = [871]
    filters = [5]
    seq_sizes = [1]
    distances = [0]
    distance_learnabilities = [None]  # "none"(None),"uniform","diverse"
    embedding_methods = ["identity"]  # single_layer, two_layers,
    embedding_sizes = [1]
    outs = ["output"]  # output, concat_input, concat_mid_layer

    tasks = []
    for algorithm, feature, sample, filter, seq_size, \
            distance, distance_learnability, embedding_method, embedding_size, out \
            in \
            confs(algorithms, features, samples, filters, seq_sizes,  
                  distances, distance_learnabilities, embedding_methods, embedding_sizes, outs):
        tasks.append(
            {
                "algorithm": algorithm,
                "feature": feature,
                "sample": sample,
                "filter": filter,
                "seq_size": seq_size,
                "distance": distance,
                "distance_learnability": distance_learnability,
                "embedding_method": embedding_method,
                "embedding_size": embedding_size,
                "out": out
            }
        )
    ev = Evaluator(tasks)
    ev.evaluate()