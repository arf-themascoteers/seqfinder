# Future - Attention

def get_default_configuration():
    return {
        "seq_size": 1,
        "distance": 0,
        "distance_learnability": None,#"none"(None),"uniform","diverse"
        "embedding_method": "identity", # single_layer, two_layers,
        "embedding_size": 1,
        "skip": None, #"none" (None), band_evaluations, mid_layer

    }

