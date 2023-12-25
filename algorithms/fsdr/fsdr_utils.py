# Future - Attention

def get_default_configuration():
    return {
        "seq_size": 1,
        "distance": 0,
        "distance_learnability": None,#"none"(None),"uniform","diverse"
        "embedding_method": "identity", # single_layer, two_layers,
        "embedding_size": 1,
        "out": "output" #output, concat_input, concat_mid_layer
    }

