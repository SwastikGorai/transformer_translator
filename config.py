from pathlib import Path
def get_config():
    return {
        "batch_size" : 8,
        "num_epochs": 20,
        "lr" : 0.0001,
        "sequence_length" : 2200,
        "d_model" : 512,
        "source_language" : "en",
        "target_language": "hi",
        "model_folder" : "weights",
        "model_basename": "tmodel_",
        "preload" : None,
        "tokenizer_file" : "tokenizer_{0}.json",
        "experiment_name" : "runs/tmodel",
    }
    
def get_weights_file_path(config, epoch):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)
    