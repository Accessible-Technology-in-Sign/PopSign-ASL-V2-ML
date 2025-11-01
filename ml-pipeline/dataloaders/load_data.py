
def load_dataset(cfg, split):
    '''
    Imports the Dataset specified by passing in configuration
    - cfg: Hydra configuration containing config details of dataset
    - split: One of "test", "train", "validation" to look for specific dataset split

    returns:

    - Dataset object: A PyTorch Dataloader based on the specified model
    '''
    dataset_name = cfg.name
    if dataset_name == "cnn":
        from .datasets import CNNDataset
        return CNNDataset(cfg, split)
    
    elif dataset_name == "lstm":
        from .datasets import LSTMDataset
        return LSTMDataset(cfg, split)
    
    elif dataset_name == "whisper":
        from .datasets import WhisperDataset
        return WhisperDataset(cfg, split)
    elif dataset_name == "homosign_cnn":
        from .homosign_datasets import CNNDataset
        return CNNDataset(cfg, split)
    elif dataset_name == "homosign_lstm":
        from .homosign_datasets import LSTMDataset
        return LSTMDataset(cfg, split) 
    elif dataset_name == "homosign_whisper":
        from .homosign_datasets import WhisperDataset
        return WhisperDataset(cfg, split)
    elif dataset_name == "homosign_holistic_lstm":
        from .homosign_holistic import LSTMDataset
        return LSTMDataset(cfg, split)

    elif dataset_name == "homosign_single_augmented_lstm":
        from .homosign_augmented import LSTMDataset
        return LSTMDataset(cfg, split)
    else:
        
        raise Exception(f"Invalid Dataset name given! Dataset name: {dataset_name}")