
def load_model(cfg):
    '''
    Imports the Model specified by passing in configuration
    - cfg: Hydra configuration containing config details of model
    '''
    model_name = cfg.name
    print(f"model_name: {cfg.name}")
    if model_name == "SCNN2D":
        from .cnn_2d import SimpleCNN2D
        return SimpleCNN2D(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "ComplexCNN2D":
        from .cnn_2d import ComplexCNN2D
        return ComplexCNN2D(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "WhisperInspiredClassifier":
        from .cnn_2d import WhisperInspiredClassifier
        return WhisperInspiredClassifier(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "SimpleLSTM":
        from .lstm import SimpleLSTM
        return SimpleLSTM(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "ComplexLSTM":
        from .lstm import ComplexLSTM
        return ComplexLSTM(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "DoubleLSTM":
        from .lstm import DoubleLSTM
        return DoubleLSTM(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "ScaledLSTM":
        from .lstm import ScaledLSTM
        return ScaledLSTM(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "TransformerClassifier":
        from .lstm import TransformerClassifier
        return TransformerClassifier(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "ComplexTransformerClassifier":
        from .lstm import ComplexTransformerClassifier
        return ComplexTransformerClassifier(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "TransformerCNNClassifier":
        from .lstm import TransformerCNNClassifier
        return TransformerCNNClassifier(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "ProjectedLSTM":
        from .lstm import ProjectedLSTM
        return ProjectedLSTM(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "WorldTestLSTM":
        from .lstm import WorldTestLSTM
        return WorldTestLSTM(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)

    elif model_name == "phonology_lstm":
        from .phonology_lstm import DoubleLSTM
        return DoubleLSTM(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs, cfg.phonology_params)
    
    elif model_name == "phonology_lstm_3":
        from .phonology_lstm import DoubleLSTM3
        return DoubleLSTM3(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs, cfg.phonology_params)
    
    elif model_name == "phonology_complex_cnn_2d":
        from .phonology_cnn_2d import ComplexCNN2D
        return ComplexCNN2D(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs, cfg.phonology_params)
    
    elif model_name == "phonology_whisper":
        from .phonology_cnn_2d import WhisperInspiredClassifier
        return WhisperInspiredClassifier(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs, cfg.phonology_params)
    
    elif model_name == "DoubleLSTM2":
        from .double_lstm import DoubleLSTM2
        return DoubleLSTM2(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)

    elif model_name == "DoubleLSTM3":
        from .double_lstm import DoubleLSTM3
        return DoubleLSTM3(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)

    elif model_name == "DoubleLSTM3FIXED":
        from .double_lstm import DoubleLSTM3FIXED
        return DoubleLSTM3FIXED(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)

    elif model_name == "DoubleLSTM4":
        from .double_lstm import DoubleLSTM4
        return DoubleLSTM4(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)

    elif model_name == "DoubleLSTM3VARIABLE":
        from .double_lstm import DoubleLSTM3VARIABLE
        return DoubleLSTM3VARIABLE(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)

    elif model_name == "DoubleLSTM4_FIXED":
        from .double_lstm import DoubleLSTM4_FIXED
        return DoubleLSTM4_FIXED(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)

    elif model_name == "DoubleLSTM3WithAttention":
        from .double_lstm import DoubleLSTM3WithAttention
        return DoubleLSTM3WithAttention(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "DoubleLSTM3_hardcoded":

        from .phonology_lstm import DoubleLSTM3_hardcoded

        return DoubleLSTM3_hardcoded(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    
    elif model_name == "DoubleLSTM_A":
        from .double_lstm import DoubleLSTM_A
        num_layers = 3
        if hasattr(cfg, "num_layers"):
            num_layers = cfg.num_layers
        return DoubleLSTM_A(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs, num_layers)
    
    elif model_name == "kaggle_transformer":
        from .kaggle_transformer import M

        return M(cfg.num_frames, cfg.num_features, cfg.num_coords, cfg.num_signs)
    else:
        raise Exception(f"Invalid Model Name given! Model name: {model_name}")
