import argparse
import torch
import os
import ai_edge_torch
from models.cnn_2d import SimpleCNN2D, ComplexCNN2D
from models.lstm import SimpleLSTM, DoubleLSTM, ComplexLSTM
import config

_model_lookup = {
        "SCNN2D": SimpleCNN2D,
        "CCNN2D": ComplexCNN2D,
        "SLSTM": SimpleLSTM,
        "CLSTM": ComplexLSTM,
        "DLSTM": DoubleLSTM,
    }
def model_lookup(model_name):
    return _model_lookup[model_name]

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch .pt model to TensorFlow Lite format.")
    parser.add_argument("--name", type=str, help="Name of the .pt model file.")
    parser.add_argument("--type", type=str, help=f"Model Type. One of {', '.join(_model_lookup.keys())}")
    parser.add_argument("--features", type=str, help="Number of features")
    parser.add_argument("--frames", type=str, help="Number of input frames")
    parser.add_argument("--signs", type=str, help="Number of output signs")
    parser.add_argument("--lstm", type=bool, help="If the model is lstm")
    args = parser.parse_args()

    os.environ["PJRT_DEVICE"] = "CPU"

    cpu = torch.device("cpu")
    # cuda = torch.device("cuda")

    _input_tensor = torch.randn(1, 1, int(args.frames), int(args.features) * 2) if not args.lstm else torch.randn(1, int(args.frames), int(args.features) * 2)

    input_cpu = (_input_tensor.to(cpu),)
    # input_cuda = (_input_tensor.to(cuda),)

    model = model_lookup(args.type)(*map(int, (args.frames, args.features, 2, args.signs)))

    # Need to load in state dict 
    model.load_state_dict(torch.load(f"{config.output_location}/pytorch/models/{args.name}.pt"))

    #model = torch.jit.load(f"{config.output_location}/pytorch/models/{args.name}.pt", map_location="cpu")
    #model.eval()

    ai_edge_torch.convert(model.to(cpu).eval(), input_cpu).export(f"{config.output_location}/tensorflow/{args.signs}-{args.name}-cpu.tflite")
    # ai_edge_torch.convert(model.to(cuda).eval(), input_cuda).export(f"/modesl/{args.name}-cuda.tflite")

    
    

if __name__ == "__main__":
    main()