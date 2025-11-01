import argparse
import torch
import os
import ai_edge_torch
from models.load_model import load_model

from omegaconf import DictConfig, OmegaConf

import hydra

config = None

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    print(f"Loading model:\n {cfg.model}")

    os.environ["PJRT_DEVICE"] = "CPU"

    cpu = torch.device("cpu")
    # cuda = torch.device("cuda")

    #_input_tensor = torch.randn(1, 1, int(args.frames), int(args.features) * 2) if not args.lstm else torch.randn(1, int(args.frames), int(args.features) * 2)

    _input_tensor = torch.randn(1, cfg.model.num_frames, cfg.model.num_features * cfg.model.num_coords)

    input_cpu = (_input_tensor.to(cpu),)
    # input_cuda = (_input_tensor.to(cuda),)

    model = load_model(cfg.model)

    # Need to load in state dict 
    model.load_state_dict(torch.load(f"{cfg.experiment.output_dir}/pytorch/checkpoints/{cfg.model.save_name}.pth"))

    #model = torch.jit.load(f"{cfg.experiment.output_dir}/pytorch/models/{cfg.model.save_name}.pt", map_location="cpu")
    model.eval()

    tf_model = ai_edge_torch.convert(model.to(cpu).eval(), input_cpu)
    
    tf_model.export(f"{cfg.experiment.output_dir}/tensorflow/{cfg.model.save_name}-cpu.tflite")
    # ai_edge_torch.convert(model.to(cuda).eval(), input_cuda).export(f"/modesl/{args.name}-cuda.tflite")

    return input_cpu, model, tf_model
    


if __name__ == "__main__":
    input_cpu, model, tf_model = main()

