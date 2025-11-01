import os
import csv
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataloaders.load_data import load_dataset
from models.load_model import load_model

import hydra
import json


class Homonyms():
    def __init__(self, homosign_file):
        
        self.label_file = "/path/to/metadata/563_sign_list.txt" # Update with corresponding path

        # I should create the homosign mapping here
        with open(homosign_file, 'r') as file:
            homosign_list = json.load(file)
        
        # Create a lookup table of sign -> homosign name
        # Create a lookup table of sign -> [homonym_0, ..., homonym_i]
        # Create a lookup of category -> [homonym_0, ..., homonym_i]
        self.homosign_lookup = {}
        self.homosign_group_lookup = {}
        self.merged_group_lookup = {}
        for homosign_group in homosign_list:
            merged_name = "_".join(homosign_group)
            for sign in homosign_group:
                self.homosign_lookup[sign] = merged_name
                self.homosign_group_lookup[sign] = homosign_group
            self.merged_group_lookup[merged_name] = homosign_group
        
        with open(self.label_file) as file:
            sign_labels = [line.strip() for line in file.readlines()]
        

        self.sign_categories = set()
        for sign_label in sign_labels:
            if sign_label in self.homosign_lookup:
                self.sign_categories.add(self.homosign_lookup[sign_label])
            else:
                self.sign_categories.add(sign_label)

        self.sign_categories = list(self.sign_categories)
        self.sign_categories.sort()

        self.label_to_idx = {}
        self.idx_to_label = {}
        for i in range(len(self.sign_categories)):
            label = self.sign_categories[i]
            self.label_to_idx[label.lower()] = i
            self.idx_to_label[i] = label.lower()


def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    """
    Loads a state dict from a checkpoint. Accepts plain state_dict or a dict with key 'state_dict'.
    Strips 'module.' prefixes if the model was saved via DataParallel.
    """
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Strip 'module.' if present
    new_state = {}
    for k, v in state.items():
        new_k = k[7:] if k.startswith("module.") else k
        new_state[new_k] = v

    # strict=False to tolerate harmless shape/name differences (e.g., heads swapped later)
    model.load_state_dict(new_state, strict=False)


@torch.no_grad()
def evaluate(model, dataloader, homonyms_list_file, v1_signs_set, v2_signs_set, device):
    model.eval()
    correct = 0
    total = 0

    # Total number of v1 signs and number of v1 signs correct (Excluding homonyms) 
    v1_correct = 0
    v1_total = 0

    #Total number of v2 signs and number of v2 signs correct (Excluding homonyms)
    v2_correct = 0
    v2_total = 0

    homonyms_lookup = Homonyms(homonyms_list_file)

    print("----------EVAL----------")
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device).float()
        targets = labels.argmax(dim=1)  # class indices

        # Forward
        outputs = model(inputs)
        B, C = outputs.size()[0], outputs.size()[1]
        for b in range(B):
            true_idx = int(targets[b].item())

            logits = outputs[b]
            output_idx = int(torch.argmax(logits).item())

            
            true_label = dataloader.dataset.idx_to_label[true_idx]
            output_label = dataloader.dataset.idx_to_label[output_idx]

            if true_label in homonyms_lookup.homosign_group_lookup:
                homonyms = homonyms_lookup.homosign_group_lookup[true_label]

                if output_label in homonyms:
                    correct += 1
            else:
                # Sign is not a homonym
                if (true_idx == output_idx):
                    correct += 1
                
                # Check if it is a v1 sign without a homonym and if it is correct
                if true_label in v1_signs_set:
                    v1_total += 1
                    if (true_idx == output_idx):
                        v1_correct += 1
                
                # Check if it is a v2 sign without a homonym and if it is correct
                if true_label in v2_signs_set:
                    v2_total += 1
                    if (true_idx == output_idx):
                        v2_correct += 1
                        
            total += 1

    print(f"correct: {correct}, total: {total}, percentage: {correct/total}")
    print(f"v1_correct: {v1_correct}, v1_total: {v1_total}, v1 percentage: {v1_correct/v1_total}")
    print(f"v2_correct: {v2_correct}, v2_total: {v2_total}, v2 percentage: {v2_correct/v2_total}")

    return {
        "correct": correct,
        "total": total,
    }


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    """
    Usage examples (Hydra overrides):
      - Use default checkpoint location:
          python eval_only.py
      - Specify an explicit checkpoint path and split:
          python eval_only.py eval.ckpt=/abs/path/to/checkpoint.pth eval.split=test
      - Override batch size:
          python eval_only.py dataloader.batch_size=64
    """
    print(OmegaConf.to_yaml(cfg))
    print("Running eval_only.py")

    # Resolve device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=========RUNNING ON {device}==========")

    # Build model
    model = load_model(cfg.model).to(device)

    # Resolve checkpoint path
    # Priority: cfg.eval.ckpt (if provided) > default training save path
    eval_cfg = cfg.get("eval", {})
    ckpt_path = (eval_cfg.get("ckpt", None)
                 if hasattr(eval_cfg, "get") else None)
    if ckpt_path is None:
        ckpt_path = os.path.join(
            cfg.experiment.output_dir,
            "pytorch",
            "checkpoints",
            f"{cfg.model.save_name}.pth",
        )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    _load_checkpoint(model, ckpt_path, device)

    # Dataloader for chosen split
    split = (eval_cfg.get("split", "test") if hasattr(eval_cfg, "get") else "test")
    print(f"Evaluating split: {split}")

    dataset = load_dataset(cfg.dataset, split)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,                          # deterministic eval
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Criterion must match training
    criterion = nn.CrossEntropyLoss()

    # Run eval
    #metrics = evaluate(model, loader, criterion, device)

    v1_signs_file = "/path/to/PopSignV2-ML/ml-pipeline/metadata/250_sign_list.txt" # Update accordingly
    v2_signs_file = "/path/to/PopSignV2-ML/ml-pipeline/metadata/313_sign_list.txt" # Update accordingly

    with open(v1_signs_file) as file:
        v1_signs_list = [line.strip() for line in file.readlines()]
    
    with open(v2_signs_file) as file:
        v2_signs_list = [line.strip() for line in file.readlines()]

    v1_signs_set = set(v1_signs_list)
    v2_signs_set = set(v2_signs_list)

    # Remove "cut" because it overlaps between v1 and v2 data collection

    v1_signs_set.remove("cut")
    v2_signs_set.remove("cut")

    homonym_path = "/path/to/PopSignV2-ML/ml-pipeline/metadata/homosigns.json" # Update accordingly
    metrics = evaluate(model, loader, homonym_path, v1_signs_set, v2_signs_set, device)
    


if __name__ == "__main__":
    main()
