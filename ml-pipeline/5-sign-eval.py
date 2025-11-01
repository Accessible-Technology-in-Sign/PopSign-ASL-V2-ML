import os
import csv
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm
import json

import numpy as np
import random
from dataloaders.load_data import load_dataset
from models.load_model import load_model

import hydra

import pdb


class Homonyms():
    def __init__(self, homosign_file):

        self.label_file = ""

        
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
def evaluate(model, dataloader, num_classes, device):
    # For this evaluation, it just picks 4 other signs to compare against
        # If you are using a model with homonyms merged, then this will be the evaluation to use
        # If you are using a model with homonyms unmerged, you may have your accuracies affected in
        # a pair of homonyms are selected within a subset of 5.

    model.eval()
    correct = 0
    total = 0
    

    print("----------EVAL----------")
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device).float()
        targets = labels.argmax(dim=1)  # class indices

        # Forward
        outputs = model(inputs)

        B, C = outputs.size()[0], outputs.size()[1]
        for b in range(B):
            
            true_idx = int(targets[b].item())
            other_idx = np.arange(num_classes)
            other_idx = np.delete(other_idx, true_idx) # Delete truth idx
            
            other_idx = np.random.choice(other_idx, 4)
            candidates = np.append(other_idx, true_idx)

            candidates_tensor = torch.tensor(candidates, device=outputs.device, dtype=torch.long)

            output_subset = outputs[b, candidates_tensor]

            # What is the prediction on the subset
            subset_predicted_idx = int(torch.argmax(output_subset).item())

            #Look up in the candidates tensor what the actual index was
            predicted_idx = candidates[subset_predicted_idx]

            is_correct = int(predicted_idx == true_idx)

            #pdb.set_trace()

            correct += is_correct
            total += 1

    print(f"correct: {correct}, total: {total}, percentage: {correct/total}")
    return correct/total
            

# I did not want to deal with 
def evaluate_homonyms_unmerged(model, dataloader, num_classes, homonyms_list_file, device):
    # Use this evaluation when using a model that does not have the homonyms merged in the prediction layer
    # It uses the homonyms list to guarantee that a set of five does not have a pair of homonyms in it

    # I want to generate a set of 5
    # If given a list of 5, I want to ensure there are no two signs in there that are homonyms to each other
    # If the homonyms are merged, then this is guaranteed not to happen, but if they were unmerged, I will need 
    # to fetch the label and try to pairwise confirm that they aren't homonyms
    # It would be easier to randomly generate based off of the homonyms merged and then use that to create a set of 5
    # Idea: Let's load the homonym code into here, have the sign_categories and such here

    # 
    
    model.eval()
    correct = 0
    total = 0


    homonyms_lookup = Homonyms(homonyms_list_file)

    print("----------EVAL----------")
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device).float()
        targets = labels.argmax(dim=1)  # class indices

        # Forward
        outputs = model(inputs)

        # For each sample b in B (Batch):

            # Extract 4 other signs to generate the mask
            # If we are given a homosign list and the label has a corresponding homonym, pick a different sign
            # I will need to be able to grab the corresponding labels for it as well. fuck.
        #print(type(outputs))

        B, C = outputs.size()[0], outputs.size()[1]
        for b in range(B):
            

            #print()
            true_idx = int(targets[b].item())

            #pdb.set_trace()

            true_label = dataloader.dataset.idx_to_label[true_idx]

            # If the true label does have homonyms, replace with category name
            if true_label in homonyms_lookup.homosign_lookup:
                true_category = homonyms_lookup.homosign_lookup[true_label]
            else:
                true_category = true_label


            # Select 4 other categories (excluding the true category)
            other_categories = homonyms_lookup.sign_categories.copy()
            other_categories.remove(true_category)
            other_categories = random.sample(other_categories, 4)


            candidate_labels = [true_label]
            for category in other_categories:
                # If the category does have homonyms, pick a random one to be the candidate label
                if category in homonyms_lookup.merged_group_lookup:
                    category_homonyms = homonyms_lookup.merged_group_lookup[category]
                    #pdb.set_trace()
                    label = random.sample(category_homonyms, 1)[0]
                    candidate_labels.append(label)
                else:
                    # Otherwise just append the category
                    candidate_labels.append(category)
                    
            
            candidates = []
            for label in candidate_labels:
                if label not in dataloader.dataset.label_to_idx:
                    pdb.set_trace()
                candidates.append(dataloader.dataset.label_to_idx[label])
            

            candidates_tensor = torch.tensor(candidates, device=outputs.device, dtype=torch.long)

            output_subset = outputs[b, candidates_tensor]

            # What is the prediction on the subset
            subset_predicted_idx = int(torch.argmax(output_subset).item())

            #Look up in the candidates tensor what the actual index was
            predicted_idx = candidates[subset_predicted_idx]

            is_correct = int(predicted_idx == true_idx)

            #pdb.set_trace()

            correct += is_correct
            total += 1

    print(f"correct: {correct}, total: {total}, percentage: {correct/total}")
    return correct/total


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

    num_classes = len(dataset.sign_categories)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,                          # deterministic eval
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


    # Run eval
    metrics = evaluate(model, loader, num_classes, device)

if __name__ == "__main__":
    main()
