import os
import csv
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

from dataloaders.load_data import load_dataset
from models.load_model import load_model

import numpy as np


import hydra


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
def evaluate(model, dataloader, criterion, device, num_classes, topk_pairs=25):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    all_preds = []
    all_labels = []

    print("----------EVAL----------")
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device).float()
        targets = labels.argmax(dim=1)  # class indices

        # Forward
        outputs = model(inputs)

        # Loss
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)

        # Accuracies
        probs = torch.softmax(outputs, dim=1)
        pred_top1 = probs.argmax(dim=1)

        k = min(5, probs.size(1))
        _, pred_topk = probs.topk(k, dim=1)

        correct_top1 += (pred_top1 == targets).sum().item()
        correct_top5 += (pred_topk == targets.unsqueeze(1)).any(dim=1).sum().item()

        total += targets.size(0)

        # For F1
        all_preds.extend(pred_top1.cpu().numpy().tolist())
        all_labels.extend(targets.cpu().numpy().tolist())

    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total

    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    #Confusion Matrix code:
    labels_range = list(range(num_classes))
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)
    cm = cm.astype(np.int64)

    # Build “most confused” list: global top off-diagonal counts
    # Each item is of the form (count, true_idx, pred_idx, rate_of_true_class)
    most_confused = []
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    rates = cm / row_sums  # per-class normalization
    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue
            c = int(cm[i, j])
            if c > 0:
                most_confused.append((c, i, j, float(rates[i, j])))

    # Sort by raw count desc, break ties by rate desc
    most_confused.sort(key=lambda x: (x[0], x[3]), reverse=True)
    most_confused_top = most_confused[:topk_pairs]


    return {
        "loss": running_loss / len(dataloader.dataset),
        "acc_top1": acc_top1,
        "acc_top5": acc_top5,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
        "correct_top1": correct_top1,
        "total": total,
        "confusion_matrix": cm,
        "most_confused_pairs": most_confused_top
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
    metrics = evaluate(model, loader, criterion, device, len(dataset.sign_categories))

    # Pretty print
    print(
        f"Eval [{split}] -> "
        f"Loss: {metrics['loss']:.4f} | "
        f"Top-1: {metrics['acc_top1']:.4f} | "
        f"Top-5: {metrics['acc_top5']:.4f} | "
        f"F1(macro): {metrics['f1_macro']:.4f} | "
        f"F1(micro): {metrics['f1_micro']:.4f} | "
        f"F1(weighted): {metrics['f1_weighted']:.4f} | "
        f"Correct/Total: {metrics['correct_top1']}/{metrics['total']}"
    )


    cm = metrics["confusion_matrix"]
    idx_to_name = dataset.idx_to_label

    # Print top globally confused pairs
    print("\nTop Confused Pairs (by raw count; ties broken by per-class rate):")
    print(" count | rate  | true -> pred | true_label -> pred_label")
    for c, i, j, r in metrics["most_confused_pairs"]:
        true_name = idx_to_name[i]
        pred_name = idx_to_name[j]
        print(f"{c:6d} | {r:0.3f} | {i:4d} -> {j:4d} | {true_name} -> {pred_name}")

    # Optional: write CSV
    out_dir = os.path.join(cfg.experiment.output_dir, "eval_metrics")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{cfg.model.save_name}_{split}.csv")
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                ["split", "loss", "acc_top1", "acc_top5",
                 "f1_macro", "f1_micro", "f1_weighted", "correct_top1", "total",
                 "checkpoint"]
            )
        w.writerow([
            split, metrics["loss"], metrics["acc_top1"], metrics["acc_top5"],
            metrics["f1_macro"], metrics["f1_micro"], metrics["f1_weighted"],
            metrics["correct_top1"], metrics["total"], ckpt_path
        ])
    print(f"Saved metrics to: {out_csv}")


if __name__ == "__main__":
    main()
