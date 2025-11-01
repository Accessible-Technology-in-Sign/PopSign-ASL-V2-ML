import os
import sys
import csv
import config
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataloaders.load_data import load_dataset
from models.load_model import load_model

# import ai_edge_torch
# os.environ['PJRT_DEVICE'] = 'CPU'

from omegaconf import DictConfig, OmegaConf

import hydra

from tqdm import tqdm


# Assuming these are imported or defined elsewhere:
# from model import SimpleCNN2D
# from data import SingleHandH5Dataset_MobileLastN

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    print("Running train.py")
    print(f"Training {cfg.model.name}")

    if (hasattr(cfg.experiment, "seed")):
        set_seed(int(cfg.experiment.seed))

    train_loader = torch.utils.data.DataLoader(
        load_dataset(cfg.dataset, "train"),
        batch_size=cfg.dataloader.batch_size, 
        shuffle=cfg.dataloader.shuffle, 
        pin_memory=True,                       # GPU: enables async H2D copies
        persistent_workers=True,               # don’t respawn workers every epoch
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
        num_workers=cfg.dataloader.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        load_dataset(cfg.dataset, "validation"),
        batch_size=cfg.dataloader.batch_size, 
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,                       # GPU: enables async H2D copies
        persistent_workers=True,               # don’t respawn workers every epoch
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        load_dataset(cfg.dataset, "test"),
        batch_size=cfg.dataloader.batch_size, 
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=True,                       # GPU: enables async H2D copies
        persistent_workers=True,               # don’t respawn workers every epoch
        prefetch_factor=4,
        worker_init_fn=worker_init_fn,
    )


    os.makedirs(os.path.join(cfg.experiment.output_dir, "train_metrics"), exist_ok=True)
    with open(cfg.experiment.train_metrics_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_accuracy", "total_correct", "total"])

    # Step 4: Initialize the model
    model = load_model(cfg.model)
    print(model)

    # Step 5: Set up the loss function and optimizer
        # TODO: Change this such that we can configure learning rate and alpha 
    loss = nn.CrossEntropyLoss()
    criterion = loss
    optimizer = optim.RMSprop(model.parameters(), lr=cfg.training.learning_rate, alpha=cfg.training.alpha)
    #optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    #optimizer = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    print(optimizer)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
    

    #Step 6: Define training step and eval
    def train_one_epoch(model, dataloader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        print("----------TRAIN----------")
        for inputs, labels in tqdm(dataloader):
            # Move inputs and labels to the device
            inputs, labels = inputs.to(device), labels.to(device).float()  # Ensure labels are float for BCEWithLogitsLoss
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels.argmax(dim=1))
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate the running loss
            running_loss += loss.item() * inputs.size(0)
            
        return running_loss / len(dataloader.dataset)


    def evaluate(model, dataloader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            print("----------EVAL----------")
            for inputs, labels in tqdm(dataloader):
                inputs, labels = inputs.to(device), labels.to(device).float()  # Ensure labels are float
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute loss
                loss = criterion(outputs, labels.argmax(dim=1))
                running_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy: convert outputs to probabilities and round them
                predicted = (torch.softmax(outputs, dim=1))  # Threshold at 0.5 for binary decision per class
                correct += (predicted.argmax(dim=1) == labels.argmax(dim=1)).sum().item()  # Count correct multi-class matches
                total += labels.size(0)
                                
        accuracy = correct / total
        #new_accuracy = new_correct / total

        return running_loss / len(dataloader.dataset), accuracy, correct, total

    # Step 7: Train and evaluate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=========RUNNING ON {device}==========")
    model.to(device)
    best_val_loss = float('inf')
    for epoch in range(cfg.training.num_epochs):
        print(f"---------Running Epoch {epoch}----------")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        #val_loss, val_accuracy, val_new_accuracy, correct, new_correct, total = evaluate(model, val_loader, criterion, device)
        val_loss, val_accuracy, correct, total = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f} | correct, total | {correct, total}")

        scheduler.step(val_loss)
        
        # Write results to CSV file
        with open(cfg.experiment.train_metrics_file, mode='a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([epoch, train_loss, val_loss, val_accuracy, correct, total])

        # Save the model if validation loss has improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{cfg.experiment.output_dir}/pytorch/checkpoints/{cfg.model.save_name}.pth")
            print(f"saved at: {cfg.experiment.output_dir}/pytorch/checkpoints/{cfg.model.save_name}.pth")

    # Step 8: Evaluate on the test set
    model.load_state_dict(torch.load(f"{cfg.experiment.output_dir}/pytorch/checkpoints/{cfg.model.save_name}.pth"))
    model.eval()
    test_loss, test_accuracy, correct, total = evaluate(model, test_loader, criterion, device)
    scripted_model = torch.jit.script(model)
    scripted_model.save(f"{cfg.experiment.output_dir}/pytorch/models/{cfg.model.save_name}.pt")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f} | correct, total | {correct, total}")

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)

# Step 1: Load the data
if __name__ == '__main__':
    main()