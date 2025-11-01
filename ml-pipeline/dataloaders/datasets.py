import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F


class BaseSingleHandDataset(Dataset):
        
    def __init__(self, cfg, split):
        self.dataset_dir = cfg.dataset_dir
        self.split = split
        self.num_frames = cfg.num_frames
        self.num_coords = cfg.num_coords
        self.padding = cfg.padding
        self.label_file = cfg.label_file

        self.files = []
        self.labels = []
        
        dataset_split_path = os.path.join(self.dataset_dir, self.split)

        # This is how the labels for the data is created
        for label_folder in os.listdir(dataset_split_path):
            label_path = os.path.join(dataset_split_path, label_folder)
            if os.path.isdir(label_path):
                label = label_folder
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.h5'):
                        self.files.append(os.path.join(label_path, file_name))
                        self.labels.append(label)

        # Read the label file and get the order of the signs
        with open(self.label_file) as f:
            self.label_order = [i.strip() for i in f.readlines()]
        
        # The idx of the label in the one-hot encoding is based on the ordering in the file
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_order)}
        self.labels = [self.label_to_idx[label.lower()] for label in self.labels]


    def __len__(self):
        return len(self.files)
    
    def num_signs(self):
        return len(self.sign_categories)
    
    def load_item(self, idx):
        with h5py.File(self.files[idx], 'r') as h5_file:
            data = h5_file['intermediate'][:][:, 0, :, :] #since we only have 1 hand remove that dimension
        
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # If we are only using two (xy) coordiantes and the data we load in has three (xyz),
        # truncate to xy (assumes data is in xyz)
        if (data_tensor.shape[2] == 3 and self.num_coords == 2):
            data_tensor = data_tensor[:,:,:2]

        # if not self.is_lstm:
        if self.padding == "default":
            # If the number of frames 
            if data_tensor.shape[0] > self.num_frames:
                data_tensor = data_tensor[-self.num_frames:]
            if data_tensor.shape[0] < self.num_frames:
                middle_idx = data_tensor.shape[0] // 2
                pad_num = self.num_frames - data_tensor.shape[0]

                # Pads out the middle frames until it reaches n frames
                data_tensor = torch.cat([
                    data_tensor[:middle_idx, :],
                    data_tensor[middle_idx:middle_idx+1, :].repeat([pad_num, *[1 for i in data_tensor.shape[1:]]]),
                    data_tensor[middle_idx:, :]
                ])
        elif self.padding == "linear-interp":
            data_tensor = data_tensor.permute(1, 2, 0)

            # Resample using linear interpolation
            resampled_tensor = F.interpolate(data_tensor, size=self.num_frames, mode='linear', align_corners=False)

            # Remove the extra dimensions (reverse the unsqueeze)
            data_tensor = resampled_tensor.squeeze(0)
            data_tensor = data_tensor.permute(2, 0, 1)

        # One Hot Encoding
        label_tensor = torch.zeros(len(self.label_order), dtype=torch.float32)
        label_tensor[self.labels[idx]] = 1.0

        # None One Hot Encoding 
        # label_tensor = torch.zeros(1, dtype=torch.int32)
        # label_tensor[0] = self.labels[idx]
        #if self.debug:
        #    return data_tensor, label_tensor, self.files[idx]
        return data_tensor, label_tensor


class CNNDataset(BaseSingleHandDataset):

    def __getitem__(self, idx):
        data_tensor, label_tensor = self.load_item(idx)
        #data_tensor = data_tensor.view(1, data_tensor.size(0), -1)
        data_tensor = torch.reshape(data_tensor, (1, data_tensor.size(0), -1))
        return data_tensor, label_tensor

class LSTMDataset(BaseSingleHandDataset):

    def __getitem__(self, idx):
        data_tensor, label_tensor = self.load_item(idx)
        new_data_tensor = torch.reshape(data_tensor, (data_tensor.size(0), -1))
        #new_data_tensor = data_tensor.view(data_tensor.size(0), -1)        
        return new_data_tensor, label_tensor
    

class WhisperDataset(BaseSingleHandDataset):
    
    def __getitem__(self, idx):
        data_tensor, label_tensor = self.load_item(idx)
        data_tensor = torch.reshape(data_tensor, (data_tensor.size(0), data_tensor.size(1), -1))
        return data_tensor, label_tensor


'''
        if self.is_lstm:
            data_tensor = data_tensor.view(data_tensor.size(0), -1)
        elif self.is_whisper:
            data_tensor = data_tensor.view(data_tensor.size(0), data_tensor.size(1), -1)
        elif self.is_gnn:
            pass
        else:
            data_tensor = data_tensor.view(1, data_tensor.size(0), -1)
'''


if __name__ == '__main__':
    print("Running datasets.py")
    
    
