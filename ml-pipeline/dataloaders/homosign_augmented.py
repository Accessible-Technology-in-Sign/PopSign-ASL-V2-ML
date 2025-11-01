import os
import h5py
import torch
import json
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F


@torch.no_grad()
def do_random_affine_torch(xyz: torch.Tensor,
                           scale=(0.8, 1.3),
                           shift=(-0.08, 0.08),
                           degree=(-16.0, 16.0),
                           p: float = 0.5):
    if p is not None and p < 1.0 and torch.rand(()) > p:
        return xyz  # no-op

    # scale: single scalar applied to all coords
    if scale is not None:
        s = torch.rand(()) * (scale[1] - scale[0]) + scale[0]
        xyz.mul_(s)

    # shift: single scalar applied to all coords
    if shift is not None:
        sh = torch.rand(()) * (shift[1] - shift[0]) + shift[0]
        xyz.add_(sh)

    # rotate last two coords
    if degree is not None:
        deg = torch.rand(()) * (degree[1] - degree[0]) + degree[0]
        rad = deg * (np.pi / 180.0)
        c, s = torch.cos(rad), torch.sin(rad)
        R_T = torch.tensor([[c,  s],
                            [-s, c]], dtype=xyz.dtype, device=xyz.device)  # transpose for row-vectors
        xy = xyz[..., :2].reshape(-1, 2)
        xy.matmul(R_T)
        xyz[..., :2] = xy.view_as(xyz[..., :2])
    return xyz


def resample_time_linear(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Resample along time to target_len without mixing landmarks.
    x: (T, L, C) -> (target_len, L, C)
    """
    # Treat each landmark as batch, coords as channels, time as length
    x_bcT = x.permute(1, 2, 0)                # (L, C, T)
    y_bcT = F.interpolate(x_bcT, size=target_len, mode='linear', align_corners=False)
    y_TLC = y_bcT.permute(2, 0, 1)            # (target_len, L, C)
    return y_TLC


class HomosignSingleHandDataset(Dataset):
        
    def __init__(self, cfg, split):
        self.dataset_dir = cfg.dataset_dir
        self.split = split
        self.num_frames = cfg.num_frames
        self.num_coords = cfg.num_coords
        self.padding = cfg.padding
        self.label_file = cfg.label_file
        self.homosign_file = cfg.homosign_file
        self.augmentations = cfg.augmentations # FOR NOW I AM SETTING THIS TO A BOOLEAN, IN THE FUTURE THIS WILL HAVE A WHOLE SUITE OF SETTINGS

        # Hacky way of disabling augmentations on non-train splits
        if split != "train":
            self.augmentations = False

        self.files = []
        
        dataset_split_path = os.path.join(self.dataset_dir, self.split)

        # I should create the homosign mapping here
        with open(cfg.homosign_file, 'r') as file:
            homosign_list = json.load(file)
        
        # Create a lookup table of sign -> homosign name
        self.homosign_lookup = {}
        for homosign_group in homosign_list:
            merged_name = "_".join(homosign_group)
            for sign in homosign_group:
                self.homosign_lookup[sign] = merged_name


        
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
        for i in range(len(self.sign_categories)):
            label = self.sign_categories[i]
            self.label_to_idx[label.lower()] = i

        
        # Add files to list

        for label_dir in os.listdir(dataset_split_path):
            label_path = os.path.join(dataset_split_path, label_dir)
            if os.path.isdir(label_path):
                label = label_dir.lower()
                for file_name in os.listdir(label_path):
                    if file_name.endswith('.h5'):
                        file_info = (os.path.join(label_path, file_name), label)
                        self.files.append(file_info)
                        

    def __len__(self):
        return len(self.files)
    
    
    def load_item(self, idx):
        file_path, label = self.files[idx]
        try:
            with h5py.File(file_path, 'r') as h5_file:
                data = h5_file['intermediate'][:][:, 0, :, :] #since we only have 1 hand remove that dimension
                data_tensor = torch.from_numpy(data)
        except Exception as e:
            print(f"Encountered Exception {e} here for file_path: {file_path}")

            return None, None
        

        # If we are only using two (xy) coordiantes and the data we load in has three (xyz),
        # truncate to xy (assumes data is in xyz)
        if (data_tensor.shape[2] == 3 and self.num_coords == 2):
            data_tensor = data_tensor[:,:,:2]


        if self.augmentations:
            data_tensor = do_random_affine_torch(data_tensor)
        


        # Do any padding / interpolation if necessary
        # We do padding instead of variable length to make it easier to collate / batch
        if self.padding == "default":
            # If the number of frames 
            if data_tensor.shape[0] > self.num_frames:
                data_tensor = data_tensor[-self.num_frames:]
            elif data_tensor.shape[0] < self.num_frames:
                middle_idx = data_tensor.shape[0] // 2
                pad_num = self.num_frames - data_tensor.shape[0]

                # Pads out the middle frames until it reaches n frames
                data_tensor = torch.cat([
                    data_tensor[:middle_idx, :],
                    data_tensor[middle_idx:middle_idx+1, :].repeat([pad_num, *[1 for i in data_tensor.shape[1:]]]),
                    data_tensor[middle_idx:, :]
                ])
        elif self.padding == "zeros":
            if (data_tensor.shape[0] > self.num_frames):
                data_tensor = data_tensor[-self.num_frames:]
            elif (data_tensor.shape[0] < self.num_frames):
                pad_shape = (self.num_frames - data_tensor.shape[0], *data_tensor.shape[1:])
                pad = data_tensor.new_zeros(pad_shape)
                data_tensor = torch.cat([data_tensor, pad], dim=0)
        

        elif self.padding == "linear-interp":
            # Drop all non-zero frames
            nonzero_mask = (torch.sum(torch.abs(data_tensor), dim=(1, 2)) != 0)
            filtered_tensor = data_tensor[nonzero_mask]

            # Check that we haven't dropped everything (if we have, then don't replace)
            if filtered_tensor.shape[0] > 0:
                data_tensor = filtered_tensor

            data_tensor = data_tensor.permute(1, 2, 0)

            # Resample using linear interpolation
            resampled_tensor = F.interpolate(data_tensor, size=self.num_frames, mode='linear', align_corners=False)

            # Remove the extra dimensions (reverse the unsqueeze)
            data_tensor = resampled_tensor.squeeze(0)
            data_tensor = data_tensor.permute(2, 0, 1)


        # One Hot Encoding
        if label in self.homosign_lookup:
            label = self.homosign_lookup[label]

        label_tensor = torch.zeros(len(self.sign_categories), dtype=torch.float32)
        label_tensor[self.label_to_idx[label]] = 1.0

        return data_tensor, label_tensor
    
    def load_metadata(self, idx):
        file_path, label = self.files[idx]
        with h5py.File(file_path, 'r') as h5_file:
            dataset = h5_file['intermediate']
        
            data = dataset[:][:, 0, :, :] #since we only have 1 hand remove that dimension
            metadata = {key: dataset.attrs[key] for key in dataset.attrs}
        metadata["file_path"] = file_path
        metadata["label"] = label
        return metadata


class CNNDataset(HomosignSingleHandDataset):

    def __getitem__(self, idx):
        data_tensor, label_tensor = self.load_item(idx)
        #data_tensor = data_tensor.view(1, data_tensor.size(0), -1)
        data_tensor = torch.reshape(data_tensor, (1, data_tensor.size(0), -1))
        return data_tensor, label_tensor

class LSTMDataset(HomosignSingleHandDataset):

    def __getitem__(self, idx):
        data_tensor, label_tensor = self.load_item(idx)
        new_data_tensor = torch.reshape(data_tensor, (data_tensor.size(0), -1))
        #new_data_tensor = data_tensor.view(data_tensor.size(0), -1)        
        return new_data_tensor, label_tensor
    



if __name__ == '__main__':
    print("Running datasets.py")
    
    
