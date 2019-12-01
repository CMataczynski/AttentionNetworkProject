import os
import torch
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class USGDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        with open(dataset, "rb") as file:
            self.whole_set = pkl.load(file)
        self.transforms = transforms
        self.length = len(self.whole_set['id'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.whole_set['lower'][idx]
        if self.transforms:
            image = self.transforms(image)
        label = torch.tensor(self.whole_set['target'][idx])
        identifier = torch.tensor(int(self.whole_set['id'][idx]))

        return {
            "image": image,
            "label": label,
            "identifier": identifier
        }


