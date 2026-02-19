'''
Load prostate MRI data from megsegbench and translate into formats suitable for training a U-net model.
'''

import numpy as np

import torch
from torch.utils.data import Dataset

from medsegbench import Promise12MSBench

def load_images(size=128):
    ''' Specify the size to which the images should be resized. Output is the image datasets in the form of lists of PIL images '''
    train_images = Promise12MSBench(split="train", download=True, size=size)
    val_images = Promise12MSBench(split="val", download=True, size=size)

    return train_images, val_images

class MRIDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitemsa__(self, idx):
        ''' For converting the PIL image lists into tensors and normalizing them for use in training the model. Also converts the masks to binary format. '''
        image, mask = self.base_dataset[idx]

        # Convert PIL image to tensor and normalize, add channel dimension
        image.convert("L")  # Ensure grayscale
        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0

        # Convert mask to binary: any non-zero value becomes 1 (as float for BCE loss)
        mask = torch.tensor((mask > 0).astype(np.float32), dtype=torch.float32).unsqueeze(0)
        return image, mask