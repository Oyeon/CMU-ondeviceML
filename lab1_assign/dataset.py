import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch

class CustomMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = self.data.iloc[:, 0].values
        self.images = self.data.iloc[:, 1:].values.astype('float32')
        self.mean = np.mean(self.images)
        self.std_dev = np.std(self.images)
        self.images = (self.images - self.mean) / self.std_dev
        self.images = self.images.reshape(-1, 28, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def create_transform(transform_type):
    if transform_type == 'resize_14':
        return transforms.Compose([
            transforms.Lambda(lambda img: torch.tensor(img).unsqueeze(0)),  # Convert NumPy array to tensor
            transforms.Resize((14, 14)),
            transforms.Normalize((0.5,), (0.5,))
        ]), 14 * 14
    elif transform_type == 'resize_20':
        return transforms.Compose([
            transforms.Lambda(lambda img: torch.tensor(img).unsqueeze(0)),  # Convert NumPy array to tensor
            transforms.Resize((20, 20)),
            transforms.Normalize((0.5,), (0.5,))
        ]), 20 * 20
    elif transform_type == 'crop_20':
        return transforms.Compose([
            transforms.Lambda(lambda img: torch.tensor(img).unsqueeze(0)),  # Convert NumPy array to tensor
            transforms.Resize((28, 28)),
            transforms.CenterCrop(20),
            transforms.Normalize((0.5,), (0.5,))
        ]), 20 * 20
    else:  # no_transform
        return transforms.Compose([
            transforms.Lambda(lambda img: torch.tensor(img).unsqueeze(0)),  # Convert NumPy array to tensor
            transforms.Normalize((0.5,), (0.5,))
        ]), 28 * 28

def get_dataloaders(config, transform_type='no_transform'):
    transform, input_dims = create_transform(transform_type)
    config['input_dims'] = input_dims
    train_dataset = CustomMNISTDataset(config['dataset_train_path'], transform=transform)
    val_dataset = CustomMNISTDataset(config['dataset_test_path'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['test_batch_size'], shuffle=False)
    return train_loader, val_loader