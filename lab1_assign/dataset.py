import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    elif transform_type == 'resize_7':
        return transforms.Compose([
            transforms.Lambda(lambda img: torch.tensor(img).unsqueeze(0)),  # Convert NumPy array to tensor
            transforms.Resize((7, 7)),
            transforms.Normalize((0.5,), (0.5,))
        ]), 7 * 7
    elif transform_type == 'crop_20':
        return transforms.Compose([
            transforms.Lambda(lambda img: torch.tensor(img).unsqueeze(0)),  # Convert NumPy array to tensor
            transforms.Resize((28, 28)),
            transforms.CenterCrop(20),
            transforms.Normalize((0.5,), (0.5,))
        ]), 20 * 20
    elif transform_type == 'crop_14':
        return transforms.Compose([
            transforms.Lambda(lambda img: torch.tensor(img).unsqueeze(0)),  # Convert NumPy array to tensor
            transforms.Resize((28, 28)),
            transforms.CenterCrop(14),
            transforms.Normalize((0.5,), (0.5,))
        ]), 14 * 14  # Corrected this line
    elif transform_type == 'crop_7':
        return transforms.Compose([
            transforms.Lambda(lambda img: torch.tensor(img).unsqueeze(0)),  # Convert NumPy array to tensor
            transforms.Resize((28, 28)),
            transforms.CenterCrop(7),
            transforms.Normalize((0.5,), (0.5,))
        ]), 7 * 7
    else:  # no_transform
        return transforms.Compose([
            transforms.Lambda(lambda img: torch.tensor(img).unsqueeze(0)),  # Convert NumPy array to tensor
            transforms.Normalize((0.5,), (0.5,))
        ]), 28 * 28

def get_dataloaders(config, transform_type='no_transform', seed=42):
    set_seed(seed)  # Set the seed for reproducibility
    transform, input_dims = create_transform(transform_type)
    config['input_dims'] = input_dims
    train_dataset = CustomMNISTDataset(config['dataset_train_path'], transform=transform)
    test_dataset = CustomMNISTDataset(config['dataset_test_path'], transform=transform)
    
    # Split train_dataset into training and validation sets (80/20 split)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['test_batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['test_batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader