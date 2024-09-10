import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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

def get_dataloaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = CustomMNISTDataset(config['dataset_train_path'], transform=transform)
    val_dataset = CustomMNISTDataset(config['dataset_test_path'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['test_batch_size'], shuffle=False)
    return train_loader, val_loader