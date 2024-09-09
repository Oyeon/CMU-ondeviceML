import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms

# Custom dataset class
class CustomMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Load the CSV file
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Assuming the first column contains labels and the rest are pixel values
        self.labels = self.data.iloc[:, 0].values  # First column is the label
        self.images = self.data.iloc[:, 1:].values.astype('float32')  # Remaining columns are pixel values

        # Normalize the pixel values (subtract the mean and divide by std dev)
        # Calculate mean and std for normalization
        self.mean = np.mean(self.images)
        self.std_dev = np.std(self.images)

        # Normalize using the calculated mean and std
        self.images = (self.images - self.mean) / self.std_dev

        # Reshape the images to 28x28
        self.images = self.images.reshape(-1, 28, 28)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            # Apply transformations (if any)
            image = self.transform(image)

        return image, label

# Define the transformations (you can modify them as needed)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Additional normalization to center data around 0
])

# Initialize the custom dataset using the CSV file path
csv_file_path = './datasets/mnist_train.csv'
csv_file_path2 = './datasets/mnist_test.csv'
mnist_dataset = CustomMNISTDataset(csv_file_path, transform=transform)
val_mnist_dataset = CustomMNISTDataset(csv_file_path2, transform=transform)

# Use DataLoader for batching and shuffling
train_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_mnist_dataset, batch_size=10, shuffle=True)


# Example: Iterating through the dataset
for images, labels in train_loader:
    print(f'Batch images shape: {images.shape}')
    print(f'Batch labels shape: {labels.shape}')
    break
