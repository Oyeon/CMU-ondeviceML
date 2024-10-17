import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    'dataset_train_path': '../datasets/lab1_dataset/mnist_train.csv',
    'dataset_test_path': '../datasets/lab1_dataset/mnist_test.csv',
    'input_dims': 28 * 28,
    'hidden_feature_dims': 1024,
    'output_classes': 10,
    'train_batch_size': 64,
    'test_batch_size': 10,
    'learning_rate': 0.001,
    'epochs': 2
}

# Custom dataset class
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

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Initialize datasets and dataloaders
def get_dataloaders(config):
    train_dataset = CustomMNISTDataset(config['dataset_train_path'], transform=transform)
    val_dataset = CustomMNISTDataset(config['dataset_test_path'], transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['test_batch_size'], shuffle=False)
    return train_loader, val_loader

train_loader, val_loader = get_dataloaders(CONFIG)

# Define the model
class GarmentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GarmentClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.input_layer(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output_layer(x)
        return x

model = GarmentClassifier(CONFIG['input_dims'], CONFIG['hidden_feature_dims'], CONFIG['output_classes'])
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

# Training and validation functions
def train_one_epoch(epoch_index, model, train_loader, optimizer, loss_fn, tb_writer):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f'  batch {i + 1} loss: {last_loss}')
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0
    return running_loss / len(train_loader)

def validate(model, val_loader, loss_fn):
    model.eval()
    running_vloss = 0.0
    running_vcorrects = 0
    total_vsamples = 0
    with torch.no_grad():
        for vinputs, vlabels in val_loader:
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()
            _, vpreds = torch.max(voutputs, 1)
            running_vcorrects += torch.sum(vpreds == vlabels).item()
            total_vsamples += vlabels.size(0)
    avg_vloss = running_vloss / len(val_loader)
    validation_accuracy = running_vcorrects / total_vsamples
    return avg_vloss, validation_accuracy

# Main training loop
def train_model(config):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
    best_vloss = float('inf')
    epoch_number = 0
    total_training_time = 0.0
    total_inference_time = 0.0
    training_accuracy = 0.0
    validation_accuracy = 0.0
    training_times = []
    inference_times_without_warmup = []

    for epoch in range(config['epochs']):
        print(f'=== EPOCH {epoch + 1} ===')
        start_train_time = time.time()
        avg_loss = train_one_epoch(epoch_number, model, train_loader, optimizer, loss_fn, writer)
        end_train_time = time.time()
        epoch_training_time = end_train_time - start_train_time
        total_training_time += epoch_training_time
        training_times.append(epoch_training_time)
        print(f'Epoch {epoch + 1} Training Time: {epoch_training_time:.4f} seconds')

        avg_vloss, validation_accuracy = validate(model, val_loader, loss_fn)
        print(f'LOSS - Train: {avg_loss:.4f}, Validation: {avg_vloss:.4f}')
        print(f'Epoch {epoch + 1} Validation Accuracy: {validation_accuracy:.4f}')

        writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Training': training_accuracy, 'Validation': validation_accuracy}, epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_{timestamp}_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path} with validation loss {avg_vloss:.4f}")

        epoch_number += 1

    print(f'Total Training Time: {total_training_time:.4f} seconds')
    print(f'Total Inference Time (excluding warm-up): {total_inference_time:.4f} seconds')

train_model(CONFIG)