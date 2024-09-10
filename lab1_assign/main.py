from config import CONFIG
from dataset import get_dataloaders
from model import GarmentClassifier
from train import train_model
import torch

def main():
    train_loader, val_loader = get_dataloaders(CONFIG)
    model = GarmentClassifier(CONFIG['input_dims'], CONFIG['hidden_feature_dims'], CONFIG['output_classes'])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    train_model(CONFIG, model, train_loader, val_loader, loss_fn, optimizer)

if __name__ == "__main__":
    main()