from config import CONFIG
from dataset import get_dataloaders
from model import GarmentClassifier, count_parameters, compute_flops
from train import train_model
import torch
import matplotlib.pyplot as plt
import os
import json
import csv

def main():
    transform_types = [CONFIG['transform_type']]
    model_configs = [
        {'num_hidden_layers': CONFIG['num_hidden_layers'], 'hidden_layer_width': CONFIG['hidden_layer_width']},  # Default
    ]
    results = {}

    for test_batch_size in CONFIG['test_batch_list']:
        CONFIG['test_batch_size'] = test_batch_size
        print(f"Running experiment with test batch size: {test_batch_size}")
        for transform_type in transform_types:
            for model_config in model_configs:
                config_name = f"{transform_type}_depth{model_config['num_hidden_layers']}_width{model_config['hidden_layer_width'] or 'default'}"
                print(f"Running experiment with config: {config_name}")
                CONFIG['transform_type'] = transform_type
                train_loader, val_loader, test_loader = get_dataloaders(CONFIG, transform_type=CONFIG['transform_type'])
                model = GarmentClassifier(CONFIG['input_dims'], CONFIG['hidden_feature_dims'], CONFIG['output_classes'], **model_config)
                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
                
                epoch_results = train_model(CONFIG, model_config, model, train_loader, val_loader, test_loader, loss_fn, optimizer)
                
                total_params = count_parameters(model)
                flops = compute_flops(model)
                
                results[config_name] = {'epoch_results': epoch_results}

if __name__ == "__main__":
    main()
