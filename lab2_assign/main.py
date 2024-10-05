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
    # transform_types = ['no_transform', 'crop_20', 'crop_14', 'crop_7', 'resize_20', 'resize_14', 'resize_7']
    # transform_types = ['crop_7', 'resize_7']
    transform_types = ['resize_7']
    model_configs = [
        {'num_hidden_layers': 2, 'hidden_layer_width': 128},  # Default
        # {'num_hidden_layers': 2, 'hidden_layer_width': 256},  # Default
        # {'num_hidden_layers': 2, 'hidden_layer_width': 512},  # Default
        # {'num_hidden_layers': 2, 'hidden_layer_width': 1024},  # Default
        # {'num_hidden_layers': 2, 'hidden_layer_width': 2048},  # Default
        # {'num_hidden_layers': 2, 'hidden_layer_width': 4096},  # Default
        # {'num_hidden_layers': 3, 'hidden_layer_width': None},  # Increased depth
        # {'num_hidden_layers': 4, 'hidden_layer_width': None},  # Increased depth
        # {'num_hidden_layers': 8, 'hidden_layer_width': None},  # Increased depth
        # {'num_hidden_layers': 16, 'hidden_layer_width': None},  # Increased depth
    ]
    results = {}

    for transform_type in transform_types:
        for model_config in model_configs:
            config_name = f"{transform_type}_depth{model_config['num_hidden_layers']}_width{model_config['hidden_layer_width'] or 'default'}"
            print(f"Running experiment with config: {config_name}")
            CONFIG['transform_type'] = transform_type
            train_loader, val_loader, test_loader = get_dataloaders(CONFIG, transform_type=CONFIG['transform_type'])
            model = GarmentClassifier(CONFIG['input_dims'], CONFIG['hidden_feature_dims'], CONFIG['output_classes'], **model_config)
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
            
            # training_losses, validation_losses, training_accuracies, validation_accuracies, training_times, inference_times, inference_times_all, batch_latency_list, batch_flops_list = train_model(CONFIG, model_config, model, train_loader, val_loader, loss_fn, optimizer)
            epoch_results = train_model(CONFIG, model_config, model, train_loader, val_loader, test_loader, loss_fn, optimizer)
            
            total_params = count_parameters(model)
            flops = compute_flops(model)
            
            results[config_name] = {'epoch_results': epoch_results}

    # flatten_and_save_to_csv(results, 'results/7by7and128_results.csv')


def flatten_and_save_to_csv(results, csv_file_path):
    # Flatten the results
    flattened_results = []
    for config_name, data in results.items():
        epoch_results = data['epoch_results']
        for epoch, metrics in epoch_results.items():
            flat_result = {'config_name': config_name, 'epoch': epoch}
            flat_result.update(metrics)
            flattened_results.append(flat_result)

    # Get the headers from the first flattened result
    headers = flattened_results[0].keys()

    # Write to CSV
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in flattened_results:
            writer.writerow(row)

if __name__ == "__main__":
    main()
