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
    transform_types = ['crop_7', 'resize_7']
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


    # Save results to a JSON file
    # with open('results/results.json', 'w') as f:
    #     json.dump(results, f, indent=4)

    flatten_and_save_to_csv(results, 'results/7by7and128_results.csv')



            # results[config_name] = {
            #     'training_losses': training_losses,
            #     'validation_losses': validation_losses,
            #     'training_accuracies': training_accuracies,
            #     'validation_accuracies': validation_accuracies,
            #     'training_times': training_times,
            #     'inference_times': inference_times,
            #     'inference_times_all': inference_times_all,
            #     'total_params': total_params,
            #     'flops': flops,
            #     'model_config': model_config  # Add model_config to results
            # }
            
            # total_train_latency = sum(training_times)
            # total_inference_latency = sum(inference_times)
            # print(f'Depth: {model_config["num_hidden_layers"]}, FLOPs: {flops}, Params: {total_params}, Train Acc: {training_accuracies[-1]}, Val Acc: {validation_accuracies[-1]}, Train Latency: {total_train_latency / CONFIG["epochs"]}, Inference Latency: {total_inference_latency / len(val_loader)}')

        # visualize_results(results)


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

def visualize_results(results):
    for key, metrics in results.items():
        epochs = range(1, CONFIG['epochs'] + 1)
        
        # Create a directory for the results if it doesn't exist
        result_dir = f'results/{key}'
        os.makedirs(result_dir, exist_ok=True)
        
        # Extract model configuration for labeling
        model_config = metrics['model_config']
        config_label = f"Depth: {model_config['num_hidden_layers']}, Width: {model_config['hidden_layer_width'] or 'default'}"
        
        # Plot training and validation loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, metrics['training_losses'], label=f'Training Loss ({config_label})')
        plt.plot(epochs, metrics['validation_losses'], label=f'Validation Loss ({config_label})')
        plt.title(f'Loss for {key}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, metrics['training_accuracies'], label=f'Training Accuracy ({config_label})')
        plt.plot(epochs, metrics['validation_accuracies'], label=f'Validation Accuracy ({config_label})')
        plt.title(f'Accuracy for {key}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{result_dir}/loss_accuracy.png')
        plt.close()

        # Plot FLOPs vs Accuracy
        plt.figure()
        plt.scatter(metrics['flops'], metrics['validation_accuracies'][-1], label=config_label)
        plt.title(f'FLOPs vs Accuracy for {key}')
        plt.xlabel('FLOPs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{result_dir}/flops_vs_accuracy.png')
        plt.close()

        # Plot FLOPs vs Latency
        plt.figure()
        plt.scatter(metrics['flops'], sum(metrics['inference_times']) / len(metrics['inference_times']), label=config_label)
        plt.title(f'FLOPs vs Latency for {key}')
        plt.xlabel('FLOPs')
        plt.ylabel('Latency')
        plt.legend()
        plt.savefig(f'{result_dir}/flops_vs_latency.png')
        plt.close()

        # Plot Latency vs Accuracy
        plt.figure()
        plt.scatter(sum(metrics['inference_times']) / len(metrics['inference_times']), metrics['validation_accuracies'][-1], label=config_label)
        plt.title(f'Latency vs Accuracy for {key}')
        plt.xlabel('Latency')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{result_dir}/latency_vs_accuracy.png')
        plt.close()

if __name__ == "__main__":
    main()

# from config import CONFIG
# from dataset import get_dataloaders
# from model import GarmentClassifier, count_parameters, compute_flops
# from train import train_model
# import torch
# import matplotlib.pyplot as plt

# def main():
#     transform_types = ['resize_14', 'resize_20', 'crop_20', 'no_transform']
#     model_configs = [
#         {'num_hidden_layers': 2, 'hidden_layer_width': None},  # Default
#         {'num_hidden_layers': 3, 'hidden_layer_width': None},  # Increased depth
#         {'num_hidden_layers': 2, 'hidden_layer_width': 2048},  # Increased width
#     ]
#     results = {}

#     for transform_type in transform_types:
#         for model_config in model_configs:
#             config_name = f"{transform_type}_depth{model_config['num_hidden_layers']}_width{model_config['hidden_layer_width'] or 'default'}"
#             print(f"Running experiment with config: {config_name}")
#             CONFIG['transform_type'] = transform_type
#             train_loader, val_loader = get_dataloaders(CONFIG, transform_type=CONFIG['transform_type'])
#             model = GarmentClassifier(CONFIG['input_dims'], CONFIG['hidden_feature_dims'], CONFIG['output_classes'], **model_config)
#             loss_fn = torch.nn.CrossEntropyLoss()
#             optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
            
#             training_losses, validation_losses, training_accuracies, validation_accuracies, training_times, inference_times = train_model(CONFIG, model_config, model, train_loader, val_loader, loss_fn, optimizer)
            
#             total_params = count_parameters(model)
#             flops = compute_flops(model)
            
#             results[config_name] = {
#                 'training_losses': training_losses,
#                 'validation_losses': validation_losses,
#                 'training_accuracies': training_accuracies,
#                 'validation_accuracies': validation_accuracies,
#                 'training_times': training_times,
#                 'inference_times': inference_times,
#                 'total_params': total_params,
#                 'flops': flops
#             }
            
#             total_train_latency = sum(training_times)
#             total_inference_latency = sum(inference_times)
#             print(f'Depth: {model_config["num_hidden_layers"]}, FLOPs: {flops}, Params: {total_params}, Train Acc: {training_accuracies[-1]}, Val Acc: {validation_accuracies[-1]}, Train Latency: {total_train_latency / CONFIG["epochs"]}, Inference Latency: {total_inference_latency / len(val_loader)}')

# def visualize_results(results):
#     for key, metrics in results.items():
#         epochs = range(1, CONFIG['epochs'] + 1)
        
#         plt.figure(figsize=(12, 4))
        
#         # Plot training and validation loss
#         plt.subplot(1, 2, 1)
#         plt.plot(epochs, metrics['training_losses'], label='Training Loss')
#         plt.plot(epochs, metrics['validation_losses'], label='Validation Loss')
#         plt.title(f'Loss for {key}')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.legend()
        
#         # Plot training and validation accuracy
#         plt.subplot(1, 2, 2)
#         plt.plot(epochs, metrics['training_accuracies'], label='Training Accuracy')
#         plt.plot(epochs, metrics['validation_accuracies'], label='Validation Accuracy')
#         plt.title(f'Accuracy for {key}')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.legend()
        
#         plt.tight_layout()
#         plt.show()

# if __name__ == "__main__":
#     main()
    
# from config import CONFIG
# from dataset import get_dataloaders
# from model import GarmentClassifier
# from train import train_model
# import torch

# def main():
#     train_loader, val_loader = get_dataloaders(CONFIG, transform_type=CONFIG['transform_type'])
#     model = GarmentClassifier(CONFIG['input_dims'], CONFIG['hidden_feature_dims'], CONFIG['output_classes'])
#     loss_fn = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
#     train_model(CONFIG, model, train_loader, val_loader, loss_fn, optimizer)

# if __name__ == "__main__":
#     main()