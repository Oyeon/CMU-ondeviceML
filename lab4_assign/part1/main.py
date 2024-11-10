import logging
import os
import json
import torch
import torch.nn.utils.prune as prune  # Import pruning utilities

from config import CONFIG
from dataset import get_dataloaders
from model import GarmentClassifier, count_parameters, compute_flops
from train import train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_size_of_model(model, label=""):
    """
    Saves the model state_dict to a temporary file, prints its size, and removes the file.

    Args:
        model (torch.nn.Module): The model to measure.
        label (str): A label for the model.
    
    Returns:
        float: Size of the model in MB.
    """
    temp_path = "temp.p"
    try:
        torch.save(model.state_dict(), temp_path)
        size = os.path.getsize(temp_path) / 1e6  # Convert to MB
        logging.info(f"Model: {label}\t Size (MB): {size:.2f}")
        return size
    except Exception as e:
        logging.error(f"Error while saving model '{label}': {e}")
        return 0.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def prune_model(model, pruning_params, amount=0.9):
    """
    Applies global unstructured pruning to the specified layers of the model.

    Args:
        model (torch.nn.Module): The model to prune.
        pruning_params (list): List of tuples specifying (module, parameter_name) to prune.
        amount (float): The proportion of connections to prune.
    """
    try:
        prune.global_unstructured(
            pruning_params,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        logging.info(f"Pruned {amount*100:.1f}% of the model parameters.")
    except Exception as e:
        logging.error(f"Pruning failed: {e}")
        raise e

def remove_pruning_reparameterization(pruning_params):
    """
    Removes pruning reparameterization to make pruning permanent.

    Args:
        pruning_params (list): List of tuples specifying (module, parameter_name) to remove pruning from.
    """
    try:
        for module, param in pruning_params:
            prune.remove(module, param)
        logging.info("Removed pruning reparameterization. Pruning is now permanent.")
    except Exception as e:
        logging.error(f"Removing pruning reparameterization failed: {e}")
        raise e

def convert_to_sparse(state_dict):
    """
    Converts relevant weight tensors in the state_dict to sparse format.

    Args:
        state_dict (dict): The state dictionary of the model.

    Returns:
        dict: The state dictionary with sparse tensors where applicable.
    """
    sparse_sd = {}
    for key, tensor in state_dict.items():
        if 'weight' in key:
            sparse_sd[key] = tensor.to_sparse()
        else:
            sparse_sd[key] = tensor
    return sparse_sd

def save_model(state_dict, path="model_sparse.pt"):
    """
    Saves the state dictionary to the specified path.

    Args:
        state_dict (dict): The state dictionary to save.
        path (str): The file path to save the state dictionary.
    """
    try:
        torch.save(state_dict, path)
        size_mb = os.path.getsize(path) / 1e6
        logging.info(f"Sparse model saved to '{path}' with size: {size_mb:.2f} MB")
    except Exception as e:
        logging.error(f"Failed to save model to '{path}': {e}")
        raise e

def load_sparse_model(path, model_config):
    """
    Loads a sparse model from the specified path and converts it back to dense.

    Args:
        path (str): The file path of the sparse model.
        model_config (dict): Configuration parameters for the model.

    Returns:
        torch.nn.Module: The loaded dense model with pruned weights.
    """
    try:
        loaded_sd = torch.load(path, weights_only=True)
        new_model = GarmentClassifier(
            CONFIG['input_dims'],
            CONFIG['hidden_feature_dims'],
            CONFIG['output_classes'],
            **model_config
        )
        new_model.load_state_dict({k: (v if v.layout == torch.strided else v.to_dense()) for k, v in loaded_sd.items()})
        logging.info(f"Loaded model from '{path}'.")
        return new_model
    except Exception as e:
        logging.error(f"Failed to load model from '{path}': {e}")
        raise e

def get_pruning_params(model):
    """
    Prepares the list of parameters to prune in the model.

    Args:
        model (torch.nn.Module): The model whose parameters are to be pruned.

    Returns:
        list: List of tuples specifying (module, parameter_name) to prune.
    """
    prune_params = []

    # Add input layer
    prune_params.append((model.input_layer, 'weight'))

    # Add all hidden layers
    for hidden_layer in model.hidden_layers:
        prune_params.append((hidden_layer, 'weight'))

    # Add output layer
    prune_params.append((model.output_layer, 'weight'))

    return prune_params

def calculate_sparsity(model):
    """
    Calculates the overall sparsity of the model.

    Args:
        model (torch.nn.Module): The model to evaluate.

    Returns:
        float: Sparsity ratio as a float between 0 and 1.
    """
    zero_params = 0
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            zero_params += torch.sum(param == 0).item()
            total_params += param.numel()
    return zero_params / total_params if total_params > 0 else 0

def verify_pruning(model):
    """
    Verifies and logs the sparsity of each layer in the model.

    Args:
        model (torch.nn.Module): The model to verify.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            zero_elements = torch.sum(weight == 0).item()
            total_elements = weight.numel()
            sparsity = 100. * zero_elements / total_elements
            logging.info(f"{name} - Sparsity: {sparsity:.2f}%")

def main():
    transform_types = [CONFIG['transform_type']]
    model_configs = [
        {
            'num_hidden_layers': CONFIG['num_hidden_layers'],
            'hidden_layer_width': CONFIG['hidden_layer_width']
        },
    ]
    results = {}

    for test_batch_size in CONFIG['test_batch_list']:
        CONFIG['test_batch_size'] = test_batch_size
        logging.info(f"Running experiment with test batch size: {test_batch_size}")
        
        for transform_type in transform_types:
            for model_config in model_configs:
                config_name = f"{transform_type}_depth{model_config['num_hidden_layers']}_width{model_config['hidden_layer_width'] or 'default'}"
                logging.info(f"Running experiment with config: {config_name}")
                CONFIG['transform_type'] = transform_type
                
                try:
                    train_loader, val_loader, test_loader = get_dataloaders(
                        CONFIG, transform_type=CONFIG['transform_type']
                    )
                except Exception as e:
                    logging.error(f"Failed to get data loaders: {e}")
                    continue

                try:
                    model = GarmentClassifier(
                        CONFIG['input_dims'],
                        CONFIG['hidden_feature_dims'],
                        CONFIG['output_classes'],
                        **model_config
                    )
                    logging.info("Initialized GarmentClassifier.")
                except Exception as e:
                    logging.error(f"Failed to initialize model: {e}")
                    continue

                # Measure model size before pruning
                full_size = print_size_of_model(model, "model_before_pruning")

                # Prepare pruning parameters
                prune_params = get_pruning_params(model)

                # Prune the model
                prune_model(model, prune_params, amount=CONFIG['pruning_amount'])

                # Make pruning permanent
                remove_pruning_reparameterization(prune_params)

                # Calculate sparsity
                sparsity = calculate_sparsity(model)
                logging.info(f"Model sparsity after pruning: {sparsity * 100:.2f}%")

                # Decide whether to convert to sparse based on sparsity threshold
                SPARSITY_THRESHOLD = CONFIG.get('sparsity_threshold', 0.5)  # Default to 50% if not set
                if sparsity >= SPARSITY_THRESHOLD:
                    # Convert to sparse representations
                    try:
                        sparse_sd = convert_to_sparse(model.state_dict())
                        logging.info("Converted model to sparse representation.")
                    except Exception as e:
                        logging.error(f"Failed to convert to sparse: {e}")
                        continue

                    # Save the sparse model
                    try:
                        save_model(sparse_sd, "model_sparse.pt")
                    except Exception as e:
                        logging.error(f"Failed to save sparse model: {e}")
                        continue
                else:
                    # Save the model in dense format
                    try:
                        torch.save(model.state_dict(), "model_dense.pt")
                        size_dense = os.path.getsize('model_dense.pt') / 1e6  # MB
                        logging.info(f"Dense model saved to 'model_dense.pt' with size: {size_dense:.2f} MB")
                    except Exception as e:
                        logging.error(f"Failed to save dense model: {e}")
                        continue

                # Load the appropriate model
                try:
                    if sparsity >= SPARSITY_THRESHOLD:
                        new_model = load_sparse_model("model_sparse.pt", model_config)
                        logging.info("Loaded sparse model from 'model_sparse.pt'.")
                    else:
                        loaded_sd = torch.load("model_dense.pt", weights_only=True)
                        new_model = GarmentClassifier(
                            CONFIG['input_dims'],
                            CONFIG['hidden_feature_dims'],
                            CONFIG['output_classes'],
                            **model_config
                        )
                        new_model.load_state_dict(loaded_sd)
                        logging.info("Loaded dense model from 'model_dense.pt'.")
                except Exception as e:
                    logging.error(f"Failed to load model: {e}")
                    continue

                # Verify pruning effectiveness
                verify_pruning(new_model)

                # Continue with training and evaluation
                loss_fn = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(new_model.parameters(), lr=CONFIG['learning_rate'])
                
                try:
                    epoch_results = train_model(
                        CONFIG, model_config, new_model,
                        train_loader, val_loader, test_loader,
                        loss_fn, optimizer
                    )
                except Exception as e:
                    logging.error(f"Training failed: {e}")
                    continue

                try:
                    total_params = count_parameters(new_model)
                    flops = compute_flops(new_model)
                except Exception as e:
                    logging.error(f"Failed to compute metrics: {e}")
                    total_params = None
                    flops = None

                results[config_name] = {
                    'epoch_results': epoch_results,
                    'total_params': total_params,
                    'flops': flops
                }

    # Save the results to a JSON file
    try:
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)
        logging.info("Saved all results to 'results.json'.")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()


# from config import CONFIG
# from dataset import get_dataloaders
# from model import GarmentClassifier, count_parameters, compute_flops
# from train import train_model
# import torch
# import matplotlib.pyplot as plt
# import os
# import json
# import csv

# def print_size_of_model(model, label=""):
#     torch.save(model.state_dict(), "temp.p")
#     size=os.path.getsize("temp.p")
#     print("model: ",label,' \t','Size (MB):', size/1e6)
#     os.remove('temp.p')
#     return size

# def main():
#     transform_types = [CONFIG['transform_type']]
#     model_configs = [
#         {'num_hidden_layers': CONFIG['num_hidden_layers'], 'hidden_layer_width': CONFIG['hidden_layer_width']},  # Default
#     ]
#     results = {}

#     for test_batch_size in CONFIG['test_batch_list']:
#         CONFIG['test_batch_size'] = test_batch_size
#         print(f"Running experiment with test batch size: {test_batch_size}")
#         for transform_type in transform_types:
#             for model_config in model_configs:
#                 config_name = f"{transform_type}_depth{model_config['num_hidden_layers']}_width{model_config['hidden_layer_width'] or 'default'}"
#                 print(f"Running experiment with config: {config_name}")
#                 CONFIG['transform_type'] = transform_type
#                 train_loader, val_loader, test_loader = get_dataloaders(CONFIG, transform_type=CONFIG['transform_type'])
#                 model = GarmentClassifier(CONFIG['input_dims'], CONFIG['hidden_feature_dims'], CONFIG['output_classes'], **model_config)

#                 full=print_size_of_model(model, "model1")
#                 pruned=print_size_of_model(model, "model2")
#                 print("{0:.2f} times smaller".format(full/pruned))                
                
#                 loss_fn = torch.nn.CrossEntropyLoss()
#                 optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
                
#                 epoch_results = train_model(CONFIG, model_config, model, train_loader, val_loader, test_loader, loss_fn, optimizer)
                
#                 total_params = count_parameters(model)
#                 flops = compute_flops(model)
                
#                 results[config_name] = {'epoch_results': epoch_results}

# if __name__ == "__main__":
#     main()


