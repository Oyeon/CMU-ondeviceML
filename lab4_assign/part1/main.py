import logging
import os
import json
import torch
import torch.nn.utils.prune as prune
import gzip
import pickle
import random
import numpy as np
import copy

from config import CONFIG
from dataset import get_dataloaders
from model import GarmentClassifier, count_parameters, compute_flops
from train import train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_size_of_model(model, label=""):
    temp_path = "temp.p"
    try:
        torch.save(model.state_dict(), temp_path)
        size = os.path.getsize(temp_path) / 1e6  # Convert to MB
        logging.info(f"Model: {label}\t Size (MB): {size:.2f} MB")
        return size
    except Exception as e:
        logging.error(f"Error while saving model '{label}': {e}")
        return 0.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def prune_model(model, pruning_params, amount=0.33, method=prune.RandomUnstructured):
    # Prune using the specified method
    try:
        prune.global_unstructured(
            pruning_params,
            pruning_method=method,
            amount=amount,
        )
        logging.info(f"Pruned {amount * 100:.1f}% of the model parameters using {method.__name__} method.")
    except Exception as e:
        logging.error(f"Pruning failed: {e}")
        raise e

def remove_pruning_reparameterization(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
            # Only attempt to remove pruning for bias if it has been pruned
            if hasattr(module, 'bias_orig'):
                prune.remove(module, 'bias')
    logging.info("Removed pruning reparameterization. Pruning is now permanent.")


def save_compressed_model(state_dict, path="model_compressed.pt.gz"):
    # Filter out any metadata keys from the state_dict
    filtered_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
    
    try:
        with gzip.GzipFile(path, 'w') as f:
            pickle.dump(filtered_state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(path) / 1e6
        logging.info(f"Compressed model saved to '{path}' with size: {size_mb:.2f} MB")
    except Exception as e:
        logging.error(f"Failed to save compressed model: {e}")
        raise e


def load_compressed_model(path, model_config):
    try:
        with gzip.GzipFile(path, 'r') as f:
            loaded_sd = pickle.load(f)
        new_model = GarmentClassifier(
            CONFIG['input_dims'],
            CONFIG['hidden_feature_dims'],
            CONFIG['output_classes'],
            **model_config
        )
        new_model.load_state_dict(loaded_sd)
        logging.info(f"Loaded compressed model from '{path}'.")
        return new_model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise e

def get_pruning_params(model):
    prune_params = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            prune_params.append((module, 'weight'))
            # If you want to prune biases as well, uncomment the following line
            # prune_params.append((module, 'bias'))
    return prune_params

def calculate_sparsity(model):
    zero_params = 0
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight_orig'):  # Check for pruned parameters
                # Calculate zeros by applying the mask
                mask = getattr(module, 'weight_mask')
                zero_params_layer = torch.sum(mask == 0).item()
                total_params_layer = mask.numel()
                logging.info(f"{name}.weight_orig - Zero Params: {zero_params_layer}, Total Params: {total_params_layer}")
                zero_params += zero_params_layer
                total_params += total_params_layer
            else:
                # For layers that are not pruned or haven't been set up with pruning
                zero_params_layer = torch.sum(module.weight == 0).item()
                total_params_layer = module.weight.numel()
                logging.info(f"{name}.weight - Zero Params: {zero_params_layer}, Total Params: {total_params_layer}")
                zero_params += zero_params_layer
                total_params += total_params_layer

    sparsity = zero_params / total_params if total_params > 0 else 0
    logging.info(f"Total Zero Params: {zero_params}, Total Params: {total_params}")
    return sparsity

def verify_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            zero_elements = torch.sum(weight == 0).item()
            total_elements = weight.numel()
            sparsity = 100. * zero_elements / total_elements
            logging.info(f"{name} - Sparsity: {sparsity:.2f}%")

def rewind_weights(model, initial_state_dict, prune_param_list):
    """
    Rewinds the weights of the model to their initial values while keeping the pruning masks.

    Args:
        model (torch.nn.Module): The model to rewind.
        initial_state_dict (dict): The initial state dictionary of the model.
        prune_param_list (list): List of parameter names that were pruned.
    """
    # Adjust parameter names to match those in the pruned model
    init_updated = {}
    for k, v in initial_state_dict.items():
        if k in prune_param_list:
            init_updated[k + '_orig'] = v  # Add '_orig' to the parameter name
        else:
            init_updated[k] = v  # Keep the parameter name as is

    # Update the model's state dict
    model_state_dict = model.state_dict()
    model_state_dict.update(init_updated)
    model.load_state_dict(model_state_dict)
    logging.info("Rewound model weights to initial state.")

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
                    train_loader, val_loader, test_loader = get_dataloaders(CONFIG, transform_type=CONFIG['transform_type'])
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

                # Save initial weights for rewinding
                initial_state_dict = copy.deepcopy(model.state_dict())
                logging.info("Saved initial model weights for rewinding.")

                full_size = print_size_of_model(model, "model_before_pruning")

                prune_params = get_pruning_params(model)

                # Create a list of parameter names that have been pruned
                prune_param_list = []
                for module_name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        param_name = 'weight'
                        full_param_name = f"{module_name}.{param_name}" if module_name else param_name
                        prune_param_list.append(full_param_name)

                # Set loss function
                loss_fn = torch.nn.CrossEntropyLoss()

                # Prune the model based on iterative or random approach
                if CONFIG['iterative_mode']:
                    logging.info("Starting Iterative Magnitude Pruning with Rewinding.")
                    num_iterations = CONFIG.get('num_iterations', 5)
                    pruning_amount = CONFIG.get('pruning_amount_per_iteration', 0.20)
                    for iteration in range(num_iterations):
                        logging.info(f"=== Iteration {iteration + 1}/{num_iterations} ===")

                        # Prune the model (accumulates masks)
                        prune_model(model, prune_params, amount=pruning_amount, method=prune.L1Unstructured)
                        verify_pruning(model)
                        # Do not remove pruning reparameterization yet
                        sparsity = calculate_sparsity(model)
                        logging.info(f"Model sparsity after iteration {iteration + 1}: {sparsity * 100:.2f}%")

                        # Rewind weights to initial untrained state
                        rewind_weights(model, initial_state_dict, prune_param_list)
                        logging.info("Rewound model weights to initial (untrained) state.")

                        # Re-initialize optimizer
                        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

                        # Retrain the model with same hyperparameters
                        try:
                            epoch_results = train_model(
                                CONFIG, model_config, model,
                                train_loader, val_loader, test_loader,
                                loss_fn=loss_fn,
                                optimizer=optimizer
                            )
                        except Exception as e:
                            logging.error(f"Training failed: {e}")
                            break  # Exit the iteration

                        # Collect and store metrics
                        results[f"{config_name}_iteration_{iteration + 1}"] = {
                            'epoch_results': epoch_results,
                            'sparsity': sparsity,
                        }

                    # After all iterations, remove pruning reparameterization
                    remove_pruning_reparameterization(model)
                    logging.info("Removed pruning reparameterization after all iterations.")

                    # Save the final pruned model
                    try:
                        save_compressed_model(model.state_dict(), "model_iterative_pruned.pt.gz")
                    except Exception as e:
                        logging.error(f"Failed to save compressed model: {e}")
                        continue

                    # Load and evaluate the model
                    try:
                        new_model = load_compressed_model("model_iterative_pruned.pt.gz", model_config)
                    except Exception as e:
                        logging.error(f"Failed to load model: {e}")
                        continue

                else:  # Randomized Non-Iterative Pruning
                    prune_model(model, prune_params, amount=CONFIG['pruning_amount'], method=prune.RandomUnstructured)
                    verify_pruning(model)
                    remove_pruning_reparameterization(model)
                    sparsity = calculate_sparsity(model)
                    logging.info(f"Model sparsity after random pruning: {sparsity * 100:.2f}%")

                    # Save the model
                    try:
                        save_compressed_model(model.state_dict(), "model_random_pruned.pt.gz")
                    except Exception as e:
                        logging.error(f"Failed to save compressed model: {e}")
                        continue

                    # Load and evaluate the model
                    try:
                        new_model = load_compressed_model("model_random_pruned.pt.gz", model_config)
                    except Exception as e:
                        logging.error(f"Failed to load model: {e}")
                        continue

                    # Continue with training and evaluation
                    optimizer = torch.optim.Adam(new_model.parameters(), lr=CONFIG['learning_rate'])

                    try:
                        epoch_results = train_model(CONFIG, model_config, new_model, train_loader, val_loader, test_loader, loss_fn=loss_fn, optimizer=optimizer)
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
                        'flops': flops,
                        'sparsity': sparsity
                    }

        try:
            with open('results.json', 'w') as f:
                json.dump(results, f, indent=4)
            logging.info("Saved all results to 'results.json'.")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()





# import logging
# import os
# import json
# import torch
# import torch.nn.utils.prune as prune
# import gzip
# import pickle
# import random
# import numpy as np
# import copy

# from config import CONFIG
# from dataset import get_dataloaders
# from model import GarmentClassifier, count_parameters, compute_flops
# from train import train_model

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def print_size_of_model(model, label=""):
#     temp_path = "temp.p"
#     try:
#         torch.save(model.state_dict(), temp_path)
#         size = os.path.getsize(temp_path) / 1e6  # Convert to MB
#         logging.info(f"Model: {label}\t Size (MB): {size:.2f} MB")
#         return size
#     except Exception as e:
#         logging.error(f"Error while saving model '{label}': {e}")
#         return 0.0
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

# def prune_model(model, pruning_params, amount=0.33, method=prune.RandomUnstructured):
#     # Set random seeds for variability in each run
#     random.seed()
#     np.random.seed()
#     torch.manual_seed(torch.initial_seed() + random.randint(1, 10000))

#     # Prune using the specified method
#     try:
#         prune.global_unstructured(
#             pruning_params,
#             pruning_method=method,
#             amount=amount,
#         )
#         logging.info(f"Pruned {amount * 100:.1f}% of the model parameters using {method.__name__} method.")
#     except Exception as e:
#         logging.error(f"Pruning failed: {e}")
#         raise e

# def remove_pruning_reparameterization(model):
#     for module in model.modules():
#         if isinstance(module, torch.nn.Linear):
#             for param_name, _ in module.named_parameters():
#                 if param_name.endswith('_orig'):
#                     prune.remove(module, param_name.replace('_orig', ''))
#     logging.info("Removed pruning reparameterization. Pruning is now permanent.")

# def save_compressed_model(state_dict, path="model_compressed.pt.gz"):
#     try:
#         with gzip.GzipFile(path, 'w') as f:
#             pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
#         size_mb = os.path.getsize(path) / 1e6
#         logging.info(f"Compressed model saved to '{path}' with size: {size_mb:.2f} MB")
#     except Exception as e:
#         logging.error(f"Failed to save compressed model: {e}")
#         raise e

# def load_compressed_model(path, model_config):
#     try:
#         with gzip.GzipFile(path, 'r') as f:
#             loaded_sd = pickle.load(f)
#         new_model = GarmentClassifier(
#             CONFIG['input_dims'],
#             CONFIG['hidden_feature_dims'],
#             CONFIG['output_classes'],
#             **model_config
#         )
#         new_model.load_state_dict(loaded_sd)
#         logging.info(f"Loaded compressed model from '{path}'.")
#         return new_model
#     except Exception as e:
#         logging.error(f"Failed to load model: {e}")
#         raise e

# def get_pruning_params(model):
#     prune_params = []
#     prune_params.append((model.input_layer, 'weight'))
#     for hidden_layer in model.hidden_layers:
#         prune_params.append((hidden_layer, 'weight'))
#     prune_params.append((model.output_layer, 'weight'))
#     return prune_params

# def calculate_sparsity(model):
#     zero_params = 0
#     total_params = 0
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Linear):
#             if hasattr(module, 'weight_orig'):  # Check for pruned parameters
#                 # Calculate zeros by applying the mask
#                 mask = getattr(module, 'weight_mask')
#                 zero_params_layer = torch.sum(mask == 0).item()
#                 total_params_layer = mask.numel()
#                 logging.info(f"{name}.weight_orig - Zero Params: {zero_params_layer}, Total Params: {total_params_layer}")
#                 zero_params += zero_params_layer
#                 total_params += total_params_layer
#             else:
#                 # For layers that are not pruned or haven't been set up with pruning
#                 zero_params_layer = torch.sum(module.weight == 0).item()
#                 total_params_layer = module.weight.numel()
#                 logging.info(f"{name}.weight - Zero Params: {zero_params_layer}, Total Params: {total_params_layer}")
#                 zero_params += zero_params_layer
#                 total_params += total_params_layer

#     sparsity = zero_params / total_params if total_params > 0 else 0
#     logging.info(f"Total Zero Params: {zero_params}, Total Params: {total_params}")
#     return sparsity


# def verify_pruning(model):
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Linear):
#             weight = module.weight.data
#             zero_elements = torch.sum(weight == 0).item()
#             total_elements = weight.numel()
#             sparsity = 100. * zero_elements / total_elements
#             logging.info(f"{name} - Sparsity: {sparsity:.2f}%")

# def rewind_weights(model, initial_state_dict):
#     model.load_state_dict(initial_state_dict, strict=False)
#     logging.info("Rewound model weights to initial state.")

# def main():
#     transform_types = [CONFIG['transform_type']]
#     model_configs = [
#         {
#             'num_hidden_layers': CONFIG['num_hidden_layers'],
#             'hidden_layer_width': CONFIG['hidden_layer_width']
#         },
#     ]
#     results = {}

#     for test_batch_size in CONFIG['test_batch_list']:
#         CONFIG['test_batch_size'] = test_batch_size
#         logging.info(f"Running experiment with test batch size: {test_batch_size}")

#         for transform_type in transform_types:
#             for model_config in model_configs:
#                 config_name = f"{transform_type}_depth{model_config['num_hidden_layers']}_width{model_config['hidden_layer_width'] or 'default'}"
#                 logging.info(f"Running experiment with config: {config_name}")
#                 CONFIG['transform_type'] = transform_type

#                 try:
#                     train_loader, val_loader, test_loader = get_dataloaders(CONFIG, transform_type=CONFIG['transform_type'])
#                 except Exception as e:
#                     logging.error(f"Failed to get data loaders: {e}")
#                     continue

#                 try:
#                     model = GarmentClassifier(
#                         CONFIG['input_dims'],
#                         CONFIG['hidden_feature_dims'],
#                         CONFIG['output_classes'],
#                         **model_config
#                     )
#                     logging.info("Initialized GarmentClassifier.")
#                 except Exception as e:
#                     logging.error(f"Failed to initialize model: {e}")
#                     continue

#                 # Save initial weights for rewinding
#                 initial_state_dict = copy.deepcopy(model.state_dict())

#                 full_size = print_size_of_model(model, "model_before_pruning")

#                 prune_params = get_pruning_params(model)

#                 # Prune the model based on iterative or random approach
#                 if CONFIG['iterative_mode']:
#                     logging.info("Starting Iterative Pruning...")
#                     num_iterations = CONFIG.get('num_iterations', CONFIG['num_iterations'])
#                     pruning_amount = CONFIG.get('pruning_amount_per_iteration', CONFIG['pruning_amount_per_iteration'])
#                     for iteration in range(num_iterations):
#                         logging.info(f"=== Iteration {iteration + 1}/{num_iterations} ===")
#                         prune_model(model, prune_params, amount=pruning_amount, method=prune.L1Unstructured)
#                         verify_pruning(model)
#                         # Do not remove pruning reparameterization yet
#                         sparsity = calculate_sparsity(model)
#                         logging.info(f"Model sparsity after iteration {iteration + 1}: {sparsity * 100:.2f}%")

#                         # Rewind weights
#                         rewind_weights(model, initial_state_dict)

#                         # Re-initialize optimizer
#                         optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

#                         # Retrain the model
#                         try:
#                             epoch_results = train_model(CONFIG, model_config, model, train_loader, val_loader, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), optimizer=optimizer)
#                         except Exception as e:
#                             logging.error(f"Training failed: {e}")
#                             break

#                     # After all iterations, remove pruning reparameterization
#                     remove_pruning_reparameterization(model)
#                     logging.info("Removed pruning reparameterization after all iterations.")

#                     # Save the model
#                     try:
#                         save_compressed_model(model.state_dict(), "model_iterative_pruned.pt.gz")
#                     except Exception as e:
#                         logging.error(f"Failed to save compressed model: {e}")
#                         continue

#                     # Load and evaluate the model
#                     try:
#                         new_model = load_compressed_model("model_iterative_pruned.pt.gz", model_config)
#                     except Exception as e:
#                         logging.error(f"Failed to load model: {e}")
#                         continue

#                 else:  # Randomized Non-Iterative Pruning
#                     prune_model(model, prune_params, amount=CONFIG['pruning_amount'], method=prune.RandomUnstructured)
#                     verify_pruning(model)
#                     remove_pruning_reparameterization(model)
#                     sparsity = calculate_sparsity(model)
#                     logging.info(f"Model sparsity after random pruning: {sparsity * 100:.2f}%")

#                     # Save the model
#                     try:
#                         save_compressed_model(model.state_dict(), "model_random_pruned.pt.gz")
#                     except Exception as e:
#                         logging.error(f"Failed to save compressed model: {e}")
#                         continue

#                     # Load and evaluate the model
#                     try:
#                         new_model = load_compressed_model("model_random_pruned.pt.gz", model_config)
#                     except Exception as e:
#                         logging.error(f"Failed to load model: {e}")
#                         continue

#                 # Continue with training and evaluation
#                 optimizer = torch.optim.Adam(new_model.parameters(), lr=CONFIG['learning_rate'])

#                 try:
#                     epoch_results = train_model(CONFIG, model_config, new_model, train_loader, val_loader, test_loader, loss_fn=torch.nn.CrossEntropyLoss(), optimizer=optimizer)
#                 except Exception as e:
#                     logging.error(f"Training failed: {e}")
#                     continue

#                 try:
#                     total_params = count_parameters(new_model)
#                     flops = compute_flops(new_model)
#                 except Exception as e:
#                     logging.error(f"Failed to compute metrics: {e}")
#                     total_params = None
#                     flops = None

#                 results[config_name] = {
#                     'epoch_results': epoch_results,
#                     'total_params': total_params,
#                     'flops': flops,
#                     'sparsity': sparsity
#                 }

#     try:
#         with open('results.json', 'w') as f:
#             json.dump(results, f, indent=4)
#         logging.info("Saved all results to 'results.json'.")
#     except Exception as e:
#         logging.error(f"Failed to save results: {e}")

# if __name__ == "__main__":
#     main()



########################################################################################
########################################################################################

# import logging
# import os
# import json
# import torch
# import torch.nn.utils.prune as prune  # Import pruning utilities
# import gzip
# import pickle
# import random
# import numpy as np

# from config import CONFIG
# from dataset import get_dataloaders
# from model import GarmentClassifier, count_parameters, compute_flops
# from train import train_model

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def print_size_of_model(model, label=""):
#     """
#     Saves the model state_dict to a temporary file, prints its size, and removes the file.

#     Args:
#         model (torch.nn.Module): The model to measure.
#         label (str): A label for the model.

#     Returns:
#         float: Size of the model in MB.
#     """
#     temp_path = "temp.p"
#     try:
#         torch.save(model.state_dict(), temp_path)
#         size = os.path.getsize(temp_path) / 1e6  # Convert to MB
#         logging.info(f"Model: {label}\t Size (MB): {size:.2f} MB")
#         return size
#     except Exception as e:
#         logging.error(f"Error while saving model '{label}': {e}")
#         return 0.0
#     finally:
#         if os.path.exists(temp_path):
#             os.remove(temp_path)

# def prune_model(model, pruning_params, amount=0.33):
#     """
#     Applies global unstructured pruning to the specified layers of the model.

#     Args:
#         model (torch.nn.Module): The model to prune.
#         pruning_params (list): List of tuples specifying (module, parameter_name) to prune.
#         amount (float): The proportion of connections to prune.
#     """
 
#     random.seed()  
#     np.random.seed()
#     torch.manual_seed(torch.initial_seed() + random.randint(1, 10000))

#     try:
#         prune.global_unstructured(
#             pruning_params,
#             # pruning_method=prune.L1Unstructured,
#             pruning_method=prune.RandomUnstructured,
#             amount=amount,
#         )
#         logging.info(f"Pruned {amount*100:.1f}% of the model parameters.")
#     except Exception as e:
#         logging.error(f"Pruning failed: {e}")
#         raise e

# def remove_pruning_reparameterization(pruning_params):
#     """
#     Removes pruning reparameterization to make pruning permanent.

#     Args:
#         pruning_params (list): List of tuples specifying (module, parameter_name) to remove pruning from.
#     """
#     try:
#         for module, param in pruning_params:
#             prune.remove(module, param)
#         logging.info("Removed pruning reparameterization. Pruning is now permanent.")
#     except Exception as e:
#         logging.error(f"Removing pruning reparameterization failed: {e}")
#         raise e

# def save_compressed_model(state_dict, path="model_compressed.pt.gz"):
#     """
#     Saves the state dictionary to the specified path with gzip compression.

#     Args:
#         state_dict (dict): The state dictionary to save.
#         path (str): The file path to save the state dictionary.
#     """
#     try:
#         with gzip.GzipFile(path, 'w') as f:
#             pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
#         size_mb = os.path.getsize(path) / 1e6
#         logging.info(f"Compressed model saved to '{path}' with size: {size_mb:.2f} MB")
#     except Exception as e:
#         logging.error(f"Failed to save compressed model to '{path}': {e}")
#         raise e

# def load_compressed_model(path, model_config):
#     """
#     Loads a compressed model from the specified path.

#     Args:
#         path (str): The file path of the compressed model.
#         model_config (dict): Configuration parameters for the model.

#     Returns:
#         torch.nn.Module: The loaded model with pruned weights.
#     """
#     try:
#         with gzip.GzipFile(path, 'r') as f:
#             loaded_sd = pickle.load(f)
#         new_model = GarmentClassifier(
#             CONFIG['input_dims'],
#             CONFIG['hidden_feature_dims'],
#             CONFIG['output_classes'],
#             **model_config
#         )
#         new_model.load_state_dict(loaded_sd)
#         logging.info(f"Loaded compressed model from '{path}'.")
#         return new_model
#     except Exception as e:
#         logging.error(f"Failed to load compressed model from '{path}': {e}")
#         raise e

# def get_pruning_params(model):
#     """
#     Prepares the list of parameters to prune in the model.

#     Args:
#         model (torch.nn.Module): The model whose parameters are to be pruned.

#     Returns:
#         list: List of tuples specifying (module, parameter_name) to prune.
#     """
#     prune_params = []

#     # Add input layer
#     prune_params.append((model.input_layer, 'weight'))

#     # Add all hidden layers
#     for hidden_layer in model.hidden_layers:
#         prune_params.append((hidden_layer, 'weight'))

#     # Add output layer
#     prune_params.append((model.output_layer, 'weight'))

#     return prune_params

# def calculate_sparsity(model):
#     """
#     Calculates the overall sparsity of the model.

#     Args:
#         model (torch.nn.Module): The model to evaluate.

#     Returns:
#         float: Sparsity ratio as a float between 0 and 1.
#     """
#     zero_params = 0
#     total_params = 0
#     for param in model.parameters():
#         if param.requires_grad:
#             zero_params += torch.sum(param == 0).item()
#             total_params += param.numel()
#     return zero_params / total_params if total_params > 0 else 0

# def verify_pruning(model):
#     """
#     Verifies and logs the sparsity of each layer in the model.

#     Args:
#         model (torch.nn.Module): The model to verify.
#     """
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Linear):
#             weight = module.weight.data
#             zero_elements = torch.sum(weight == 0).item()
#             total_elements = weight.numel()
#             sparsity = 100. * zero_elements / total_elements
#             logging.info(f"{name} - Sparsity: {sparsity:.2f}%")

# def main():
#     transform_types = [CONFIG['transform_type']]
#     model_config = {
#         'num_hidden_layers': CONFIG['num_hidden_layers'],
#         'hidden_layer_width': CONFIG['hidden_layer_width']
#     }
#     results = {}

#     for test_batch_size in CONFIG['test_batch_list']:
#         CONFIG['test_batch_size'] = test_batch_size
#         logging.info(f"Running experiment with test batch size: {test_batch_size}")
        
#         for transform_type in transform_types:
#             config_name = f"{transform_type}_depth{model_config['num_hidden_layers']}_width{model_config['hidden_layer_width'] or 'default'}"
#             logging.info(f"Running experiment with config: {config_name}")
#             CONFIG['transform_type'] = transform_type
            
#             try:
#                 train_loader, val_loader, test_loader = get_dataloaders(
#                     CONFIG, transform_type=CONFIG['transform_type']
#                 )
#             except Exception as e:
#                 logging.error(f"Failed to get data loaders: {e}")
#                 continue

#             try:
#                 model = GarmentClassifier(
#                     CONFIG['input_dims'],
#                     CONFIG['hidden_feature_dims'],
#                     CONFIG['output_classes'],
#                     **model_config
#                 )
#                 logging.info("Initialized GarmentClassifier.")
#             except Exception as e:
#                 logging.error(f"Failed to initialize model: {e}")
#                 continue

#             # Measure model size before pruning
#             full_size = print_size_of_model(model, "model_before_pruning")

#             # Prepare pruning parameters
#             prune_params = get_pruning_params(model)

#             # Prune the model with amount=0.33
#             prune_model(model, prune_params, amount=CONFIG['pruning_amount'])

#             # Make pruning permanent
#             remove_pruning_reparameterization(prune_params)

#             # Calculate sparsity
#             sparsity = calculate_sparsity(model)
#             logging.info(f"Model sparsity after pruning: {sparsity * 100:.2f}%")

#             # Save the model in dense format with compression
#             try:
#                 save_compressed_model(model.state_dict(), "model_compressed.pt.gz")
#             except Exception as e:
#                 logging.error(f"Failed to save compressed model: {e}")
#                 continue

#             # Load the compressed model
#             try:
#                 new_model = load_compressed_model("model_compressed.pt.gz", model_config)
#             except Exception as e:
#                 logging.error(f"Failed to load model: {e}")
#                 continue

#             # Verify pruning effectiveness
#             verify_pruning(new_model)

#             # Continue with training and evaluation
#             loss_fn = torch.nn.CrossEntropyLoss()
#             optimizer = torch.optim.Adam(new_model.parameters(), lr=CONFIG['learning_rate'])
            
#             try:
#                 epoch_results = train_model(
#                     CONFIG, model_config, new_model,
#                     train_loader, val_loader, test_loader,
#                     loss_fn, optimizer
#                 )
#             except Exception as e:
#                 logging.error(f"Training failed: {e}")
#                 continue

#             try:
#                 total_params = count_parameters(new_model)
#                 flops = compute_flops(new_model)
#             except Exception as e:
#                 logging.error(f"Failed to compute metrics: {e}")
#                 total_params = None
#                 flops = None

#             results[config_name] = {
#                 'epoch_results': epoch_results,
#                 'total_params': total_params,
#                 'flops': flops
#             }

#     # Save the results to a JSON file
#     try:
#         with open('results.json', 'w') as f:
#             json.dump(results, f, indent=4)
#         logging.info("Saved all results to 'results.json'.")
#     except Exception as e:
#         logging.error(f"Failed to save results: {e}")

# if __name__ == "__main__":
#     main()




########################################################################################
########################################################################################
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


