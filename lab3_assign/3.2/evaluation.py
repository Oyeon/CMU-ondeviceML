import torch
import time
import os

def evaluation(model, val_loader, loss_fn, device='cpu', input_dims=(3, 224, 224)):
    """
    Validates the model on the validation dataset.
    
    Args:
        model (torch.nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for validation data.
        loss_fn (callable): Loss function.
        device (str, optional): Device to perform inference on. Defaults to 'cpu'.
        input_dims (tuple, optional): Dimensions of the input tensor for FLOPs calculation. Defaults to (3, 224, 224).
    
    Returns:
        tuple: Average validation loss, validation accuracy, and list of inference times.
    """
    model.to(device)
    model.eval()
    
    running_vloss = 0.0
    running_vcorrects = 0
    total_vsamples = 0
    inference_times = []  # List to store inference times

    with torch.no_grad():
        for vinputs, vlabels in val_loader:
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)  # Move data to the specified device
            start_time = time.time()  # Start time for inference latency
            voutputs = model(vinputs)
            inference_time = time.time() - start_time  # End time for inference latency
            inference_times.append(inference_time)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()
            _, vpreds = torch.max(voutputs, 1)
            running_vcorrects += torch.sum(vpreds == vlabels).item()
            total_vsamples += vlabels.size(0)
    
    avg_vloss = running_vloss / len(val_loader)
    validation_accuracy = running_vcorrects / total_vsamples
    return avg_vloss, validation_accuracy, inference_times