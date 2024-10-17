import time
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import train_one_epoch, validate
from evaluation import evaluation #,plot_metrics
from model import GarmentClassifier, count_parameters, compute_flops
from thop import profile
from copy import deepcopy
import os
import torch.nn as nn
import numpy as np

import torch.quantization
from torch.ao.quantization import get_default_qconfig_mapping, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx



def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size

def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(config, model_config, model, train_loader, val_loader, test_loader, loss_fn, optimizer):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
    best_vloss = float('inf')
    epoch_number = 0
    total_training_time = 0.0
    total_inference_time = 0.0
    training_times = []
    inference_times_list = []
    inference_times_all = []
    loss_list = []

    total_params = count_parameters(model)
    flops = compute_flops(model)    

    epoch_results = {}

    for epoch in range(config['epochs']):
        print(f'=== EPOCH {epoch + 1} ===')
        avg_loss, train_accuracy, epoch_time, avg_vloss, validation_accuracy = train_one_epoch(epoch_number, model, train_loader, val_loader, optimizer, loss_fn, writer)
        training_times.append(epoch_time)
        total_training_time += epoch_time
        print(f'Epoch {epoch + 1} Training Time: {epoch_time:.4f} seconds')

        if config['quantization'] == 'dynamic':
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
        elif config['quantization'] == 'static':
            qconfig_mapping = get_default_qconfig_mapping("fbgemm")
            example_inputs = (torch.randn(1, config['input_dims']),)
            prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
            
            # Calibrate the model with test data
            model.eval()  
            with torch.no_grad():
                for inputs, _ in test_loader:
                    prepared_model(inputs)  # Forward pass to calibrate

            quantized_model = convert_fx(prepared_model)  
        elif config['quantization'] == 'qat':
            qconfig_mapping = get_default_qat_qconfig_mapping("fbgemm")
            example_inputs = (torch.randn(1, config['input_dims']),)
            prepared_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)
            # Train the model with QAT
            for _ in range(config['qat_epochs']):  # Add a new config for QAT epochs
                avg_loss, train_accuracy, epoch_time, avg_vloss, validation_accuracy = train_one_epoch(epoch_number, prepared_model, train_loader, val_loader, optimizer, loss_fn, writer)
            quantized_model = convert_fx(prepared_model)  # Convert the model after QAT
        else:
            quantized_model = model

        inference_times = []
        for _ in range(5):
            avg_tloss, test_accuracy, times = evaluation(quantized_model, test_loader, loss_fn, device=config['inference_device'], input_dims=config['input_dims'])
            inference_times.extend(times)
        
        avg_inference_time = np.mean(inference_times) #sum(inference_times) / len(inference_times)
        std_inference_time = np.std(inference_times) #(sum((x - avg_inference_time) ** 2 for x in inference_times) / len(inference_times)) ** 0.5

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_checkpoints/model_{timestamp}_epoch_{epoch + 1}.pth'

            if config['model_save']:
                torch.save(model.state_dict(), model_path)

            print(f'Epoch {epoch + 1} Average Inference Time: {avg_inference_time * 1e6:.2f} µs')
            print(f'Epoch {epoch + 1} Inference Time Std Dev: {std_inference_time * 1e6:.2f} µs')
            print(f'Epoch {epoch + 1} Training Accuracy: {train_accuracy:.4f}')
            print(f'Epoch {epoch + 1} Test Accuracy: {test_accuracy:.4f}')

            print_size_of_model(quantized_model, "INT8")
            total_params = count_parameters(quantized_model)
            print(f'Total Parameters: {total_params}')

            

    return epoch_results

