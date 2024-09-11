import time
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import train_one_epoch, validate
from evaluation import evaluate_model #,plot_metrics
from model import GarmentClassifier, count_parameters, compute_flops
from thop import profile

def train_model(config, model_config, model, train_loader, val_loader, loss_fn, optimizer):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
    best_vloss = float('inf')
    epoch_number = 0
    total_training_time = 0.0
    total_inference_time = 0.0
    training_times = []
    inference_times = []

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    single_batch_latency_list = []
    single_batch_flops_list = []
    total_params = count_parameters(model)
    flops = compute_flops(model)    
    # flops = compute_flops(model, input_size=(1, config['input_dims']))

    for epoch in range(config['epochs']):
        print(f'=== EPOCH {epoch + 1} ===')
        avg_loss, train_accuracy, epoch_time = train_one_epoch(epoch_number, model, train_loader, optimizer, loss_fn, writer)
        training_times.append(epoch_time)
        total_training_time += epoch_time
        print(f'Epoch {epoch + 1} Training Time: {epoch_time:.4f} seconds')

        avg_vloss, validation_accuracy, avg_inference_time = validate(model, val_loader, loss_fn)
        inference_times.append(avg_inference_time)
        total_inference_time += avg_inference_time
        print(f'LOSS - Train: {avg_loss:.4f}, Validation: {avg_vloss:.4f}')
        print(f'Epoch {epoch + 1} Training Accuracy: {train_accuracy:.4f}')
        print(f'Epoch {epoch + 1} Validation Accuracy: {validation_accuracy:.4f}')
        print(f'Epoch {epoch + 1} Average Inference Time: {avg_inference_time:.4f} seconds')

        writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Training': train_accuracy, 'Validation': validation_accuracy}, epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_{timestamp}_epoch_{epoch + 1}.pth'
            # torch.save(model.state_dict(), model_path)
            # print(f"Model saved at {model_path} with validation loss {avg_vloss:.4f}")

        # Store metrics for plotting
        training_losses.append(avg_loss)
        validation_losses.append(avg_vloss)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        total_train_latency = sum(training_times)
        total_inference_latency = sum(inference_times)
        print(f'Depth: {model_config["num_hidden_layers"]}, FLOPs: {flops}, Params: {total_params}, Train Acc: {train_accuracy}, Val Acc: {validation_accuracy}, Train Latency: {total_train_latency / (epoch + 1)}, Inference Latency: {total_inference_latency / len(val_loader)}')

        epoch_number += 1

        # ---- Single-Batch Inference ----
        # Perform single-batch inference and measure FLOPs and latency
        model.eval()
        single_batch = next(iter(val_loader))
        inputs, labels = single_batch

        with torch.no_grad():
            start_single_batch_time = time.time()  # Start timing single-batch inference

            # Forward pass for single batch
            outputs = model(inputs)

            end_single_batch_time = time.time()  # End timing
            single_batch_latency = end_single_batch_time - start_single_batch_time

            # Calculate FLOPs for single-batch inference
            single_batch_flops, _ = profile(model, inputs=(inputs,), verbose=False)

        # Store latency and FLOPs
        single_batch_latency_list.append(single_batch_latency)
        single_batch_flops_list.append(single_batch_flops)

        print(f'Single-Batch Inference for {model_config["num_hidden_layers"]} layers: FLOPs: {single_batch_flops}, Latency: {single_batch_latency:.6f} seconds')

    return training_losses, validation_losses, training_accuracies, validation_accuracies, training_times, inference_times