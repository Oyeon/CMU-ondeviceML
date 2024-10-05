import time
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import train_one_epoch, validate
from evaluation import evaluate_model #,plot_metrics
from model import GarmentClassifier, count_parameters, compute_flops
from thop import profile
from copy import deepcopy

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
    # flops = compute_flops(model, input_size=(1, config['input_dims']))

    epoch_results = {}

    for epoch in range(config['epochs']):
        print(f'=== EPOCH {epoch + 1} ===')
        avg_loss, train_accuracy, epoch_time, avg_vloss, validation_accuracy = train_one_epoch(epoch_number, model, train_loader, val_loader, optimizer, loss_fn, writer)
        training_times.append(epoch_time)
        total_training_time += epoch_time
        print(f'Epoch {epoch + 1} Training Time: {epoch_time:.4f} seconds')

        avg_tloss, test_accuracy, inference_times = validate(model, test_loader, loss_fn)
        inference_times_list.append(inference_times)
        # inference_times.append(avg_inference_time)
        # inference_times_all.append(avg_inference_time_all)
        # total_inference_time += avg_inference_time
        avg_inference_time = sum(inference_times) / (len(inference_times))
        print(f'LOSS - Train: {avg_loss:.4f}, Test: {avg_tloss:.4f}')
        print(f'Epoch {epoch + 1} Training Accuracy: {train_accuracy:.4f}')
        print(f'Epoch {epoch + 1} Validation Accuracy: {validation_accuracy:.4f}')
        print(f'Epoch {epoch + 1} Test Accuracy: {test_accuracy:.4f}')
        print(f'Epoch {epoch + 1} Average Inference Time: {avg_inference_time:.4f} seconds')
        # print(f'Epoch {epoch + 1} Average Inference Time (all batches): {avg_inference_time_all:.4f} seconds')

        writer.add_scalars('Training vs. Test Loss', {'Training': avg_loss, 'Test': avg_tloss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Training': train_accuracy, 'Test': test_accuracy}, epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_checkpoints/model_{timestamp}_epoch_{epoch + 1}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path} with Test loss {avg_tloss:.4f}")

            dummy_input = torch.randn(1, config['input_dims'])
            onnx_model_path = f'model_checkpoints/model_{timestamp}_epoch_{epoch + 1}.onnx'

            model.eval()  # Set to evaluation mode before exporting
            torch.onnx.export(
                model,
                dummy_input,
                onnx_model_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"ONNX model exported at {onnx_model_path}")


        # Store metrics for plotting
        total_train_latency = sum(training_times)
        # total_inference_latency = sum(inference_times) / (len(inference_times))
        # total_inference_latency_all = sum(inference_times_all)
        print(f'Depth: {model_config["num_hidden_layers"]}, FLOPs: {flops}, Params: {total_params}, Train Acc: {train_accuracy}, Test Acc: {test_accuracy}, Train Latency: {total_train_latency / (epoch + 1)}, Inference Latency: {avg_inference_time}')

        # ---- Multiple-Batch Inference ----
        batch_sizes = [4, 8, 16, 32, 64]
        batch_latency_list = []
        batch_flops_list = []

        for batch_size in batch_sizes:
            model.eval()
            test_loader_iter = iter(test_loader)
            batch_latencies = []
            batch_flops_list = []

            # Iterate over multiple batches
            for _ in range(len(test_loader)):
                inputs, labels = next(test_loader_iter)
                inputs = inputs[:batch_size]
                labels = labels[:batch_size]

                with torch.no_grad():
                    start_batch_time = time.time()  # Start timing batch inference

                    # Forward pass for batch
                    outputs = model(inputs)

                    end_batch_time = time.time()  # End timing
                    batch_latency = end_batch_time - start_batch_time

                    # Calculate FLOPs for batch inference
                    # batch_flops, _ = profile(model, inputs=(inputs,), verbose=False)

                # Store latency and FLOPs, excluding the first batch
                if _ > 0:  # Skip the first batch
                    batch_latencies.append(batch_latency)
                    # batch_flops_list.append(batch_flops)

            # Calculate the average latency and FLOPs for the remaining batches
            avg_batch_latency = sum(batch_latencies) / len(batch_latencies)
            # avg_batch_flops = sum(batch_flops_list) / len(batch_flops_list)

            # Store the average latency and FLOPs
            batch_latency_list.append(avg_batch_latency)
            # batch_flops_list.append(batch_flops)

            # print(f'Batch Inference for batch size {batch_size}: FLOPs: {batch_flops}, Latency: {batch_latency:.6f} seconds')

        # Store epoch results
        epoch_results[epoch + 1] = {
            'flops': flops,
            'params': total_params,
            'training_loss': avg_loss,
            'training_accuracy': train_accuracy,
            'validation_loss': avg_vloss,
            'validation_accuracy': validation_accuracy,            
            'test_loss': avg_tloss,            
            'test_accuracy': test_accuracy,
            'training_time': epoch_time,
            'average_inference_time': avg_inference_time,                        
            'batch_latency_list': batch_latency_list,
            'inference_time': inference_times_list,            
        }

        # Create a deepcopy of epoch_results without the 'inference_time' key
        display_results = deepcopy(epoch_results[epoch + 1])
        del display_results['inference_time']

        print('epoch_results', display_results)        
        # print('epoch_results', epoch_results)

    return epoch_results

