import time
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from utils import train_one_epoch, validate
from evaluation import evaluate_model, plot_metrics

def train_model(config, model, train_loader, val_loader, loss_fn, optimizer):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/fashion_trainer_{timestamp}')
    best_vloss = float('inf')
    epoch_number = 0
    total_training_time = 0.0
    total_inference_time = 0.0
    training_times = []
    inference_times_without_warmup = []

    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    for epoch in range(config['epochs']):
        print(f'=== EPOCH {epoch + 1} ===')
        start_train_time = time.time()
        avg_loss = train_one_epoch(epoch_number, model, train_loader, optimizer, loss_fn, writer)
        end_train_time = time.time()
        epoch_training_time = end_train_time - start_train_time
        total_training_time += epoch_training_time
        training_times.append(epoch_training_time)
        print(f'Epoch {epoch + 1} Training Time: {epoch_training_time:.4f} seconds')

        avg_vloss, validation_accuracy = validate(model, val_loader, loss_fn)
        print(f'LOSS - Train: {avg_loss:.4f}, Validation: {avg_vloss:.4f}')
        print(f'Epoch {epoch + 1} Validation Accuracy: {validation_accuracy:.4f}')

        writer.add_scalars('Training vs. Validation Loss', {'Training': avg_loss, 'Validation': avg_vloss}, epoch + 1)
        writer.add_scalars('Accuracy', {'Training': 0, 'Validation': validation_accuracy}, epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'model_{timestamp}_epoch_{epoch + 1}.pth'
            # torch.save(model.state_dict(), model_path)
            # print(f"Model saved at {model_path} with validation loss {avg_vloss:.4f}")

        # Store metrics for plotting
        training_losses.append(avg_loss)
        validation_losses.append(avg_vloss)
        training_accuracies.append(0)  # Placeholder, replace with actual training accuracy if available
        validation_accuracies.append(validation_accuracy)

        epoch_number += 1

    print(f'Total Training Time: {total_training_time:.4f} seconds')
    print(f'Total Inference Time (excluding warm-up): {total_inference_time:.4f} seconds')

    # Plot metrics
    plot_metrics(training_losses, validation_losses, training_accuracies, validation_accuracies)