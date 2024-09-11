import torch
import matplotlib.pyplot as plt

def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = running_corrects / total_samples
    return avg_loss, accuracy

# def plot_metrics(training_losses, validation_losses, training_accuracies, validation_accuracies):
#     epochs = range(1, len(training_losses) + 1)

#     plt.figure(figsize=(12, 5))

#     # Plot training and validation loss
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, training_losses, 'r', label='Training loss')
#     plt.plot(epochs, validation_losses, 'b', label='Validation loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()

#     # Plot training and validation accuracy
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, training_accuracies, 'r', label='Training accuracy')
#     plt.plot(epochs, validation_accuracies, 'b', label='Validation accuracy')
#     plt.title('Training and Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     plt.tight_layout()
#     plt.show()