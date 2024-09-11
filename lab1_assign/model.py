import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG
import torch
from thop import profile

class GarmentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=2, hidden_layer_width=None):
        super(GarmentClassifier, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        if hidden_layer_width is None:
            hidden_layer_width = hidden_size
        
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(hidden_size, hidden_layer_width))
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
        
        self.output_layer = nn.Linear(hidden_layer_width, output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor dynamically
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_flops(model):
    input_tensor = torch.randn(1, CONFIG['input_dims'])  # Ensure this matches the input dimensions
    flops, _ = profile(model, inputs=(input_tensor,), verbose=False)
    return flops


# class GarmentClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(GarmentClassifier, self).__init__()
#         self.input_layer = nn.Linear(input_size, hidden_size)
#         self.fc1 = nn.Linear(hidden_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.output_layer = nn.Linear(hidden_size, output_size)
        
#     def forward(self, x):
#         x = x.view(-1, self.input_layer.in_features)  # Use dynamic input dimensions
#         x = F.relu(self.input_layer(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.output_layer(x)
#         return x