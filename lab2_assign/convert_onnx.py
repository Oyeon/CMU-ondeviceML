import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import time
import sys
import matplotlib.pyplot as plt
import numpy as np

img_size = (28, 28)
num_labels = 10
learning_rate = 1e-3
batch_size = 64
num_layers = 2
hidden_size = 1024
num_epochs = 2
# print(type(img_size[0] * img_size[1]))
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class MNISTNetwork(nn.Module):
    def __init__(self, image_size=img_size, hidden_layers=2, hidden_size=1024, num_labels=10):
        super(MNISTNetwork, self).__init__()
        # First layer input size must be the dimension of the image
        self.flatten = nn.Flatten()
        # Define NN layers based on the number of layers and hidden size
        flatten_size = image_size[0] * image_size[1] # int
        self.NN_layers = []
        self.NN_layers.append(flatten_size) # first element is input
        for i in range(hidden_layers):
            self.NN_layers.append(hidden_size)
        self.NN_layers.append(num_labels) # output layer size
        NN = []
        for i in range(len(self.NN_layers)-1):
            # [784, 1024] -> ReLU -> [1024, 1024] -> ReLU -> [1024, 10]
            NN.append(nn.Linear(self.NN_layers[i],self.NN_layers[i+1]))
            if i < (hidden_layers):
                NN.append(nn.ReLU())
        self.sequential = nn.Sequential(*NN)
                
    def forward(self, x):
        x = self.flatten(x)
        logits = self.sequential(x)
        return logits
    
    def countflops(self):
        # Count FLOPs per layer [784,1024,1024,10]
        # Linear layer: Ax + b
        # ReLU: b
        # dim(A)[0] * dim(A)[1] + dim(B) + dim(B)
        flop_count = 0
        for i in range(len(self.NN_layers)-1): # 0 to 2
            A = self.NN_layers[i]
            B = self.NN_layers[i+1]
            flop_count += A*A*B+B # linear layer
            if i < (len(self.NN_layers)-2): # before output layer no ReLU
                flop_count += 2*B
        print(f"FLOPs: {flop_count}")
        return flop_count


#Function to Convert to ONNX 
def convert(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(64, 1, 28, 28, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "MNIST_base.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes 
                                'output' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')




model = MNISTNetwork()
model.load_state_dict(torch.load('./MNIST_model.pth'))
convert()