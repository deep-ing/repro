import torch 
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self, abstract_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3,3,1)
        self.conv2 = nn.Conv2d(32, 16, 3,3,1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(576, abstract_dim)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
         
        