import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class Phrases(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = self.data[idx][0]  # Get the features from the input data
        label = self.data[idx][1]     # Get the label from the input data
        
        # Perform any necessary preprocessing or transformations on the features and label
        # For example, you can convert them to tensors
        features = torch.Tensor(features)
        label = torch.Tensor([label])
        
        return features, label

class rev1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.RNN(input_size, hidden_size)
        self.fc2 = nn.RNN(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

batch_size = 8
