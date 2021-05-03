import torch
from torch import nn


class RegressionNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_size: int = 1024,
                 emb_size: int = 1024,
                 output_dim: int = 1,
                 activation_function: str = 'erf'
                 ):
        super(RegressionNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, emb_size)
        self.regression = nn.Linear(emb_size, output_dim, bias=False)
        self.activation = torch.relu
        if activation_function == 'erf':
            self.activation = torch.erf
        elif activation_function == 'relu':
            self.activation = torch.relu
        elif activation_function == 'tanh':
            self.activation = torch.tanh
        else:
            raise ValueError('Undefined activation function')
        self.activation_function = activation_function

    def forward(self, x):
        feature = self.extract_feature(x)
        mu = self.regression(feature)
        return mu

    def extract_feature(self, x):
        h1 = self.fc1(x)
        x1 = self.activation(h1)
        h2 = self.fc2(x1)
        if self.activation_function == 'relu':
            h2 = self.activation(h2)
        # x2 = torch.tanh(h2)
        return h2
