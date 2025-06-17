import numpy as np
import torch
import torch.nn as nn
import torch.func as func

class ACnet(nn.Module):
    def __init__(self, dim_in, width, dim_out, beta, bias = False):
        super(ACnet, self).__init__()
        self.dim_in = dim_in
        self.width = width
        self.beta = beta
        self.dim_out = dim_out
        self.bias = bias # final bias

        self.wb = nn.Linear(self.dim_in, self.width)
        self.c = nn.Linear(self.width, self.dim_out, bias=bias)
        self.c.weight.data = torch.as_tensor(np.random.uniform(-1, 1, size=self.c.weight.shape), dtype=torch.float32)*(self.width**(-self.beta))
        self.wb.weight.data = torch.as_tensor(np.random.normal(0, 1, size=self.wb.weight.shape),  dtype=torch.float32)
        self.wb.bias.data = torch.as_tensor(np.random.normal(0, 1, size=self.wb.bias.shape) , dtype=torch.float32)

    def forward(self, x):
        x = self.wb(x)
        x = torch.sigmoid(x)
        x = self.c(x)
        return x

#generic multi-layer feedforward neural network that does not have compact final weight support
class SimpleNN(nn.Module):
    def __init__(self, dim_in, num_layers, num_neurons, dim_out, bias = False):
        super(SimpleNN, self).__init__()
        self.dim_in = dim_in
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.dim_out = dim_out
        self.bias = bias #final bias

        layers = []

        # input layer
        layers.append(nn.Linear(self.dim_in, self.num_neurons))

        # hidden layers
        for _ in range(num_layers-1):
            layers.extend([nn.Sigmoid(), nn.Linear(self.num_neurons, self.num_neurons)])

        # output layer
        layers.extend([nn.Sigmoid(), nn.Linear(self.num_neurons, self.dim_out, bias = bias)])

        # build the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(1)
        return self.network(x).squeeze()