import numpy as np
import pandas as pd
import torch; torch.manual_seed(42)
from torch import nn

def get_arch(input_size, hidden_dims, activation_function = nn.ReLU()):
    """
    Util to get different architectures for NN
    """
    layers = []
    for idx, d in enumerate(hidden_dims):
        if idx == 0:
            layers.append(nn.Linear(input_size, d))
            layers.append(activation_function)
        else:
            layers.append(nn.Linear(hidden_dims[idx-1], d))
            layers.append(activation_function)
        
    layers.append(nn.Linear(in_features = hidden_dims[-1] , out_features = 1))
    # layers.append(nn.Softplus())
    
    return nn.Sequential(*layers)