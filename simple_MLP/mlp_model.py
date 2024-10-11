# -*- coding: utf-8 -*-
"""
Created on 2024-09-27 (Fri) 22:41:54

@author: I.Azuma
"""
# %%
import gc
import time
import scanpy as sc
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
import torch.optim as optim

# %%
# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        super(MLP, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# Function to freeze layers
def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

