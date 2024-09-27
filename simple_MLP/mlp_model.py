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
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.feature_extractor = nn.Linear(input_size, hidden_size1)
        self.hidden_layer = nn.Linear(hidden_size1, hidden_size2)
        self.classifier = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        features = self.extract_features(x)
        x = torch.relu(self.hidden_layer(features))
        logits = self.classifier(x)
        return logits
    
    def extract_features(self, x):
        x = torch.relu(self.feature_extractor(x))
        return x

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
    
    def forward(self, outputs, targets):
        # normalize outputs and targets
        outputs_normalized = torch.nn.functional.normalize(outputs, p=2, dim=1)
        targets_normalized = torch.nn.functional.normalize(targets, p=2, dim=1)
        
        # calculate cosine similarity
        similarity = self.cosine_similarity(outputs_normalized, targets_normalized)
        
        # loss = 1 - similarity
        loss = 1 - similarity.mean()
        return loss

def calc_deconv_loss(theta_tensor, prop_tensor):
    ext_theta = theta_tensor[:,0:prop_tensor.shape[1]]

    mse = torch.mean((ext_theta - prop_tensor) ** 2)
    rmse = torch.sqrt(mse)
    cel = -torch.mean(torch.log(ext_theta) * prop_tensor)

    cos_sim = 1 - F.cosine_similarity(ext_theta, prop_tensor)
    cos_sim = cos_sim.mean()

    return {'mse':mse, 'rmse':rmse, 'cel':cel, 'cos_sim':cos_sim}