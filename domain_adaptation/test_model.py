# -*- coding: utf-8 -*-
"""
Created on 2024-09-27 (Fri) 14:04:26

@author: I.Azuma
"""
# %%
BASE_DIR = '/workspace/mnt/cluster/HDD/azuma/TopicModel_Deconv'

import gc
import time
import scanpy as sc
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from matplotlib import colors as mcolors
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import sys
sys.path.append(BASE_DIR+'/github/GSTMDec')
from domain_adaptation import common_utils
from nsem_gmhtm_deconvolution import nsem_gmhtm_dec_dev5 as nsem_gmhtm_dec


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def main(test_x, test_y, target_genes, model_path='/path/to/checkpoint.pth'):
    _, d = test_x.shape
    #target_genes = adata.var_names
    vocab_dict = dict(zip([i for i in range(d)], target_genes))

    epochs = 1000
    batch_size = 1024  
    lr = 0.01

    topic_num_1 = 8
    topic_num_2 = 32
    topic_num_3 = 64
    hidden_num = 128

    model = nsem_gmhtm_dec.AMM_no_dag(reader=None, vocab_dict=vocab_dict, model_path=None, 
                                    word_embed=None, topic_num_1=topic_num_1, topic_num_2=topic_num_2, topic_num_3=topic_num_3, 
                                    vocab_num=d, epochs=epochs, hidden_num=hidden_num, batch_size=batch_size, learning_rate=lr)
    # load model checkpoint
    chekcpoint = torch.load(model_path)
    model.Net.load_state_dict(chekcpoint)

    # evaluate test data 1
    deconv_df, y_df, corr_df = common_utils.eval_data(model, test_x, test_y)
    dec_name_list = [[0],[1],[2],[3],[4]]
    val_name_list = [["Monocytes"],["CD4Tcells"],["Bcells"],["NK"],["CD8Tcells"]]
    # flatten
    flat_dec = []
    for dec in dec_name_list:
        flat_dec.extend(dec)
    deconv_df = deconv_df[flat_dec]
    deconv_df = deconv_df.div(deconv_df.sum(axis=1), axis=0)  # normalize to make it sum to 1 (row-wise)

    flat_val = []
    for val in val_name_list:
        flat_val.extend(val)
    y_df = y_df[flat_val]
    y_df = y_df.div(y_df.sum(axis=1), axis=0)  # normalize to make it sum to 1 (row-wise)

    overall_res = common_utils.eval_deconv(dec_name_list = dec_name_list, val_name_list = val_name_list, deconv_df=deconv_df, y_df=y_df)

    # summarize the results
    for i, cell in enumerate(val_name_list):
        print(f"{cell[0]}: R={overall_res[i][0]['R']}")

