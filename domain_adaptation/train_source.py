# -*- coding: utf-8 -*-
"""
Created on 2024-09-27 (Fri) 12:51:19

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
sys.path.append(BASE_DIR+'/github/GSTMDec/nsem_gmhtm_deconvolution')

import utils
#import nsem_gmhtm_dec_dev4 as nsem_gmhtm_dec
import nsem_gmhtm_dec_dev5 as nsem_gmhtm_dec
from learning_utils import *
from reader import TextReader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def summarize_loss(model, loss_dic, prop_df):
    # loss of Gaussian Mixture VAE
    gmvae_loss = loss_dic['recon_loss'] + loss_dic['gauss_loss'] #+ loss_dic['cat_loss']
    # sparse loss
    sparse_loss = (
        1 * torch.mean(torch.abs(model.Net.adj_A))
        + 1 * torch.mean(torch.abs(model.Net.adj_A_2))
        + 1 * torch.mean(torch.abs(model.Net.adj_A_3))
    )
    # deconvolution loss
    theta_tensor = model.Net.get_doc_topic_dist(level=1)  # NOTE: middle layer
    prop_tensor = torch.tensor(prop_df.values).to(device)

    assert theta_tensor.shape[0] == prop_tensor.shape[0], "Batch size is different"
    deconv_loss_dic = calc_deconv_loss(theta_tensor, prop_tensor)
    deconv_loss = 1.0*deconv_loss_dic['cos_sim'] + 0.0*deconv_loss_dic['rmse']

    # scaling
    gmvae_loss = 0.01*gmvae_loss
    sparse_loss = 1*sparse_loss
    deconv_loss = 100*deconv_loss
    loss = gmvae_loss + sparse_loss + deconv_loss

    return (loss, gmvae_loss, sparse_loss, deconv_loss)
    

def main(train_x, train_y, valid_x, valid_y, target_genes, model_path):
    """ 
    NOTE: train_x and valid_x should be from the same domain.
    """
    _, d = train_x.shape
    #target_genes = adata.var_names
    vocab_dict = dict(zip([i for i in range(d)], target_genes))

    epochs = 1000
    batch_size = 1024  
    lr = 0.01

    topic_num_1 = 8
    topic_num_2 = 32
    topic_num_3 = 64
    hidden_num = 128

    # set model
    model = nsem_gmhtm_dec.AMM_no_dag(reader=None, vocab_dict=vocab_dict, model_path=model_path, 
        word_embed=None, topic_num_1=topic_num_1, topic_num_2=topic_num_2, topic_num_3=topic_num_3, 
        vocab_num=d, epochs=epochs, hidden_num=hidden_num, batch_size=batch_size, learning_rate=lr)
    
    train_generator = utils.get_batches(train_data=train_x, batch_size=model.batch_size, device=device, rand=True)  # NOTE: random sampling

    data_size = train_x.shape[0]
    n_batchs = data_size // model.batch_size
    if n_batchs == 0:
        n_batchs = 1
    
    
    # optimizer settings
    optimizer = optim.Adam(model.Net.parameters(), lr=model.lr)
    optimizer2 = optim.Adam(
        [model.Net.adj_A, model.Net.adj_A_2, model.Net.adj_A_3], lr=model.lr * 0.2
    )
    # scheduler settings
    scheduler = StepLR(optimizer2, step_size=100, gamma=0.1)

    best_loss = 1e10
    clipper = WeightClipper(frequency=1)
    loss_history = []
    gmvae_loss_history = []
    sparse_loss_history = []
    deconv_loss_history = []

    t_begin = time.time()
    for epoch in tqdm(range(model.epochs)):

        model.Net.train()
        epoch_word_all = 0

        if epoch % (3) < 1:  #
            model.Net.adj_A.requires_grad = False
            model.Net.adj_A_2.requires_grad = False
            model.Net.adj_A_3.requires_grad = False

        else:
            model.Net.adj_A.requires_grad = True
            model.Net.adj_A_2.requires_grad = True
            model.Net.adj_A_3.requires_grad = True
        
        # Training loop
        running_loss = 0.0
        running_gmvae_loss = 0.0
        running_sparse_loss = 0.0
        running_deconv_loss = 0.0

        for i in range(n_batchs):
            # Training loop
            optimizer.zero_grad()
            optimizer2.zero_grad()
            temperature = max(0.95 ** epoch, 0.5)

            batch_idx, ori_train = next(train_generator)
            train_y_batch = train_y.iloc[batch_idx]

            # loss of training data
            train_loss_dic, dep_mats = model.Net(
                ori_train, temperature = temperature
            )
            (loss_t, gmvae_loss_t, sparse_loss_t, deconv_loss_t) = summarize_loss(model, train_loss_dic, train_y_batch)
            
            # backpropagation
            total_loss = gmvae_loss_t + sparse_loss_t + deconv_loss_t
            total_loss.backward()

            # add each loss to runnung loss
            running_loss += total_loss.item()
            running_gmvae_loss += gmvae_loss_t.item()
            running_sparse_loss += sparse_loss_t.item()
            running_deconv_loss += deconv_loss_t.item()


        if epoch % (3) < 1:
            optimizer.step()
        else:
            optimizer2.step()
        if True:
            model.Net.generative.decoder[0].apply(clipper)
            model.Net.generative.decoder_2[0].apply(clipper)
        scheduler.step()

        # evaluation
        model.Net.eval()
        with torch.no_grad():
            ori_valid = torch.from_numpy(valid_x).to(device)
            valid_loss_dic, dep_mats = model.Net(ori_valid)
            (loss_v, gmvae_loss_v, sparse_loss_v, deconv_loss_v) = summarize_loss(model, valid_loss_dic, valid_y)
            total_loss_v = gmvae_loss_v + sparse_loss_v + deconv_loss_v

        if epoch+1 >= 20: # skip initial N epochs
            loss_history.append((loss_t.item(), loss_v.item()))  # collect both train and valid loss
            gmvae_loss_history.append((gmvae_loss_t.item(), gmvae_loss_v.item()))
            sparse_loss_history.append((sparse_loss_t.item(), sparse_loss_v.item()))
            deconv_loss_history.append((deconv_loss_t.item(), deconv_loss_v.item()))
        
        if (epoch + 1) % 50 == 0:
            if total_loss_v < best_loss:
                best_loss = total_loss_v
                model.save_model()
        
        gc.collect()

    t_end = time.time()
    print("Time of training: {} secs".format(round(t_end - t_begin,3)))

    #  Visualization of loss history
    plt.plot([t[0] for t in loss_history],label='all', alpha=0.8, color='tab:blue')
    plt.plot([t[1] for t in loss_history],label='all', alpha=0.8, color='tab:blue', linestyle='--')
    plt.plot([t[0] for t in gmvae_loss_history],label='gmvae', alpha=0.8, color='tab:orange')
    plt.plot([t[1] for t in gmvae_loss_history],label='gmvae', alpha=0.8, color='tab:orange', linestyle='--')
    plt.plot([t[0] for t in sparse_loss_history],label='sparse', alpha=0.8, color='tab:green')
    plt.plot([t[1] for t in sparse_loss_history],label='sparse', alpha=0.8, color='tab:green', linestyle='--')
    plt.plot([t[0] for t in deconv_loss_history],label='deconv', alpha=0.8, color='tab:red')
    plt.plot([t[0] for t in deconv_loss_history],label='deconv', alpha=0.8, color='tab:red', linestyle='--')
    plt.title("Training and Validation Loss")
    plt.legend(loc="best")
    plt.show()

    fig, axes = plt.subplots(2,2,figsize=(10,8))
    axes[0,0].plot(loss_history,label='all', alpha=0.8)
    axes[0,0].set_title('Total Loss')
    axes[0,1].plot(gmvae_loss_history,label='gmvae', alpha=0.8)
    axes[0,1].set_title('GMVAE Loss')
    axes[1,0].plot(sparse_loss_history,label='sparse', alpha=0.8)
    axes[1,0].set_title('Sparse Loss')
    axes[1,1].plot(deconv_loss_history,label='deconv', alpha=0.8)
    axes[1,1].set_title('Deconvolution Loss')
    plt.show()

    gc.collect()
