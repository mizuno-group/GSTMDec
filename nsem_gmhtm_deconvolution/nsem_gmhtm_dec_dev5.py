# -*- coding: utf-8 -*-
"""
Created on 2024-08-02 (Fri) 10:19:29

NSEM-GMHTM: Nonlinear Structural Equation Model guided Gaussian Mixture Hierarchical Topic Model

Version 5
- Phi scaling

@author: I.Azuma
"""
# %%
from audioop import bias
import os
from pickle import FALSE, TRUE
from re import X
import time
import scipy.sparse as sp
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch.nn.functional as F
from torch import nn, optim

from scipy import sparse
import numpy as np
from pyparsing import Word
import torch.optim as optim
import yaml
from numpy.random import normal

# import km
from sklearn import metrics
from torch.autograd import Variable
from torch.nn import init
from tqdm import tqdm

from pathlib import Path
BASE_DIR = Path(__file__).parent
print(BASE_DIR)
print("!! NSEM-GMHTM Deconvolution -v4 !!")

import utils
from customized_linear import CustomizedLinear
from learning_utils import *


Tensor = torch.cuda.FloatTensor
np.random.seed(0)
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inv_flag = True
print("inv_flag",inv_flag)

# %%
def kl_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

class LossFunctions:
    def reconstruction_loss(self, real, predicted, dropout_mask=None, rec_type="mse"):
        if rec_type == "ce_sum":
            loss = -torch.sum(torch.log(predicted) * real)
        elif rec_type == "ce_mean":
            loss = -torch.mean(torch.log(predicted) * real)
        elif rec_type == "mse":
            if dropout_mask is None:
                loss = torch.sum((real - predicted).pow(2)).mean()
            else:
                loss = torch.sum((real - predicted).pow(2) * dropout_mask) / torch.sum(
                    dropout_mask
                )

        elif rec_type == "bce":
            loss = F.binary_cross_entropy(predicted, real, reduction="none").mean()
        else:
            raise Exception
        return loss

    def log_normal(self, x, mu, var, eps=1e-8):
        """Logarithm of normal distribution with mean=mu and variance=var
            log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
            x: (array) corresponding array containing the input
            mu: (array) corresponding array containing the mean 
            var: (array) corresponding array containing the variance

        Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                    of all the sample losses or an array with the losses per sample
        """
        if eps > 0.0:
            var = var + eps
        return -0.5 * torch.mean(
            torch.log(torch.FloatTensor([2.0 * np.pi]).cuda()).sum(0)
            + torch.log(var)
            + torch.pow(x - mu, 2) / var,
            dim=-1,
        )

    def gaussian_loss(
        self, z, z_mu, z_var, z_mu_prior, z_var_prior
    ):  
        """Variational loss when using labeled data without considering reconstruction loss
            loss = log q(z|x,y) - log p(z) - log p(y)

        Args:
            z: (array) array containing the gaussian latent variable
            z_mu: (array) array containing the mean of the inference model
            z_var: (array) array containing the variance of the inference model
            z_mu_prior: (array) array containing the prior mean of the generative model
            z_var_prior: (array) array containing the prior variance of the generative mode
            
        """
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        return loss.sum()

    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.sum(torch.sum(targets * log_q, dim=-1))

class GumbelSoftmax(nn.Module):

  def __init__(self, f_dim, c_dim):
    super(GumbelSoftmax, self).__init__()
    self.logits = nn.Linear(f_dim, c_dim)
    self.f_dim = f_dim
    self.c_dim = c_dim
     
  def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
      U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, logits, temperature):
    y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
    return F.softmax(y / temperature, dim=-1)

  def gumbel_softmax(self, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    #categorical_dim = 10
    y = self.gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard 
  
  def forward(self, x, temperature=1.0, hard=False):
    logits = self.logits(x).view(-1, self.c_dim)
    prob = F.softmax(logits, dim=-1)
    y = self.gumbel_softmax(logits, temperature, hard)
    return logits, prob, y

class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.var = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.var(x)
        return mu, logvar

# Encoder
class InferenceNet(nn.Module):
    def __init__(self,topic_num_1,topic_num_2,topic_num_3,hidden_num,y_dim,nonLinear):
        super(InferenceNet, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(topic_num_1,topic_num_2), nn.BatchNorm1d(topic_num_2), nonLinear)
        self.encoder_2 = nn.Sequential(nn.Linear(topic_num_2,topic_num_3), nn.BatchNorm1d(topic_num_3), nonLinear)
        self.inference_qyx3 = torch.nn.ModuleList(
            [
                nn.Linear(topic_num_3, hidden_num),  # 64 1
                nn.BatchNorm1d(hidden_num),
                nonLinear,
                GumbelSoftmax(hidden_num, y_dim),  # 1 256
            ]
        )
        self.inference_qzyx3 = torch.nn.ModuleList(
            [
                nn.Linear(topic_num_3 + y_dim, hidden_num),
                nn.BatchNorm1d(hidden_num),
                nonLinear,
                Gaussian(hidden_num, topic_num_3),
            ]
        )

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z

    # q(y|x)
    def qyx3(self, x,temperature,hard):
        num_layers = len(self.inference_qyx3)
        for i, layer in enumerate(self.inference_qyx3):
            if i == num_layers - 1:
                x = layer(x, temperature, hard)
            else:
                x = layer(x)
        return x
    
    # q(z|x,y)
    def qzxy3(self, x, y):
        concat = torch.cat((x.squeeze(2), y), dim=1)
        for layer in self.inference_qzyx3:
            concat = layer(concat)
        return concat


    def forward(self, x, adj, adj_2, adj_3, temperature, hard=0):
        if inv_flag ==True:
            x_1 = torch.matmul(adj.to(torch.float32),x.squeeze(2).T).T
            x_2 = self.encoder(x_1)
            x_2 = torch.matmul(adj_2.to(torch.float32),x_2.T).T
            x_3 = self.encoder_2(x_2)
            x_3 = torch.matmul(adj_3.to(torch.float32),x_3.T).T
        else:
            x_1 = x.squeeze(2)
            x_2 = self.encoder(x_1)
            x_3 = self.encoder_2(x_2)                     

        logits_3, prob_3, y_3  = self.qyx3(x_3,temperature, hard = 0)
        mu_3, logvar_3 = self.qzxy3(x_3.view(x_3.size(0), -1, 1), y_3)
        var_3 = torch.exp(logvar_3)
        # reparameter: td1
        z_3 = self.reparameterize(mu_3, var_3)
        output_3 = {"mean": mu_3, "var": var_3, "gaussian": z_3, "categorical": y_3,'logits': logits_3, 'prob_cat': prob_3}
        return output_3   

# Decoder
class GenerativeNet(nn.Module):
    def __init__(self, topic_num_1,topic_num_2,topic_num_3, y_dim=256, nonLinear=None):
        super(GenerativeNet, self).__init__()
        self.y_mu_1 = nn.Sequential(nn.Linear(y_dim, topic_num_3))
        self.y_var_1 = nn.Sequential(nn.Linear(y_dim, topic_num_3))
        self.decoder = nn.Sequential(CustomizedLinear(torch.ones(topic_num_3,topic_num_2),bias=False), nn.BatchNorm1d(topic_num_2), nonLinear)
        self.decoder_2 = nn.Sequential(CustomizedLinear(torch.ones(topic_num_2,topic_num_1),bias=False), nn.BatchNorm1d(topic_num_1), nonLinear)

        if True:
            print('Constraining decoder to positive weights', flush=True)

            self.decoder[0].reset_params_pos()
            self.decoder[0].weight.data *= self.decoder[0].mask        
            self.decoder_2[0].reset_params_pos()    
            self.decoder_2[0].weight.data *= self.decoder_2[0].mask 

        self.generative_pxz = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_3),
                nonLinear,
            ]
        )
        self.generative_pxz_1 = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_2),
                nonLinear,
            ]
        )
        self.generative_pxz_2 = torch.nn.ModuleList(
            [
                nn.BatchNorm1d(topic_num_1),
                nonLinear,
            ]
        )

    def pzy1(self, y):
        y_mu = self.y_mu_1(y)
        y_logvar = self.y_var_1(y)
        return y_mu, y_logvar
    def pxz(self, z):
        for layer in self.generative_pxz:
            z = layer(z)
        return z
    def pxz_1(self, z):
        for layer in self.generative_pxz_1:
            z = layer(z)
        return z
    def pxz_2(self, z):
        for layer in self.generative_pxz_2:
            z = layer(z)
        return z

    def forward(
        self,
        z,
        y_3,
        adj_A_t_inv_2,
        adj_A_t_inv_1,
        adj_A_t_3,
    ):
        y_mu_3, y_logvar_3 = self.pzy1(y_3)
        y_var_3 = torch.exp(y_logvar_3)

        if inv_flag ==True:
            z = torch.matmul(adj_A_t_3.to(torch.float32), z.T).T
            out_1 = self.pxz(z)
            z_2 = self.decoder(z)
            z_2 = torch.matmul(adj_A_t_inv_2.to(torch.float32), z_2.T).T
            out_2 = self.pxz_1(z_2)
            z_3 = self.decoder_2(z_2)
            z_3 =  torch.matmul(adj_A_t_inv_1.to(torch.float32), z_3.T).T
            out_3 = self.pxz_2(z_3)
        else:
            out_1 = self.pxz(z)
            z_2 = self.decoder(z)
            out_2 = self.pxz_1(z_2)
            z_3 = self.decoder_2(z_2)
            out_3 = self.pxz_2(z_3)
        
        m0 = self.decoder[0].weight.data
        m1 = self.decoder_2[0].weight.data

        # torch.Size([batch_size, topic_n])
        output_1 = {"x_rec": out_1}  
        output_2 = {"x_rec": out_2}
        output_3 = {"y_mean": y_mu_3, "y_var": y_var_3, "x_rec": out_3}
        output_4 = {"m_0":m0, "m_1":m1}  # dependency matrix

        return output_1, output_2, output_3, output_4


class net(nn.Module):
    def __init__(
        self,
        batch_size=None,
        adj_A=None,
        adj_A_2=None,
        adj_A_3=None,
        mask=None,
        topic_num_1=None,
        topic_num_2=None,
        topic_num_3=None,
        word_embed=None,
        vocab_num=None,
        hidden_num=None,
        prior_beta=None,
        **kwargs,
    ):
        super(net, self).__init__()
        self.dropout = nn.Dropout(0.1)
        xavier_init = torch.distributions.Uniform(-0.05,0.05)
        
        if word_embed is not None:
            self.word_embed = nn.Parameter(word_embed)
        else:
            self.word_embed = nn.Parameter(torch.rand(hidden_num, vocab_num))
        self.topic_embed = nn.Parameter(xavier_init.sample((topic_num_1, hidden_num)))
        self.topic_embed_1 = nn.Parameter(xavier_init.sample((topic_num_2, hidden_num)))
        self.topic_embed_2 = nn.Parameter(xavier_init.sample((topic_num_3, hidden_num)))

        #self.phi_1 = nn.Parameter(xavier_init.sample((topic_num_1, vocab_num)))
        #self.phi_2 = nn.Parameter(xavier_init.sample((topic_num_2, vocab_num)))
        #self.phi_3 = nn.Parameter(xavier_init.sample((topic_num_3, vocab_num)))

        self.adj_A = nn.Parameter(
            Variable(torch.from_numpy(adj_A).float(), requires_grad=True, name="adj_A")
        )
        self.adj_A_2 = nn.Parameter(
            Variable(
                torch.from_numpy(adj_A_2).float(), requires_grad=True, name="adj_A_2"
            )
        )
        self.adj_A_3 = nn.Parameter(
            Variable(
                torch.from_numpy(adj_A_3).float(), requires_grad=True, name="adj_A_3"
            )
        )

        self.encoder = nn.Sequential(nn.Linear(vocab_num, topic_num_1), nn.BatchNorm1d(topic_num_1), nn.Tanh())
        y_dim = 10  # x:  y:  z: # FIXME ??

        self.inference = InferenceNet(topic_num_1,topic_num_2,topic_num_3,hidden_num,y_dim,nn.Tanh())
        self.generative = GenerativeNet(topic_num_1,topic_num_2,topic_num_3,y_dim,nn.Tanh())

        self.losses = LossFunctions()
        for m in self.modules():
            if (
                type(m) == nn.Linear
                or type(m) == nn.Conv2d
                or type(m) == nn.ConvTranspose2d
            ):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    init.constant_(m.bias, 0)

    def to_np(self,x):
        return x.cpu().detach().numpy()

    def get_topic_word_dist(self, level=2):
        """ Phi : (n_topics, V)"""
        if level == 2:
            return  torch.softmax(self.topic_embed @ self.word_embed, dim=1)
        elif level == 1:
            return torch.softmax(self.topic_embed_1 @ self.word_embed, dim=1)
        elif level == 0:
            return torch.softmax(self.topic_embed_2 @ self.word_embed, dim=1)

    def get_doc_topic_dist(self, level=2):
        """ Theta : (batch_size, n_topics)"""
        if level == 2:
            return self.theta_3  # (D, 8)
        elif level == 1:
            return self.theta_2  # (D, 32)
        elif level == 0:
            return self.theta_1  # (D, 128)

    def encode(self, x):
        p1 = self.encoder(x)
        return p1
    
    def decode(self, x_ori, out_1, out_2, out_3):
        out_1 = torch.softmax(out_1, dim=1)
        out_2 = torch.softmax(out_2, dim=1)
        out_3 = torch.softmax(out_3, dim=1)

        self.theta_1 = out_1
        self.theta_2 = out_2
        self.theta_3 = out_3

        beta_1 = torch.softmax(self.topic_embed @ self.word_embed, dim=1)  # NOTE: phi
        beta_2 = torch.softmax(self.topic_embed_1 @ self.word_embed, dim=1)
        beta_3 = torch.softmax(self.topic_embed_2 @ self.word_embed, dim=1)

        p1 = out_3 @ beta_1 
        p2 = out_2 @ beta_2 
        p3 = out_1 @ beta_3
        p_fin = (p1.T+p2.T+p3.T)/3.0

        return p_fin.T
    

    def _one_minus_A_t(self, adj):
        adj_normalized = abs(adj) 
        adj_normalized = Tensor(np.eye(adj_normalized.shape[0])).cuda() - (adj_normalized.transpose(0, 1)).cuda()   
        return adj_normalized

    def forward(self, x, dropout_mask=None, temperature=1.0, hard=0):
        x_ori = x  # >> (batch_size, V)
        x = self.encode(x)  # >> (batch_size, max_topic_n)
        x = x.view(x.size(0), -1, 1)  # (batch_size, max_topic_n, 1)
        
        topic_num_1 = self.adj_A.shape[0]
        mask = Variable(
            torch.from_numpy(np.ones(topic_num_1) - np.eye(topic_num_1)).float(),
            requires_grad=False,
        ).cuda() 

        adj_A_t = self._one_minus_A_t(self.adj_A * mask)
        adj_A_t_inv = torch.inverse(adj_A_t)

        topic_num_2 = self.adj_A_2.shape[0]
        mask_1 = Variable(
            torch.from_numpy(np.ones(topic_num_2) - np.eye(topic_num_2)).float(), requires_grad=False
        ).cuda()
        adj_A_t_2 = self._one_minus_A_t(self.adj_A_2 * mask_1)
        adj_A_t_inv_2 = torch.inverse(adj_A_t_2)

        topic_num_3 = self.adj_A_3.shape[0]
        mask_2 = Variable(
            torch.from_numpy(np.ones(topic_num_3) - np.eye(topic_num_3)).float(), requires_grad=False
        ).cuda()
        adj_A_t_3 = self._one_minus_A_t(self.adj_A_3 * mask_2)
        adj_A_t_inv_3 = torch.inverse(adj_A_t_3)

        # inference
        #output_3 = {"mean": mu_3, "var": var_3, "gaussian": z_3, "categorical": y_3,'logits': logits_3, 'prob_cat': prob_3}
        out_inf_1 = self.inference(
            x, adj_A_t, adj_A_t_2, adj_A_t_3, temperature, x_ori.view(x.size(0), -1, 1)
        )

        z_3, y_3 = out_inf_1["gaussian"], out_inf_1["categorical"]
        self.z_3 = z_3  # (D, 128) --> Lth layer
        self.y_3 = y_3  # (D, 10) --> categorical dim

        # collect each hd
        output_1, output_2, output_3, dep_mats = self.generative(
            z_3,
            y_3,
            adj_A_t_inv_2,
            adj_A_t_inv,
            adj_A_t_inv_3,
        )

        dec_1 = output_1["x_rec"]  # (D, 128)
        dec_2 = output_2["x_rec"]  # (D, 32)
        dec_3 = output_3["x_rec"]  # (D, 8)
        dec_res = self.decode(x_ori, dec_1, dec_2, dec_3)

        loss_rec_1 = self.losses.reconstruction_loss(
            x_ori, dec_res, dropout_mask, "ce_mean"
        )
        loss_gauss_3 = (
            self.losses.gaussian_loss(
                z_3,                  # out_inf_1["gaussian"]
                out_inf_1["mean"],
                out_inf_1["var"],
                output_3["y_mean"],
                output_3["y_var"],
            )
            * 1)
        loss_cat_3 = (-self.losses.entropy(out_inf_1['logits'], out_inf_1['prob_cat']) - np.log(0.1)) 
        loss_dic = {'recon_loss':loss_rec_1, 'gauss_loss':loss_gauss_3, 'cat_loss':loss_cat_3}

        return loss_dic, dep_mats

class AMM_no_dag(object):
    def __init__(
        self,
        reader=None,
        vocab_dict=None,
        model_path=None,
        word_embed=None,
        topic_num_1=None,
        topic_num_2=None,
        topic_num_3=None,
        epochs=None,
        batch_size=None,
        learning_rate=None,
        rho_max=None,
        rho=None,
        phi=None,
        epsilon=None,
        lam=None,
        threshold_1=None,
        threshold_2=None,
        **kwargs,
    ):
        self.reader = reader
        self.vocab_dict = vocab_dict
        self.model_path = model_path
        #self.n_classes = self.reader.get_n_classes()  # document class
        self.topic_num_1 = topic_num_1
        self.topic_num_2 = topic_num_2
        self.topic_num_3 = topic_num_3

        self.word_embed = word_embed.to(device) if word_embed is not None else None

        self.epochs = epochs
        self.learning_rate = learning_rate

        self.adj = self.initialize_A(topic_num_1)
        self.adj_2 = self.initialize_A(topic_num_2)  # topic_num_2
        self.adj_3 = self.initialize_A(topic_num_3)  # topic_num_3
        print("AMM_no_dag init model.")

        self.Net = net(
            batch_size,
            adj_A=self.adj,
            adj_A_2=self.adj_2,
            adj_A_3=self.adj_3,
            topic_num_1=self.topic_num_1,
            topic_num_2=self.topic_num_2,
            topic_num_3=self.topic_num_3,
            word_embed=self.word_embed,
            **kwargs,
        ).to(device)

        self.pi_ave = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        # optimizer uses ADAM

    def initialize_A(self, topic_nums=16):
        A = np.ones([topic_nums, topic_nums]) / (topic_nums - 1) + (
            np.random.rand(topic_nums * topic_nums) * 0.0002
        ).reshape([topic_nums, topic_nums])
        for i in range(topic_nums):
            A[i, i] = 0
        return A

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.Net.state_dict(), f"{self.model_path}/model.pth")
        with open(f"{self.model_path}/info.txt", "w") as f:
            f.write(f"Topic Num 1: {str(self.topic_num_1)}\n")
            f.write(f"Topic Num 2: {str(self.topic_num_2)}\n")
            f.write(f"Topic Num 3: {str(self.topic_num_3)}\n")
            f.write(f"Epochs: {str(self.epochs)}\n")
            f.write(f"Learning Rate: {str(self.learning_rate)}\n")
            f.write(str(self.Net))

        np.save(f"{self.model_path}/pi_ave.npy", self.pi_ave)
        #print(f"Models save to  {self.model_path}/model.pkl")

    def load_model(self, model_filename="model.pkl"):
        model_path = os.path.join(self.model_path, model_filename)

        self.Net.load_state_dict(torch.load(model_path))
        # self.Net = torch.load(model_path)
        with open(f"{self.model_path}/info.txt", "r") as f:
            self.topic_num_1 = int(f.read())  # FIXME
        self.pi_ave = np.load(f"{self.model_path}/pi_ave.npy")
        print("AMM_no_dag model loaded from {}.".format(model_path))

    def get_topic_dist(self, level=2):
        # topic_dist = self.Net.get_topic_dist()[self.topics]
        topic_dist = self.Net.get_topic_word_dist(level)
        return topic_dist

    def get_topic_word(self, level=2, top_k=15, vocab_dict=None):
        topic_dist = self.get_topic_dist(level)
        vals, indices = torch.topk(topic_dist, top_k, dim=1)
        indices = self.to_np(indices).tolist()
        topic_words = [
            [vocab_dict[idx] for idx in indices[i]]
            for i in range(topic_dist.shape[0])
        ]
        return topic_words

    def get_topic_parents(self, mat):
        return 0

    def evaluate(self):
        for level in range(3):
            topic_word = self.get_topic_word(
                top_k=10, level=level, vocab_dict=self.vocab_dict
            )
            for k, top_word_k in enumerate(topic_word):
                print(f"Topic {k}:{top_word_k}")

    # NPMI
    def sampling(self, flag, best_coherence=-1, test_data=None):

        #test_data, test_label, _ = self.reader.get_matrix("test", mode="count")

        # for level in range(3):
        topic_dist_2 = self.to_np(self.get_topic_dist(level=2)) 
        topic_dist_1 = self.to_np(self.get_topic_dist(level=1))
        topic_dist_0 = self.to_np(self.get_topic_dist(level=0))

        # concat each level
        topic_dist = np.concatenate(
            (np.concatenate((topic_dist_2, topic_dist_1), axis=0), topic_dist_0), axis=0
        )

        train_coherence_2 = utils.evaluate_NPMI(test_data, topic_dist_2)
        train_coherence_1 = utils.evaluate_NPMI(test_data, topic_dist_1)
        train_coherence_0 = utils.evaluate_NPMI(test_data, topic_dist_0)
        train_coherence = utils.evaluate_NPMI(test_data, topic_dist)

        if flag == 1:
            TU2 = utils.evaluate_TU(topic_dist_2)
            TU1 = utils.evaluate_TU(topic_dist_1)
            TU0 = utils.evaluate_TU(topic_dist_0)
            TU = utils.evaluate_TU(topic_dist)

            print("TU level 2: " + str(TU2))
            print("TU level 1: " + str(TU1))
            print("TU level 0: " + str(TU0))
            print("TU: " + str(TU))
            print("Topic coherence  level 2: ", train_coherence_2)
            print("Topic coherence  level 1: ", train_coherence_1)
            print("Topic coherence  level 0: ", train_coherence_0)
        
        print("Topic coherence:", train_coherence)
        avg_coherence = (train_coherence_2 + train_coherence_1 + train_coherence_0)/3
        print("Avg. coherence:", avg_coherence)

        if avg_coherence > best_coherence:
            best_coherence = avg_coherence
            print("New best coherence found!!")
            self.save_model()

        return best_coherence

# %%
