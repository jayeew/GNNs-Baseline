'''
Author : jayee
Date : 2022-4-16

'''

import argparse
from ast import walk
from audioop import bias, mul
from cgi import test
from multiprocessing.sharedctypes import Value
from textwrap import fill
from tkinter import Y
from turtle import forward
from unicodedata import bidirectional
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import time
import torch
import torch.nn as nn
from torch.nn import LSTM, Parameter, GRU, Linear
from dataset_utils import DataLoader
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_dense_adj, dense_to_sparse, k_hop_subgraph, subgraph, degree, add_self_loops, dense_to_sparse, to_networkx
from PointerNet import *
from torch_sparse import fill_diag, SparseTensor, mul
from torch_sparse import sum as sparsesum
import torch_geometric.transforms as T
from torch_cluster import random_walk
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from pylab import *
from gensim.models import Word2Vec
import networkx as nx
import random
from utils import random_planetoid_splits
from tqdm import tqdm
from sklearn import preprocessing
import scipy.sparse as sp

class NetMF(torch.nn.Module):
    def __init__(self, dimension, window_size, rank, negative, is_large=False):
        super(NetMF, self).__init__()
        self.dimension = dimension
        self.window_size = window_size
        self.rank = rank
        self.negative = negative
        self.is_large = is_large
    
    def train(self, G):
        A = sp.csr_matrix(nx.adjacency_matrix(G))
        if not self.is_large:
            print("Running NetMF for a small window size...")
            deepwalk_matrix = self._compute_deepwalk_matrix(
                A, window=self.window_size, b=self.negative
            )

        else:
            print("Running NetMF for a large window size...")
            vol = float(A.sum())
            evals, D_rt_invU = self._approximate_normalized_laplacian(
                A, rank=self.rank, which="LA"
            )
            deepwalk_matrix = self._approximate_deepwalk_matrix(
                evals, D_rt_invU, window=self.window_size, vol=vol, b=self.negative
            )
        # factorize deepwalk matrix with SVD
        u, s, _ = sp.linalg.svds(deepwalk_matrix, self.dimension)
        self.embeddings = sp.diags(np.sqrt(s)).dot(u.T).T
        return self.embeddings

    def _compute_deepwalk_matrix(self, A, window, b):
        # directly compute deepwalk matrix
        n = A.shape[0]
        vol = float(A.sum())
        L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
        # X = D^{-1/2} A D^{-1/2}
        X = sp.identity(n) - L
        S = np.zeros_like(X)
        X_power = sp.identity(n)
        for i in range(window):
            print("Compute matrix %d-th power", i + 1)
            X_power = X_power.dot(X)
            S += X_power
        S *= vol / window / b
        D_rt_inv = sp.diags(d_rt ** -1)
        M = D_rt_inv.dot(D_rt_inv.dot(S).T).todense()
        M[M <= 1] = 1
        Y = np.log(M)
        return sp.csr_matrix(Y)

    def _approximate_normalized_laplacian(self, A, rank, which="LA"):
        # perform eigen-decomposition of D^{-1/2} A D^{-1/2} and keep top rank eigenpairs
        n = A.shape[0]
        L, d_rt = sp.csgraph.laplacian(A, normed=True, return_diag=True)
        # X = D^{-1/2} W D^{-1/2}
        X = sp.identity(n) - L
        print("Eigen decomposition...")
        evals, evecs = sp.linalg.eigsh(X, rank, which=which)
        print(
            "Maximum eigenvalue %f, minimum eigenvalue %f", np.max(evals), np.min(evals)
        )
        print("Computing D^{-1/2}U..")
        D_rt_inv = sp.diags(d_rt ** -1)
        D_rt_invU = D_rt_inv.dot(evecs)
        return evals, D_rt_invU

    def _deepwalk_filter(self, evals, window):
        for i in range(len(evals)):
            x = evals[i]
            evals[i] = 1.0 if x >= 1 else x * (1 - x ** window) / (1 - x) / window
        evals = np.maximum(evals, 0)
        print(
            "After filtering, max eigenvalue=%f, min eigenvalue=%f",
            np.max(evals),
            np.min(evals),
        )
        return evals

    def _approximate_deepwalk_matrix(self, evals, D_rt_invU, window, vol, b):
        # approximate deepwalk matrix
        evals = self._deepwalk_filter(evals, window=window)
        X = sp.diags(np.sqrt(evals)).dot(D_rt_invU.T).T
        M = X.dot(X.T) * vol / b
        M[M <= 1] = 1
        Y = np.log(M)
        print("Computed DeepWalk matrix with %d non-zero elements", np.count_nonzero(Y))
        return sp.csr_matrix(Y)

class MLP_net(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MLP_net, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.lin = Linear(in_channel, out_channel)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x):
        out = self.lin(x)
        return F.log_softmax(out, dim=1)

def train(model, optimizer, data, x):
    model.train()
    optimizer.zero_grad()
    out = model(x)
    nll = F.nll_loss(out[data.train_mask], data.y[data.train_mask])# Negative Log Likelihood Loss，softmax+log+nll_loss = CrossEntropyLoss
    loss = nll
    loss.backward()

    optimizer.step()
    del out

def test(model, data, x):
    model.eval()
    logits, accs, losses, preds = model(x), [], [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].argmax(dim=1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        loss = F.nll_loss(model(x)[mask], data.y[mask])

        preds.append(pred.detach().cpu())#阻断反向传播，数据迁移至cpu
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, preds, losses

if __name__=='__main__':
    netmf = NetMF(dimension=128,
                window_size=5,
                rank=256,
                negative=1,
                is_large=False)

    dataset, data = DataLoader('squirrel')
    print(data)

    G = to_networkx(data, 
                    to_undirected = True, 
                    remove_self_loops = False)

    embedding = torch.from_numpy(netmf.train(G)).float().cuda()
    # print(embedding.size(), torch.typename(embedding))
    
    train_rate = 0.6
    val_rate = 0.2
    percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))#平均每类的训练数量
    val_lb = int(round(val_rate*len(data.y)))#总的验证量
    TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
    print('True Label rate: ', TrueLBrate) # 训练集+验证集 占总数据集的比例
    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb)

    data = data.cuda()
    model = MLP_net(128, dataset.num_classes).cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=0.01,
                                     weight_decay=0.0005)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    early_stopping = 200
    for epoch in tqdm(range(1000)):
        train(model, optimizer, data, embedding)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data, embedding)
        # print(f'Epoch {epoch} : Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}, Val Acc = {val_acc:.2f}')
        if val_loss < best_val_loss:#根据验证集评估模型
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if early_stopping > 0 and epoch > early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break #如果超过early_stopping部分的平均验证损失小于当前的验证损失，说明模型不再提高
    
    print(f'Test acc = {test_acc:.4f} val acc = {best_val_acc:.4f}')
