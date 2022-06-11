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

class DeepWalk(torch.nn.Module):
    def __init__(self, dimension, walk_length, walk_num, window_size, worker, iteration):
        super(DeepWalk, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.window_size = window_size
        self.worker = worker
        self.iteration = iteration

    def train(self, G):
        self.G = G
        print(self.G)
        walks = self._simulate_walks(self.walk_length, self.walk_num)

        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(walks, vector_size=self.dimension, window=self.window_size, min_count=0, sg=1, workers=self.worker, epochs=self.iteration)
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([model.wv[str(id2node[i])] for i in range(len(id2node))])
        return embeddings

    def _walk(self, start_node, walk_length):
        # Simulate a random walk starting from start node.
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) == 0:
                break
            k = int(np.floor(np.random.rand()*len(cur_nbrs)))
            walk.append(cur_nbrs[k])
        return walk


    def _simulate_walks(self, walk_length, num_walks):
        # Repeatedly simulate random walks from each node.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('node number:', len(nodes))
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._walk(node, walk_length))
        return walks
        

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
    deepwalk = DeepWalk(dimension=128,
                        walk_length=80,
                        walk_num=40,
                        window_size=5,
                        worker=10,
                        iteration=10)
    dataset, data = DataLoader('chameleon')
    print(data)

    G = to_networkx(data, 
                    to_undirected = True, 
                    remove_self_loops = False)
    embedding = torch.from_numpy(deepwalk.train(G)).cuda()
    
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

