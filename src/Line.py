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

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int64)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

class Line(torch.nn.Module):
    def __init__(self, dimension, walk_length, walk_num, negative, batch_size, alpha, order):
        super(Line, self).__init__()
        self.dimension = dimension
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.negative = negative
        self.batch_size = batch_size
        self.init_alpha = alpha
        self.order = order

    def train(self, G):
        # run LINE algorithm, 1-order, 2-order or 3(1-order + 2-order)
        self.G = G
        self.is_directed =  nx.is_directed(self.G)
        self.num_node = G.number_of_nodes()
        self.num_edge = G.number_of_edges()
        self.num_sampling_edge = self.walk_length * self.walk_num * self.num_node

        node2id = dict([(node, vid) for vid, node in enumerate(G.nodes())])
        self.edges = [[node2id[e[0]], node2id[e[1]]] for e in self.G.edges()]
        self.edges_prob = np.asarray([G[u][v].get("weight", 1.0) for u, v in G.edges()])
        self.edges_prob /= np.sum(self.edges_prob)
        self.edges_table, self.edges_prob = alias_setup(self.edges_prob)

        degree_weight = np.asarray([0] * self.num_node)
        for u, v in G.edges():
            degree_weight[node2id[u]] += G[u][v].get("weight", 1.0)
            if not self.is_directed:
                degree_weight[node2id[v]] += G[u][v].get("weight", 1.0)
        self.node_prob = np.power(degree_weight, 0.75)
        self.node_prob /= np.sum(self.node_prob)
        self.node_table, self.node_prob = alias_setup(self.node_prob)

        if self.order == 3:
            self.dimension = int(self.dimension / 2)
        if self.order == 1 or self.order==3:
            print("train line with 1-order")
            print(type(self.dimension))
            self.emb_vertex = (np.random.random((self.num_node, self.dimension)) - 0.5) / self.dimension
            self._train_line(order=1)
            embedding1 = preprocessing.normalize(self.emb_vertex, "l2")

        if self.order == 2 or self.order == 3:
            print("train line with 2-order")
            self.emb_vertex = (np.random.random((self.num_node, self.dimension)) - 0.5) / self.dimension
            self.emb_context = self.emb_vertex
            self._train_line(order=2)
            embedding2 = preprocessing.normalize(self.emb_vertex, "l2")

        if self.order == 1:
            self.embeddings = embedding1
        elif self.order == 2:
            self.embeddings = embedding2
        else:
            print("concatenate two embedding...")
            self.embeddings = np.hstack((embedding1, embedding2))
        return self.embeddings

    def _update(self, vec_u, vec_v, vec_error, label):
        # update vetex embedding and vec_error
        f = 1 / (1 + np.exp(-np.sum(vec_u * vec_v, axis=1)))
        g = (self.alpha * (label - f)).reshape((len(label), 1))
        vec_error += g * vec_v
        vec_v += g * vec_u

    def _train_line(self, order):
        # train Line model with order
        self.alpha = self.init_alpha
        batch_size = self.batch_size
        t0 = time.time()
        num_batch = int(self.num_sampling_edge / batch_size)
        epoch_iter = tqdm(range(num_batch))
        for b in epoch_iter:
            if b % 100 == 0:
                epoch_iter.set_description(f"Progress: {b *1.0/num_batch * 100:.4f}%, alpha: {self.alpha:.6f}, time: {time.time() - t0:.4f}")
                self.alpha = self.init_alpha  * max((1 - b *1.0/num_batch), 0.0001)
            u, v = [0] * batch_size, [0] * batch_size
            for i in range(batch_size):
                edge_id = alias_draw(self.edges_table, self.edges_prob)
                u[i], v[i] = self.edges[edge_id]
                if not self.is_directed and np.random.rand() > 0.5:
                    v[i], u[i] = self.edges[edge_id]

            vec_error = np.zeros((batch_size, self.dimension))
            label, target = np.asarray([1 for i in range(batch_size)]), np.asarray(v)
            for j in range(self.negative):
                if j != 0:
                    label = np.asarray([0 for i in range(batch_size)])
                    for i in range(batch_size):
                        target[i] = alias_draw(self.node_table, self.node_prob)
                if order == 1:
                    self._update(self.emb_vertex[u], self.emb_vertex[target], vec_error, label)
                else:
                    self._update(self.emb_vertex[u], self.emb_context[target], vec_error, label)
            self.emb_vertex[u] += vec_error

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
    line = Line(dimension=128,
                walk_length=80,
                walk_num=40,
                negative=5,
                batch_size=1000,
                alpha=0.025,
                order=3)
    dataset, data = DataLoader('chameleon')
    print(data)

    G = to_networkx(data, 
                    to_undirected = True, 
                    remove_self_loops = False)
    embedding = torch.from_numpy(line.train(G)).float().cuda()
    
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