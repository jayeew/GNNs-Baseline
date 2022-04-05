#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np

def to_sparse_tensor(data):
    """Convert edge_index to sparse matrix"""
    edge_index = data.edge_index
    sparse_mx = to_scipy_sparse_matrix(edge_index)
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not isinstance(sparse_mx, sp.coo_matrix):
        sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.array([sparse_mx.row, sparse_mx.col]))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape
    )

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)# 数据集中，标签类别为i的索引号 规模：[Num_i]
        index = index[torch.randperm(index.size(0))]# 打乱上述索引号
        indices.append(index)# 最终是长度为num_classes的列表

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)#每一类取出percls_trn条数据，拼成总的训练集，每类可能取不够percls_trn

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]#取剩余数据，索引打乱

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)# 剩余数据取前val_lb条做验证
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)# 最后剩下的做测试
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)# 每一类都取出val_lb条做验证
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)# 再剩下的做测试
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data
