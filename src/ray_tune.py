#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

"""

use ray tune to adjust the hyperparameters

"""

import argparse
from ast import arg
from tabnanny import verbose
from dataset_utils import DataLoader
from utils import random_planetoid_splits, to_sparse_tensor
from GNN_models import *
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import ExperimentAnalysis

def train(model, optimizer, data, dprate):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    nll = F.nll_loss(out[data.train_mask], data.y[data.train_mask])# Negative Log Likelihood Loss，softmax+log+nll_loss = CrossEntropyLoss
    loss = nll
    loss.backward()

    optimizer.step()
    del out

def test(model, data):
    model.eval()
    logits, accs, losses, preds = model(data), [], [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].argmax(dim=1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        loss = F.nll_loss(model(data)[mask], data.y[mask])

        preds.append(pred.detach().cpu())#阻断反向传播，数据迁移至cpu
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, preds, losses

def RunExp(config, data):
    args = data
    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'FAGCN':
        Net = FAGCN_Net
    elif gnn_name == 'H2GCN':
        Net = H2GCN_Net

    dname = args.dataset
    dataset, data = DataLoader(dname)

    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    alpha = args.alpha
    train_rate = args.train_rate
    val_rate = args.val_rate
    percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))#平均每类的训练数量
    val_lb = int(round(val_rate*len(data.y)))#总的验证量
    TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
    print('True Label rate: ', TrueLBrate) # 训练集+验证集 占总数据集的比例

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    Results0 = []

    appnp_net = Net(dataset, args)#初始化网络模型，包含数据集特征数、类别数，模型超参
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)# 得到train_mask, test_mask, val_mask

    if args.net == 'H2GCN':# 强制改edge_index为稀疏张量表示
        # print(data)
        adj = to_sparse_tensor(data)
        data.edge_index = adj
    
    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config["lr"],
                                     weight_decay=config["weight_decay"])

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    # str = args.net + '_' + args.dataset + '_{}'.format(RP)
    # writer = SummaryWriter(os.path.abspath('..') + '/tensorboard/' + str)

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)

        tune.report(mean_accuracy=val_acc)

        if val_loss < best_val_loss:#根据验证集评估模型
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'GPRGNN':
                TEST = appnp_net.prop1.temp.clone()
                Alpha = TEST.detach().cpu().numpy()
            else:
                Alpha = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break #如果超过early_stopping部分的平均验证损失小于当前的验证损失，说明模型不再提高


    # return test_acc, best_val_acc, Gamma_0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.025)
    parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1) # APPNP
    parser.add_argument('--epsilon', type=float, default=0.1) # FAGCN
    parser.add_argument('--k', type=float, default=2) # H2GCN
    parser.add_argument('--dprate', type=float, default=0.5)# 没用到，也不知道干啥的
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None) # GPRGCN
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='Cora')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN', 'FAGCN', 'H2GCN'],
                        default='GPRGNN')

    args = parser.parse_args()

    search_space = {
        "lr": tune.grid_search([0.0005, 0.001, 0.002, 0.05, 0.1, 0.5]),
        "weight_decay": tune.grid_search([0, 0.0001, 0.0005]),
        }
    analysis = tune.run(
            tune.with_parameters(
                RunExp,
                data=args,  #传入额外参数
            ),
            
            num_samples=3,  # 每种超参组合的实验次数
            scheduler=ASHAScheduler(
                metric="mean_accuracy", 
                mode="max",
                time_attr='training_iteration',
                max_t=200,  #最大epoch
                grace_period=70,    #最小epoch 
                reduction_factor=3, #进一步对比grace_period*reduction_factor个epoch
                brackets=1),
            config=search_space,  # 超参搜索空间
            resources_per_trial={'cpu':16, 'gpu': 2},
            name="my_exp", 
            local_dir="/home/lab/wjy/GPRGNN-master/src/tune_results",
            verbose=2,
        )


    # analysis = ExperimentAnalysis(
    #             experiment_checkpoint_path="/home/lab/wjy/GPRGNN-master/src/tune_results/my_exp",
    #             default_mode = 'max')

    # best_trial = analysis.best_trial  # Get best trial
    # best_config = analysis.best_config  # Get best trial's hyperparameters
    # best_logdir = analysis.best_logdir  # Get best trial's logdir
    # best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
    # best_result = analysis.best_result  # Get best trial's last results

    # 实验结果输出
    # print("Best trial is:", best_trial)
    # print("Best config is:", best_config)
    # print("Best logdir is:", best_logdir)
    # print("Best checkpoint is:", best_checkpoint)
    # print("Best result is:", best_result)


    # for RP in tqdm(range(RPMAX)):#重复训练RPMAX次，每次有args.epochs轮

    #     test_acc, best_val_acc, Gamma_0 = RunExp(args, dataset, data, Net, percls_trn, val_lb, RP)
    #     Results0.append([test_acc, best_val_acc, Gamma_0])

    # test_acc_mean, val_acc_mean, _ = np.mean(Results0, axis=0) * 100 #axis=0，纵向计算
    # test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100 #标准差
    # confidence_interval = 1.96 * test_acc_std/np.sqrt(RPMAX) # 0.95置信区间
    # print(f'{gnn_name} on dataset {args.dataset}, in {RPMAX} repeated experiment:')
    # # print(f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f}')
    # print(f'Test acc = {test_acc_mean:.4f} ± {confidence_interval:.4f} \t val acc mean = {val_acc_mean:.4f}')

    # # save experiments
    # from datetime import datetime
    # save_file_name = '{}_{}_{}.txt'.format(args.net, args.dataset, datetime.today().date())
    # print('experiments results have been saved in : results/{}.'.format(save_file_name))
    # file = open(os.path.abspath('..') + "/results/" + save_file_name, "a")
    # print('{}{}{}'.format('*'*20, datetime.today(), '*'*20), file=file)
    # for k in args.__dict__:
    #     print(k + ": " + str(args.__dict__[k]), file=file)
    # print(f'Test acc = {test_acc_mean:.4f} ± {confidence_interval:.4f} \t val acc mean = {val_acc_mean:.4f}', file=file)
