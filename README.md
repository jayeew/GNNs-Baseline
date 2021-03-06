# GNNs-Baseline
A Pytorch based implementation of classical GNNs.  
GNN基准测试模型.  
Some components of this code are adapted from [GPRGNN](https://github.com/jianhao2016/GPRGNN) and [H2GCN](https://github.com/GitEventhandler/H2GCN-PyTorch).

## Requirement
This project should be able to run without any modification after following packages installed.  
```
pytorch
torch_geometric
networkx
torch-sparse
```

## GNNs supported
```
H2GCN FAGCN APPNP GCN GAT ChebNet APPNP JKNet GPRGNN
```

## Unsupervised models supported
[Network Embedding as Matrix Factorization: Unifying DeepWalk, LINE, PTE, and node2vec](https://github.com/xptree/NetMF)
```
DeepWalk Line NetMF
```

## Datasets supported
Download by yourself, dataset_utils.py will help you.
```
cora, citeseer, pubmed, computers, photo, chameleon, squirrel, film, texas, cornell, wisconsin
```

## Run experiment with GCN & Cora:
go to folder `src`
```
python train_model.py --RPMAX 1 \
        --net GCN \
        --train_rate 0.05 \
        --val_rate 0.05 \
        --dataset cora 
```
## Visualization of Confusion Matrix
![](https://github.com/jayeew/GNNs-Baseline/blob/main/pics/H2GCN_chamelon.png)
