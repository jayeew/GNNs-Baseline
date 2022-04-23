#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import torch.nn.functional as F
import torch_sparse

from torch import FloatTensor
from torch import nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv, FAConv, APPNP
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, to_dense_adj, dense_to_sparse
from torch_geometric.nn.inits import glorot, zeros


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)

class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT_prop(MessagePassing):
    '''
    out : 原始GAT，仅考虑1阶
    out2: 仅考虑2阶的注意力
    out3: 1阶 拼 2接 的注意力
    '''
    def __init__(
        self,
        in_channels,
        out_channels,
        heads = 1,
        concat = True,
        negative_slope = 0.2,
        dropout = 0.0,
        add_self_loops = True,
        bias = True,
        **kwargs
    ):
        super(GAT_prop, self).__init__(aggr='add', node_dim=0, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        # self.bias = bias

        self.lin_src = Linear(in_channels, heads*out_channels,
                            bias=False, weight_initializer='glorot')
        self.lin_dst = Linear(in_channels, heads*out_channels,
                            bias=False, weight_initializer='glorot')
        self.lin_dst2 = Linear(in_channels, heads*out_channels,
                            bias=False, weight_initializer='glorot')
        # self.lin_dst = Linear(in_channels, heads*out_channels,
        #                     bias=False, weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:   
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst2 = Parameter(torch.Tensor(1, heads, out_channels))

        # self.lin_edge = None
        # self.register_parameter('att_edge', None)
            
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # self._alpha = None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.lin_dst2.reset_parameters()

        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_dst2)
        # glorot(self.att_edge)
        zeros(self.bias)
        
    def forward(self, x, edge_index, size=None):
        H, C = self.heads, self.out_channels

        x_src = self.lin_src(x).view(-1, H, C) # N*H*C
        x_dst = self.lin_dst(x).view(-1, H, C)
        x_dst2 = self.lin_dst2(x).view(-1, H, C)

        x = (x_src, x_dst)
        
        
        alpha_src = (x_src * self.att_src).sum(dim=-1) # ? N*H
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha_dst2 = (x_dst2 * self.att_dst2).sum(dim=-1)

        alpha = (alpha_src, alpha_dst)
        
        if self.add_self_loops:
            num_nodes = x_src.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size) # N*H*C
        
        adj = to_dense_adj(edge_index=edge_index).squeeze(0)
        adj = torch.spmm(adj, adj)
        edge_index = dense_to_sparse(adj)[0]
        x = (x_src, x_dst2)
        alpha = (alpha_src, alpha_dst2)
        out2 = self.propagate(edge_index, x=x, alpha=alpha, size=size)
        

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels) # N*(H*C)
            out2 = out2.view(-1, self.heads * self.out_channels)
        
        else:
            out = out.mean(1)
            out2 = out2.mean(1)
            
        # if self.bias is not None:
        #     out += self.bias
        # print('out {}, out2 {}, out3 {}'.format(out.size(), out2.size(), out3.size()))    
        out3 = torch.cat((out, out2), dim=-1)
        return out3
        
    def message(self, x_j, alpha_j, alpha_i,
                index, ptr, size_i):
        # x_j: E*H*C
        # print('index {}, ptr {}, size_i {}'.format(index.size(), ptr, size_i))
        alpha = alpha_i + alpha_j # E*H
        # print('alpha1 size {}'.format(alpha.size()))
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i) # E*H
        # print('alpha2 size {}'.format(alpha.size()))

        alpha = F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(-1) #E*H*1
        # print('alpha3 size {}'.format(alpha.size()))
        # print('x_j {}'.format(x_j.size()))
        return x_j * alpha

class GAT_Net2(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net2, self).__init__()
        self.conv1 = GAT_prop(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GAT_prop(
            args.hidden * args.heads * 2, # args.hidden * args.heads * 2 when use out3
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = nn.Linear(dataset.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

class GCN_JKNet(torch.nn.Module):
    def __init__(self, dataset, args):
        in_channels = dataset.num_features
        out_channels = dataset.num_classes

        super(GCN_JKNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin1 = torch.nn.Linear(16, out_channels)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=16,
                                   num_layers=4
                                   )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)

class FAGCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(FAGCN_Net, self).__init__()
        in_channels = dataset.num_features
        out_channels = dataset.num_classes
        self.eps = args.epsilon
        self.layer_num = 2
        self.dropout = args.dropout
        self.hidden = args.hidden

        self.layers = torch.nn.ModuleList()
        for _ in range(self.layer_num):
            self.layers.append(FAConv(self.hidden,self.eps,self.dropout))

        self.lin1 = torch.nn.Linear(in_channels, self.hidden)
        self.lin2 = torch.nn.Linear(self.hidden, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lin1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.lin2.weight, gain=1.414)

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = torch.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h,raw,edge_index)
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)

class H2GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(H2GCN_Net, self).__init__()
        dropout = args.dropout
        hidden_dim = args.hidden
        feat_dim = dataset.num_features
        class_dim = dataset.num_classes
        k = args.k

        self.dropout = dropout
        self.k = k
        self.w_embed = torch.nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = torch.nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.w_embed)
        torch.nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        device = adj.device
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        ).to(device)
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, data):
        adj, x = data.edge_index, data.x # (adj: torch.sparse.Tensor, x: FloatTensor) -> FloatTensor:
        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [F.relu(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(torch.cat([r1, r2], dim=1))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        return F.log_softmax(torch.mm(r_final, self.w_classify), dim=1)