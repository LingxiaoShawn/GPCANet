import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_sparse import SparseTensor

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(nlayer):
            self.convs.append(
                GCNConv(nhid if i>0 else nfeat, nhid if i<nlayer-1 else nclass, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, **kwargs):
        x, adj = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj)
            # x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj)
        return x

    
class GPCALayer(nn.Module):
    def __init__(self, nin, nout, alpha, beta, center=True, n_powers=50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(nin, nout))
        self.bias = nn.Parameter(torch.FloatTensor(1, nout))
        self.nin = nin
        self.nhid = nout
        self.alpha = alpha
        self.beta = beta
        self.center = center
        self.n_powers = n_powers
        # init default parameters
        self.reset_parameters()
    
    def freeze(self, requires_grad=False):
        self.weight.requires_grad = requires_grad
        self.bias.requires_grad = requires_grad
        
    def approximate_invphi_x(self, A, x, y=None):
        """
        Adding another option: use truncated taylor expansion when alpha<1
        """
        if y is not None and self.beta>0:
            yyt_normalizer = y.matmul(y.sum(dim=0).view(-1,1)) + 1e-8
        # init
        invphi_x = x
        # power method
        for _ in range(self.n_powers):
            part1 = A.matmul(invphi_x)
            if self.beta > 0:
                part2 = y.matmul(y.t().matmul(invphi_x))/yyt_normalizer
                part1 = (1-self.beta)*part1 + self.beta*part2
            invphi_x = self.alpha/(1+self.alpha)*part1 + 1/(1+self.alpha)*x
        return invphi_x
                
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0) 

    def forward(self, data, return_invphi_x=False):
        """
            Assume data.adj is SparseTensor and normalized
        """
        # inputs
        A, x, y, train_idx = data.adj, data.x, data.y, data.train_idx
        n, c = data.num_nodes, data.num_classes
        # one hot encoding of training labels
        y_train = None if y is None else SparseTensor(row=train_idx, 
                   col=y.squeeze()[train_idx], sparse_sizes=(n, c))
        # center
        if self.center:
            x = x - x.mean(dim=0)
            
        # calculate inverse of phi times x
        invphi_x = self.approximate_invphi_x(A, x, y_train)
        if return_invphi_x:
            return invphi_x
        else:     
            # AXW + bias
            return invphi_x.mm(self.weight) + self.bias
    
    def init(self, data):
        """
        Later need considering init the network batch-wise
        """
        x = data.x
        invphi_x = self.forward(data, True)
        eig_val, eig_vec = torch.symeig(x.t().mm(invphi_x), eigenvectors=True)
        if self.nhid <= 2*self.nin:
            #weight = eig_vec[:, -self.nhid:] #when 
            weight = torch.cat([eig_vec[:,-self.nhid//2:], -eig_vec[:,-self.nhid//2:]], dim=-1)
        else:
            # get more eigvectors
            # 1. eigen of x.t times x
            # 2. eigen of invphi_x.t times invphi_x
            # 3. negative and positive eigenvectors for relu
            raise ValueError('Larger hidden size is not supported yet.')
        
        # assign 
        self.weight.data = weight
                 
class GPCANet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, alpha, beta, 
                 dropout=0, n_powers=10, center=True, act='Identity', **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(nlayer):
            self.convs.append(
                GPCALayer(nhid if i>0 else nfeat, nhid, alpha, beta, center, n_powers))
            
        self.out_mlp = nn.Sequential(nn.Linear(nhid, nclass)) # consider increasing layers
        self.dropout = nn.Dropout(dropout)
        self.relu = getattr(nn,act)()
        self.cache = None # cannot be used when use batch training
        self.freeze_status = False
    
    def freeze(self, requires_grad=False):
        self.freeze_status = not requires_grad
        for conv in self.convs:
            conv.freeze(requires_grad)
             
    def forward(self, data, use_cache=False):
        # inputs
        A, x, y, train_idx = data.adj, data.x, data.y, data.train_idx
        n, c = data.num_nodes, data.num_classes
        original_x = x
        
        if self.freeze_status and use_cache and self.cache is not None:
            return self.out_mlp(self.cache)
        
        for i, conv in enumerate(self.convs):
            x = conv(data)
            x = self.relu(x)
            x = self.dropout(x)
            data.x = x
        self.cache = data.x

        out = self.out_mlp(data.x)
        data.x = original_x # restore 

        return out
    
    def init(self, data):
        """
        Later need considering init the network batch-wise
        """
        original_x = data.x
        for i, conv in enumerate(self.convs):
            conv.init(data) # init using GPCA
            x = conv(data)
            x = self.relu(x)
            x = self.dropout(x)
            data.x = x
        data.x = original_x # restore 
        