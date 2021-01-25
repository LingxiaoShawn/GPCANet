import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

from torch_sparse import SparseTensor

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout, mode, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(nlayer):
            self.convs.append(GCNConv(nhid if i>0 else nfeat, 
                              nhid if i<nlayer-1 else nclass, normalize=False))
        self.dropout = dropout
        self.mode = mode
        self.nclass = nclass
        
        # Fix them currently to reduce the number of experiments
        # because init won't affect much. Can test increase beta later.
        self.alpha = 1
        self.beta = 0 
        self.n_powers = 20

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            
    def forward(self, data, minibatch=False, **kwargs):
        x = data.x
        if minibatch:
            A = to_normalized_sparsetensor(data.edge_index, x.size(0), self.mode)
        else:
            A = data.adj
        # here can also do full batch, if gpu is not big enough, use cpu
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, A)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, A)
        return x
    
    def init(self, full_data, center=True, posneg=False, approximate=True, **kwargs):
        # posneg = True is better for relu
        with torch.no_grad():
            # use GPCANet to init GCN
            A, x, y = full_data.adj, full_data.x, full_data.y
            if approximate:
                for i, conv in enumerate(self.convs[:-1]):
                    # use A as inv_phi
                    x_ = x - x.mean(dim=0) if center else x
                    eig_val0, eig_vec0 = torch.symeig(x_.t().mm(A).mm(x_), eigenvectors=True)
                    if conv.out_channels <= (int(posneg)+1)*conv.in_channels:
                        weight = torch.cat((eig_vec0[:, -conv.out_channels//2:], 
                                     -eig_vec0[:, -conv.out_channels//2:]), dim=-1) \
                               if posneg else eig_vec0[:, -conv.out_channels:] 
                    elif conv.out_channels <= 2*(int(posneg)+1)*conv.in_channels:
                        eig_val1, eig_vec1 = torch.symeig(x_.t().mm(x_), eigenvectors=True)
                        m = conv.out_channels % ((int(posneg)+1)*conv.in_channels)
                        weight = torch.cat((eig_vec0, -eig_vec0, eig_vec1[:,-m//2:], 
                                            -eig_vec1[:,-m//2:]) , dim=-1) \
                               if posneg else torch.cat((eig_vec0, eig_vec1[:,-m:]) , dim=-1)
                    else:
                        raise ValueError('Larger hidden size is not supported yet.')
                    conv.weight.data = weight
                    # pass to next layer
                    x = conv(x, A)
                    x = F.relu(x) if posneg else x            
            else:
                raise NotImplementedError("Power method is not supported yet.")
#                 if self.beta > 0:
#                     n, c = x.size(0), self.nclass
#                     train_idx = full_data.train_mask.nonzero(as_tuple=False).squeeze()
#                     y_train = None if y is None else SparseTensor(row=train_idx, 
#                                    col=y.squeeze()[train_idx], sparse_sizes=(n, c))
#                     yyt_normalizer = y_train.matmul(y_train.sum(dim=0).view(-1,1)) + 1e-8
#                 for i, conv in enumerate(self.convs[:-1]):

            
def to_normalized_sparsetensor(edge_index, N, mode='DA'):
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5) 
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    if mode == 'DA':
        return deg_inv_sqrt.view(-1,1) * deg_inv_sqrt.view(-1,1) * adj
    if mode == 'DAD':
        return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

    
    
# here maybe I should set center = False
class GPCALayer(nn.Module):
    def __init__(self, nin, nout, alpha, beta, num_classes, center=True, n_powers=50, mode='DA'):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(nin, nout))
        self.bias = nn.Parameter(torch.FloatTensor(1, nout))
        self.nin = nin
        self.nhid = nout
        self.alpha = alpha
        self.beta = beta
        self.center = center
        self.n_powers = n_powers
        self.num_classes = num_classes
        self.mode = mode
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

    def forward(self, data, return_invphi_x=False, minibatch=False):
        """
            Assume data.adj is SparseTensor and normalized
        """
        # inputs
        n, c = data.x.size(0), self.num_classes
        if minibatch:
            edge_index, x, y, train_mask = data.edge_index, data.x, data.y, data.train_mask
            A = to_normalized_sparsetensor(edge_index, n, self.mode )
        else:    
            A, x, y, train_mask = data.adj, data.x, data.y, data.train_mask
        
        # one hot encoding of training labels
        if not hasattr(data, 'y_train'):
            train_idx = train_mask.nonzero(as_tuple=False).squeeze()
            y_train = None if y is None else SparseTensor(row=train_idx, 
                       col=y.squeeze()[train_idx], sparse_sizes=(n, c))
            data.y_train = y_train
        else:
            y_train = data.y_train
        # center
        if self.center:
            x = x - x.mean(dim=0)
            
        # calculate inverse of phi times x
        invphi_x = self.approximate_invphi_x(A, x, y_train)
        if return_invphi_x:
            return invphi_x, x
        else:     
            # AXW + bias
            return invphi_x.mm(self.weight) + self.bias
    
    def init(self, full_data, posneg=False):
        """
        Init always use full batch, same as inference/test. 
        """
        with torch.no_grad():
            invphi_x, x = self.forward(full_data, True)
            eig_val, eig_vec = torch.symeig(x.t().mm(invphi_x), eigenvectors=True)
            if self.nhid <= (int(posneg)+1)*self.nin:
                weight = torch.cat([eig_vec[:,-self.nhid//2:], -eig_vec[:,-self.nhid//2:]], dim=-1) \
                      if posneg else eig_vec[:, -self.nhid:] #when 

            elif self.nhid <= 2*(int(posneg)+1)*self.nin:
                eig_val1, eig_vec1 = torch.symeig(x.t().mm(x), eigenvectors=True)
                m = self.nhid % ((int(posneg)+1)*self.nin)
                weight = torch.cat([eig_vec, -eig_vec, eig_vec1[:, -m//2:], -eig_vec1[:, -m//2:]], dim=-1) \
                      if posneg else torch.cat([eig_vec, eig_vec1[:, -m:]], dim=-1)
                                
            elif self.nhid <= 3*(int(posneg)+1)*self.nin:
                eig_val1, eig_vec1 = torch.symeig(x.t().mm(x), eigenvectors=True)
                eig_val2, eig_vec2 = torch.symeig(invphi_x.t().mm(invphi_x), eigenvectors=True)
                m = self.nhid % ((int(posneg)+1)*self.nin)
                weight = torch.cat([eig_vec, eig_vec1, eig_vec2[:, -m//2:]
                             -eig_vec, -eig_vec1, -eig_vec2[:, -m//2:]], dim=-1) \
                      if posneg else torch.cat([eig_vec, eig_vec1, eig_vec2[:, -m:]], dim=-1)
            else:
                raise ValueError('Larger hidden size is not supported yet.')

            # assign 
            self.weight.data = weight
        return invphi_x.mm(self.weight) + self.bias
        
class GPCANet(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, alpha, beta, 
                 dropout=0, n_powers=10, center=True, act='ReLU', 
                 mode='DA', out_nlayer=1, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(nlayer-1):
            self.convs.append(
                GPCALayer(nhid if i>0 else nfeat, nhid, alpha, beta, nclass, center, n_powers, mode))
        # last layer
        self.convs.append(
            GPCALayer(nhid, nclass if out_nlayer==0 else nhid, alpha, beta, nclass, center, n_powers, mode))
        """
        out_nlayer = 0 should only be used for non frezzed setting
        """
         
        self.dropout = nn.Dropout(dropout)
        self.relu = getattr(nn,act)()
        
        # fc layers
        if out_nlayer == 0:
            self.out_mlp = nn.Identity()
        elif out_nlayer == 1:
            self.out_mlp = nn.Sequential(nn.Linear(nhid, nclass)) 
        else: 
            self.out_mlp = nn.Sequential(nn.Linear(nhid, nhid), self.relu, 
                                         self.dropout, nn.Linear(nhid, nclass))        
        
        self.cache = None # cannot be used when use batch training
        self.freeze_status = False # only support full batch

    def freeze(self, requires_grad=False):
        self.freeze_status = not requires_grad
        for conv in self.convs:
            conv.freeze(requires_grad)
             
    def forward(self, data, minibatch=False):
        # inputs
#         A, x, y, train_mask = data.adj, data.x, data.y, data.train_mask
#         n, c = data.num_nodes, data.num_classes
        if minibatch:
            self.freeze_status = False
            
        if self.freeze_status and self.cache is not None:
            return self.out_mlp(self.dropout(self.cache))

        original_x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(data, minibatch=minibatch)
            if not self.freeze_status:
                # don't do this when use plain GPCANet
                x = self.relu(x)
                x = self.dropout(x)
            data.x = x
    
        if self.freeze_status:
            self.cache = data.x
        
        out = self.out_mlp(self.dropout(data.x))
        data.x = original_x # restore 

        return out
        
    def init(self, full_data, posneg=False, **kwargs):
        """
        Init always use full batch, same as inference/test. 
        Btw, should we increase the scale of weight based on relu scale?
        Think about this later. 
        """
        self.eval()
        with torch.no_grad():
            original_x = full_data.x
            for i, conv in enumerate(self.convs):
                full_data.x = conv.init(full_data, posneg) # init using GPCA
                #----- init without relu and dropout?
                x = self.relu(x) if posneg else x
#                 x = self.dropout(x)
            full_data.x = original_x # restore 
        