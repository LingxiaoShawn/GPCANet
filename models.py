import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

from torch_sparse import SparseTensor
from torch_sparse import spmm 
from torch_scatter import scatter_softmax
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.data import NeighborSampler


class APPNP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout=0, alpha=1, n_powers=10, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nlayer-1):
            self.layers.append(nn.Linear(nfeat if i==0 else nhid, nhid))
        self.layers.append(nn.Linear(nhid, nclass))

        self.dropout = dropout
        self.alpha = alpha
        self.n_powers = n_powers

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data, minibatch=False, **kwargs):
        # inputs
        if minibatch:
            edge_index, x = data.edge_index, data.x
            A = to_normalized_sparsetensor(edge_index, data.x.size(0), 'DA')
        else:    
            A, x = data.adj, data.x

        for layer in self.layers[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(layer(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return approximate_invphi_x(A, x, None, self.alpha, 0, self.n_powers)

    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout, nhead=4, **kwargs):
        super().__init__()
        self.in_linear = nn.Linear(nfeat, nhid)
        self.convs = torch.nn.ModuleList()
        for i in range(nlayer-1):
            # self.convs.append(GATConv(nhid if i>0 else nfeat, nhid, nhead, dropout))
            self.convs.append(GATConv(nhid, nhid, nhead, dropout))
        self.convs.append(GATConv(nhid, nclass, 1, dropout))
        self.dropout = dropout
        self.nlayer = nlayer

    def forward(self, data, **kwargs):
        # here can also do full batch, if gpu is not big enough, use cpu
        x, edge_index = data.x, data.edge_index
        # edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=len(x))
        x = self.in_linear(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x 

    def init(self, full_data, center=True, posneg=False, **kwargs):
        # now only support full batch init
        self.eval()
        with torch.no_grad():  
            x, edge_index = full_data.x, full_data.edge_index
            x = self.in_linear(x)
            x = F.elu(x)
            for i, conv in enumerate(self.convs[:-1]):
                x = conv.init(x, edge_index, posneg=posneg)
                x = F.elu(x)
            x = self.convs[-1].init(x, edge_index, posneg=posneg)
    
    def inference(self, full_data, device, batch_size=2**14, num_workers=8):
        # For products which is really large, we need to use mini-batch to do inference,
        # This is very slow
        subgraph_loader = NeighborSampler(full_data.edge_index, 
                                          node_idx=None, sizes=[-1],
                                          batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)
        all_x = full_data.x
        all_x = self.in_linear(all_x.to(device))
        all_x = F.elu(all_x).cpu()

        for i, conv in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = all_x[n_id].to(device)
                x = conv(x, edge_index)[:size[1]]
                if i < self.nlayer - 1:
                    x = F.elu(x)            
                xs.append(x.cpu())
            all_x = torch.cat(xs, dim=0)
        return all_x

class GATConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout):
        super().__init__()
        assert out_features % heads == 0
        out_perhead = out_features // heads
        self.graph_atts = nn.ModuleList([GraphAttConvOneHead(
               in_features, out_perhead, dropout=dropout) for _ in range(heads)])

    def forward(self, x, edge_index):
        output = torch.cat([att(x, edge_index) for att in self.graph_atts], dim=1)
        return output

    def init(self, x, edge_index, posneg=False, **kwargs):
        self.eval() 
        with torch.no_grad():
            output = torch.cat([att.init(x, edge_index, posneg=posneg) for att in self.graph_atts], dim=1)
        return output

class GraphAttConvOneHead(nn.Module):
    def __init__(self, in_features, out_features, dropout=0, alpha=0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, 2*in_features)))
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        # init 
        nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('relu')) # look at here
        nn.init.xavier_uniform_(self.a.data, gain=nn.init.calculate_gain('relu'))
        
        self.nin = in_features
        self.nhid = out_features
         
    def forward(self, x, edge_index, return_invphi_x=False):
        n = len(x)
        # h = torch.mm(x, self.weight) 
        h = x
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge_index[0]], h[edge_index[1]]), dim=1).t() # edge_h: 2*D x E
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze()) # E
        alpha = scatter_softmax(alpha, edge_index[0])
        output = spmm(edge_index, alpha, n, n, h) # h_prime: N x out
        if return_invphi_x:
            return output, h
        output = output.mm(self.weight)
        return output

    def init(self, x, edge_index, posneg=False, **kwargs):
        self.eval() 
        with torch.no_grad():
            invphi_x, x = self.forward(x, edge_index, return_invphi_x=True)
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
        return invphi_x.mm(self.weight)




class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout, mode, alpha=1, beta=0, n_powers=10, **kwargs):
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
        self.alpha = alpha
        self.beta = beta
        self.n_powers = n_powers

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
        self.eval() 
        with torch.no_grad():
            # use GPCANet to init GCN
            A, x, y = full_data.adj, full_data.x, full_data.y
            n, c = x.size(0), self.nclass
            train_idx = full_data.train_mask.nonzero(as_tuple=False).squeeze()
            y_train = None if y is None else SparseTensor(row=train_idx, 
                           col=y.squeeze()[train_idx], sparse_sizes=(n, c))
            for i, conv in enumerate(self.convs[:-1]):
                x_ = x - x.mean(dim=0) if center else x
                if approximate:
                    # use A as inv_phi, only support alpha=1 and beta=0
                    invphi_x = A.matmul(x_)
                else:
                    invphi_x = approximate_invphi_x(A, x_, y_train, self.alpha, self.beta, self.n_powers)  
                    
#                 eig_val0, eig_vec0 = torch.symeig(x_.t().mm(x_), eigenvectors=True)
                eig_val0, eig_vec0 = torch.symeig(x_.t().mm(invphi_x), eigenvectors=True)
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

def approximate_invphi_x(A, x, y, alpha=1, beta=0, n_powers=10):
    """
    Adding another option: use truncated taylor expansion when alpha<1
    """
    if y is not None and beta>0:
        yyt_normalizer = y.matmul(y.sum(dim=0).view(-1,1)) + 1e-8
    # init
    invphi_x = x
    # power method
    for _ in range(n_powers):
        part1 = A.matmul(invphi_x)
        if beta > 0:
            part2 = y.matmul(y.t().matmul(invphi_x))/yyt_normalizer
            part1 = (1-beta)*part1 + beta*part2
        invphi_x = alpha/(1+alpha)*part1 + 1/(1+alpha)*x
    return invphi_x    
    
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
                
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0) 

    def forward(self, data, return_invphi_x=False, minibatch=False, center=True):
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
        
        if center:
            x = x - x.mean(dim=0)
        # calculate inverse of phi times x
        if return_invphi_x:
            if center:
                x = x - x.mean(dim=0) # center
            invphi_x = approximate_invphi_x(A, x, y_train, self.alpha, self.beta, self.n_powers)
            return invphi_x, x
        else:     
            # AXW + bias
            invphi_x = approximate_invphi_x(A, x, y_train, self.alpha, self.beta, self.n_powers)
            return invphi_x.mm(self.weight) + self.bias
    
    def init(self, full_data, center=True, posneg=False):
        """
        Init always use full batch, same as inference/test. 
        """
        self.eval() 
        with torch.no_grad():
            invphi_x, x = self.forward(full_data, return_invphi_x=True, center=center)
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
            GPCALayer(nhid if nlayer>1 else nfeat, 
                      nclass if out_nlayer==0 else nhid, alpha, beta, nclass, center, n_powers, mode))
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
        
    def init(self, full_data, center=True, posneg=False, **kwargs):
        """
        Init always use full batch, same as inference/test. 
        Btw, should we increase the scale of weight based on relu scale?
        Think about this later. 
        """
        self.eval()
        with torch.no_grad():
            original_x = full_data.x
            for i, conv in enumerate(self.convs):
                # init
                conv.init(full_data, center, posneg) # init using GPCA
                # next layer
                x = conv(full_data)
                #----- init without relu and dropout?
                full_data.x = self.relu(x) if posneg else x
#                 x = self.dropout(x)
            full_data.x = original_x # restore 
        
