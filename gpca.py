import torch, os, shutil
import torch.nn as nn
from torch_sparse import SparseTensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.models.regression import LogisticRegression

class GPCA:
    def __init__(self, name, nhid, nlayer, alpha=1, beta=0, 
                 batch_size=1024, epochs=50, learning_rate=0.001, weight_decay=0,
                 act=nn.Identity(), **kwargs):
        self.nhid = nhid
        self.nlayer = nlayer
        self.alpha = alpha
        self.beta = beta
        self.act = act
        
        self.max_epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.model = None
        self.trainer = None
        self.working_dir = os.path.join('results', name)
        shutil.rmtree(self.working_dir, ignore_errors=True)
            
    def run(self, data, gpu=True, train_again=False):
        # data - a graph, need to create loader from the original data for clusterGCN
        device = torch.device("cuda"if torch.cuda.is_available() and gpu else "cpu")
        gpca_embeddings = gpca(data.to(device), self.nhid, self.nlayer, self.alpha, self.beta, self.act).to('cpu')
        data = data.to('cpu')
        # step 1: create dataloader 
        train_dataset = TabularDataset(gpca_embeddings[data.train_idx], data.y[data.train_idx])
        valid_dataset = TabularDataset(gpca_embeddings[data.valid_idx], data.y[data.valid_idx])
        test_dataset = TabularDataset(gpca_embeddings[data.test_idx], data.y[data.test_idx])
        nw = 6
        # DataLoader needs dataset in cpu when num_workers > 1
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=nw, pin_memory=True) 
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=nw, pin_memory=True) 
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=nw, pin_memory=False)
        
        # step 2: create model and trainer, only train the first time
        if self.trainer is None or train_again:
            self.trainer = pl.Trainer(gpus=1 if gpu else 0, max_epochs=self.max_epochs, default_root_dir=self.working_dir,
                                      callbacks=[ModelCheckpoint(monitor='val_acc', mode='max',dirpath=self.working_dir)])
        if self.model is None or train_again:
            self.model = LogisticRegression(input_dim=self.nhid, num_classes=data.num_classes, 
                                            learning_rate=self.learning_rate,
                                            l2_strength=self.weight_decay)
            self.trainer.fit(self.model, train_loader, valid_loader)
           
        
        # step 3: train and test
        valid_performance = self.trainer.test(self.model, test_dataloaders=valid_loader)
        test_performance = self.trainer.test(self.model, test_dataloaders=test_loader)
        return test_performance, valid_performance

def gpca(data, nhid=64, nlayer=1, alpha=1., beta=0, act=nn.Identity()):
    """
    Currently needs the help of SparseTensor package. 
    TODO:
        1. Extend to supervised GPCA: when beta> 0 we use supervised GPCA [different from Leman's] [Done]
        2. Test add some non-linearity
        3. Adapt to torch_geometric datasets
        4. Study different normalization of x and A
        5. Think about how to do this batch-wise, which is important for initialization,
           also when can not fit into GPU
    Inputs:
        data.x - torch.Tensor
        data.y - torch.Tensor
        data.adj - torch_sparse.SparseTensor
    Output:
        x - new embeddings after gpca
    """
    x = data.x    
    y_train = SparseTensor(row=data.train_idx, col=data.y.squeeze()[data.train_idx], 
                           sparse_sizes=(data.num_nodes, data.num_classes)) # one hot 
    for _ in range(nlayer):
        x = x - x.mean(dim=0)
        # x = x / (x.std(dim=0)+1e-6) # standarize to test 
        # pre_x = x
        inv_phi_times_x = power_method_with_beta(data.adj, x, y_train, alpha, beta)
        eig_val, eig_vec = torch.symeig(x.t().mm(inv_phi_times_x), eigenvectors=True)
        #weight = eig_vec[:,-nhid:] #when nhid is large than previous hidden size, we need to sample some eigenvectors.
        weight = torch.cat([eig_vec[:,-nhid//2:], -eig_vec[:,-nhid//2:]], dim=-1)
        x = inv_phi_times_x.mm(weight) 
        x = act(x)
        # x += pre_x
    return x

def power_method_with_beta(A, x, y, alpha=1, beta=0.1, t=50):
    # here y should only include training labels, nxc one hot
    inv_phi_times_x = x
    yyt_normalizer = y.matmul(y.sum(dim=0).view(-1,1)) + 1e-6
    for _ in range(t):
        part1 = A.matmul(inv_phi_times_x)
        if beta > 0:
            part2 = y.matmul(y.t().matmul(inv_phi_times_x))/yyt_normalizer
            part1 = (1-beta)*part1 + beta*part2
        inv_phi_times_x = alpha/(1+alpha)*part1 + 1/(1+alpha)*x
    return inv_phi_times_x

class TabularDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.transform = transform
        self.x = x
        self.y = y.squeeze()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx] 
    
if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    import argparse
    from data import load_data
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='arxiv', help='{cora, pubmed, citeseer, arxiv}.')
    parser.add_argument('--nhid', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--nlayer', type=int, default=3, help='Number of layers, works for Deep model.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--alpha', type=float, default=1, help='GPCA trade-off')
    parser.add_argument('--beta', type=float, default=0, help='Supervised GPCA trade-off')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay (L2 loss on parameters).')
    args = parser.parse_args()
    
    description = f"D[{args.data}]-h[{args.nhid}]-l[{args.nlayer}]-a[{args.alpha}]-b[{args.beta}]"
    model = GPCA(description, 
                nhid=args.nhid, 
                nlayer=args.nlayer, 
                alpha=args.alpha, 
                beta=args.beta, 
                batch_size=args.batch_size, 
                epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=args.wd)
    mode =  'DA'
    data = load_data(args.data, mode)
    test_performance, valid_performance = model.run(data)
