import os, torch, logging, argparse, json
import numpy as np
from data import load_data
from models import GPCANet, GCN
from utils import *

# inputs
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='arxiv', help='{cora, pubmed, citeseer, arxiv}.')
parser.add_argument('--nhid', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--nlayer', type=int, default=3, help='Number of layers, works for Deep model.')
parser.add_argument('--alpha', type=float, default=1, help='GPCA trade-off')
parser.add_argument('--beta', type=float, default=0, help='Supervised GPCA trade-off')
parser.add_argument('--model', type=str, default='GPCANet', help='{GCN, GPCA}')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--log', type=str, default='debug', help='{info, debug}')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate.')
parser.add_argument('--powers', type=int, default=10, help='for approximaing inverse')
parser.add_argument('--freeze', action='store_true', default=False, help='Whether freeze weights of GPCANet')
parser.add_argument('--act', type=str, default='Identity', help='Activitation function in torch.nn')

args = parser.parse_args()

# out dir 
OUT_PATH = "results/"
if args.log == 'info':
    OUT_PATH = os.path.join(OUT_PATH, 'benchmarks')

# info 
description = f"D[{args.data}]-M[{args.model}]-h[{args.nhid}]" + \
              f"-l[{args.nlayer}]-a[{args.alpha}]-b[{args.beta}]" + \
              f"-lr[{args.lr}]-wd[{args.wd}]-drop[{args.dropout}]-freeze[{args.freeze}]"

# create work space
workspace = os.path.join(OUT_PATH, description)
if not os.path.isdir(workspace):
    os.makedirs(workspace)
    
# save args into file to use later
args_file = os.path.join(workspace, 'args.txt')
with open(args_file, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# load args
# with open(args_file, 'r') as f:
#     args.__dict__ = json.load(f)

# create logging file
logging_file = f'log-{args.model}'
if args.model == 'GPCANet':
    if args.nlayer == 1 and args.freeze:
        logging_file = 'log-GPCA-Logistic'
    elif not args.freeze:
        logging_file = 'log-GPCANet-Finetune'
    else:
        logging_file = 'log-GPCANet-Plain'
logging_file += '.txt'

# setup logger
logging.basicConfig(format='%(message)s', filename=logging_file if args.log=='info' else None,
                    level=getattr(logging, args.log.upper())) 

logging.info("-"*50)
logging.info(description)

# later consider normalize when use it
mode =  'DA'
data = load_data(args.data, mode)

# model 
# problem: how to split generate embedding from logistic regression. 
net = eval(args.model)(nfeat=data.num_features,
                       nhid=args.nhid, 
                       nclass=data.num_classes,
                       nlayer=args.nlayer, 
                       dropout=args.dropout,
                       alpha=args.alpha, 
                       beta=args.beta,
                       n_powers=args.powers,
                       act=args.act)

# cuda 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data.to(device)
net.to(device)

# freeze the model and init the model for GPCANet
if args.model == 'GPCANet':
    if args.freeze:
        net.freeze()
    net.init(data)
    
"""
    Here we can change GPCANet to embedder, and then we train logistic regression
    on top of it. 
    Pros: fast and scalable to large scale dataset, can use mini-batch to train LR which
    converges a lot faster.
    Cons: cannot modify embeddings based on the training loss.
    This is achieved in gpca.py
    -------------------------------
    The current implementation uses init based way. Problem is that training is full-batch,
    and I still didn't see the goodness of init with GPCANet. Also training GPCANet needs 
    more memory and cannot reach a really high number of layers.
"""

# optimizer and criterion
optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wd)
criterion = torch.nn.CrossEntropyLoss()

# saving 
best_val_acc = 0
best_checkpoint = os.path.join(workspace, 'best_checkpoint.pkl')
last_checkpoint = os.path.join(workspace, 'last_checkpoint.pkl')
# record training curve
records, records_file = [], os.path.join(workspace, 'training_curves.npy')

# training: need to split full-batch and mini-batch
for epoch in range(args.epochs):
    try:
        train_loss, train_acc = train(net, optimizer, criterion, data)
        train_acc, val_acc, test_acc = evalulate(net, criterion, data)
        logging.debug(f'Epoch: {epoch:02d}, '
                      f'Loss: {train_loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * val_acc:.2f}%,'
                      f'Test: {100 * test_acc:.2f}% ')
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save(net.state_dict(), best_checkpoint)
        records.append([train_loss, train_acc, val_acc, test_acc])
    except KeyboardInterrupt:
        break
        
torch.save(net.state_dict(), last_checkpoint)
np.save(records_file, np.array(records))

# test 
net.load_state_dict(torch.load(best_checkpoint))
train_acc, val_acc, test_acc = evalulate(net, criterion, data)

logging.info("-"*50)
logging.info( f'! Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * val_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}% ')

