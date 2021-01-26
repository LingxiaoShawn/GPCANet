import os, torch, logging, argparse, json, random
import numpy as np
from data import load_data
from models import GPCANet, GCN
from utils import *
from torch_geometric.data import ClusterData, ClusterLoader, NeighborSampler

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
parser.add_argument('--init', action='store_true', default=False, help='Init GCN')
parser.add_argument('--act', type=str, default='ReLU', help='Activitation function in torch.nn')
parser.add_argument('--seed', type=int, default=1010, help='Random seed to use')
#TODO: gpu device
parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use')
# for minibatch
parser.add_argument('--minibatch', action='store_true', default=False, help='Whether use minibatch to train')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_partitions', type=int, default=15000)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--eval_steps', type=int, default=5)
# adj normalization
parser.add_argument('--adjmode', type=str, default='DA', help='{DA, DAD}')
# init settings
parser.add_argument('--posneg', action='store_true', default=False, help='Whether use +- eigenvectors')
parser.add_argument('--approx', action='store_true', default=False, help='Use truncated taylor to init GCN')
args = parser.parse_args()
# later change
args.posneg=True
args.approx=True

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# out dir 
OUT_PATH = f"results/gpu{args.gpu}"
if args.log == 'info':
    OUT_PATH = os.path.join(OUT_PATH, 'benchmarks')
    
# create model name 
model_name = ''
if args.model == 'GPCANet':
    if args.nlayer == 1 and args.freeze:
        model_name = 'GPCA'
    elif not args.freeze:
        model_name = 'GPCANet-Finetune'
    else:
        model_name = 'GPCANet-Plain'
else:
    model_name = args.model
    if args.init:
        model_name += '-Init'

# info 
description = f"D[{args.data}]-M[{model_name}]-h[{args.nhid}]" + \
              f"-l[{args.nlayer}]-a[{args.alpha}]-b[{args.beta}]" + \
              f"-lr[{args.lr}]-wd[{args.wd}]-drop[{args.dropout}]-freeze[{args.freeze}]-seed[{args.seed}]" + \
              f"-posneg[{args.posneg}]-approx[{args.approx}] "
print(description)
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
LOG_PATH = f"logs/gpu{args.gpu}"
if not os.path.isdir(LOG_PATH):
    os.makedirs(LOG_PATH)
logging_file = f'log-{model_name}'
logging_file += f'-{args.data}'
logging_file += '.txt'
logging_file = os.path.join(LOG_PATH, logging_file)


# setup logger
logging.basicConfig(format='%(message)s', filename=logging_file if args.log=='info' else None,
                    level=getattr(logging, args.log.upper())) 

logging.info("-"*50)
logging.info(description)

# later consider normalize when use it
data, dataset = load_data(args.data, args.adjmode)
logging.debug(f"Data statistics:  #features {dataset.num_features}, #nodes {data.x.size(0)}, #class {dataset.num_classes}")

# minibatch support
if args.minibatch:
    cluster_data = ClusterData(data, num_parts=args.num_partitions,
                               recursive=False, save_dir=dataset.processed_dir)
    dataloader = ClusterLoader(cluster_data, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)
    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=args.num_workers)
else:
    dataloader = data

# model 
# problem: how to split generate embedding from logistic regression. 
net = eval(args.model)(nfeat=data.num_features,
                       nhid=args.nhid, 
                       nclass=dataset.num_classes,
                       nlayer=args.nlayer, 
                       dropout=args.dropout,
                       alpha=args.alpha, 
                       beta=args.beta,
                       n_powers=args.powers,
                       act=args.act,
                       mode=args.adjmode,
                       out_nlayer=2 if args.data in ['arxiv', 'products'] else 1)

# cuda 
args.gpu = min(args.gpu, torch.cuda.device_count()-1)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
data.edge_index = None # delete to save memory
data.to(device)
net.to(device)

# freeze the model and init the model for GPCANet
if args.model == 'GPCANet':
    if args.freeze:
        net.freeze()
        net.init(data, center=True, posneg=False)
    else:
        net.init(data, center=True, posneg=args.posneg)

# init GCN (combine with GPCANet later)
if args.init:
    # use both ceter and posneg.maybe need ablation study later
    net.init(data, center=True, posneg=args.posneg, approximate=args.approx) 
    
# move data to cpu to save memory if minibatch
if args.minibatch:
    data = data.to('cpu')
    
# optimizer and criterion
optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wd)
criterion = torch.nn.CrossEntropyLoss()

# saving 
best_val_acc = 0
best_checkpoint = os.path.join(workspace, 'best_checkpoint.pkl')
# last_checkpoint = os.path.join(workspace, 'last_checkpoint.pkl')
# record training curve
records, records_file = [], os.path.join(workspace, 'training_curves.npy')

# training: need to split full-batch and mini-batch
for epoch in range(args.epochs):
    try:
        train_loss, train_acc = train(net, optimizer, criterion, dataloader, device, args.minibatch)
        # full batch evaluation: set a frequency
        if  epoch % args.eval_steps == 0:
            if args.minibatch:
                # clear cache for full batch operation
                torch.cuda.empty_cache()# use before full batch evaluation
            train_acc, val_acc, test_acc = evaluate(net, criterion, data, device, args.minibatch)
            logging.debug(f'Epoch: {epoch:04d}, '
                          f'Loss: {train_loss:.4f}, '
                          f'Train: {100 * train_acc:.4f}%, '
                          f'Valid: {100 * val_acc:.4f}%,'
                          f'Test: {100 * test_acc:.4f}% ')
            if best_val_acc <= val_acc:
                # do not use = to save running time
                best_val_acc = val_acc
                torch.save(net.state_dict(), best_checkpoint)
            records.append([epoch, train_loss, train_acc, val_acc, test_acc])
    except KeyboardInterrupt:
        break
        
# torch.save(net.state_dict(), last_checkpoint)
np.save(records_file, np.array(records))

# test 
net.load_state_dict(torch.load(best_checkpoint))
train_acc, val_acc, test_acc = evaluate(net, criterion, data, device, args.minibatch)

# remove saved model to save space
os.remove(best_checkpoint)

logging.info("-"*50)
logging.info( f'! Train: {100 * train_acc:.5f}%, '
                  f'Valid: {100 * val_acc:.5f}%, '
                  f'Test: {100 * test_acc:.5f}% ')

