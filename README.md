# GPCANet
Code for paper [Connecting Graph Convolutional Networks and Graph-Regularized PCA](https://arxiv.org/abs/2006.12294)

**Supported datasets**: `cora`, `citeseer`, `pubmed`, `arxiv`, `products`.   
The last two are from OGB datasets. And `products` dataset needs `--minibatch` option. 


### run GPCA 
Note that --freeze and --nlayer 1 are required for plain GPCA. 
```python 
python main.py --data arxiv --model GPCANet --nlayer 1 --lr 0.1 --freeze --alpha 20 
```

### run GPCANet 
```python
python main.py --data arxiv --model GPCANet --lr 0.005 --nlayer 3 --alpha 1 
```

### run GCN (or GAT APPNP)
```python 
python main.py --data arxiv --model GCN --lr 0.005 --nlayer 3
python main.py --data arxiv --model GAT --lr 0.005 --nlayer 3
python main.py --data arxiv --model APPNP --lr 0.005 --nlayer 3
```

### run GPCANet-initialized GCN
```python 
python main.py --data arxiv --model GCN --lr 0.005 --nlayer 3 --init
```

### Bash scripts with hyperparameter search
See `scripts` folder. 



