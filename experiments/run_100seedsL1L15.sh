# maybe just run it for cora+arxiv. Ignore products. 
gpu=0
# best configs GCN
seeds=(9529 3885 4681 6030 1521 2346 9231  714 3869 3335 1361
       8281 5991 7675 9245 3894 5398 3518 9928 1899 6186 5613
       1392 7419 4856 4483 8347 3245 8354  505  755 6950 6107
       5460 3712 2885  653 9436  892 2300 3388 9663 2761 9273
       1101 9601 2426   16 5782 6984 1808  521 3529  350 2038
       8654 1791 1662  753 9994 4640 4065 7819 1429 6826 9824
       5860  736 3468 6004 3418  318  994 9445 2239 4005 3022
       7902 4666 8527 3692 3318 9107 1607  604 8331 3880 3903
       7355 3377 7583 1993 3076 4867 5932   99 4762 5732 4354
       1657)

# Cora:     GCN: L2- h128 wd0.005 drop0.5  | L15'h256 wd0.0005 drop0.5
# GCN-init       L2- h128 wd0.05 drop0.5   | L15 h256 wd0.005 drop0.0
for seed in "${seeds[@]}"; do
    python main.py --log info --data cora --model GCN --nlayer 2 --nhid 128 --wd 0.005 --drop 0.5 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu
    python main.py --log info --data cora --model GCN --nlayer 15 --nhid 256 --wd 0.0005 --drop 0.5 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu
    python main.py --log info --data cora --model GCN --nlayer 2 --nhid 128 --wd 0.05 --drop 0.5 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu --init
    python main.py --log info --data cora --model GCN --nlayer 15 --nhid 256 --wd 0.005 --drop 0 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu --init
done
 

# Citeseer: GCN L2- h128 wd0.05 drop0.5 | L15 h128 wd0.0005 drop0.0 
# GCN-init      L2- h256 wd0.05 drop0.5 | L15 h256 wd0.0005 drop0.0
for seed in "${seeds[@]}"; do
    python main.py --log info --data citeseer --model GCN --nlayer 2 --nhid 128 --wd 0.05 --drop 0.5 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu
    python main.py --log info --data citeseer --model GCN --nlayer 15 --nhid 128 --wd 0.0005 --drop 0 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu
    python main.py --log info --data citeseer --model GCN --nlayer 2 --nhid 256 --wd 0.05 --drop 0.5 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu --init
    python main.py --log info --data citeseer --model GCN --nlayer 15 --nhid 256 --wd 0.0005 --drop 0 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu --init
done


# PubMed:   GCN L2- h128 wd0.005 drop0.5 | L15 h128 wd0.005 drop0.0 
# GCN-init      L2- h128 wd0.005 drop0.5 | L15 h128 wd0.0005 drop0.0

for seed in "${seeds[@]}"; do
    python main.py --log info --data pubmed --model GCN --nlayer 2 --nhid 128 --wd 0.005 --drop 0.5 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu
    python main.py --log info --data pubmed --model GCN --nlayer 15 --nhid 128 --wd 0.005 --drop 0 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu
    python main.py --log info --data pubmed --model GCN --nlayer 2 --nhid 128 --wd 0.005 --drop 0.5 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu --init
    python main.py --log info --data pubmed --model GCN --nlayer 15 --nhid 128 --wd 0.0005 --drop 0 --lr 0.001 --epochs 1000 --seed $seed --gpu $gpu --init
done

# Arxiv:    GCN L2- h256 drop0.2 |  L15 h128 drop0.2
# GCN-init      L2- h256 drop0.2 |  L15 h128 drop0.2
for seed in "${seeds[@]}"; do
    python main.py --log info --data arxiv --model GCN --nlayer 2 --nhid 256 --wd 0 --drop 0.2 --lr 0.005 --epochs 500 --seed $seed --gpu $gpu
    python main.py --log info --data arxiv --model GCN --nlayer 15 --nhid 128 --wd 0 --drop 0.2 --lr 0.005 --epochs 500 --seed $seed --gpu $gpu
    python main.py --log info --data arxiv --model GCN --nlayer 2 --nhid 256 --wd 0 --drop 0.2 --lr 0.005 --epochs 500 --seed $seed --gpu $gpu --init
    python main.py --log info --data arxiv --model GCN --nlayer 15 --nhid 128 --wd 0 --drop 0.2 --lr 0.005 --epochs 500 --seed $seed --gpu $gpu --init
done

