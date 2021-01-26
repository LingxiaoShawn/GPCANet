# set seed and gpu
seeds=(0105 15213 15217 9195 1010)
seed=${seeds[$1]}
gpu=$1
####################################################
# cora citeseer and pubmed
datasets=(cora citeseer pubmed)
hiddens=(128 256)
weightdecays=('5e-4' '5e-3' '5e-2')
dropouts=(0 0.5)
layers=(2 3 5 10 15) # prefer larger number of layers
# test alpba and beta later
# when alpha = 0, it's pca init
# --------------------------------------------------
# use both ceter and posneg to init
for data in "${datasets[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for wd in "${weightdecays[@]}"; do
    python main.py --log info --data $data --model GCN --lr 0.001 --wd $wd --nlayer $layer --nhid $hid --dropout $drop --epochs 1000 --seed $seed --gpu $gpu --init
done 
done 
done
done
done


####################################################
# arxiv
dropouts=(0 0.2) # prefer a small dropout, not easy to overfit
# --------------------------------------------------
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data arxiv --model GCN --lr 0.005 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu --init
done 
done 
done

# makeup
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data arxiv --model GCN --lr 0.005 --wd 0 --nlayer 15 --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu
done 
done 



####################################################
# products
dropouts=(0 0.1) # prefer small dropout
hiddens=(128 256) # 256 is too large for memory, need 3090

for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data products --model GCN --minibatch --lr 0.001 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 100 --seed $seed --gpu $gpu --init
done 
done 
done

# makeup
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data products --model GCN --minibatch --lr 0.001 --wd 0 --nlayer 15 --nhid $hid --dropout $drop --epochs 100 --seed $seed --gpu $gpu
done 
done 