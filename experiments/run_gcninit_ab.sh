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
betas=(0.1 0.2)
# test alpba and beta later
# when alpha = 0, it's pca init
# --------------------------------------------------
# use both ceter and posneg to init
for seed in "${seeds[@]}"; do
for data in "${datasets[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for wd in "${weightdecays[@]}"; do
for b in "${betas[@]}"; do
    python main.py --log info --data $data --model GCN --lr 0.001 --wd $wd --nlayer $layer --nhid $hid --dropout $drop --epochs 1000 --seed $seed --gpu $gpu --init --beta $b
done 
done 
done
done
done
done
done




alphas=(0.5 5)
for seed in "${seeds[@]}"; do
for data in "${datasets[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for wd in "${weightdecays[@]}"; do
for a in "${alphas[@]}"; do
    python main.py --log info --data $data --model GCN --lr 0.001 --wd $wd --nlayer $layer --nhid $hid --dropout $drop --epochs 1000 --seed $seed --gpu $gpu --init --alpha $a 
done 
done 
done
done
done
done
done
