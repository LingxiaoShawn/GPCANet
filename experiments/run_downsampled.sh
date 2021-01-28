# set seed and gpu
seeds=(0105 15213 15217 9195 1010)
seed=${seeds[$1]}
gpu=$1

hiddens=(128 256)
layers=(2 3 5 10 15) # prefer larger number of layers
###################################################
# arxiv
dropouts=(0.2) # prefer a small dropout, not easy to overfit
# --------------------------------------------------
for seed in "${seeds[@]}"; do 
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data arxiv --model GCN --lr 0.005 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu --init
    python main.py --log info --data arxiv --model GCN --lr 0.005 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu
done 
done 
done
done


####################################################
# products
dropouts=(0.1) # prefer small dropout
# --------------------------------------------------
for seed in "${seeds[@]}"; do 
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data products --model GCN --minibatch --lr 0.001 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 100 --seed $seed --gpu $gpu --init --batch 1024
    python main.py --log info --data products --model GCN --minibatch --lr 0.001 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 100 --seed $seed --gpu $gpu --batch 1024
done 
done 
done
done
