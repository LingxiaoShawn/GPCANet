# Focus on finetuned version
# We can fix some parameter. 
# We focus on tuning alpha and beta.

# we focus on using lower number of layers with larger alpha. 

# L=1:    a=[10 20 30]
# L=2:    a=[5 10 15]
# L=3:    a=[3 5 10]
# beta [0 0.1 0.2]

seeds=(0105 15213 15217 9195 1010)
seed=${seeds[$1]}
gpu=$1

hiddens=(128 256)
weightdecays=('5e-4' '5e-3' '5e-2')
####################################################
# cora citeseer and pubmed
datasets=(cora citeseer pubmed)
dropouts=(0 0.5)
betas=(0 0.1 0.2)
# test alpba and beta later
# when alpha = 0, it's pca init
# --------------------------------------------------
# use both ceter and posneg to init
for seed in "${seeds[@]}"; do
for data in "${datasets[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for wd in "${weightdecays[@]}"; do
for b in "${betas[@]}"; do

alphas=(10 20 30)
for a in "${alphas[@]}"; do
    python main.py --log info --data $data --model GPCANet --lr 0.001 --wd $wd --nlayer 1 --nhid $hid --dropout $drop --epochs 1000 --seed $seed --gpu $gpu --beta $b --alpha a --powers 5
done 
alphas=(5 10 15)
for a in "${alphas[@]}"; do
    python main.py --log info --data $data --model GPCANet --lr 0.001 --wd $wd --nlayer 2 --nhid $hid --dropout $drop --epochs 1000 --seed $seed --gpu $gpu --beta $b --alpha a --powers 5
done 
alphas=(3 5 10)
for a in "${alphas[@]}"; do
    python main.py --log info --data $data --model GPCANet --lr 0.001 --wd $wd --nlayer 3 --nhid $hid --dropout $drop --epochs 1000 --seed $seed --gpu $gpu --beta $b --alpha a --powers 5
done 

done 
done
done
done
done
done


####################################################
# arxiv
dropouts=(0 0.2) # prefer a small dropout, not easy to overfit
# --------------------------------------------------

for seed in "${seeds[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do

alphas=(10 20 30)
for a in "${alphas[@]}"; do
    python main.py --log info --data arxiv --model GPCANet --lr 0.005 --wd 0 --nlayer 1 --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu  --alpha a --powers 5
done 
alphas=(5 10 15)
for a in "${alphas[@]}"; do
    python main.py --log info --data arxiv --model GPCANet --lr 0.005 --wd 0 --nlayer 2 --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu  --alpha a --powers 5
done 
alphas=(3 5 10)
for a in "${alphas[@]}"; do
    python main.py --log info --data arxiv --model GPCANet --lr 0.005 --wd 0 --nlayer 3 --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu  --alpha a --powers 5
done 

done 
done
done


####################################################
# products
dropouts=(0 0.1) # prefer small dropout
# --------------------------------------------------

for seed in "${seeds[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do

alphas=(10 20 30)
for a in "${alphas[@]}"; do
    python main.py --log info --data products --model GPCANet --lr 0.001 --wd 0 --nlayer 1 --nhid $hid --dropout $drop --epochs 100 --seed $seed --gpu $gpu  --alpha a --powers 5 --minibatch
done 
alphas=(5 10 15)
for a in "${alphas[@]}"; do
    python main.py --log info --data products --model GPCANet --lr 0.001 --wd 0 --nlayer 2 --nhid $hid --dropout $drop --epochs 100 --seed $seed --gpu $gpu  --alpha a --powers 5 --minibatch
done 
alphas=(3 5 10)
for a in "${alphas[@]}"; do
    python main.py --log info --data products --model GPCANet --lr 0.001 --wd 0 --nlayer 3 --nhid $hid --dropout $drop --epochs 100 --seed $seed --gpu $gpu  --alpha a --powers 5 --minibatch
done     
    
done 
done 
done
