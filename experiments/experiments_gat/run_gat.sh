seeds=(1010 0105 15213 15217 9195)

gpu=4
seeds=(9195)
hiddens=(128 256)
layers=(2 3)
alphas=(1 5 10 20 50) # add 50 

# -------------------------- GAT Cora CiteSeer PubMed -------------------------
# datasets=(cora citeseer pubmed)
# weightdecays=('5e-4' '5e-3')
# dropouts=(0 0.5)

# alphas=(1 5 10 20 50) # add 50 

# for seed in "${seeds[@]}"; do
# for data in "${datasets[@]}"; do
# for layer in "${layers[@]}"; do
# for hid in "${hiddens[@]}"; do
# for drop in "${dropouts[@]}"; do
# for wd in "${weightdecays[@]}"; do
#     python main.py --log info --data $data --model GAT --lr 0.001 --wd $wd --nlayer $layer --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu 
# done 
# done 
# done
# done
# done
# # done 

# # for seed in "${seeds[@]}"; do
# for data in "${datasets[@]}"; do
# for a in "${alphas[@]}"; do
# for hid in "${hiddens[@]}"; do
# for drop in "${dropouts[@]}"; do
# for wd in "${weightdecays[@]}"; do
#     python main.py --log info --data $data --model APPNP --lr 0.001 --wd $wd  --nhid $hid --alpha $a --dropout $drop --epochs 500 --seed $seed --gpu $gpu 
# done 
# done 
# done
# done
# done
# done 

# python main.py --model GAT --data arxiv --nlayer 4 --nhid 256 --gpu $gpu --lr 0.005 --wd 0 --dropout 0 --epochs 1000


# --------------------------  Arxiv -------------------------
# hiddens=(128 256)
dropouts=(0 0.2)
for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data arxiv --model GAT --lr 0.005 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 800 --seed $seed --gpu $gpu
done 
done
done
done


for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for a in "${alphas[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data arxiv --model APPNP --lr 0.01 --wd 0 --nlayer $layer --nhid $hid --alpha $a --dropout $drop --epochs 800 --seed $seed --gpu $gpu  
done 
done
done
done 
done


--------------------------  Product -------------------------
dropouts=(0 0.1)
# too large to be run, cannot fit into memory for full-batch forward 
for seed in "${seeds[@]}"; do
for a in "${alphas[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data products --model APPNP --lr 0.001 --wd 0 --nlayer 2  --nhid $hid --alpha $a --dropout $drop --epochs 100 --seed $seed --gpu $gpu  --minibatch 
done 
done
done
done 




# hiddens=(128 256)
# for seed in "${seeds[@]}"; do
# for layer in "${layers[@]}"; do
# for hid in "${hiddens[@]}"; do
#     python main.py --log info --data products --model GAT --minibatch --lr 0.001 --wd 0 --nlayer $layer --nhid $hid --dropout 0 --epochs 100 --seed $seed --gpu $gpu 
# done 
# done
# done