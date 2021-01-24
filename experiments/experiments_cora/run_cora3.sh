
gpu=3
# full batch, fix learning rate 0.1, epochs 500
# Global configurations:
datasets=(cora citeseer pubmed)
seeds=(9195)  #0105 15213 15217 9195)
hiddens=(128 256)
weightdecays=('5e-4' '5e-3' '5e-2')
dropouts=(0 0.5)
alphas=(1 5 10 20 50) # add 50 
betas=(0 0.1 0.2)

# -------------------------- GCN -------------------------
layers=(2 3 5 10 15)
# ----
for seed in "${seeds[@]}"; do
for data in "${datasets[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for wd in "${weightdecays[@]}"; do
    python main.py --log info --data $data --model GCN --lr 0.001 --wd $wd --nlayer $layer --nhid $hid --dropout $drop --epochs 1000 --seed $seed --gpu $gpu
done 
done 
done
done
done
done 

# -------------------------- GPCA -------------------------
alphas=(1 5 10 20 50) # add 50 
betas=(0 0.1 0.2)
# ----
for seed in "${seeds[@]}"; do
for data in "${datasets[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for wd in "${weightdecays[@]}"; do
for a in "${alphas[@]}"; do 
for b in "${betas[@]}"; do
    python main.py --log info --data $data --model GPCANet --nlayer 1 --lr 0.1 --wd $wd --nhid $hid --dropout $drop --alpha $a --beta $b --freeze --epochs 500 --seed $seed --gpu $gpu
done
done
done
done
done
done
done


# -------------------------- GPCANet ------------------------
# Later can add alpha0.1, as growing layers equals to growing alpha, or just fix alpha.
layers=(2 3 5 10 15)
alphas=(1 5 10 20 50) # add 50   
betas=(0 0.1 0.2)
# ----
for seed in "${seeds[@]}"; do
for data in "${datasets[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for wd in "${weightdecays[@]}"; do
for a in "${alphas[@]}"; do 
for b in "${betas[@]}"; do
    python main.py --log info --data $data --model GPCANet --nlayer $layer --lr 0.1 --wd $wd --nhid $hid --dropout $drop --alpha $a --beta $b --freeze --epochs 500 --seed $seed --gpu $gpu
done
done
done
done
done
done
done
done

# -------------------------- GPCANet + FineTune ------------------------
layers=(2 3 5 10 15) # only tune beta to save time, fix alpha to be 1
betas=(0 0.1 0.2) # later we can choose the best configuration of GPCANet of a and b and then run
# ----
for seed in "${seeds[@]}"; do
for data in "${datasets[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for wd in "${weightdecays[@]}"; do
for b in "${betas[@]}"; do
    python main.py --log info --data $data --model GPCANet --lr 0.001 --wd $wd --nlayer $layer --nhid $hid --dropout $drop --epochs 1000 --seed $seed --act ReLU --alpha 1 --beta $b --gpu $gpu
done 
done 
done
done
done
done
done 
