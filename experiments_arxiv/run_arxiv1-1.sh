# problem: GPCANet+finetune train slow [make powers 5]

# Need to use two-layer MLP for out, also small amount of dropout helps

gpu=1

seeds=(15213) #1010 0105 15213 15217 9195)
hiddens=(128 256)
dropouts=(0 0.2) # prefer a small dropout, not easy to overfit
layers=(2 3 5 10) # 15 is too large for hidden size 256
alphas=(1 5 10 20 50) # add 50 
betas=(0 0.1 0.2)

# -------------------------- GCN -------------------------
layers=(2 3 5 10) # 15 is too large for hidden size 256
# ----
# for seed in "${seeds[@]}"; do
# for layer in "${layers[@]}"; do
# for hid in "${hiddens[@]}"; do
# for drop in "${dropouts[@]}"; do
#     python main.py --log info --data arxiv --model GCN --lr 0.005 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu
# done 
# done 
# done
# done

# -------------------------- GPCA -------------------------
alphas=(1 5 10 20 50) # add 50 
betas=(0 0.1 0.2)
# ----
# for seed in "${seeds[@]}"; do
# for hid in "${hiddens[@]}"; do
# for drop in "${dropouts[@]}"; do
# for a in "${alphas[@]}"; do 
# for b in "${betas[@]}"; do
#     python main.py --log info --data arxiv --model GPCANet --nlayer 1 --lr 0.1 --wd 0 --nhid $hid --dropout $drop --alpha $a --beta $b --freeze --epochs 1000 --seed $seed --gpu $gpu
# done
# done
# done
# done
# done

# -------------------------- GPCANet ------------------------
# Later can add alpha0.1, as growing layers equals to growing alpha, or just fix alpha.
layers=(2 3 5 10) # 15 is too large for hidden size 256
#alphas=(1 5 10 20 50) # add 50   
alphas=(0.5) # add 50   
betas=(0 0.1 0.2)
# ----
layers=(10) # continue
hiddens=(256)

for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for a in "${alphas[@]}"; do 
for b in "${betas[@]}"; do
    python main.py --log info --data arxiv --model GPCANet --nlayer $layer --lr 0.1 --wd 0 --nhid $hid --dropout $drop --alpha $a --beta $b --freeze --epochs 1000 --seed $seed --gpu $gpu
done
done
done
done
done
done


# -------------------------- GPCANet + FineTune ------------------------
layers=(2 3 5 10) # only tune beta to save time, fix alpha to be 1
hiddens=(128 256)
betas=(0 0.1 0.2) # later we can choose the best configuration of GPCANet of a and b and then run
alphas=(0.5 5) # add 50 
# ----
for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for a in "${alphas[@]}"; do 
for b in "${betas[@]}"; do
    python main.py --log info --data arxiv --model GPCANet --lr 0.005 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 500 --seed $seed --act ReLU --alpha $a --beta $b --powers 5 --gpu $gpu
done 
done 
done
done
done
done
