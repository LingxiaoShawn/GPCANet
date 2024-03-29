gpu=0

seeds=(1010 0105 15213 15217 9195)
layers=(2 3 5 10)
dropouts=(0 0.1) # prefer small dropout
hiddens=(128) # 256 is too large for memory

alphas=(1 5 10 20 50) # add 50 
betas=(0)

# -------------------------- GCN -------------------------
# ----
for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data products --model GCN --minibatch --lr 0.001 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 100 --seed $seed --gpu $gpu
done 
done 
done
done

# -------------------------- GPCA -------------------------
# ----
for seed in "${seeds[@]}"; do
for hid in "${hiddens[@]}"; do
for a in "${alphas[@]}"; do 
for b in "${betas[@]}"; do
    python main.py --log info --data products --model GPCANet --nlayer 1 --lr 0.01 --wd 0 --nhid $hid --alpha $a --beta $b --freeze --epochs 1000 --seed $seed --gpu $gpu
done
done
done
done

# -------------------------- GPCANet + Finetune ------------------------
# ----
for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for a in "${alphas[@]}"; do 
for b in "${betas[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data products --model GPCANet --nlayer $layer --alpha $a --lr 0.001 --wd 0 --nhid $hid  --beta $b --epochs 100 --minibatch --dropout $drop --act ReLU --seed $seed --gpu $gpu
done
done
done
done
done
done

