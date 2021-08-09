gpu=0

seeds=(1010 0105 15213 15217 9195)
hiddens=(128 256)
dropouts=(0 0.2) # prefer a small dropout, not easy to overfit
layers=(2 3 5 10) # 15 is too large for hidden size 256
alphas=(1 5 10 20 50) # add 50 
betas=(0)

# -------------------------- GCN -------------------------
for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data arxiv --model GCN --lr 0.005 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu
done 
done 
done
done

# -------------------------- GPCA -------------------------
for seed in "${seeds[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for a in "${alphas[@]}"; do 
for b in "${betas[@]}"; do
    python main.py --log info --data arxiv --model GPCANet --nlayer 1 --lr 0.1 --wd 0 --nhid $hid --dropout $drop --alpha $a --beta $b --freeze --epochs 1000 --seed $seed --gpu $gpu
done
done
done
done
done

# -------------------------- GPCANet + FineTune ------------------------
for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for b in "${betas[@]}"; do
    python main.py --log info --data arxiv --model GPCANet --lr 0.005 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 500 --seed $seed --act ReLU --alpha 1 --beta $b --powers 5 --gpu $gpu
done 
done 
done
done
done

# -------------------------- GPCANet initialized GCN -------------------------
for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data arxiv --model GCN --lr 0.005 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu --init
done 
done 
done
done

