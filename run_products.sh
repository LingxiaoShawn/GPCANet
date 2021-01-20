
# -------------------------- GCN -------------------------
layers=(2 3 5 10 15)
hiddens=(128)
dropouts=(0 0.5)
# ----
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data products --model GCN --minibatch --lr 0.001 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 50
done 
done 
done

# -------------------------- GPCA -------------------------
hiddens=(128)
alphas=(1 5 10 20)
betas=(0 0.05 0.1 0.15)
# ----
for hid in "${hiddens[@]}"; do
for a in "${alphas[@]}"; do 
for b in "${betas[@]}"; do
    python main.py --log info --data products --model GPCANet --nlayer 1 --lr 0.01 --wd 0 --nhid $hid --alpha $a --beta $b --freeze --epochs 500
done
done
done

# -------------------------- GPCANet ------------------------
hiddens=(128)
layers=(2 3 5 10 15)
betas=(0 0.05 0.1 0.15)
# ----
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for b in "${betas[@]}"; do
    python main.py --log info --data products --model GPCANet --nlayer $layer --alpha 1 --lr 0.01 --wd 0 --nhid $hid  --beta $b --freeze --epochs 500
done
done
done

# -------------------------- GPCANet + Finetune ------------------------
hiddens=(128)
layers=(2 3 5 10 15)
betas=(0 0.05 0.1 0.15)
dropouts=(0 0.5)
# ----
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for b in "${betas[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data products --model GPCANet --nlayer $layer --alpha 1 --lr 0.001 --wd 0 --nhid $hid  --beta $b --epochs 50 --minibatch --dropout $drop
done
done
done
done


