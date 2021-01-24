
gpu=2
# Need to set eval_steps
# Need to use two-layer MLP for out, also small amount of dropout helps
# For products, beta doesn't work, so we remove it to save time. This should because of 
# the hard split of the data. 

# prefer small dropout (0.1)

seeds=(15217) #  1010 0105 15213 15217 9195)
layers=(2 3 5 10)
dropouts=(0 0.1) # prefer small dropout
hiddens=(128) # 256 is too large for memory

alphas=(1 5 10 20 50) # add 50 
betas=(0)

# -------------------------- GCN -------------------------
# ----
# for seed in "${seeds[@]}"; do
# for layer in "${layers[@]}"; do
# for hid in "${hiddens[@]}"; do
# for drop in "${dropouts[@]}"; do
#     python main.py --log info --data products --model GCN --minibatch --lr 0.001 --wd 0 --nlayer $layer --nhid $hid --dropout $drop --epochs 100 --seed $seed --gpu $gpu
# done 
# done 
# done
# done

# -------------------------- GPCA -------------------------
# ----
dropouts=(0.1) # makeup 1

for seed in "${seeds[@]}"; do
for hid in "${hiddens[@]}"; do
for a in "${alphas[@]}"; do 
for b in "${betas[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data products --model GPCANet --nlayer 1 --lr 0.01 --wd 0 --nhid $hid --alpha $a --beta $b --freeze --epochs 1000 --seed $seed --gpu $gpu --dropout $drop
done
done
done
done
done

# -------------------------- GPCANet ------------------------ #here can add more alpha
# hiddens=(128)
# layers=(2 3 5 10 15)
# betas=(0 0.05 0.1 0.15)

alphas=(0.5 5) # additional 1     
dropouts=(0.1) # prefer small dropout
# # ----
for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for a in "${alphas[@]}"; do 
for b in "${betas[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data products --model GPCANet --nlayer $layer --alpha $a --lr 0.01 --wd 0 --nhid $hid  --beta $b --freeze --epochs 1000 --seed $seed --gpu $gpu --dropout $drop
done
done
done
done
done
done

# -------------------------- GPCANet + Finetune ------------------------
# ----
# alphas=(0.5 5) # additional 1     
# dropouts=(0 0.1) # prefer small dropout

# for seed in "${seeds[@]}"; do
# for layer in "${layers[@]}"; do
# for hid in "${hiddens[@]}"; do
# for a in "${alphas[@]}"; do 
# for b in "${betas[@]}"; do
# for drop in "${dropouts[@]}"; do
#     python main.py --log info --data products --model GPCANet --nlayer $layer --alpha $a --lr 0.001 --wd 0 --nhid $hid  --beta $b --epochs 100 --minibatch --dropout $drop --act ReLU --seed $seed --gpu $gpu
# done
# done
# done
# done
# done
# done

python main.py --log info --data products --model GPCANet --nlayer 10 --alpha 5 --lr 0.001 --wd 0 --nhid 128  --beta 0 --epochs 100 --minibatch --dropout 0.1 --act ReLU --seed $seed --gpu $gpu