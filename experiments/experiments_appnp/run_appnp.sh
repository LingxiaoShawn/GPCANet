gpu=0


seeds=(1010 0105 15213 15217 9195)
hiddens=(128 256)
weightdecays=('5e-4' '5e-3')
dropouts=(0 0.5)
datasets=(cora citeseer pubmed)

alphas=(1 5 10 20 50) # add 50 
for seed in "${seeds[@]}"; do
for data in "${datasets[@]}"; do
for a in "${alphas[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for wd in "${weightdecays[@]}"; do
    python main.py --log info --data $data --model APPNP --lr 0.01 --wd $wd  --nhid $hid --alpha $a --dropout $drop --epochs 500 --seed $seed --gpu $gpu 
done 
done 
done
done
done
done 