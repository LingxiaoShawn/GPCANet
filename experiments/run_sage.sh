
gpu=0
# full batch, fix learning rate 0.1, epochs 500
# Global configurations:
datasets=(cora citeseer pubmed)
seeds=(1010 0105 15213 15217 9195)
hiddens=(128 256)
weightdecays=('5e-4' '5e-3' )
dropouts=(0 0.5)

# -------------------------- GCN -------------------------
layers=(2 3 5)
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