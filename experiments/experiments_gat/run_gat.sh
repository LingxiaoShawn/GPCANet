gpu=0


seeds=(1010 0105 15213 15217 9195)
hiddens=(64 128)
layers=(2 3 5 10 15)

# -------------------------- GAT Cora CiteSeer PubMed -------------------------
datasets=(cora citeseer pubmed)
weightdecays=('5e-4' '5e-3')
dropouts=(0 0.5)

for seed in "${seeds[@]}"; do
for data in "${datasets[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
for wd in "${weightdecays[@]}"; do
    python main.py --log info --data $data --model GAT --lr 0.001 --wd $wd --nlayer $layer --nhid $hid --dropout $drop --epochs 500 --seed $seed --gpu $gpu 
done 
done 
done
done
done
done 

# python main.py --model GAT --data arxiv --nlayer 4 --nhid 256 --gpu $gpu --lr 0.005 --wd 0 --dropout 0 --epochs 1000


# -------------------------- GAT Arxiv -------------------------
hiddens=(128 256)
dropouts=(0 0.2)
for seed in "${seeds[@]}"; do
for layer in "${layers[@]}"; do
for hid in "${hiddens[@]}"; do
for drop in "${dropouts[@]}"; do
    python main.py --log info --data arxiv --model GAT --lr 0.005 --wd 0 --nlayer $layer --nhid $hid --dropout 0 --epochs 800 --seed $seed --gpu $gpu
done 
done
done
done


# -------------------------- GAT Product -------------------------
# too large to be run, cannot fit into memory for full-batch forward 


# hiddens=(128 256)
# for seed in "${seeds[@]}"; do
# for layer in "${layers[@]}"; do
# for hid in "${hiddens[@]}"; do
#     python main.py --log info --data products --model GAT --minibatch --lr 0.001 --wd 0 --nlayer $layer --nhid $hid --dropout 0 --epochs 100 --seed $seed --gpu $gpu 
# done 
# done
# done