#!/bin/bash
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta16:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL

#echo commands to stdout
set -x

#move to your appropriate pylon5 directory
cd /pylon5/cc5pigp/lingxiao/GPCANet

# load anaconda
module load anaconda3

# activate environment
source activate pytorch1.6

# source experiments_products/run_products0.sh
# source experiments_products/run_products1.sh
# source experiments_products/run_products2.sh
source experiments_products/run_products3.sh
# nohup source experiments_products/run_products0.sh &
# nohup source experiments_products/run_products1.sh &
# nohup source experiments_products/run_products2.sh &
# nohup source experiments_products/run_products3.sh &

