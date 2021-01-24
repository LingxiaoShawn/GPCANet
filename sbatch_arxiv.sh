#!/bin/bash
#SBATCH --partition=GPU-AI
#SBATCH --nodes=1
#SBATCH --gres=gpu:volta16:1
#SBATCH --time=8:00:00
#SBATCH --mail-type=ALL

#echo commands to stdout
set -x

#move to your appropriate pylon5 directory
cd /pylon5/cc5pigp/lingxiao/GPCANet

# load anaconda
module load anaconda3

# activate environment
source activate pytorch1.6

# source experiments_arxiv/run_arxiv0.sh
# source experiments_arxiv/run_arxiv1.sh
# source experiments_arxiv/run_arxiv2.sh
# source experiments_arxiv/run_arxiv3.sh

# source experiments_arxiv/run_arxiv0-1.sh
# source experiments_arxiv/run_arxiv1-1.sh
# source experiments_arxiv/run_arxiv2-1.sh
source experiments_arxiv/run_arxiv3-1.sh