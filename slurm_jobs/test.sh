#!/bin/bash
#SBATCH --time=0-1:00
#SBATCH --account=def-r6buchan
#SBATCH --mem=16G
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=2
#SBATCH --output=test.log
#SBATCH --mail-user=k24liang@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

cd ..
module load StdEnv/2020 python/3.9.6
module load gcc/9.3.0 cuda/11.4 opencv/4.7.0
source lg6d-env/bin/activate

python train.py --config config/Denoiser.py