#!/bin/bash
#SBATCH --time=1-00:00
#SBATCH --account=def-r6buchan
#SBATCH --partition=gpubase_bynode_b4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=498G
#SBATCH --output=test.log
#SBATCH --error=test.log
#SBATCH --mail-user=k24liang@uwaterloo.ca
#SBATCH --mail-type=BEGIN,END,FAIL

cd ..
module load StdEnv/2020 python/3.9.6
module load gcc/9.3.0 cuda/11.4 opencv/4.7.0
source lg6d-env/bin/activate

python train.py --config config/Denoiser.py
