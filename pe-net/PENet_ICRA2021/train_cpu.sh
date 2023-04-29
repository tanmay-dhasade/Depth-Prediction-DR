#!/bin/bash
#SBATCH --output=simple.out
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:0
#SBATCH -J enet
module load cuda11.2/toolkit/11.2.2
module load cudnn8.1-cuda11.2/8.1.1.33 
source activate base
srun -l python main.py -b 4 -n e --data-folder "/home/trdhasade/DR/kitti_depth/depth" --data-folder-rgb "/home/trdhasade/DR/kitti_depth/kitti_raw" --epochs 1 --data-folder-save "/home/trdhasade/DR/PENet_ICRA2021/submit_test"