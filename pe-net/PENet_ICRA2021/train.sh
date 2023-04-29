#!/bin/bash
#SBATCH --output=dpt_fusedtest5.out
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=80G
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH -C A100
#SBATCH -J enet_rs

module load cuda11.7/toolkit/11.7.1
module load cudnn8.5-cuda11.7/8.5.0.96
# conda activate pe_emv
export CUDA_VISIBLE_DEVICES="0"
# srun -l python main.py -b 1 -n pe --evaluate /home/trdhasade/DR/kitti_depth/Zip/pe.pth.tar --test --data-folder "/home/trdhasade/DR/kitti_depth/depth" --data-folder-rgb "/home/trdhasade/DR/kitti_depth/kitti_raw" --data-folder-save "/home/trdhasade/DR/PENet_ICRA2021/submit_test"
# srun -l python main.py -b 1 -n pe --evaluate "/home/trdhasade/DR/results_latest/input=rgbd.criterion=l2.lr=0.001.bs=10.wd=1e-06.jitter=0.1.time=2023-02-20@03-16/model_best.pth.tar" --data-folder "/home/trdhasade/DR/kitti_depth/depth" --data-folder-rgb "/home/trdhasade/DR/kitti_depth/kitti_raw" --data-folder-save "/home/trdhasade/DR/PENet_ICRA2021/submit_test" --test
# srun -l  python main.py -b 10 -n pe -he 160 -w 576 --resume "/home/trdhasade/DR/results/input=rgbd.criterion=l2.lr=0.001.bs=10.wd=1e-06.jitter=0.1.time=2023-03-19@14-47/checkpoint-73.pth.tar" --data-folder "/home/trdhasade/DR/kitti_depth/depth" --data-folder-rgb "/home/trdhasade/DR/kitti_depth/kitti_raw" --data-folder-save "/home/trdhasade/DR/PENet_ICRA2021/submit_test" --epochs 76
srun -l python main.py -b 4 -n e --data-folder "/home/trdhasade/DR/kitti_depth/depth" --data-folder-rgb "/home/trdhasade/DR/kitti_depth/kitti_raw" --epochs 30 --resume "/home/trdhasade/DR/results/input=rgbd.criterion=l2.lr=0.001.bs=4.wd=1e-06.jitter=0.1.time=2023-04-17@21-35/checkpoint-6.pth.tar"
# 128952