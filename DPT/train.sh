#!/bin/bash
#SBATCH --output=t_t3est.out
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:0
#SBATCH -C A100
#SBATCH -J enet_rs

# module load cuda11.2/toolkit/11.2.2
# module load cudnn8.1-cuda11.2/8.1.1.33
conda activate transf_env
export CUDA_VISIBLE_DEVICES="0"
# srun -l python main.py -b 1 -n pe --evaluate /home/trdhasade/DR/kitti_depth/Zip/pe.pth.tar --test --data-folder "/home/trdhasade/DR/kitti_depth/depth" --data-folder-rgb "/home/trdhasade/DR/kitti_depth/kitti_raw" --data-folder-save "/home/trdhasade/DR/PENet_ICRA2021/submit_test"
srun -l python run_monodepth.py -i "/home/trdhasade/DR/kitti_depth/depth/data_depth_selection/val_selection_cropped/image" -m '/home/trdhasade/DR/DPT/weights/dpt_hybrid_kitti-cb926ef4.pt'
# run_segmentation.py -i "/home/trdhasade/DR/kitti_depth/depth/data_depth_selection/val_selection_cropped/image" -m "/home/trdhasade/DR/DPT/weights/dpt_hybrid-ade20k-53898607.pt"
# srun -l  python main.py -b 10 -n pe -he 160 -w 576 --resume "/home/trdhasade/DR/results/input=rgbd.criterion=l2.lr=0.001.bs=10.wd=1e-06.jitter=0.1.time=2023-03-16@23-44/checkpoint-45.pth.tar" --data-folder "/home/trdhasade/DR/kitti_depth/depth" --data-folder-rgb "/home/trdhasade/DR/kitti_depth/kitti_raw" --data-folder-save "/home/trdhasade/DR/PENet_ICRA2021/submit_test" --epochs 76
