#!/bin/sh
#PBS -N ssd_sim_long
#PBS -l walltime=36:00:00
#PBS -l select=1:ncpus=256:mem=512gb

cd /rds/general/user/dkornai/home/Projects/SSD-in-Neurons

module load anaconda3/personal
source activate ssdpy

python cb_scan.py --model model_1 --delta 0.50 --replicates 10000 --time 30000
