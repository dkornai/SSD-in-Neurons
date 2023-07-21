#!/bin/sh
#PBS -N ssd_sim_test
#PBS -l walltime=00:15:00
#PBS -l select=1:ncpus=96:mem=256gb

cd /rds/general/user/dkornai/home/Projects/SSD-in-Neurons

module load anaconda3/personal
source activate ssdpy

python hpc.py
