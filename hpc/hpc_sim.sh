#!/bin/sh
#PBS -N ssd_sim_test
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=192:mem=512gb

cd /rds/general/user/dkornai/home/Projects/SSD-in-Neurons

module load anaconda3/personal
source activate ssdpy

python hpc.py
