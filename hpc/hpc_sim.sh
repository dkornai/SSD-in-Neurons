#!/bin/sh
#PBS -N ssd_sim_test
#PBS -l walltime=00:15:00
#PBS -l select=1:ncpus=48:mem=128gb

cd /rds/general/user/dkornai/home/github_clone/SSD-in-Neurons

module load anaconda3/personal
source activate ssdpy

python hpc.py
