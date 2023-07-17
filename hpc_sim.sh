#!/bin/sh
#PBS -N ssd_sim_test
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=1:mem=24gb

set -e

module load anaconda3/personal
source activate stat-rethink

cd /rds/general/user/ag5818/projects/rleetmaa/live/ALI_DATA/NMR_Cambridge/results

wget -m ftp://68.195.245.254/results/* --ftp-user=K_Ungerleider_20230325 --ftp-password=WxsCcwg-8\>P\"WZ\'V

