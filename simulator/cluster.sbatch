#!/bin/bash
#SBATCH --job-name=Serval           # Job name
#SBATCH -N1 --ntasks-per-node=2          # Number of nodes and >
#SBATCH --mem-per-cpu=16G           # Memory per core
#SBATCH --time=18:00:00                        # Duration of the job >
#SBATCH -oReport-%j.out                  # Combined output and >
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=aerol3@gatech.edu # E-mail address for noti>
#SBATCH --account=hive-dmahajan7               # Change to worki>
module load anaconda3
# Set up Conda environment
source activate
conda activate satSim

# clear log directory
rm -rf logs/*
rm -rf log/*

# Install dependencies
pip install networkx skyfield rtree shapely pyastronomy astropy cartopy itur
dir
python3 run.py


