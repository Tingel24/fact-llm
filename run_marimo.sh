#!/bin/bash
#SBATCH --job-name=marimo
#SBATCH --output=marimo-%j.out
#SBATCH --partition=scc-gpu
#SBATCH -G A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --time=1:00:00
#SBATCH -C inet

# module load or otherwise set up environment
module load miniforge3

source activate /mnt/vast-standard/home/lucajoshua.francis/u25472/fact-llm/llm-env


python faith_shop.py 
