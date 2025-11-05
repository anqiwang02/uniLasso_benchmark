#!/bin/bash
#SBATCH --job-name=unilasso_22data_coef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # Request 24 cores
#SBATCH --partition=jsteinhardt                 # request partition
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aqwang@berkeley.edu
#SBATCH -o unilasso_22data_coef.out
#SBATCH -e unilasso_22data_coef.err


# Run your R script
/scratch/users/aqwang/conda/envs/r_package/bin/Rscript unilasso_22data_coef.R
