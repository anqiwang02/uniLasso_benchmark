#!/bin/bash
#SBATCH --job-name=unilasso_17data_loop_coef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    # Request 32 cores
#SBATCH --partition=jsteinhardt                 # request partition
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aqwang@berkeley.edu
#SBATCH -o unilasso_17data_loop_basic_train50percent_coef.out
#SBATCH -e unilasso_17data_loop_basic_train50precent_coef.err


# Run your R script
/scratch/users/aqwang/conda/envs/r_package/bin/Rscript unilasso_17data_loop_coef.R
