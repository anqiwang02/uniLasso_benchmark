#!/bin/bash
#SBATCH --job-name=unilasso_17data_bootstrap_train50percent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    # Request 32 cores
#SBATCH --partition=jsteinhardt                 # request partition
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aqwang@berkeley.edu
#SBATCH -o unilasso_17databootstrap_train50percent.out
#SBATCH -e unilasso_17databootstrap_train50percent.err


# Run your R script
/scratch/users/aqwang/conda/envs/r_package/bin/Rscript unilasso_17data_loop_bootstrap.R