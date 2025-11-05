#!/bin/bash
#SBATCH --job-name=unilasso_12data_ownlambda_coef
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32    # Request 32 cores
#SBATCH --partition=jsteinhardt                 # request partition
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aqwang@berkeley.edu
#SBATCH -o unilasso_12data_ownlambda_coef.out
#SBATCH -e unilasso_12data_ownlambda_coef.err


# Run your R script
/scratch/users/aqwang/conda/envs/r_package/bin/Rscript unilasso_12data_ownlambda_coef.R