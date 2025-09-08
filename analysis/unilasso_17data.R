#%%
if (R.home() != "/scratch/users/aqwang/conda/envs/r_package/lib/R") {
  system("/scratch/users/aqwang/conda/envs/r_package/bin/Rscript unilasso_17data.R")
  quit("no")
}

# %%
x <- rnorm(100)
mean(x)

# %%
plot(x)