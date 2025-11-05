# set working directory
setwd("/accounts/grad/aqwang/unilasso/analysis/unilasso_22data_default_lambda")

# Loop to run uniLasso analysis on all 22 datasets with 100 random splits (PARALLELIZED)
if (R.home() != "/scratch/users/aqwang/conda/envs/r_package/lib/R") {
  system("/scratch/users/aqwang/conda/envs/r_package/bin/Rscript unilasso_22data_100ran_split.R")
  quit("no")
}

# check the R environment is r_package
R.home()

# load packages 
library(uniLasso) # for unilasso 
library(glmnet) # for cv lasso 
library(parallel) # for parallel processing
library(doParallel) # for parallel processing with foreach
library(foreach) # for parallel loops

# Setup parallel processing
n_cores <- 32
cat("Setting up parallel processing with", n_cores, "cores\n")
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# Export necessary objects to all workers
clusterEvalQ(cl, {
  library(uniLasso)
  library(glmnet)
})

# Define all dataset names (without "data_" prefix)
dataset_names <- c(
  "airfoil", "appliances_energy", "auto_mpg", "automobile", "beijing_pm2.5",
  "bike_sharing", "concrete", "crime", "crime_unnormalized", "daily_demand", "facebook_metrics", "forest_fires", "infrared_thermo_temp",
   "liver_disorders", "metro_traffic", "parkinsons", "power_consumption", "power_plant", "real_estate",
  "servo", "stock_portfolio", "superconductivity")

# Number of random splits
N_SPLITS <- 100

# Function to run analysis for a single split
run_single_split <- function(split_id, X, y, dataset_name) {
  # Split data into training and test sets (different seed for each split)
  set.seed(split_id)
  # change the training size here if needed
  training_ratio <- 0.5
  training_size <- floor(training_ratio * nrow(X))
  train_indices <- sample(seq_len(nrow(X)), size = training_size)
  Xtr <- X[train_indices, ]
  ytr <- y[train_indices]
  Xte <- X[-train_indices, ]
  yte <- y[-train_indices]
  
  # Initialize results for this split
  split_results <- list()
  
  # ensuring the same CV folds for all methods within this split
  set.seed(42 + split_id)
  K <- 10
  n <- nrow(Xtr)
  foldid <- sample(rep(1:K, length.out = n))
  
  # Obtain leave one out coefficients from uniInfo
  tryCatch({
    uni_coefs <- uniInfo(Xtr, ytr,
                           family = "gaussian",
                           loo = FALSE)$beta  
    uni_coefs <- as.numeric(uni_coefs)
  }, error = function(e) {
    uni_coefs <<- rep(0, ncol(Xtr))  # fallback to zeros if uniInfo fails
  })
  
  # 1. UniLasso with loo=TRUE
  tryCatch({
    unilasso_cv_loo_true <- cv.uniLasso(
      Xtr, ytr,
      family = "gaussian",
      loo = TRUE,
      lower.limits = 0,
      standardize = FALSE,
      foldid = foldid,
      nlambda = 100
    )
    
    yte_pred_CV <- predict(unilasso_cv_loo_true, newx = Xte, s = "lambda.min")
    split_results$unilasso_loo_true_mse <- mean((yte_pred_CV - yte)^2)
    split_results$unilasso_loo_true_support <- sum(coef(unilasso_cv_loo_true, s = "lambda.min")[-1] != 0) # ignore the intercept 
    
    # Calculate sign differences
    beta_coef <- as.numeric(coef(unilasso_cv_loo_true, s = "lambda.min")[-1])  # p × 1 vector, exclude intercept
    active_index <- which(beta_coef != 0 & uni_coefs != 0)
    sign_beta_coef <- sign(beta_coef[active_index])
    sign_uni_coefs <- sign(uni_coefs[active_index])
    split_results$unilasso_loo_true_sign_diff <- sum(sign_beta_coef != sign_uni_coefs)  # number of sign differences
    
  }, error = function(e) {
    split_results$unilasso_loo_true_mse <<- NA
    split_results$unilasso_loo_true_support <<- NA
    split_results$unilasso_loo_true_sign_diff <<- NA
  })
  
  # 2. UniLasso with loo=FALSE
  tryCatch({
    unilasso_cv_loo_false <- cv.uniLasso(
      Xtr, ytr,
      family = "gaussian",
      loo = FALSE,
      lower.limits = 0,
      standardize = FALSE,
      foldid = foldid,
      nlambda = 100
    )
    
    yte_pred_CV <- predict(unilasso_cv_loo_false, newx = Xte, s = "lambda.min")
    split_results$unilasso_loo_false_mse <- mean((yte_pred_CV - yte)^2)
    split_results$unilasso_loo_false_support <- sum(coef(unilasso_cv_loo_false, s = "lambda.min")[-1] != 0)
    
    # Set sign difference to 0 for loo=FALSE (as specified in your code)
    split_results$unilasso_loo_false_sign_diff <- 0
    
  }, error = function(e) {
    split_results$unilasso_loo_false_mse <<- NA
    split_results$unilasso_loo_false_support <<- NA
    split_results$unilasso_loo_false_sign_diff <<- NA
  })
  
  # 3. Polish UniLasso
  tryCatch({
    pol <- polish.uniLasso(Xtr, ytr,
                          family = "gaussian",
                          foldid = foldid,
                          nlambda = 100, 
                          standardize = FALSE,
                          intercept = TRUE,
                          loo = TRUE)
    
    yte_pred_pol <- predict(pol, newx = Xte, s = "lambda.min")
    split_results$polish_unilasso_mse <- mean((yte_pred_pol - yte)^2)
    split_results$polish_unilasso_support <- sum(coef(pol, s = "lambda.min")[-1] != 0)
    
    # Calculate sign differences
    beta_coef <- as.numeric(coef(pol, s = "lambda.min")[-1])  # p × 1 vector, exclude intercept
    active_index <- which(beta_coef != 0 & uni_coefs != 0)
    sign_beta_coef <- sign(beta_coef[active_index])
    sign_uni_coefs <- sign(uni_coefs[active_index])
    split_results$polish_unilasso_sign_diff <- sum(sign_beta_coef != sign_uni_coefs)  # number of sign differences
    
  }, error = function(e) {
    split_results$polish_unilasso_mse <<- NA
    split_results$polish_unilasso_support <<- NA
    split_results$polish_unilasso_sign_diff <<- NA
  })
  
  # 4. Lasso with CV
  tryCatch({
    lasso_cv <- cv.glmnet(
      Xtr, ytr,
      family = "gaussian",
      alpha = 1,
      standardize = TRUE, #standardize for lasso
      foldid = foldid,
      nlambda = 100
    )
    
    yte_pred_CV <- predict(lasso_cv, newx = Xte, s = "lambda.min")
    split_results$lasso_cv_mse <- mean((yte_pred_CV - yte)^2)
    split_results$lasso_cv_support <- sum(coef(lasso_cv, s = "lambda.min")[-1] != 0)
    
    # Calculate sign differences
    beta_coef <- as.numeric(coef(lasso_cv, s = "lambda.min")[-1])  # p × 1 vector, exclude intercept
    active_index <- which(beta_coef != 0 & uni_coefs != 0)
    sign_beta_coef <- sign(beta_coef[active_index])
    sign_uni_coefs <- sign(uni_coefs[active_index])
    split_results$lasso_cv_sign_diff <- sum(sign_beta_coef != sign_uni_coefs)  # number of sign differences
    
  }, error = function(e) {
    split_results$lasso_cv_mse <<- NA
    split_results$lasso_cv_support <<- NA
    split_results$lasso_cv_sign_diff <<- NA
  })
  
  # 5. UniReg with CV
  tryCatch({
    uniReg_cv <- cv.uniReg(
      Xtr, ytr,
      family = "gaussian",
      standardize = FALSE,
      foldid = foldid,
      nlambda = 100,
      loo = TRUE
    )
    
    yte_pred_CV <- predict(uniReg_cv, newx = Xte, s = 0) # use s=0 for unireg
    split_results$unireg_mse <- mean((yte_pred_CV - yte)^2)
    split_results$unireg_support <- sum(coef(uniReg_cv, s = 0)[-1] != 0) # exclude intercept
    
    # Calculate sign differences
    beta_coef <- as.numeric(coef(uniReg_cv, s = 0)[-1])  # p × 1 vector, exclude intercept
    active_index <- which(beta_coef != 0 & uni_coefs != 0)
    sign_beta_coef <- sign(beta_coef[active_index])
    sign_uni_coefs <- sign(uni_coefs[active_index])
    split_results$unireg_sign_diff <- sum(sign_beta_coef != sign_uni_coefs)  # number of sign differences
    
  }, error = function(e) {
    split_results$unireg_mse <<- NA
    split_results$unireg_support <<- NA
    split_results$unireg_sign_diff <<- NA
  })
  
  # 6. Least Squares Regression (no regularization)
  tryCatch({
    # Convert to data frames for lm()
    Xtr_df <- as.data.frame(Xtr)
    Xte_df <- as.data.frame(Xte)
    
    ls_fit <- lm(ytr ~ ., data = Xtr_df)  # use . to include all variables
    yte_pred_lm <- predict(ls_fit, newdata = Xte_df)
    split_results$least_squares_mse <- mean((yte_pred_lm - yte)^2)
    split_results$least_squares_support <- sum(coef(ls_fit)[-1] != 0, na.rm = TRUE)  # exclude intercept
    
    # Calculate sign differences
    beta_coef <- as.numeric(coef(ls_fit)[-1])  # p × 1 vector, exclude intercept
    active_index <- which(beta_coef != 0 & uni_coefs != 0)
    sign_beta_coef <- sign(beta_coef[active_index])
    sign_uni_coefs <- sign(uni_coefs[active_index])
    split_results$least_squares_sign_diff <- sum(sign_beta_coef != sign_uni_coefs)  # number of sign differences
    
  }, error = function(e) {
    split_results$least_squares_mse <<- NA
    split_results$least_squares_support <<- NA
    split_results$least_squares_sign_diff <<- NA
  })
  
  # Return results as a data frame (updated to include sign differences)
  return(data.frame(
    dataset = dataset_name,
    split_id = split_id,
    unilasso_loo_true_mse = split_results$unilasso_loo_true_mse,
    unilasso_loo_true_support = split_results$unilasso_loo_true_support,
    unilasso_loo_true_sign_diff = split_results$unilasso_loo_true_sign_diff,
    unilasso_loo_false_mse = split_results$unilasso_loo_false_mse,
    unilasso_loo_false_support = split_results$unilasso_loo_false_support,
    unilasso_loo_false_sign_diff = split_results$unilasso_loo_false_sign_diff,
    polish_unilasso_mse = split_results$polish_unilasso_mse,
    polish_unilasso_support = split_results$polish_unilasso_support,
    polish_unilasso_sign_diff = split_results$polish_unilasso_sign_diff,
    lasso_cv_mse = split_results$lasso_cv_mse,
    lasso_cv_support = split_results$lasso_cv_support,
    lasso_cv_sign_diff = split_results$lasso_cv_sign_diff,
    unireg_mse = split_results$unireg_mse,
    unireg_support = split_results$unireg_support,
    unireg_sign_diff = split_results$unireg_sign_diff,
    least_squares_mse = split_results$least_squares_mse,
    least_squares_support = split_results$least_squares_support,
    least_squares_sign_diff = split_results$least_squares_sign_diff,
    stringsAsFactors = FALSE
  ))
}

# Initialize results storage for all splits (add sign difference columns)
results_all_splits <- data.frame(
  dataset = character(),
  split_id = integer(),
  unilasso_loo_true_mse = numeric(),
  unilasso_loo_true_support = numeric(),
  unilasso_loo_true_sign_diff = numeric(),
  unilasso_loo_false_mse = numeric(),
  unilasso_loo_false_support = numeric(),
  unilasso_loo_false_sign_diff = numeric(),
  polish_unilasso_mse = numeric(),
  polish_unilasso_support = numeric(),
  polish_unilasso_sign_diff = numeric(),
  lasso_cv_mse = numeric(),
  lasso_cv_support = numeric(),
  lasso_cv_sign_diff = numeric(),
  unireg_mse = numeric(),
  unireg_support = numeric(),
  unireg_sign_diff = numeric(),
  least_squares_mse = numeric(),
  least_squares_support = numeric(),
  least_squares_sign_diff = numeric(),
  stringsAsFactors = FALSE
)

# Start timing
start_time <- Sys.time()

# Loop through all datasets
for (i in seq_along(dataset_names)) {
  dataset_name <- dataset_names[i]

  # Load data
  X_path <- paste0("/accounts/grad/aqwang/unilasso/datasets_22/", dataset_name, "/X.csv")
  y_path <- paste0("/accounts/grad/aqwang/unilasso/datasets_22/", dataset_name, "/y.csv")
  
  X <- read.csv(X_path)
  X <- as.matrix(X)
  y <- read.csv(y_path)
  y <- as.numeric(unlist(y))  # Ensure y is numeric vector

  training_ratio <- 0.5

  # Visual separator for output 
  cat("\n", strrep("=", 80), "\n")
  cat("Processing dataset:", dataset_name, "(", i, "/", length(dataset_names), ")\n")
  cat("Running", N_SPLITS, "random train/test splits in parallel...\n")
  cat("Training ratio:", training_ratio, "\n")
  cat(strrep("=", 80), "\n")
  
  cat("Data dimensions: X =", dim(X), ", y =", length(y), "\n")
  
  # Run all splits for this dataset in parallel
  dataset_start_time <- Sys.time()
  
  # Use foreach for parallel execution
  dataset_results <- foreach(split_id = 1:N_SPLITS, 
                           .combine = rbind,  # Combine as data frame
                           .packages = c("uniLasso", "glmnet")) %dopar% {
    run_single_split(split_id, X, y, dataset_name)
  }
  
  dataset_end_time <- Sys.time()
  cat("Dataset completed in:", format(dataset_end_time - dataset_start_time), "\n")
  
  # Add to main results
  results_all_splits <- rbind(results_all_splits, dataset_results)
  
  # Summary statistics for this dataset
  cat("\nSummary statistics for", dataset_name, "(", N_SPLITS, "splits):\n")
  cat("  UniLasso (LOO=T):     Mean MSE =", sprintf("%.6f", mean(dataset_results$unilasso_loo_true_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$unilasso_loo_true_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$unilasso_loo_true_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$unilasso_loo_true_support, na.rm = TRUE)), ")\n")
  cat("                        Mean Sign Diff =", sprintf("%.2f", mean(dataset_results$unilasso_loo_true_sign_diff, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$unilasso_loo_true_sign_diff, na.rm = TRUE)), ")\n")
  
  cat("  UniLasso (LOO=F):     Mean MSE =", sprintf("%.6f", mean(dataset_results$unilasso_loo_false_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$unilasso_loo_false_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$unilasso_loo_false_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$unilasso_loo_false_support, na.rm = TRUE)), ")\n")
  cat("                        Mean Sign Diff =", sprintf("%.2f", mean(dataset_results$unilasso_loo_false_sign_diff, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$unilasso_loo_false_sign_diff, na.rm = TRUE)), ")\n")
  
  cat("  Polish UniLasso:      Mean MSE =", sprintf("%.6f", mean(dataset_results$polish_unilasso_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$polish_unilasso_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$polish_unilasso_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$polish_unilasso_support, na.rm = TRUE)), ")\n")
  cat("                        Mean Sign Diff =", sprintf("%.2f", mean(dataset_results$polish_unilasso_sign_diff, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$polish_unilasso_sign_diff, na.rm = TRUE)), ")\n")
  
  cat("  Lasso CV:             Mean MSE =", sprintf("%.6f", mean(dataset_results$lasso_cv_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$lasso_cv_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$lasso_cv_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$lasso_cv_support, na.rm = TRUE)), ")\n")
  cat("                        Mean Sign Diff =", sprintf("%.2f", mean(dataset_results$lasso_cv_sign_diff, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$lasso_cv_sign_diff, na.rm = TRUE)), ")\n")
  
  cat("  UniReg:               Mean MSE =", sprintf("%.6f", mean(dataset_results$unireg_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$unireg_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$unireg_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$unireg_support, na.rm = TRUE)), ")\n")
  cat("                        Mean Sign Diff =", sprintf("%.2f", mean(dataset_results$unireg_sign_diff, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$unireg_sign_diff, na.rm = TRUE)), ")\n")
  
  cat("  Least Squares:        Mean MSE =", sprintf("%.6f", mean(dataset_results$least_squares_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$least_squares_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$least_squares_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$least_squares_support, na.rm = TRUE)), ")\n")
  cat("                        Mean Sign Diff =", sprintf("%.2f", mean(dataset_results$least_squares_sign_diff, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$least_squares_sign_diff, na.rm = TRUE)), ")\n")
}

# Stop timing
end_time <- Sys.time()
total_time <- end_time - start_time

# Stop the cluster
stopCluster(cl)

cat("\n", strrep("=", 80), "\n")
cat("ANALYSIS COMPLETE FOR ALL", length(dataset_names), "DATASETS!\n")
cat("Total number of experiments:", nrow(results_all_splits), "\n")
cat("Total time elapsed:", format(total_time), "\n")
cat("Average time per dataset:", format(total_time / length(dataset_names)), "\n")
cat(strrep("=", 80), "\n")

# Save all individual results
training_ratio <- "50percent"
filename <- paste0("unilasso_22datasets_100splits_train", training_ratio, "_results.csv")
write.csv(results_all_splits, filename, row.names = FALSE)
cat("All individual results saved to:", filename, "\n")