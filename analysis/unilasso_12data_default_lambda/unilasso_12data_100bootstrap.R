# Loop to run uniLasso analysis on all 12 datasets with bootstrap sampling (PARALLELIZED)
if (R.home() != "/scratch/users/aqwang/conda/envs/r_package/lib/R") {
  system("/scratch/users/aqwang/conda/envs/r_package/bin/Rscript unilasso_12data_100bootstrap.R")
  quit("no")
}

# check the R environment is r_package
R.home()

# set working directory
setwd("/accounts/grad/aqwang/unilasso/analysis/unilasso_12data_default_lambda")

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

# Define all dataset names
dataset_names <- c(
"ca_housing", "computer",
  "debutanizer", "diamond", "elevator", "energy_efficiency",
  "insurance", "kin8nm", "miami_housing", "naval_propulsion",
 "protein_structure", "qsar")

# Number of bootstrap samples
N_BOOTSTRAP <- 100


# Function to run analysis for a single bootstrap sample
run_single_bootstrap <- function(bootstrap_id, Xtr_original, ytr_original, Xte, yte, dataset_name) {
  # Bootstrap sample the training data (sample with replacement)
  set.seed(bootstrap_id)
  n_train <- nrow(Xtr_original)
  bootstrap_indices <- sample(1:n_train, size = n_train, replace = TRUE)
  
  # Find out-of-bag (OOB) indices - training points NOT in bootstrap sample
  oob_indices <- setdiff(1:n_train, unique(bootstrap_indices))
  
  Xtr_boot <- Xtr_original[bootstrap_indices, ]
  ytr_boot <- ytr_original[bootstrap_indices]
  
  # Initialize results for this bootstrap sample
  boot_results <- list()
  
  # ensuring the same CV folds for all methods within this bootstrap
  set.seed(42 + bootstrap_id)
  K <- 10
  n <- nrow(Xtr_boot)
  foldid <- sample(rep(1:K, length.out = n))
  
  # 1. UniLasso with loo=TRUE
  tryCatch({
    unilasso_cv_loo_true <- cv.uniLasso(
      Xtr_boot, ytr_boot,
      family = "gaussian",
      loo = TRUE,
      lower.limits = 0,
      standardize = FALSE,
      foldid = foldid,
      nlambda = 100
    )
    
    # Test set predictions
    yte_pred_CV <- predict(unilasso_cv_loo_true, newx = Xte, s = "lambda.min")
    boot_results$unilasso_loo_true_mse <- mean((yte_pred_CV - yte)^2)
    boot_results$unilasso_loo_true_support <- sum(coef(unilasso_cv_loo_true, s = "lambda.min")[-1] != 0)
    
    # Out-of-bag predictions and residuals
    if (length(oob_indices) > 0) {
      Xoob <- Xtr_original[oob_indices, , drop = FALSE]
      yoob <- ytr_original[oob_indices]
      yoob_pred <- predict(unilasso_cv_loo_true, newx = Xoob, s = "lambda.min")
      oob_residuals <- yoob - as.vector(yoob_pred)
      
      # Add random OOB residual to each test prediction for uncertainty
      sampled_residual <- sample(oob_residuals, size = length(yte), replace = TRUE)
      yte_pred_with_oob <- as.vector(yte_pred_CV) + sampled_residual
    } else {
      # If no OOB samples, use regular prediction
      yte_pred_with_oob <- as.vector(yte_pred_CV)
    }
    
    boot_results$unilasso_loo_true_predictions <- yte_pred_with_oob
    boot_results$unilasso_loo_true_oob_count <- length(oob_indices)
    
  }, error = function(e) {
    boot_results$unilasso_loo_true_mse <<- NA
    boot_results$unilasso_loo_true_support <<- NA
    boot_results$unilasso_loo_true_predictions <<- rep(NA, length(yte))
    boot_results$unilasso_loo_true_oob_count <<- 0
  })
  
  # 2. UniLasso with loo=FALSE
  tryCatch({
    unilasso_cv_loo_false <- cv.uniLasso(
      Xtr_boot, ytr_boot,
      family = "gaussian",
      loo = FALSE,
      lower.limits = 0,
      standardize = FALSE,
      foldid = foldid,
      nlambda = 100
    )
    
    yte_pred_CV <- predict(unilasso_cv_loo_false, newx = Xte, s = "lambda.min")
    boot_results$unilasso_loo_false_mse <- mean((yte_pred_CV - yte)^2)
    boot_results$unilasso_loo_false_support <- sum(coef(unilasso_cv_loo_false, s = "lambda.min")[-1] != 0)
    
    # Add OOB residuals
    if (length(oob_indices) > 0) {
      Xoob <- Xtr_original[oob_indices, , drop = FALSE]
      yoob <- ytr_original[oob_indices]
      yoob_pred <- predict(unilasso_cv_loo_false, newx = Xoob, s = "lambda.min")
      oob_residuals <- yoob - as.vector(yoob_pred)
      sampled_residual <- sample(oob_residuals, size = length(yte), replace = TRUE)
      yte_pred_with_oob <- as.vector(yte_pred_CV) + sampled_residual
    } else {
      yte_pred_with_oob <- as.vector(yte_pred_CV)
    }
    
    boot_results$unilasso_loo_false_predictions <- yte_pred_with_oob
    
  }, error = function(e) {
    boot_results$unilasso_loo_false_mse <<- NA
    boot_results$unilasso_loo_false_support <<- NA
    boot_results$unilasso_loo_false_predictions <<- rep(NA, length(yte))
  })
  
  # 3. Polish UniLasso
  tryCatch({
    pol <- polish.uniLasso(Xtr_boot, ytr_boot,
                          family = "gaussian",
                          foldid = foldid,
                          nlambda = 100, 
                          standardize = FALSE,
                          intercept = TRUE,
                          loo = TRUE)
    
    yte_pred_pol <- predict(pol, newx = Xte, s = "lambda.min")
    boot_results$polish_unilasso_mse <- mean((yte_pred_pol - yte)^2)
    boot_results$polish_unilasso_support <- sum(coef(pol, s = "lambda.min")[-1] != 0)
    
    # Add OOB residuals for prediction interval 
    if (length(oob_indices) > 0) {
      Xoob <- Xtr_original[oob_indices, , drop = FALSE]
      yoob <- ytr_original[oob_indices]
      yoob_pred <- predict(pol, newx = Xoob, s = "lambda.min")
      oob_residuals <- yoob - as.vector(yoob_pred)
      sampled_residual <- sample(oob_residuals, size = length(yte), replace = TRUE)
      yte_pred_with_oob <- as.vector(yte_pred_pol) + sampled_residual
    } else {
      yte_pred_with_oob <- as.vector(yte_pred_pol)
    }
    
    boot_results$polish_unilasso_predictions <- yte_pred_with_oob
    
  }, error = function(e) {
    boot_results$polish_unilasso_mse <<- NA
    boot_results$polish_unilasso_support <<- NA
    boot_results$polish_unilasso_predictions <<- rep(NA, length(yte))
  })
  
  # 4. Lasso with CV
  tryCatch({
    lasso_cv <- cv.glmnet(
      Xtr_boot, ytr_boot,
      family = "gaussian",
      alpha = 1,
      standardize = TRUE, #always standardize for lasso
      foldid = foldid,
      nlambda = 100
    )
    
    yte_pred_CV <- predict(lasso_cv, newx = Xte, s = "lambda.min")
    boot_results$lasso_cv_mse <- mean((yte_pred_CV - yte)^2)
    boot_results$lasso_cv_support <- sum(coef(lasso_cv, s = "lambda.min")[-1] != 0)
    
    # Add OOB residuals
    if (length(oob_indices) > 0) {
      Xoob <- Xtr_original[oob_indices, , drop = FALSE]
      yoob <- ytr_original[oob_indices]
      yoob_pred <- predict(lasso_cv, newx = Xoob, s = "lambda.min")
      oob_residuals <- yoob - as.vector(yoob_pred)
      sampled_residual <- sample(oob_residuals, size = length(yte), replace = TRUE)
      yte_pred_with_oob <- as.vector(yte_pred_CV) + sampled_residual
    } else {
      yte_pred_with_oob <- as.vector(yte_pred_CV)
    }
    
    boot_results$lasso_cv_predictions <- yte_pred_with_oob
    
  }, error = function(e) {
    boot_results$lasso_cv_mse <<- NA
    boot_results$lasso_cv_support <<- NA
    boot_results$lasso_cv_predictions <<- rep(NA, length(yte))
  })

  # 5. UniReg with CV
  tryCatch({
  uniReg_cv <- cv.uniReg(
    Xtr_boot, ytr_boot,
    family = "gaussian",
    standardize = FALSE,
    foldid = foldid,
    nlambda = 100,
    loo = TRUE
  )

  yte_pred_CV <- predict(uniReg_cv, newx = Xte, s = 0)
  boot_results$unireg_mse <- mean((yte_pred_CV - yte)^2)
  boot_results$unireg_support <- sum(coef(uniReg_cv, s = 0)[-1] != 0) # exclude intercept

  # Add OOB residuals
  if (length(oob_indices) > 0) {
    Xoob <- Xtr_original[oob_indices, , drop = FALSE]
    yoob <- ytr_original[oob_indices]
    yoob_pred <- predict(uniReg_cv, newx = Xoob, s = 0)
    oob_residuals <- yoob - as.vector(yoob_pred)
    sampled_residual <- sample(oob_residuals, size = length(yte), replace = TRUE)
    yte_pred_with_oob <- as.vector(yte_pred_CV) + sampled_residual
  } else {
    yte_pred_with_oob <- as.vector(yte_pred_CV)
  }

  boot_results$unireg_predictions <- yte_pred_with_oob

  }, error = function(e) {
  boot_results$unireg_mse <<- NA
  boot_results$unireg_support <<- NA
  boot_results$unireg_predictions <<- rep(NA, length(yte))
  })

  # 6. Least Squares Regression (no regularization)
  tryCatch({
  # Convert to data frames for lm()
  Xtr_boot_df <- as.data.frame(Xtr_boot)
  Xte_df <- as.data.frame(Xte)

  ls_fit <- lm(ytr_boot ~ ., data = Xtr_boot_df)  # use . to include all variables
  yte_pred_lm <- predict(ls_fit, newdata = Xte_df)
  boot_results$least_squares_mse <- mean((yte_pred_lm - yte)^2)
  boot_results$least_squares_support <- sum(coef(ls_fit)[-1] != 0, na.rm = TRUE)  # exclude intercept

  # Add OOB residuals
  if (length(oob_indices) > 0) {
    Xoob_df <- as.data.frame(Xtr_original[oob_indices, , drop = FALSE])
    yoob <- ytr_original[oob_indices]
    yoob_pred <- predict(ls_fit, newdata = Xoob_df)
    oob_residuals <- yoob - yoob_pred
    sampled_residual <- sample(oob_residuals, size = length(yte), replace = TRUE)
    yte_pred_with_oob <- yte_pred_lm + sampled_residual
  } else {
    yte_pred_with_oob <- yte_pred_lm
  }

  boot_results$least_squares_predictions <- yte_pred_with_oob

  }, error = function(e) {
  boot_results$least_squares_mse <<- NA
  boot_results$least_squares_support <<- NA
  boot_results$least_squares_predictions <<- rep(NA, length(yte))
  })
  
# Return results as a list (not data frame) to handle predictions
return(list(
  summary = data.frame(
    dataset = dataset_name,
    bootstrap_id = bootstrap_id,
    unilasso_loo_true_mse = boot_results$unilasso_loo_true_mse,
    unilasso_loo_true_support = boot_results$unilasso_loo_true_support,
    unilasso_loo_false_mse = boot_results$unilasso_loo_false_mse,
    unilasso_loo_false_support = boot_results$unilasso_loo_false_support,
    polish_unilasso_mse = boot_results$polish_unilasso_mse,
    polish_unilasso_support = boot_results$polish_unilasso_support,
    lasso_cv_mse = boot_results$lasso_cv_mse,
    lasso_cv_support = boot_results$lasso_cv_support,
    unireg_mse = boot_results$unireg_mse,                     # ADD
    unireg_support = boot_results$unireg_support,             # ADD
    least_squares_mse = boot_results$least_squares_mse,       # ADD
    least_squares_support = boot_results$least_squares_support, # ADD
    oob_sample_count = boot_results$unilasso_loo_true_oob_count,
    stringsAsFactors = FALSE
  ),
  predictions = list(
    unilasso_loo_true = boot_results$unilasso_loo_true_predictions,
    unilasso_loo_false = boot_results$unilasso_loo_false_predictions,
    polish_unilasso = boot_results$polish_unilasso_predictions,
    lasso_cv = boot_results$lasso_cv_predictions,
    unireg = boot_results$unireg_predictions,                 # ADD
    least_squares = boot_results$least_squares_predictions    # ADD
  )
))
}

# Update the results storage to include new methods
results_all_bootstrap <- data.frame(
  dataset = character(),
  bootstrap_id = integer(),
  unilasso_loo_true_mse = numeric(),
  unilasso_loo_true_support = numeric(),
  unilasso_loo_false_mse = numeric(),
  unilasso_loo_false_support = numeric(),
  polish_unilasso_mse = numeric(),
  polish_unilasso_support = numeric(),
  lasso_cv_mse = numeric(),
  lasso_cv_support = numeric(),
  unireg_mse = numeric(),                    # ADD
  unireg_support = numeric(),                # ADD
  least_squares_mse = numeric(),             # ADD
  least_squares_support = numeric(),         # ADD
  oob_sample_count = numeric(),
  stringsAsFactors = FALSE
)

# Initialize storage for prediction intervals
all_prediction_intervals <- data.frame()

# Start timing
start_time <- Sys.time()

# Loop through all datasets
for (i in seq_along(dataset_names)) {
  dataset_name <- dataset_names[i]

  # Load data
  X_path <- paste0("/accounts/grad/aqwang/unilasso/datasets_12/", dataset_name, "/X.csv")
  y_path <- paste0("/accounts/grad/aqwang/unilasso/datasets_12/", dataset_name, "/y.csv")
  
  X <- read.csv(X_path)
  X <- as.matrix(X)
  y <- read.csv(y_path, header = FALSE) # y has no column names, so the first row is read as the data 
  y <- as.numeric(unlist(y)) # Unlist: Convert the data frame to a vector, which removes the data frame structure and column names.

  # Create SINGLE predefined train/test split (2/3 - 1/3 split with fixed seed)
  set.seed(123)  # Fixed seed for consistent train/test split across all analyses
  training_ratio <- 0.5
  training_size <- floor(training_ratio * nrow(X))
  train_indices <- sample(seq_len(nrow(X)), size = training_size)
  
  Xtr_original <- X[train_indices, ]
  ytr_original <- y[train_indices]
  Xte <- X[-train_indices, ]  # Test set remains FIXED
  yte <- y[-train_indices]

  # Visual separator for output 
  cat("\n", strrep("=", 80), "\n")
  cat("Processing dataset:", dataset_name, "(", i, "/", length(dataset_names), ")\n")
  cat("Running", N_BOOTSTRAP, "bootstrap samples on training set in parallel...\n")
  #training ratio 
  cat("Training ratio:", training_ratio, "\n")
  cat("Test set remains FIXED for all bootstrap samples\n")
  cat(strrep("=", 80), "\n")
  
  cat("Data dimensions: X =", dim(X), ", y =", length(y), "\n")
  
  # Run all bootstrap samples for this dataset in parallel
  dataset_start_time <- Sys.time()
  
  # Use foreach for parallel execution - now returns list of results
  bootstrap_results_list <- foreach(bootstrap_id = 1:N_BOOTSTRAP, 
                                   .packages = c("uniLasso", "glmnet")) %dopar% {
    run_single_bootstrap(bootstrap_id, Xtr_original, ytr_original, Xte, yte, dataset_name)
  }
  
  dataset_end_time <- Sys.time()
  cat("Dataset completed in:", format(dataset_end_time - dataset_start_time), "\n")
  
  # Extract summary results
  dataset_results <- do.call(rbind, lapply(bootstrap_results_list, function(x) x$summary))
  
  # Add to overall results
  results_all_bootstrap <- rbind(results_all_bootstrap, dataset_results)
  
  # Extract prediction matrices for interval computation (ADD 2 new matrices)
  n_test <- length(yte)
  pred_unilasso_loo_true <- matrix(NA, nrow = n_test, ncol = N_BOOTSTRAP)
  pred_unilasso_loo_false <- matrix(NA, nrow = n_test, ncol = N_BOOTSTRAP)
  pred_polish_unilasso <- matrix(NA, nrow = n_test, ncol = N_BOOTSTRAP)
  pred_lasso_cv <- matrix(NA, nrow = n_test, ncol = N_BOOTSTRAP)
  pred_unireg <- matrix(NA, nrow = n_test, ncol = N_BOOTSTRAP)           # ADD
  pred_least_squares <- matrix(NA, nrow = n_test, ncol = N_BOOTSTRAP)    # ADD

  
  for (j in 1:N_BOOTSTRAP) {
    pred_unilasso_loo_true[, j] <- bootstrap_results_list[[j]]$predictions$unilasso_loo_true
    pred_unilasso_loo_false[, j] <- bootstrap_results_list[[j]]$predictions$unilasso_loo_false
    pred_polish_unilasso[, j] <- bootstrap_results_list[[j]]$predictions$polish_unilasso
    pred_lasso_cv[, j] <- bootstrap_results_list[[j]]$predictions$lasso_cv
    pred_unireg[, j] <- bootstrap_results_list[[j]]$predictions$unireg               # ADD
    pred_least_squares[, j] <- bootstrap_results_list[[j]]$predictions$least_squares # ADD
  }
  
  # Compute prediction intervals (95% confidence)
  cat("Computing prediction intervals...\n")
  
  # Function to compute intervals and coverage for one method
  compute_intervals <- function(pred_matrix, method_name, confidence_level = 0.95) {
    alpha <- 1 - confidence_level
    lower_q <- alpha / 2
    upper_q <- 1 - alpha / 2
    
    n_valid_boots <- rowSums(!is.na(pred_matrix)) # count nonmissing predictions per test point 
    valid_points <- n_valid_boots >= max(10, N_BOOTSTRAP * 0.5)
    
    if (sum(valid_points) == 0) {
      return(data.frame(
        dataset = dataset_name,
        method = method_name,
        coverage = NA,
        avg_interval_width = NA,
        median_interval_width = NA,
        sd_interval_width = NA,  # Add SD of interval widths
        valid_test_points = 0,
        total_test_points = n_test
      ))
    }
    
    # Compute quantiles for each test point
    lower_bounds <- rep(NA, n_test)
    upper_bounds <- rep(NA, n_test)
    
    for (k in 1:n_test) {
      if (valid_points[k]) {
        valid_preds <- pred_matrix[k, !is.na(pred_matrix[k, ])] # select the predictions for one test data point, only consider non-missing predictions
        lower_bounds[k] <- quantile(valid_preds, lower_q, na.rm = TRUE)
        upper_bounds[k] <- quantile(valid_preds, upper_q, na.rm = TRUE)
      }
    }
    
    # Compute coverage and interval widths
    in_interval <- (yte >= lower_bounds & yte <= upper_bounds) #  a logical vector indicating whether each true test response falls within its computed prediction interval:
    coverage <- mean(in_interval[valid_points], na.rm = TRUE) # Coverage is the fraction of valid test points where the true response falls within the prediction interval:
    
    interval_widths <- upper_bounds - lower_bounds
    avg_width <- mean(interval_widths[valid_points], na.rm = TRUE)
    median_width <- median(interval_widths[valid_points], na.rm = TRUE)
    sd_width <- sd(interval_widths[valid_points], na.rm = TRUE)  # Add standard deviation of interval widths
    
    return(data.frame(
      dataset = dataset_name,
      method = method_name,
      coverage = coverage, # proportion of valid test points covered
      avg_interval_width = avg_width, # average width of intervals across valid test points
      median_interval_width = median_width, # median width of intervals across valid test points
      sd_interval_width = sd_width, # standard deviation of interval widths across valid test points
      valid_test_points = sum(valid_points), # number of test points with valid intervals
      total_test_points = n_test
    ))
  }
  
  # Compute intervals for all methods (ADD 2 new methods)
  intervals_this_dataset <- rbind(
    compute_intervals(pred_unilasso_loo_true, "uniLasso_loo_true"),
    compute_intervals(pred_unilasso_loo_false, "uniLasso_loo_false"),
    compute_intervals(pred_polish_unilasso, "polish_uniLasso"),
    compute_intervals(pred_lasso_cv, "lasso_cv"),
    compute_intervals(pred_unireg, "uniReg"),                    # ADD
    compute_intervals(pred_least_squares, "least_squares")      # ADD
  )
    
  # Add to overall prediction intervals
  all_prediction_intervals <- rbind(all_prediction_intervals, intervals_this_dataset)
  
  # Summary statistics for this dataset
  cat("\nBootstrap summary statistics for", dataset_name, "(", N_BOOTSTRAP, "bootstrap samples):\n")
  cat("  UniLasso (LOO=T):     Mean MSE =", sprintf("%.6f", mean(dataset_results$unilasso_loo_true_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$unilasso_loo_true_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$unilasso_loo_true_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$unilasso_loo_true_support, na.rm = TRUE)), ")\n")
  
  cat("  UniLasso (LOO=F):     Mean MSE =", sprintf("%.6f", mean(dataset_results$unilasso_loo_false_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$unilasso_loo_false_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$unilasso_loo_false_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$unilasso_loo_false_support, na.rm = TRUE)), ")\n")
  
  cat("  Polish UniLasso:      Mean MSE =", sprintf("%.6f", mean(dataset_results$polish_unilasso_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$polish_unilasso_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$polish_unilasso_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$polish_unilasso_support, na.rm = TRUE)), ")\n")
  
  cat("  Lasso CV:             Mean MSE =", sprintf("%.6f", mean(dataset_results$lasso_cv_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$lasso_cv_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$lasso_cv_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$lasso_cv_support, na.rm = TRUE)), ")\n")

    # Add to existing summary statistics
  cat("  UniReg:               Mean MSE =", sprintf("%.6f", mean(dataset_results$unireg_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$unireg_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$unireg_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$unireg_support, na.rm = TRUE)), ")\n")

  cat("  Least Squares:        Mean MSE =", sprintf("%.6f", mean(dataset_results$least_squares_mse, na.rm = TRUE)), 
      " (SD =", sprintf("%.6f", sd(dataset_results$least_squares_mse, na.rm = TRUE)), ")\n")
  cat("                        Mean Support =", sprintf("%.2f", mean(dataset_results$least_squares_support, na.rm = TRUE)), 
      " (SD =", sprintf("%.2f", sd(dataset_results$least_squares_support, na.rm = TRUE)), ")\n")
  
# Print prediction interval results for this dataset
cat("\n  Prediction Interval Summary (95% confidence):\n")
for (k in 1:nrow(intervals_this_dataset)) {
  row <- intervals_this_dataset[k, ]
  cat(sprintf("    %-20s: Coverage = %.3f, Avg Width = %.4f (SD = %.4f)\n",
              row$method, row$coverage, row$avg_interval_width, row$sd_interval_width))
}
}

# Stop timing
end_time <- Sys.time()
total_time <- end_time - start_time

# Stop the cluster
stopCluster(cl)

cat("\n", strrep("=", 80), "\n")
cat("BOOTSTRAP ANALYSIS WITH PREDICTION INTERVALS COMPLETE FOR ALL", length(dataset_names), "DATASETS!\n")
cat("Total number of bootstrap experiments:", nrow(results_all_bootstrap), "\n")
cat("Total time elapsed:", format(total_time), "\n")
cat("Average time per dataset:", format(total_time / length(dataset_names)), "\n")
cat(strrep("=", 80), "\n")

# Save all individual results
filename <- "unilasso_12data_100bootstrap.csv"
write.csv(results_all_bootstrap, filename, row.names = FALSE)
cat("All bootstrap results saved to:", filename, "\n")

# Create summary statistics for each dataset
library(dplyr)

# Save prediction interval results
write.csv(all_prediction_intervals, "unilasso_12data_100bootstrap_pred_int.csv", row.names = FALSE)
cat("Prediction interval results saved to: unilasso_12data_100bootstrap_pred_int.csv\n")
