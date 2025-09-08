# Script to extract and save coefficients only from UniLasso analysis
# Uses identical seeds to main analysis script (unilasso_17data_loop_basic.R) for exact reproducibility

# set working directory
setwd("/accounts/grad/aqwang/unilasso/analysis")

if (R.home() != "/scratch/users/aqwang/conda/envs/r_package/lib/R") {
  system("/scratch/users/aqwang/conda/envs/r_package/bin/Rscript unilasso_17data_loop_coef.R")
  quit("no")
}

# check the R environment is r_package
R.home()

# ----------------------------
# Libraries
# ----------------------------
library(uniLasso) # for uniLasso / uniReg / polish
library(glmnet)   # for lasso baseline
library(parallel)
library(doParallel)
library(foreach)

# ----------------------------
# Parallel setup
# ----------------------------
n_cores <- 32
cat("Setting up parallel processing with", n_cores, "cores\n")
cl <- makeCluster(n_cores)
registerDoParallel(cl)
on.exit({
  try(stopCluster(cl), silent = TRUE)
}, add = TRUE)

# Export necessary packages on workers
clusterEvalQ(cl, {
  library(uniLasso)
  library(glmnet)
})

# ----------------------------
# Datasets & config
# ----------------------------
dataset_names <- c(
  "airfoil", "ca_housing", "computer", "concrete",
  "debutanizer", "diamond", "elevator", "energy_efficiency",
  "insurance", "kin8nm", "miami_housing", "naval_propulsion",
  "parkinsons", "powerplant", "protein_structure", "qsar",
  "superconductor"
)

N_SPLITS <- 100
TRAINING_RATIO <- 0.5
K_FOLDS <- 10

# Output dirs
coef_csv_dir <- "coef_csv"
dir.create(coef_csv_dir, showWarnings = FALSE, recursive = TRUE)

# ----------------------------
# Per-split extractor
# ----------------------------
extract_coefficients_single_split <- function(split_id, X, y, dataset_name) {
  # IDENTICAL data splitting as main script
  set.seed(split_id)
  training_size <- floor(TRAINING_RATIO * nrow(X))
  train_indices <- sample(seq_len(nrow(X)), size = training_size)
  Xtr <- X[train_indices, , drop = FALSE]
  ytr <- y[train_indices]

  # IDENTICAL CV folds as main script
  set.seed(42 + split_id)
  n_tr <- nrow(Xtr)
  foldid <- sample(rep(1:K_FOLDS, length.out = n_tr))

  # Initialize coefficient storage for this split
  split_coeffs <- list()

  # 0) Univariate LOO slopes (p x n_tr) -> collapse to length p
  tryCatch({
    uni_loo_mat <- uniInfo(Xtr, ytr, family = "gaussian", loo = TRUE)$beta  # p x n_tr
    split_coeffs$uni_loo <- as.numeric(uni_loo_mat)            # length p
  }, error = function(e) {
    split_coeffs$uni_loo <- rep(0, ncol(Xtr))
  })

  # 1) UniLasso LOO=TRUE
  tryCatch({
    fit <- cv.uniLasso(
      Xtr, ytr,
      family = "gaussian",
      loo = TRUE,
      lower.limits = 0,
      standardize = FALSE,
      foldid = foldid,
      nlambda = 100
    )
    split_coeffs$unilasso_loo_true <- as.numeric(coef(fit, s = "lambda.min")[-1])
  }, error = function(e) {
    split_coeffs$unilasso_loo_true <- rep(NA_real_, ncol(Xtr))
  })

  # 2) UniLasso LOO=FALSE
  tryCatch({
    fit <- cv.uniLasso(
      Xtr, ytr,
      family = "gaussian",
      loo = FALSE,
      lower.limits = 0,
      standardize = FALSE,
      foldid = foldid,
      nlambda = 100
    )
    split_coeffs$unilasso_loo_false <- as.numeric(coef(fit, s = "lambda.min")[-1])
  }, error = function(e) {
    split_coeffs$unilasso_loo_false <- rep(NA_real_, ncol(Xtr))
  })

  # 3) Polish UniLasso
  tryCatch({
    fit <- polish.uniLasso(
      Xtr, ytr,
      family = "gaussian",
      foldid = foldid,
      nlambda = 100,
      standardize = FALSE,
      intercept = TRUE,
      loo = TRUE
    )
    split_coeffs$polish_unilasso <- as.numeric(coef(fit, s = "lambda.min")[-1])
  }, error = function(e) {
    split_coeffs$polish_unilasso <- rep(NA_real_, ncol(Xtr))
  })

  # 4) Lasso (CV)
  tryCatch({
    fit <- cv.glmnet(
      Xtr, ytr,
      family = "gaussian",
      alpha = 1,
      standardize = FALSE,
      foldid = foldid,
      nlambda = 100
    )
    split_coeffs$lasso_cv <- as.numeric(coef(fit, s = "lambda.min")[-1])
  }, error = function(e) {
    split_coeffs$lasso_cv <- rep(NA_real_, ncol(Xtr))
  })

  # 5) UniReg (CV)
  tryCatch({
    fit <- cv.uniReg(
      Xtr, ytr,
      family = "gaussian",
      standardize = FALSE,
      foldid = foldid,
      nlambda = 100,
      loo = TRUE
    )
    split_coeffs$unireg <- as.numeric(coef(fit, s = "lambda.min")[-1])
  }, error = function(e) {
    split_coeffs$unireg <- rep(NA_real_, ncol(Xtr))
  })

  # 6) Least Squares
  tryCatch({
    Xtr_df <- as.data.frame(Xtr)
    df_train <- data.frame(y = ytr, Xtr_df)
    ls_fit <- lm(y ~ ., data = df_train)
    split_coeffs$least_squares <- as.numeric(coef(ls_fit)[-1])  # drop intercept
  }, error = function(e) {
    split_coeffs$least_squares <- rep(NA_real_, ncol(Xtr))
  })

  split_coeffs
}

# ----------------------------
# Driver loop
# ----------------------------
all_datasets_coefficients <- list()
combined_coeffs <- list()  # list of data.frames to rbind at end

start_time <- Sys.time()

for (i in seq_along(dataset_names)) {
  dataset_name <- dataset_names[i]

  cat("\n", strrep("=", 80), "\n")
  cat("Extracting coefficients for dataset:", dataset_name, "(", i, "/", length(dataset_names), ")\n")
  cat("Running", N_SPLITS, "random train/test splits in parallel...\n")
  cat(strrep("=", 80), "\n")

  # Load data
  X_path <- paste0("/accounts/grad/aqwang/unilasso/datasets_17/", dataset_name, "/X.csv")
  y_path <- paste0("/accounts/grad/aqwang/unilasso/datasets_17/", dataset_name, "/y.csv")

  X <- as.matrix(read.csv(X_path))
  y <- as.numeric(unlist(read.csv(y_path, header = FALSE)))

  n_features <- ncol(X)
  feat_names <- colnames(X)
  if (is.null(feat_names) || length(feat_names) != n_features) {
    feat_names <- paste0("beta_", seq_len(n_features))
  }

  cat("Data dimensions: X =", paste(dim(X), collapse = "x"), ", y =", length(y), "\n")

  dataset_start_time <- Sys.time()

  # ---- foreach with a proper combiner (flat list of length N_SPLITS)
  coeff_results <- foreach(
    split_id = 1:N_SPLITS,
    .combine = 'c',              # concatenate lists
    .multicombine = TRUE,
    .maxcombine = N_SPLITS,
    .packages = c("uniLasso", "glmnet"),
    .export = c("extract_coefficients_single_split", "TRAINING_RATIO", "K_FOLDS")
  ) %dopar% {
    list(extract_coefficients_single_split(split_id, X, y, dataset_name))
  }

  stopifnot(length(coeff_results) == N_SPLITS)

  dataset_end_time <- Sys.time()
  cat("Dataset completed in:", format(dataset_end_time - dataset_start_time), "\n")

  # Methods in desired order
  methods <- c("uni_loo", "unilasso_loo_true", "unilasso_loo_false",
               "polish_unilasso", "lasso_cv", "unireg", "least_squares")

  # Build stacked coefficient frame: rows = N_SPLITS per method
  stacked_coeffs <- NULL
  method_labels <- character(0)
  split_labels <- integer(0)

  for (method in methods) {
    method_mat <- matrix(NA_real_, nrow = N_SPLITS, ncol = n_features)
    for (split_id in seq_len(N_SPLITS)) {
      vec <- coeff_results[[split_id]][[method]]
      if (!is.null(vec)) {
        # Ensure length matches n_features
        if (length(vec) != n_features) {
          # pad or truncate defensively (shouldn't happen after fixes)
          len <- min(length(vec), n_features)
          method_mat[split_id, seq_len(len)] <- vec[seq_len(len)]
        } else {
          method_mat[split_id, ] <- vec
        }
      }
    }
    stacked_coeffs <- rbind(stacked_coeffs, method_mat)
    method_labels <- c(method_labels, rep(method, N_SPLITS))
    split_labels  <- c(split_labels, 1:N_SPLITS)
  }

  # Create final data.frame for this dataset
  final_coeffs <- data.frame(
    dataset = dataset_name,
    method  = method_labels,
    split_id = split_labels,
    stringsAsFactors = FALSE
  )
  colnames(stacked_coeffs) <- feat_names
  final_coeffs <- cbind(final_coeffs, as.data.frame(stacked_coeffs, check.names = FALSE))

  # Save per-dataset CSV
  per_ds_csv <- file.path(coef_csv_dir, sprintf("coefficients_%s.csv", dataset_name))
  write.csv(final_coeffs, per_ds_csv, row.names = FALSE)
  cat("Saved per-dataset coefficients to:", per_ds_csv, "\n")

  # Keep in memory structures
  all_datasets_coefficients[[dataset_name]] <- final_coeffs
  combined_coeffs[[dataset_name]] <- final_coeffs
  cat("Coefficients stored - Shape:", dim(final_coeffs), "\n")
}

# ----------------------------
# Save combined outputs
# ----------------------------
end_time <- Sys.time()
total_time <- end_time - start_time

# DON'T combine datasets with different feature counts into one CSV
# Instead, save separate files and a summary
cat("\n", strrep("=", 80), "\n")
cat("Creating summary of coefficient data...\n")

# Create summary of datasets
dataset_summary <- data.frame(
  dataset = character(),
  n_features = integer(),
  n_rows = integer(),
  stringsAsFactors = FALSE
)

for (ds_name in names(all_datasets_coefficients)) {
  ds_data <- all_datasets_coefficients[[ds_name]]
  n_features <- sum(grepl("^beta_|^V\\d+", names(ds_data)))  # count coefficient columns
  dataset_summary <- rbind(dataset_summary, data.frame(
    dataset = ds_name,
    n_features = n_features,
    n_rows = nrow(ds_data),
    stringsAsFactors = FALSE
  ))
}

# Save summary
write.csv(dataset_summary, "dataset_coefficient_summary.csv", row.names = FALSE)

# RData (list by dataset) - this works fine
save(all_datasets_coefficients, file = "unilasso_17datasets_coefficients_only.RData")

cat("COEFFICIENT EXTRACTION COMPLETE!\n")
cat("Per-dataset CSVs in:", normalizePath(coef_csv_dir), "\n")
cat("Dataset summary saved to: dataset_coefficient_summary.csv\n")
cat("RData saved to:", normalizePath("unilasso_17datasets_coefficients_only.RData"), "\n")

# Print summary
cat("\nDataset Summary:\n")
print(dataset_summary)
cat("Total datasets processed:", nrow(dataset_summary), "\n")
cat("Total time elapsed:", format(total_time), "\n")
cat("Average time per dataset:", format(total_time / length(dataset_names)), "\n")

# Example usage instructions
cat("\n", strrep("=", 80), "\n")
cat("USAGE EXAMPLES:\n")
cat('# Load all coefficients:\n')
cat('load("unilasso_17datasets_coefficients_only.RData")\n')
cat('\n# Access coefficients for a specific dataset:\n')
cat('airfoil_coeffs <- all_datasets_coefficients[["airfoil"]]\n')
cat('\n# Get all UniLasso LOO=TRUE coefficients for airfoil:\n')
cat('unilasso_coeffs <- airfoil_coeffs[airfoil_coeffs$method == "unilasso_loo_true", ]\n')
cat('\n# Extract just the coefficient values (excluding metadata):\n')
cat('coeff_cols <- !names(unilasso_coeffs) %in% c("dataset", "method", "split_id")\n')
cat('coeff_matrix <- as.matrix(unilasso_coeffs[, coeff_cols])\n')
cat('# Dimensions: 100 splits x p features\n')
cat(strrep("=", 80), "\n")