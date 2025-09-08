# we made intercept=FALSE assumption in escv.uniLasso just like in ESCV_glmnet_vf.R file
# so we need to make sure the data is centered (has a mean of zero)

if (R.home() != "/scratch/users/aqwang/conda/envs/r_package/lib/R") {
  system("/scratch/users/aqwang/conda/envs/r_package/bin/Rscript escv_unilasso_new.R")
  quit("no")
}

library(uniLasso)
library(glmnet)
# X and y are centered data !!!
escv.uniLasso <- function(X, y,
                          family = c("gaussian","binomial","cox"),
                          k = 10, nlambda = 100,
                          loo = TRUE,
                          lower.limits = 0,
                          standardize = FALSE) {
  stopifnot(is.matrix(X) || inherits(X, "dgCMatrix"))
  family <- match.arg(family)
  y <- as.numeric(y)

  if (!requireNamespace("glmnet", quietly = TRUE))
    stop("Please install 'glmnet'.")

  # ----- Stage 1: build uniLasso features on FULL data (to get a common lambda path)
  # info$F is the NxP matrix of LOO univariate fits when loo=TRUE (or ALO for non-Gaussian)
  # Otherwise we construct the univariate linear predictors on the full data.
  info <- uniInfo(X, y, family = family, loo = loo)
  # the info object has components $beta0 (intercept) and $beta (univariate coefs); so we want to center the predictions to be consistent with intercept=FALSE in escv.uniLasso


# For loo = FALSE: ignore beta0 and use only slopes with the centered Xc.
# For loo = TRUE: the default in uniLasso is to use info$F, which already includes univariate intercepts. To keep intercept = FALSE clean, strip the column means of F so each feature has mean 0
  if (loo) {
    xp_full_raw <- info$F  # N x P matrix of features, This is the N×P matrix of univariate fitted predictors from the first stage of uniLasso when you set loo=TRUE; Each column is the leave-one-out fitted values for one predictor
    mu_F_full   <- colMeans(xp_full_raw)
    xp_full     <- sweep(xp_full_raw, 2, mu_F_full, "-")
    # Centering the columns of xp_full to have mean zero, since we are assuming intercept=FALSE in escv.uniLasso
    # This is important because glmnet with intercept=FALSE assumes that the input features are centered.
    # If we didn’t subtract the column means, then each fold’s univariate fits would include different intercept-like shifts, which would mess up the ESCV stability metric.
  } else {
    ones <- rep(1, nrow(X))
    xp_full <- X * outer(ones, info$beta) # N x P, no +beta0 term since we have intercept=FALSE in ESCV; safe to ignore the beta0 term since we center the features first 
  }
  dimnames(xp_full) <- dimnames(X)

  # Full-data second-stage path: defines the lambda grid everyone will use (this is the full solution path in ESCV, meaning the betas for all lambda values)
  fit_full <- glmnet::glmnet(
    xp_full, y,
    family        = family,
    lower.limits  = lower.limits,
    standardize   = standardize,
    intercept     = FALSE,
    nlambda       = nlambda
  )
  lambdavec  <- fit_full$lambda
  beta_full  <- as.matrix(fit_full$beta)   # P x L (L is length of lambdavec)
  a0_full    <- fit_full$a0                # length L
  L          <- length(lambdavec)
  n          <- nrow(X)
  p          <- ncol(X)

  # ----- k-fold split
  grps <- cut(1:n, k, labels = FALSE)[sample(n)]

  # storage
  pe_mat   <- matrix(0, nrow = L, ncol = k)         # CV error per lambda per fold
  yhat_all <- vector("list", k)                     # fitted values on ALL rows, per fold

  # below is cross validation loop: each loop represents a unilasso fit on k-1 folds, and prediction on the held-out fold ("omit")
  for (i in seq_len(k)) {
    omit <- which(grps == i)

    # Stage 1 on TRAINING rows (to avoid leakage):
    info_i <- uniInfo(X[-omit, , drop = FALSE], y[-omit],
                      family = family, loo = loo)

    # Build *training* and *full* feature matrices using TRAINING coefs
    if (loo) {
      # With loo=TRUE, we want features that don't reuse y; best approximation:
      # use info_i$F for training rows, and extend to all rows with the linear predictors
      # computed from TRAINING univariate coefs (as uniLasso does when loo=FALSE).
      # center the LOO fectures by the traiing column means to be consistent with intercept=FALSE
      xp_tr_raw  <- info_i$F
      mu_F_tr    <- colMeans(xp_tr_raw)
      xp_tr <- sweep(xp_tr_raw, 2, mu_F_tr, "-")
      ones   <- rep(1, n)
      # For all rows, build raw features from training univariate coefs,
      # then subtract the **same** training means:
      xp_all_raw <- X * outer(ones, info_i$beta) + outer(ones, info_i$beta0)   # <- include beta0 here
      xp_all     <- sweep(xp_all_raw, 2, mu_F_tr, "-")
      # Centering the columns of xp_all to have mean zero, since we are assuming intercept=FALSE in escv.uniLasso
      # This is important because glmnet with intercept=FALSE assumes that the input features are centered.
      # If we didn’t subtract the column means, then each fold’s univariate fits would include different intercept-like shifts, which would mess up the ESCV stability metric.
    } else {
      ones_tr <- rep(1, nrow(X) - length(omit))
      xp_tr   <- X[-omit, , drop = FALSE] * outer(ones_tr, info_i$beta)
      ones_all <- rep(1, n)
      xp_all   <- X * outer(ones_all, info_i$beta)
    }

    # Second-stage on TRAINING rows, but *force the same lambda grid*
    # in folds:
    fit_i <- glmnet::glmnet(
      xp_tr, y[-omit],
      family        = family,
      lower.limits  = lower.limits,
      standardize   = standardize,
      intercept     = FALSE,
      lambda        = lambdavec
    )

    # Predictions on the held-out rows for CV error, for ALL lambdas
    yhat_omit <- stats::predict(fit_i, newx = xp_all[omit, , drop = FALSE], s = lambdavec)
    # yhat_omit is |omit| x L
    pe_mat[, i] <- colMeans((matrix(y[omit], nrow = length(omit), ncol = L) - yhat_omit)^2)

    # Fitted values on ALL rows (needed for ES metric)
    yhat_all[[i]] <- stats::predict(fit_i, newx = xp_all, s = lambdavec)  # n x L
  }

  # ----- CV curve (mean over folds)
  CV <- rowMeans(pe_mat)       # length L
  CV.index <- unname(which.min(CV))

  # ----- ES(λ): stability of fitted values across folds
  # stack n x L matrices from each fold
  # yhat_mean: n x L average fitted value across folds
  yhat_mean <- Reduce(`+`, yhat_all) / k

  # numerator: mean over folds of ||yhat_i - yhat_mean||^2  (per column / per lambda)
  es_num <- matrix(0, nrow = k, ncol = L)
  for (i in seq_len(k)) {
    dif <- yhat_all[[i]] - yhat_mean     # n x L
    es_num[i, ] <- colSums(dif * dif)
  }
  ES_num <- colMeans(es_num)             # length L
  ES_den <- colSums(yhat_mean * yhat_mean) + 1e-12   # avoid 0/0
  ES     <- ES_num / ES_den              # length L

  # ----- "convergence" index as in your escv.glmnet
  ESgrad     <- diff(ES)
  con.idx.v  <- which(ESgrad < 0)
  con.index  <- if (length(con.idx.v) == 0) 1 else con.idx.v[1]
  if (con.index > CV.index) con.index <- CV.index

  ESCV.index <- unname(which.min(ES[con.index:CV.index]) + con.index - 1)

  # ----- BIC / EBIC on the FULL path (optional, same as your code)
  l0norm <- function(v) sum(v != 0)
  df     <- apply(beta_full != 0, 2, sum)                 # degrees of freedom (#nonzero thetas)
  # Compute fitted values on full data (from full path) for RSS:
  yhat_full <- stats::predict(fit_full, newx = xp_full, s = lambdavec)  # n x L
  RSS       <- colSums((matrix(y, nrow = n, ncol = L) - yhat_full)^2)

  BIC   <- n * log(RSS) + log(n) * df
  EBIC1 <- n * log(RSS) + log(n) * df + 2 * 0.5 * log(choose(p, pmin(df, p)))
  EBIC2 <- n * log(RSS) + log(n) * df + 2 * 1.0 * log(choose(p, pmin(df, p)))

  BIC.index   <- unname(which.min(BIC))
  EBIC.index1 <- unname(which.min(EBIC1))
  EBIC.index2 <- unname(which.min(EBIC2))

  # ----- return results

  out <- list()
  out$glmnet   <- fit_full      # full second-stage path on uniLasso features
  out$info     <- info          # stage-1 info used to create features
  out$lambda   <- lambdavec
  out$ES       <- ES
  out$CV       <- CV
  out$selindex <- c(ESCV = ESCV.index,
                    CV   = CV.index,
                    BIC  = BIC.index,
                    EBIC1 = EBIC.index1,
                    EBIC2 = EBIC.index2)
  class(out) <- "escv.uniLasso"
  out
}
