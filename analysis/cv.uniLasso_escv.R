# change the cv.uniLasso function to adopt the escv.glmnet function from escv_glmnet_vf.R 
# X, y are centered data

if (R.home() != "/scratch/users/aqwang/conda/envs/r_package/lib/R") {
  system("/scratch/users/aqwang/conda/envs/r_package/bin/Rscript cv.unilasso_escv.R")
  quit("no")
}

library(uniLasso)
library(glmnet)

# set working directory
setwd("/accounts/grad/aqwang/unilasso/analysis")

# need to import escv function from the ESCV file.
source("ESCV_code2/ESCV_glmnet_vf.R")

cv.uniLasso_escv <- function(x, y,
                             family = c("gaussian","binomial","cox"),
                             weights = NULL,
                             loo = TRUE,
                             lower.limits = 0,
                             standardize = FALSE,
                             info = NULL,
                             k = 10,
                             nlambda = 100,
                             ...) {
  this.call <- match.call()
  family    <- match.arg(family)
  if (family != "gaussian")
    warning("This ESCV implementation is tuned for 'gaussian'.")

  stopifnot(is.matrix(x), is.numeric(y), nrow(x) == length(y))

  # ---- Stage 1: uniLasso info (x,y are already centered)
  if (is.null(info)) {
    info <- uniInfo(x, y, family = family, weights = weights, loo = loo)
  } else {
    if (!is.null(info$F))
      warning("Supplied 'info' has an 'F' component; ignoring it and using $beta/$beta0 only.")
    if (is.null(info$beta0)) info$beta0 <- rep(0, length(info$beta))
    loo <- FALSE
  }

  # ---- Stage 2 features (intercept-free path)
  if (loo) {
  xp_raw <- info$F
   } else {
  ones <- rep(1, nrow(x))
  xp_raw <- x * outer(ones, info$beta)  # (drop beta0 in Option A)
    }
  mu_xp <- colMeans(xp_raw)
  xp     <- sweep(xp_raw, 2, mu_xp, "-")
  dimnames(xp) <- dimnames(x)

  # ---- ESCV on xp (assumes escv.glmnet uses intercept=FALSE internally)
  escv <- escv.glmnet(xp, y, k = k, nlambda = nlambda)

  gfit      <- escv$glmnet                 # glmnet full path on xp
  lambdavec <- gfit$lambda
  selindex  <- escv$selindex

  # ---- multiply theta by beta to get gamma: Map theta-path -> gamma-path (γ_j = β_uni_j * θ_j)
  # beta is updated with gamma; no intercept in the final model 
  gfit$beta <- gfit$beta * outer(info$beta, rep(1, length(lambdavec)))
  gfit$a0   <- rep(0, length(lambdavec))   # intercept ~ 0 (x,y centered; intercept=FALSE)

  fit <- list(
    glmnet.fit = gfit,
    info       = info[c("beta0","beta")],
    xp_colmeans = mu_xp,  # column means of the uniLasso features (for centering new data)
    loo        = loo,
    selindex   = selindex,
    call       = this.call
  )
  class(fit) <- c("cv.uniLasso", class(fit))
  fit
}
