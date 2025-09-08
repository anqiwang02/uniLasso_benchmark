For all things involved ESCV
1) data preparation: need to make sure the data is centered (has a mean of zero) because the function makes intercept=FALSE assumption
2) for ESCV and lasso: load the ESCV function from the ESCV_code2/ESCV_glmnet_vf.R file the ESCV author made
   - this file also has the lasso(cv) version, so we don't have to use the cv function from glmnet package separately 
3) for ESCV and uniLasso: load the ESCV function from the escv_unilasso_new.R file I created (combine escv and unilasso functions)


Note:
- We need to write a escv function for uniLasso 